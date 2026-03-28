#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for DQN on Pong.
Objective: minimize time (seconds) to reach MEAN_REWARD_BOUND.
"""
import gymnasium as gym
from lib import dqn_model
from lib import wrappers

from dataclasses import dataclass
import time
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from typing import cast

import optuna

torch.set_float32_matmul_precision("high")

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

# Fixed constants (not tuned)
REPLAY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

# Safety: max wall-clock seconds per trial before pruning
MAX_TRIAL_SECONDS = 900  # 15 minutes

State = np.ndarray
Action = int
BatchTensors = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


class Agent:
    def __init__(self, env: gym.vector.VectorEnv, exp_buffer: ExperienceBuffer,
                 n_steps: int, gamma: float, n_envs: int):
        self.env = env
        self.exp_buffer = exp_buffer
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_envs = n_envs
        self.states: np.ndarray | None = None
        self.total_rewards = np.zeros(n_envs)
        self.total_steps = np.zeros(n_envs, dtype=int)
        self.step_buffers: list[collections.deque[Experience]] = [
            collections.deque(maxlen=n_steps) for _ in range(n_envs)
        ]
        self._reset()

    def _reset(self):
        self.states, _ = self.env.reset()
        self.total_rewards = np.zeros(self.n_envs)
        self.total_steps = np.zeros(self.n_envs, dtype=int)
        for buf in self.step_buffers:
            buf.clear()

    def _flush_steps(self, buf: collections.deque[Experience]):
        if not buf:
            return
        reward = 0.0
        for exp in reversed(buf):
            reward = exp.reward + self.gamma * reward
        first = buf[0]
        last = buf[-1]
        self.exp_buffer.append(Experience(
            state=first.state,
            action=first.action,
            reward=reward,
            done_trunc=last.done_trunc,
            new_state=last.new_state,
        ))

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> list[tuple[float, int]]:
        assert self.states is not None
        done_episodes: list[tuple[float, int]] = []

        if np.random.random() < epsilon:
            actions = self.env.action_space.sample()
        else:
            torch.compiler.cudagraph_mark_step_begin()
            states_v = torch.as_tensor(self.states).to(device)
            q_vals_v = net(states_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            actions = act_v.cpu().numpy()

        new_states, rewards, is_done, is_tr, infos = self.env.step(actions)
        self.total_rewards += rewards
        self.total_steps += 1

        for i in range(self.n_envs):
            done_trunc = bool(is_done[i]) or bool(is_tr[i])
            if done_trunc and "final_observation" in infos:
                last_new_state = infos["final_observation"][i]
            else:
                last_new_state = new_states[i]
            exp = Experience(
                state=self.states[i], action=int(actions[i]), reward=float(rewards[i]),
                done_trunc=done_trunc, new_state=last_new_state
            )
            self.step_buffers[i].append(exp)

            if done_trunc:
                self._flush_steps(self.step_buffers[i])
                self.step_buffers[i].clear()
                done_episodes.append((float(self.total_rewards[i]), int(self.total_steps[i])))
                self.total_rewards[i] = 0.0
                self.total_steps[i] = 0
            elif len(self.step_buffers[i]) == self.n_steps:
                self._flush_steps(self.step_buffers[i])
                self.step_buffers[i].popleft()

        self.states = new_states
        return done_episodes


def batch_to_tensors(batch: list[Experience], device: torch.device) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return (states_t.to(device, non_blocking=True),
            actions_t.to(device, non_blocking=True),
            rewards_t.to(device, non_blocking=True),
            dones_t.to(device, non_blocking=True),
            new_states_t.to(device, non_blocking=True))


def calc_loss(batch: list[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device, n_steps: int, gamma: float) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    with torch.autocast(device_type=device.type):
        all_states = torch.cat([states_t, new_states_t])
        all_q = net(all_states)
        q_current, q_next = all_q[:len(batch)], all_q[len(batch):]

        state_action_values = q_current.gather(
            1, actions_t.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            best_actions = q_next.argmax(1, keepdim=True)
            next_state_values = tgt_net(new_states_t).gather(1, best_actions).squeeze(-1)
            next_state_values[dones_t] = 0.0

        expected_state_action_values = next_state_values * (gamma ** n_steps) + rewards_t
        return nn.MSELoss()(state_action_values, expected_state_action_values)


def objective(trial: optuna.Trial) -> float:
    """
    Single Optuna trial: train DQN with sampled hyperparameters and
    return the wall-clock time (seconds) to reach MEAN_REWARD_BOUND.
    """
    device = torch.device("cuda")

    # --- Fixed (cudagraphs requires constant tensor shapes) ---
    batch_size = 64
    n_envs = 8

    # --- Sample hyperparameters ---
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    n_steps = trial.suggest_int("n_steps", 1, 5)
    replay_start_size = trial.suggest_int("replay_start_size", 5000, 20000, step=5000)
    epsilon_decay_last_frame = trial.suggest_int(
        "epsilon_decay_last_frame", 50000, 1200000, step=25000
    )

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | params: gamma={gamma:.4f}, "
          f"lr={learning_rate:.6f}, tau={tau:.4f}, n_steps={n_steps}, "
          f"replay_start={replay_start_size}, "
          f"eps_decay={epsilon_decay_last_frame}")
    print(f"{'='*60}")

    # --- Build environment and networks ---
    env = gym.vector.AsyncVectorEnv(
        [wrappers.make_env_fn(DEFAULT_ENV_NAME) for _ in range(n_envs)])
    assert isinstance(env.single_observation_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Discrete)

    raw_net = dqn_model.DQN(env.single_observation_space.shape,
                            int(env.single_action_space.n)).to(device)
    net = cast(dqn_model.DQN, torch.compile(raw_net, backend="cudagraphs"))
    tgt_net = cast(dqn_model.DQN, torch.compile(
        dqn_model.DQN(env.single_observation_space.shape,
                      int(env.single_action_space.n)).to(device),
        backend="cudagraphs"))

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer, n_steps=n_steps, gamma=gamma, n_envs=n_envs)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scaler = GradScaler("cuda")
    total_rewards: list[float] = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    epsilon = EPSILON_START
    start_ts = ts
    speed = 0.0

    try:
        while True:
            frame_idx += n_envs
            epsilon = max(EPSILON_FINAL,
                          EPSILON_START - frame_idx / epsilon_decay_last_frame)

            episodes = agent.play_step(net, device, epsilon)
            if episodes:
                now = time.time()
                elapsed = now - ts
                if elapsed > 0:
                    speed = (frame_idx - ts_frame) / elapsed
                ts_frame = frame_idx
                ts = now
            for reward, steps in episodes:
                total_rewards.append(reward)
                m_reward = float(np.mean(total_rewards[-100:]))

                elapsed_str = time.strftime(
                    "%H:%M:%S", time.gmtime(time.time() - start_ts))
                print(f"  T{trial.number} {elapsed_str} {frame_idx}: "
                      f"done {len(total_rewards)} games, reward {m_reward:.3f}, "
                      f"eps {epsilon:.2f}, speed {speed:.2f} f/s")

                # Report intermediate value for pruning
                trial.report(m_reward, len(total_rewards))
                if trial.should_prune():
                    print(f"  Trial {trial.number} pruned at episode "
                          f"{len(total_rewards)}, reward={m_reward:.2f}")
                    env.close()
                    raise optuna.TrialPruned()

                if m_reward > MEAN_REWARD_BOUND:
                    solve_time = time.time() - start_ts
                    print(f"  Trial {trial.number} SOLVED in {solve_time:.1f}s "
                          f"({frame_idx} frames, {len(total_rewards)} episodes)")
                    env.close()
                    return solve_time

            # Time-limit guard
            if time.time() - start_ts > MAX_TRIAL_SECONDS:
                print(f"  Trial {trial.number} timed out after {MAX_TRIAL_SECONDS}s, "
                      f"best mean reward={np.mean(total_rewards[-100:]) if total_rewards else -21:.2f}")
                env.close()
                return float(MAX_TRIAL_SECONDS)

            if len(buffer) < replay_start_size:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(batch_size)
            loss_t = calc_loss(batch, net, tgt_net, device,
                               n_steps=n_steps, gamma=gamma)
            scaler.scale(loss_t).backward()
            scaler.step(optimizer)
            scaler.update()

            # Soft update target network
            with torch.no_grad():
                for p, p_tgt in zip(net.parameters(), tgt_net.parameters()):
                    p_tgt.data.mul_(1 - tau).add_(tau * p.data)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        env.close()
        return float(MAX_TRIAL_SECONDS)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="dqn_pong_tuning",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
        ),
        storage="sqlite:///optuna_dqn_pong.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50)

    print("\n" + "=" * 60)
    print("STUDY COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best time to solve: {study.best_trial.value:.1f} seconds")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Print top 5 trials
    print(f"\nTop 5 trials:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value  # type: ignore
    )
    for t in sorted_trials[:5]:
        print(f"  Trial #{t.number}: {t.value:.1f}s | {t.params}")
