#!/usr/bin/env python3
import gymnasium as gym
from lib import dqn_model
from lib import wrappers

from dataclasses import dataclass
import argparse
import time
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from typing import cast

from torch.utils.tensorboard.writer import SummaryWriter


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 100000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

N_ENVS = 8

EPSILON_DECAY_LAST_FRAME = 150000 * N_ENVS
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

State = np.ndarray
Action = int
BatchTensors = tuple[
    torch.Tensor,               # current state
    torch.Tensor,               # actions
    torch.Tensor,               # rewards
    torch.Tensor,               # done || trunc
    torch.Tensor                # next state
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
    def __init__(self, env: gym.vector.VectorEnv, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.states: np.ndarray | None = None
        self.total_rewards = np.zeros(N_ENVS)
        self._reset()

    def _reset(self):
        self.states, _ = self.env.reset()
        self.total_rewards = np.zeros(N_ENVS)

    @torch.no_grad()
    def play_step(self, net: dqn_model.DQN, device: torch.device,
                  epsilon: float = 0.0) -> list[float]:
        assert self.states is not None
        done_rewards: list[float] = []

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

        for i in range(N_ENVS):
            done_trunc = bool(is_done[i]) or bool(is_tr[i])
            # VectorEnv autoreset: new_states[i] is the first obs of the next
            # episode when done. The true final obs is stored in infos.
            if done_trunc and "final_observation" in infos:
                last_new_state = infos["final_observation"][i]
            else:
                last_new_state = new_states[i]
            exp = Experience(
                state=self.states[i], action=int(actions[i]), reward=float(rewards[i]),
                done_trunc=done_trunc, new_state=last_new_state
            )
            self.exp_buffer.append(exp)
            if done_trunc:
                done_rewards.append(float(self.total_rewards[i]))
                self.total_rewards[i] = 0.0

        self.states = new_states
        return done_rewards


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
    return states_t.to(device, non_blocking=True), actions_t.to(device, non_blocking=True), \
           rewards_t.to(device, non_blocking=True), dones_t.to(device, non_blocking=True), \
           new_states_t.to(device, non_blocking=True)


def calc_loss(batch: list[Experience], net: dqn_model.DQN, tgt_net: dqn_model.DQN,
              device: torch.device) -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    with torch.autocast(device_type=device.type):
        state_action_values = net(states_t).gather(
            1, actions_t.unsqueeze(-1)
        ).squeeze(-1)
        with torch.no_grad():
            next_state_values = tgt_net(new_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0

        expected_state_action_values = next_state_values * GAMMA + rewards_t
        return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device name, default=cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device(args.dev)

    env = gym.vector.AsyncVectorEnv(
        [wrappers.make_env_fn(args.env) for _ in range(N_ENVS)])
    assert isinstance(env.single_observation_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Discrete)
    raw_net = dqn_model.DQN(env.single_observation_space.shape, env.single_action_space.n).to(device)
    net = cast(dqn_model.DQN, torch.compile(raw_net, backend="cudagraphs"))
    tgt_net = cast(dqn_model.DQN, torch.compile(
        dqn_model.DQN(env.single_observation_space.shape, env.single_action_space.n).to(device),
        backend="cudagraphs"))
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler("cuda")
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_ts = ts
    best_m_reward = None
    last_sync = 0
    solved = False
    speed = 0.0

    while not solved:
        frame_idx += N_ENVS
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        rewards = agent.play_step(net, device, epsilon)
        if rewards:
            now = time.time()
            elapsed = now - ts
            if elapsed > 0:
                speed = (frame_idx - ts_frame) / elapsed
            ts_frame = frame_idx
            ts = now
        for reward in rewards:
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_ts))
            print(f"{elapsed} {frame_idx}: done {len(total_rewards)} games, "
                  f"reward {m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s")
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(raw_net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                solved = True
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx - last_sync >= SYNC_TARGET_FRAMES:
            tgt_net.load_state_dict(net.state_dict())
            last_sync = frame_idx

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        scaler.scale(loss_t).backward()
        scaler.step(optimizer)
        scaler.update()
    writer.close()
