#!/usr/bin/env python3
"""
Rainbow DQN — all 6 components in one file:
  1. Double DQN
  2. Multi-step returns (n-step)
  3. NoisyNet (factorized Gaussian noise)
  4. Dueling architecture
  5. Distributional RL (C51)
  6. Prioritized Experience Replay (SumTree)

Reuses lib/wrappers.py for Atari preprocessing.
"""
import gymnasium as gym
from lib import wrappers

from dataclasses import dataclass
import argparse
import math
import time
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import cast

from torch.utils.tensorboard.writer import SummaryWriter


# ── Hyperparameters ──────────────────────────────────────────────────────────
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_SIZE = 100000
LEARNING_RATE = 1e-4
TAU = 0.005
REPLAY_START_SIZE = 10000
N_STEPS = 3
N_ENVS = 8

# C51 distributional
N_ATOMS = 51
V_MIN = -10.0
V_MAX = 10.0

# Prioritized replay
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 1_000_000

# NoisyNet
SIGMA_INIT = 0.5

State = np.ndarray
Action = int


# ── NoisyLinear ──────────────────────────────────────────────────────────────
class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer."""
    weight_epsilon: torch.Tensor
    bias_epsilon: torch.Tensor

    def __init__(self, in_features: int, out_features: int, sigma_init: float = SIGMA_INIT):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        mu_range = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(in_features))

        self.reset_noise()

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ── Dueling Distributional DQN ───────────────────────────────────────────────
class DuelingDistributionalDQN(nn.Module):
    support: torch.Tensor

    def __init__(self, input_shape, n_actions: int,
                 n_atoms: int = N_ATOMS, v_min: float = V_MIN, v_max: float = V_MAX):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer(
            "support", torch.linspace(v_min, v_max, n_atoms))

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out = self.conv(torch.zeros(1, *input_shape)).size(-1)

        # Value stream
        self.val_fc1 = NoisyLinear(conv_out, 512)
        self.val_fc2 = NoisyLinear(512, n_atoms)

        # Advantage stream
        self.adv_fc1 = NoisyLinear(conv_out, 512)
        self.adv_fc2 = NoisyLinear(512, n_actions * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities: (batch, n_actions, n_atoms)."""
        xx = x / 255.0
        features = self.conv(xx)

        val = F.relu(self.val_fc1(features))
        val = self.val_fc2(val).unsqueeze(1)           # (B, 1, n_atoms)

        adv = F.relu(self.adv_fc1(features))
        adv = self.adv_fc2(adv).view(-1, self.n_actions, self.n_atoms)  # (B, A, n_atoms)

        # Dueling combination in logit space, then log_softmax over atoms
        logits = val + adv - adv.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Expected Q-values for action selection: (batch, n_actions)."""
        log_probs = self(x)
        probs = log_probs.exp()
        return (probs * self.support).sum(dim=2)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ── SumTree ──────────────────────────────────────────────────────────────────
class SumTree:
    """Binary tree for O(log N) prioritized sampling."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ── Prioritized Replay Buffer ───────────────────────────────────────────────
@dataclass
class Experience:
    state: State
    action: int
    reward: float
    done_trunc: bool
    new_state: State


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0
        self._capacity = capacity

    def __len__(self):
        return self.tree.size

    def append(self, experience: Experience):
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size: int, beta: float
               ) -> tuple[list[Experience], np.ndarray, np.ndarray]:
        """Returns (experiences, tree_indices, IS_weights)."""
        experiences = []
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, prio, data = self.tree.get(s)
            if data is None:
                # Fallback: if we hit an empty slot, resample from full range
                s = np.random.uniform(0, total)
                idx, prio, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = prio
            experiences.append(data)

        # Importance-sampling weights
        probs = priorities / total
        weights = (self.tree.size * probs) ** (-beta)
        weights /= weights.max()
        return experiences, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        for idx, prio in zip(indices, priorities):
            self.tree.update(int(idx), float(prio))
        self.max_priority = max(self.max_priority, float(priorities.max()))


# ── Agent ────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, env: gym.vector.VectorEnv, exp_buffer: PrioritizedReplayBuffer,
                 n_steps: int = N_STEPS, gamma: float = GAMMA):
        self.env = env
        self.exp_buffer = exp_buffer
        self.n_steps = n_steps
        self.gamma = gamma
        self.states: np.ndarray | None = None
        self.total_rewards = np.zeros(N_ENVS)
        self.step_buffers: list[collections.deque[Experience]] = [
            collections.deque(maxlen=n_steps) for _ in range(N_ENVS)
        ]
        self._reset()

    def _reset(self):
        self.states, _ = self.env.reset()
        self.total_rewards = np.zeros(N_ENVS)
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
            state=first.state, action=first.action, reward=reward,
            done_trunc=last.done_trunc, new_state=last.new_state,
        ))

    @torch.no_grad()
    def play_step(self, net: DuelingDistributionalDQN,
                  device: torch.device) -> list[float]:
        assert self.states is not None
        done_rewards: list[float] = []

        # NoisyNet exploration — no epsilon needed
        net.reset_noise()
        states_v = torch.as_tensor(self.states).to(device)
        q_vals = net.q_values(states_v)
        actions = q_vals.argmax(dim=1).cpu().numpy()

        new_states, rewards, is_done, is_tr, infos = self.env.step(actions)
        self.total_rewards += rewards

        for i in range(N_ENVS):
            done_trunc = bool(is_done[i]) or bool(is_tr[i])
            if done_trunc and "final_observation" in infos:
                last_new_state = infos["final_observation"][i]
            else:
                last_new_state = new_states[i]
            exp = Experience(
                state=self.states[i], action=int(actions[i]),
                reward=float(rewards[i]),
                done_trunc=done_trunc, new_state=last_new_state,
            )
            self.step_buffers[i].append(exp)

            if done_trunc:
                self._flush_steps(self.step_buffers[i])
                self.step_buffers[i].clear()
                done_rewards.append(float(self.total_rewards[i]))
                self.total_rewards[i] = 0.0
            elif len(self.step_buffers[i]) == self.n_steps:
                self._flush_steps(self.step_buffers[i])
                self.step_buffers[i].popleft()

        self.states = new_states
        return done_rewards


# ── Batch conversion ─────────────────────────────────────────────────────────
def batch_to_tensors(batch: list[Experience], device: torch.device):
    states, actions, rewards, dones, new_states = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_states.append(e.new_state)
    return (
        torch.as_tensor(np.asarray(states)).to(device, non_blocking=True),
        torch.LongTensor(actions).to(device, non_blocking=True),
        torch.FloatTensor(rewards).to(device, non_blocking=True),
        torch.BoolTensor(dones).to(device, non_blocking=True),
        torch.as_tensor(np.asarray(new_states)).to(device, non_blocking=True),
    )


# ── C51 Distributional Loss ─────────────────────────────────────────────────
def calc_loss(batch: list[Experience],
              net: DuelingDistributionalDQN,
              tgt_net: DuelingDistributionalDQN,
              device: torch.device,
              is_weights: np.ndarray,
              n_steps: int = N_STEPS,
              gamma: float = GAMMA) -> tuple[torch.Tensor, np.ndarray]:
    """
    C51 categorical distributional loss with Double DQN action selection.
    Returns (weighted_loss, per_sample_td_errors_for_PER).
    """
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)
    batch_size = len(batch)

    support = net.support                    # (n_atoms,)
    n_atoms = net.n_atoms
    delta_z = (V_MAX - V_MIN) / (n_atoms - 1)

    # Current distribution: log p(s, a)
    log_probs = net(states_t)                              # (B, A, n_atoms)
    log_probs_a = log_probs[range(batch_size), actions_t]  # (B, n_atoms)

    with torch.no_grad():
        # Double DQN: online net selects best action for next state
        next_q = net.q_values(new_states_t)                # (B, A)
        best_actions = next_q.argmax(dim=1)                # (B,)

        # Target net provides the distribution for that action
        tgt_log_probs = tgt_net(new_states_t)
        tgt_probs_a = tgt_log_probs[range(batch_size), best_actions].exp()  # (B, n_atoms)

        # Categorical projection of Tᵢ = r + γⁿ zⱼ onto the support
        Tz = rewards_t.unsqueeze(1) + (gamma ** n_steps) * support.unsqueeze(0)
        Tz[dones_t] = rewards_t[dones_t].unsqueeze(1)
        Tz = Tz.clamp(V_MIN, V_MAX)

        b = (Tz - V_MIN) / delta_z                        # (B, n_atoms)
        lo = b.floor().long()
        up = b.ceil().long()
        # Handle edge case where lo == up
        lo[(up > 0) & (lo == up)] -= 1
        up[(lo < (n_atoms - 1)) & (lo == up)] += 1

        # Distribute probability
        target_probs = torch.zeros_like(tgt_probs_a)
        offset = torch.arange(batch_size, device=device).unsqueeze(1) * n_atoms
        target_probs.view(-1).index_add_(
            0, (lo + offset).view(-1), (tgt_probs_a * (up.float() - b)).view(-1))
        target_probs.view(-1).index_add_(
            0, (up + offset).view(-1), (tgt_probs_a * (b - lo.float())).view(-1))

    # Cross-entropy loss per sample
    per_sample_loss = -(target_probs * log_probs_a).sum(dim=1)  # (B,)

    # Importance-sampling weighted loss
    weights_t = torch.as_tensor(is_weights, device=device)
    loss = (per_sample_loss * weights_t).mean()

    td_errors = per_sample_loss.detach().cpu().numpy()
    return loss, td_errors


# ── Main training loop ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device name, default=cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device(args.dev)
    torch.set_float32_matmul_precision("high")

    env = gym.vector.AsyncVectorEnv(
        [wrappers.make_env_fn(args.env) for _ in range(N_ENVS)])
    assert isinstance(env.single_observation_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Discrete)

    obs_shape = env.single_observation_space.shape
    n_actions = int(env.single_action_space.n)

    raw_net = DuelingDistributionalDQN(obs_shape, n_actions).to(device)
    net = cast(DuelingDistributionalDQN, torch.compile(raw_net, backend="inductor"))
    tgt_net = cast(DuelingDistributionalDQN, torch.compile(
        DuelingDistributionalDQN(obs_shape, n_actions).to(device),
        backend="inductor"))
    writer = SummaryWriter(comment="-" + args.env + "-rainbow")
    print(raw_net)

    buffer = PrioritizedReplayBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(raw_net.parameters(), lr=LEARNING_RATE)
    total_rewards: list[float] = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    start_ts = ts
    best_m_reward: float | None = None
    solved = False
    speed = 0.0

    while not solved:
        frame_idx += N_ENVS
        beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

        rewards = agent.play_step(net, device)
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
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_ts))
            print(f"{elapsed_str} {frame_idx}: done {len(total_rewards)} games, "
                  f"reward {m_reward:.3f}, speed {speed:.2f} f/s")
            writer.add_scalar("beta", beta, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(raw_net.state_dict(), args.env + "-rainbow-best.dat")
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = float(m_reward)
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                solved = True
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        # Reset noise for training forward passes
        raw_net.reset_noise()

        optimizer.zero_grad()
        batch, tree_indices, is_weights = buffer.sample(BATCH_SIZE, beta)
        loss_t, td_errors = calc_loss(batch, net, tgt_net, device, is_weights)
        loss_t.backward()
        optimizer.step()

        buffer.update_priorities(tree_indices, td_errors)
        writer.add_scalar("loss", loss_t.item(), frame_idx)

        # Polyak averaging: soft update target network
        with torch.no_grad():
            for p, p_tgt in zip(raw_net.parameters(), tgt_net.parameters()):
                p_tgt.data.mul_(1 - TAU).add_(TAU * p.data)
    writer.close()
