#!/usr/bin/env python3
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter

#ENV_NAME = "FrozenLake-v1"
ENV_NAME = "FrozenLake8x8-v1"      # uncomment for larger version
GAMMA = 0.9
TEST_EPISODES = 20

State = int
Action = int
RewardKey = tuple[State, Action, State]
TransitKey = tuple[State, Action]


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        assert isinstance(self.env.observation_space, gym.spaces.Discrete)
        self.n_actions: int = int(self.env.action_space.n)
        self.n_states: int = int(self.env.observation_space.n)
        self.state, _ = self.env.reset()
        self.rewards: dict[RewardKey, float] = defaultdict(float)
        self.transits: dict[TransitKey, Counter] = defaultdict(Counter)
        self.values: dict[State, float] = defaultdict(float)

    def play_n_random_steps(self, n: int):
        for _ in range(n):
            action = self.env.action_space.sample()
            new_state, reward, is_done, is_trunc, _ = self.env.step(action)
            rw_key = (self.state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (self.state, action)
            self.transits[tr_key][new_state] += 1
            if is_done or is_trunc:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def calc_action_value(self, state: State, action: Action) -> float:
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            rw_key = (state, action, tgt_state)
            reward = self.rewards[rw_key]
            val = reward + GAMMA * self.values[tgt_state]
            action_value += (count / total) * val
        return action_value

    def select_action(self, state: State) -> Action:
        best_action: Action | None = None
        best_value: float | None = None
        for action in range(self.n_actions):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        assert best_action is not None
        return best_action

    def play_episode(self, env: gym.Env) -> float:
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            rw_key = (state, action, new_state)
            self.rewards[rw_key] = float(reward)
            tr_key = (state, action)
            self.transits[tr_key][new_state] += 1
            total_reward += float(reward)
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.n_states):
            state_values = [
                self.calc_action_value(state, action)
                for action in range(self.n_actions)
            ]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"{iter_no}: Best reward updated {best_reward:.3} -> {reward:.3}")
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
