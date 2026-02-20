import gymnasium as gym
import random

class RandomActionWrapper(gym.ActionWrapper):
    def __init__(
            self,
            env: gym.Env,
            epsilon: float = 0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(
            self,
            action: gym.core.WrapperActType) -> gym.core.WrapperActType:
        if random.random() < self.epsilon:
            action = self.env.action_space.sample()
            print(f"Random action {action}")
            return action
        return action

def main():
    env = RandomActionWrapper(gym.make("CartPole-v1", render_mode="rgb_array"))
    env = gym.wrappers.HumanRendering(env)
    total_reward = 0.0
    total_steps = 0
    obs, _ = env.reset()

    while True:
        obs, reward, is_done, is_trunc, _ = env.step(0)
        total_reward += reward
        total_steps += 1
        if is_done or is_trunc:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward:.2f}")


if __name__ == "__main__":
    main()
