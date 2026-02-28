#!/usr/bin/env python3
import gymnasium as gym
import ptan
from typing import Any

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import dqn_model, common

NAME = "01_baseline"

BEST_PONG = common.Hyperparams(
    env_name="PongNoFrameskip-v4",
    stop_reward=18.0,
    run_name="pong",
    replay_size=100_000,
    replay_initial=10_000,
    target_net_sync=1000,
    epsilon_frames=100_000,
    epsilon_final=0.02,
    learning_rate=9.932831968547505e-05,
    gamma=0.98,
    episodes_to_solve=340,
)


def train(params: common.Hyperparams,
          device: torch.device, _: dict) -> int | None:
    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    net = dqn_model.DQN(env.observation_space.shape, int(env.action_space.n)).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, env_seed=common.SEED)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    result: Any = engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
    if result.solved:
        return result.episode


if __name__ == "__main__":
    args = common.argparser().parse_args()
    common.train_or_tune(args, train, BEST_PONG)
