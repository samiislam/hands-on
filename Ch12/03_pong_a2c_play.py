#!/usr/bin/env python3
import gymnasium as gym
import argparse
import numpy as np
import collections

import torch

from lib import a2c_model
from lib import wrappers

AtariA2C = a2c_model.AtariA2C

DEFAULT_ENV_NAME = "ALE/Pong-v5"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", required=True, help="Directory for video")
    args = parser.parse_args()

    env = wrappers.make_env(gym.make(args.env, frameskip=1, render_mode="rgb_array"),
                   episodic_life=False)
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    net = AtariA2C(env.observation_space.shape, int(env.action_space.n))
    state = torch.load(args.model, map_location=lambda stg, _: stg, weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c: collections.Counter[int] = collections.Counter()

    while True:
        state_v = torch.tensor(np.expand_dims(state, 0))
        logits, _ = net(state_v)
        action = int(logits.argmax(dim=1).item())
        c[action] += 1
        state, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += float(reward)
        if is_done or is_trunc:
            break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()
