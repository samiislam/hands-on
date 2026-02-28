import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import dataclasses
from datetime import timedelta, datetime
from typing import Any
from collections.abc import Callable, Iterable, Iterator

import ptan.ignite as ptan_ignite
from ptan.actions import EpsilonGreedyActionSelector
from ptan.experience import ExperienceFirstLast, \
    ExperienceSourceFirstLast, ExperienceReplayBuffer

from ignite.engine import Engine
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ray import tune

SEED = 123


# Hyperparameter configuration dataclass — required fields define the
# environment and core DQN knobs, optional fields provide sensible defaults.
@dataclasses.dataclass
class Hyperparams:
    env_name: str
    stop_reward: float
    run_name: str
    replay_size: int
    replay_initial: int
    target_net_sync: int
    epsilon_frames: int

    learning_rate: float = 0.0001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_final: float = 0.1

    tuner_mode: bool = False
    episodes_to_solve: int = 500


# Pre-configured hyperparameter sets for different Atari games.
# 'pong' is the fastest to train; 'breakout' and 'invaders' need
# larger replay buffers and longer epsilon decay schedules.
GAME_PARAMS = {
    'pong': Hyperparams(
        env_name="PongNoFrameskip-v4",
        stop_reward=19.0,
        run_name="pong",
        replay_size=100_000,
        replay_initial=10_000,
        target_net_sync=1000,
        epsilon_frames=100_000,
        epsilon_final=0.02,
    ),
    'breakout-small': Hyperparams(
        env_name="BreakoutNoFrameskip-v4",
        stop_reward=500.0,
        run_name="breakout-small",
        replay_size=300_000,
        replay_initial=20_000,
        target_net_sync=1000,
        epsilon_frames=1_000_000,
        batch_size=64,
    ),
    'breakout': Hyperparams(
        env_name="BreakoutNoFrameskip-v4",
        stop_reward=500.0,
        run_name='breakout',
        replay_size=1_000_000,
        replay_initial=50_000,
        target_net_sync=10_000,
        epsilon_frames=10_000_000,
        learning_rate=0.00025,
    ),
    'invaders': Hyperparams(
        env_name="SpaceInvadersNoFrameskip-v4",
        stop_reward=500.0,
        run_name='invaders',
        replay_size=10_000_000,
        replay_initial=50_000,
        target_net_sync=10_000,
        epsilon_frames=10_000_000,
        learning_rate=0.00025,
    ),
}


# Decompose a batch of (s, a, r, s') transitions into separate numpy arrays.
# For terminal transitions (last_state is None), we substitute the current
# state as a placeholder — those entries are masked out later via done_mask.
def unpack_batch(batch: list[ExperienceFirstLast]) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = exp.state  # the result will be masked anyway
        else:
            lstate = exp.last_state
        last_states.append(lstate)
    return np.asarray(states), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=bool), np.asarray(last_states)


# Standard DQN loss: MSE between predicted Q(s,a) and the Bellman target
# r + gamma * max_a' Q_tgt(s', a'). Terminal states have next-state value
# zeroed out via done_mask so the target reduces to just the reward.
def calc_loss_dqn(
        batch: list[ExperienceFirstLast], net: nn.Module, tgt_net: nn.Module,
        gamma: float, device: torch.device) -> torch.Tensor:
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    # Move numpy arrays to tensors on the target device
    states_v = torch.as_tensor(states).to(device)
    next_states_v = torch.as_tensor(next_states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    # Q(s, a) — gather the Q-value for the action that was actually taken
    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    # max_a' Q_tgt(s', a') — computed with the frozen target network
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[done_mask] = 0.0

    # Bellman target: r + gamma * max_a' Q_tgt(s', a')
    bellman_vals = rewards_v + gamma * next_state_vals.detach() 
    return nn.MSELoss()(state_action_vals, bellman_vals)


# Linearly decays epsilon from epsilon_start to epsilon_final over
# epsilon_frames steps for the epsilon-greedy exploration strategy.
class EpsilonTracker:
    def __init__(self, selector: EpsilonGreedyActionSelector, params: Hyperparams):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - frame_idx / self.params.epsilon_frames
        self.selector.epsilon = max(self.params.epsilon_final, eps)


# Infinite generator that feeds the ignite training loop.
# First fills the replay buffer with `initial` transitions, then on each
# iteration adds one new transition and yields a random mini-batch.
def batch_generator(buffer: ExperienceReplayBuffer, initial: int, batch_size: int) -> \
        Iterator[list[ExperienceFirstLast]]:
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)  # type: ignore[misc]


# Estimate the average best Q-value over a set of reference states.
# Splits states into 64 mini-batches to avoid GPU OOM, takes the max
# Q-value per state, and returns the overall mean. Used as a training
# progress metric — a steadily rising value indicates learning.
@torch.no_grad()
def calc_values_of_states(states: np.ndarray, net: nn.Module, device: torch.device):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


# Wire up pytorch-ignite event handlers for logging, TensorBoard, and
# early stopping. This is the glue between the training loop and monitoring.
def setup_ignite(
        engine: Engine, params: Hyperparams, exp_source: ExperienceSourceFirstLast,
        run_name: str, extra_metrics: Iterable[str] = (),
        tuner_reward_episode: int = 100, tuner_reward_min: float = -19,
):
    # Attach ptan handlers: tracks episode boundaries and FPS
    handler = ptan_ignite.EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    ptan_ignite.EpisodeFPSHandler().attach(engine)

    # Log episode stats (reward, steps, speed) to console after each episode
    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        state: Any = trainer.state
        passed = state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, speed=%.1f f/s, elapsed=%s" % (
            state.episode, state.episode_reward,
            state.episode_steps, state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    # Stop training when the average reward reaches the target (stop_reward)
    @engine.on(ptan_ignite.EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        state: Any = trainer.state
        passed = state.metrics['time_passed']
        print("Game solved in %s, after %d episodes and %d iterations!" % (
            timedelta(seconds=int(passed)), state.episode,
            trainer.state.iteration))
        trainer.should_terminate = True
        state.solved = True

    # Set up TensorBoard logging with a timestamped run directory
    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)

    # Track running average of the training loss
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    # Log per-episode metrics (reward, steps, avg_reward) to TensorBoard
    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(tag="episodes", metric_names=metrics)
    event = ptan_ignite.EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # Log training metrics (loss, fps, plus any extras) every 100 iterations
    ptan_ignite.PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(tag="train", metric_names=metrics,
                                      output_transform=lambda a: a)
    event = ptan_ignite.PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # In tuner mode, add early termination: stop if the agent hasn't
    # reached a minimum reward threshold after a warmup period, or if
    # it exceeds the episode budget (episodes_to_solve * 1.1).
    if params.tuner_mode:
        @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
        def episode_completed(trainer: Engine):
            state: Any = trainer.state
            avg_reward = state.metrics.get('avg_reward', 0.0)
            max_episodes = params.episodes_to_solve * 1.1
            if state.episode > tuner_reward_episode and \
                    avg_reward < tuner_reward_min:
                trainer.should_terminate = True
                state.solved = False
            elif state.episode > max_episodes:
                trainer.should_terminate = True
                state.solved = False
            if trainer.should_terminate:
                print(f"Episode {state.episode}, "
                      f"avg_reward {avg_reward:.2f}, terminating")


# Type alias for training functions: takes hyperparams, device, and an
# extra config dict; returns episode count to solve (or None if unsolved).
TrainFunc = Callable[
    [Hyperparams, torch.device, dict],
    int | None
]

# Default Ray Tune search space shared across all tuning runs
BASE_SPACE = {
    "learning_rate": tune.loguniform(1e-5, 1e-4),
    "gamma": tune.choice([0.9, 0.92, 0.95, 0.98, 0.99, 0.995]),
}

def tune_params(
        base_params: Hyperparams, train_func: TrainFunc, device: torch.device,
        samples: int = 10, extra_space: dict[str, Any] | None = None,
):
    """
    Perform hyperparameters tune.
    :param train_func: Train function, has to return "episodes" key with metric
    :param device: torch device
    :param samples: count of samples to perform
    :param extra_space: additional search space
    """
    # Merge base search space with any extra dimensions
    search_space = dict(BASE_SPACE)
    if extra_space is not None:
        search_space.update(extra_space)
    config = tune.TuneConfig(num_samples=samples)

    # Objective: override base_params with sampled config values,
    # run training, and report episodes needed (lower is better).
    def objective(config: dict, device: torch.device) -> dict:
        keys = dataclasses.asdict(base_params).keys()
        upd = {"tuner_mode": True}
        for k, v in config.items():
            if k in keys:
                upd[k] = v
        params = dataclasses.replace(base_params, **upd)
        res = train_func(params, device, config)
        return {"episodes": res if res is not None else 10**6}

    # Bind the device param and optionally request a GPU resource
    obj = tune.with_parameters(objective, device=device)
    if device.type == "cuda":
        obj = tune.with_resources(obj, {"gpu": 1})

    # Launch the tuning run and print the best result
    tuner = tune.Tuner(obj, param_space=search_space, tune_config=config)
    results = tuner.fit()
    best = results.get_best_result(metric="episodes", mode="min")
    print(best.config)
    print(best.metrics)


# CLI argument parser shared by all Ch8 training scripts.
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument(
        "--params", choices=('common', 'best'), default="best",
        help="Params to use for training or tuning, default=best"
    )
    parser.add_argument(
        "--tune", type=int, help="Steps of params tune")
    return parser


# Entry point: seeds RNG, selects params (common vs best), and either
# runs a single training session or launches a Ray Tune hyperparameter search.
def train_or_tune(
        args: argparse.Namespace,
        train_func: TrainFunc,
        best_params: Hyperparams,
        extra_params: dict | None = None,
        extra_space: dict | None = None,
):
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.dev)

    if args.params == "common":
        params = GAME_PARAMS['pong']
    else:
        params = best_params

    if extra_params is None:
        extra_params = {}
    if args.tune is None:
        train_func(params, device, extra_params)
    else:
        tune_params(params, train_func, device, samples=args.tune,
                    extra_space=extra_space)
