# Differences from the Original Chapter 06 — Vanilla DQN (`02_dqn_pong.py`)

This file documents only the engineering and training speedup changes applied to the vanilla DQN (no Rainbow components).

Original source: [PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition – Chapter06](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter06/)

---

## 1. Vectorized environments (`N_ENVS = 8`)

> Collects 8 transitions per iteration in parallel, increasing environment throughput without additional GPU cost.

The original uses a single `gym.Env`. Ours uses `gym.vector.AsyncVectorEnv` with 8 parallel environments, collecting `N_ENVS` transitions per iteration.

This required rewriting the `Agent` class:

- `play_step` processes all envs in one batched call and returns a **list** of completed-episode rewards instead of a single `Optional[float]`.
- `states` is a 2-D array (one row per env) instead of a single observation.
- Epsilon-greedy action selection runs a single batched forward pass and `torch.max` over the batch, rather than unsqueezing a single state.
- Uses `VectorEnv` auto-reset handling: when an env is done, `final_observation` from `infos` is used as the true terminal observation (the regular `new_states[i]` is already the first obs of the next episode).

## 2. Polyak averaging (soft target updates) instead of hard sync

> Continuously blends target weights toward the online network, avoiding the periodic stale-target intervals that slow early learning.

| | Original | Ours |
|---|---|---|
| Strategy | Hard copy every `SYNC_TARGET_FRAMES = 1000` steps | Soft update every step with `TAU = 0.005` |
| Code | `tgt_net.load_state_dict(net.state_dict())` | `p_tgt.mul_(1 - TAU).add_(TAU * p.data)` |
| Placement | Before the gradient step | After the gradient step (so the update incorporates the latest gradient) |

## 3. Mixed-precision training

> Halves memory bandwidth and doubles GPU arithmetic throughput by running most operations in fp16 instead of fp32.

Ours wraps the loss computation in `torch.autocast` and uses `GradScaler` for AMP. The original uses full fp32 throughout.

## 4. `torch.compile` with cudagraphs backend

> Captures the entire forward pass as a single CUDA graph, eliminating per-step CPU kernel-launch overhead.

Ours compiles both `net` and `tgt_net` with `torch.compile(backend="cudagraphs")` and inserts `torch.compiler.cudagraph_mark_step_begin()` before inference in `play_step`. The original runs eagerly.

## 5. Hyperparameter changes

> Larger batch size better utilizes GPU parallelism; larger replay buffer improves sample diversity and training stability.

| Parameter | Original | Ours |
|---|---|---|
| `BATCH_SIZE` | 32 | 64 |
| `REPLAY_SIZE` | 10 000 | 100 000 |
| Default device | `cpu` | `cuda` |
| `EPSILON_DECAY_LAST_FRAME` | 150 000 | `150000 * N_ENVS * 0.75` (900 000) |

## 6. Non-blocking tensor transfers

> Overlaps CPU-to-GPU data transfers with computation, reducing idle time on both sides.

`batch_to_tensors` uses `non_blocking=True` on `.to(device)` calls. The original does blocking transfers.

## 7. Logging and model saving

- Elapsed wall-clock time (`HH:MM:SS`) is printed alongside each episode completion.
- Model is saved as `{env}-vanilla-best.dat` (single file, overwritten) instead of `{env}-best_{reward}.dat` (one file per new best).

## 8. Type hints modernized

`tt.Optional`, `tt.List`, `tt.Tuple` replaced with native Python 3.10+ equivalents (`X | None`, `list[X]`, `tuple[X, ...]`).
