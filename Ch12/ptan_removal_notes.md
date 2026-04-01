# Removing ptan from A2C Pong

`02_pong_a2c_noptan.py` is a fully self-contained rewrite of `02_pong_a2c.py` with all ptan and `lib.common` dependencies inlined.

## What was replaced

| ptan / lib component | Replacement |
|---|---|
| `ptan.agent.PolicyAgent` | `Agent` class — runs the network, applies softmax, samples actions via `np.random.choice` |
| `ptan.experience.VectorExperienceSourceFirstLast` | `Agent.play_step()` + `NStepTracker` — steps all vectorized envs and computes n-step discounted returns |
| `ptan.common.utils.TBMeanTracker` | `TBMeanTracker` class — accumulates scalar values and writes their mean to TensorBoard every `batch_size` calls |
| `ptan.experience.ExperienceFirstLast` | `ExperienceFirstLast` dataclass with fields: `state`, `action`, `reward`, `last_state` (None if terminal) |
| `ptan.common.wrappers.ImageToPyTorch` | `ImageToPyTorch` class — moves channel axis from HWC to CHW |
| `ptan.common.wrappers.BufferWrapper` | `BufferWrapper` class — stacks the last N frames into a single observation |
| `lib.common.RewardTracker` | `RewardTracker` class — tracks episode rewards, prints progress, logs to TensorBoard, stops at target reward |
| `lib.common.AtariA2C` | `AtariA2C` class — shared-trunk CNN with separate policy and value heads |
| `lib.common.unpack_batch` | `unpack_batch()` function — converts experience batch into tensors with bootstrapped value targets |

## How NStepTracker works

ptan's `VectorExperienceSourceFirstLast` internally maintains per-environment deques to compute n-step returns. `NStepTracker` does the same:

- Each env has a deque of `(state, action, reward)` tuples with `maxlen=steps_count`.
- On a normal step (not done), once the deque is full, it packs the oldest n transitions into a single `ExperienceFirstLast` with the discounted reward sum and the current observation as `last_state`.
- On a terminal step (done or truncated), it flushes all pending transitions with `last_state=None` since there is no future state to bootstrap from.

## What stayed the same

- All hyperparameters (gamma, learning rate, entropy beta, batch size, n-step count, grad clip, number of envs)
- The training loop structure: collect experiences, unpack batch, compute policy/value/entropy losses with separate backward passes, clip gradients, log to TensorBoard
- CLI arguments (`--dev`, `--use-async`, `-n`)
- Environment setup (ALE/Pong-v5 with frameskip=1, AtariPreprocessing, frame stacking)
