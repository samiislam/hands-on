# DQN Pong - Changes Log

## Files Modified
- `Ch6/02_dqn_pong.py` — main training script
- `Ch6/lib/dqn_model.py` — network architecture

## Changes Applied (current state)

### 1. N-Step Returns
**What:** Instead of storing single-step transitions `(s, a, r, s')`, the agent accumulates `N_STEPS=3` transitions per environment and stores a folded experience with the discounted n-step reward: `r_0 + γr_1 + γ²r_2`.

**Why:** Propagates sparse rewards (Pong only gives +1/-1 when a point is scored) back to earlier states faster, without relying on the network having learned accurate intermediate value estimates.

**How:**
- Each of the 8 parallel environments has its own `step_buffers[i]` deque (maxlen=N_STEPS)
- `_flush_steps()` folds the buffer into a single experience: `(s_0, a_0, R_n, done, s_n)`
- When a buffer is full, one n-step experience is emitted and the oldest step is dropped
- When an episode ends early (< n steps), the partial buffer is flushed immediately
- `calc_loss` bootstraps with `γ^n` instead of `γ`

### 2. Double DQN
**What:** Decouples action selection from action evaluation in the target computation to reduce overestimation bias.

**Why:** Standard DQN uses `max` over the target network's Q-values, which is biased upward when Q-estimates are noisy. Double DQN uses the online network to select the best action and the target network to evaluate it.

**How:**
- In `calc_loss`, both `states_t` and `new_states_t` are concatenated and passed through `net` in a single batched forward pass (optimization to avoid two separate passes)
- The online network's Q-values for next states select the best action via `argmax`
- The target network evaluates that action via `gather`

### 3. Polyak Averaging (Soft Target Updates)
**What:** Replaced hard target network sync (copy every N frames) with continuous exponential moving average updates using `TAU=0.005`.

**Why:** Provides smoother, more stable target updates compared to periodic hard copies. The target network continuously tracks the online network instead of being stale for long periods then jumping.

**How:**
- After every training step: `θ_target = (1 - τ) * θ_target + τ * θ_online`
- Removed `SYNC_TARGET_FRAMES` constant and `last_sync` variable

### 4. Single Best Model Save
**What:** The training loop saves a single `PongNoFrameskip-v4-best.dat` file, overwriting it each time a new best mean reward is achieved.

**Why:** Previously created a separate file for every new best reward (e.g., `-best_-21.dat`, `-best_-20.dat`), leading to many files.

## Changes Tried and Reverted

### Dueling DQN (reverted)
**What:** Split the FC head into separate value V(s) and advantage A(s,a) streams, combined as `Q(s,a) = V(s) + A(s,a) - mean(A)`.

**Why reverted:** Added a small but measurable per-step overhead (extra linear layers + mean computation). For Pong's small action space (6 actions), the learning efficiency gain was not worth the speed cost. Dueling DQN is more beneficial in environments with many actions.

## Current Hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| GAMMA | 0.99 | Discount factor |
| BATCH_SIZE | 64 | |
| REPLAY_SIZE | 100,000 | |
| LEARNING_RATE | 1e-4 | Adam optimizer |
| TAU | 0.005 | Polyak averaging rate |
| REPLAY_START_SIZE | 10,000 | Fill buffer before training |
| N_STEPS | 3 | N-step returns |
| N_ENVS | 8 | Parallel environments |
| EPSILON_START | 1.0 | |
| EPSILON_FINAL | 0.01 | |
| EPSILON_DECAY_LAST_FRAME | 150000 * N_ENVS * 0.75 | |
