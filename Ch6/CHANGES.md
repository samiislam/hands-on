# Differences from the Original `02_dqn_pong.py`

Original source: [PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition – Chapter06](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition/blob/main/Chapter06/)

---

## `lib/dqn_model.py`

| Area | Original | Ours |
|---|---|---|
| `forward` type hint | `x: torch.ByteTensor` | `x: torch.Tensor` |

No other changes.

---

## `02_dqn_pong.py`

### 1. Vectorized environments (`N_ENVS = 8`)

The original uses a single `gym.Env`. Ours uses `gym.vector.AsyncVectorEnv` with 8 parallel environments, collecting `N_ENVS` transitions per iteration.

This required rewriting the `Agent` class:

- `play_step` processes all envs in one batched call and returns a **list** of completed-episode rewards instead of a single `Optional[float]`.
- `states` is a 2-D array (one row per env) instead of a single observation.
- Epsilon-greedy action selection runs a single batched forward pass and `torch.max` over the batch, rather than unsqueezing a single state.
- Uses `VectorEnv` auto-reset handling: when an env is done, `final_observation` from `infos` is used as the true terminal observation (the regular `new_states[i]` is already the first obs of the next episode).

### 2. N-step returns (`N_STEPS = 3`)

The original stores 1-step transitions. Ours accumulates `n_steps` consecutive experiences per environment in a deque (`step_buffers`), then folds them into a single n-step transition with a discounted cumulative reward before appending to the replay buffer.

- `_flush_steps` computes `r_0 + γ r_1 + γ² r_2 + ...` and stores `(s_0, a_0, R_n, done_n, s_n)`.
- On episode termination the partial buffer (possibly < n steps) is flushed immediately.
- `calc_loss` uses `gamma ** n_steps` instead of `gamma` when bootstrapping.

### 3. Double DQN

The original selects **and** evaluates the next action using `tgt_net`:

```python
next_state_values = tgt_net(new_states_t).max(1)[0]
```

Ours uses Double DQN — the online `net` selects the best next action, `tgt_net` evaluates it:

```python
best_actions = q_next.argmax(1, keepdim=True)                      # net selects
next_state_values = tgt_net(new_states_t).gather(1, best_actions)   # tgt_net evaluates
```

Additionally, ours batches the current-state and next-state forward passes through `net` into a single `torch.cat` call to reduce kernel-launch overhead.

### 4. Polyak averaging (soft target updates) instead of hard sync

| | Original | Ours |
|---|---|---|
| Strategy | Hard copy every `SYNC_TARGET_FRAMES = 1000` steps | Soft update every step with `TAU = 0.005` |
| Code | `tgt_net.load_state_dict(net.state_dict())` | `p_tgt.mul_(1 - TAU).add_(TAU * p.data)` |
| Placement | Before the gradient step | After the gradient step (so the update incorporates the latest gradient) |

### 5. Mixed-precision training

Ours wraps the loss computation in `torch.autocast` and uses `GradScaler` for AMP. The original uses full fp32 throughout.

### 6. `torch.compile` with cudagraphs backend

Ours compiles both `net` and `tgt_net` with `torch.compile(backend="cudagraphs")` and inserts `torch.compiler.cudagraph_mark_step_begin()` before inference in `play_step`. The original runs eagerly.

### 7. Hyperparameter changes

| Parameter | Original | Ours |
|---|---|---|
| `BATCH_SIZE` | 32 | 64 |
| `REPLAY_SIZE` | 10 000 | 100 000 |
| Default device | `cpu` | `cuda` |
| `EPSILON_DECAY_LAST_FRAME` | 150 000 | `150000 * N_ENVS * 0.75` (900 000) |

### 8. Non-blocking tensor transfers

`batch_to_tensors` uses `non_blocking=True` on `.to(device)` calls. The original does blocking transfers.

### 9. Logging and model saving

- Elapsed wall-clock time (`HH:MM:SS`) is printed alongside each episode completion.
- Model is saved as `{env}-best.dat` (single file, overwritten) instead of `{env}-best_{reward}.dat` (one file per new best).

### 10. Type hints modernized

`tt.Optional`, `tt.List`, `tt.Tuple` replaced with native Python 3.10+ equivalents (`X | None`, `list[X]`, `tuple[X, ...]`).

---

## Changes Tried and Reverted

### Dueling DQN

Split the FC head into separate value V(s) and advantage A(s,a) streams, combined as `Q(s,a) = V(s) + A(s,a) - mean(A)`. **Reverted** because the added per-step overhead (extra linear layers + mean computation) wasn't worth it for Pong's small action space (6 actions). Dueling DQN is more beneficial in environments with many actions.
