# Why DQN Converges Faster Than A2C on Pong

## Q: What is the reason that Ch6/02_dqn_pong.py takes only approx. 16 minutes to converge to a mean reward of 19 and Ch12/02_pong_a2c.py takes so much longer?

The core reason is **sample efficiency vs. wall-clock throughput** -- DQN and A2C are fundamentally different algorithms.

### DQN advantages in this setup

1. **Replay buffer** -- Each experience is reused many times. With `REPLAY_SIZE=100000`, a single transition gets sampled in multiple batches. A2C uses each experience **exactly once** then discards it.

2. **`torch.compile` + `cudagraphs` + `GradScaler`** -- The DQN script uses CUDA graph compilation and mixed precision. The A2C script uses none of these optimizations, so it runs significantly slower per forward/backward pass.

3. **`AsyncVectorEnv` by default** -- DQN uses async env stepping, parallelizing CPU work. A2C defaults to `SyncVectorEnv` unless you pass `--use-async`.

4. **8 envs vs 50 envs** -- DQN runs 8 envs (less CPU overhead), while A2C runs 50. More envs means more CPU time per step, and since the GPU is already fast, the CPU becomes the bottleneck.

### A2C inherent disadvantages

5. **On-policy** -- A2C can only learn from data collected by the *current* policy. Once the policy updates, all old data is stale. DQN's off-policy replay buffer gets 100-1000x more learning per environment step.

6. **No target network stabilization** -- DQN has a target net (Polyak averaging) that stabilizes training. A2C bootstraps from its own rapidly-changing value estimates, which can be noisier.

**In short:** DQN squeezes far more learning out of each environment interaction (replay), and the DQN script is also more optimized at the systems level (compile, mixed precision, async). A2C's advantage is *simplicity* and that it can eventually achieve better final performance in some settings -- but it needs more environment frames to get there.

---

## Q: Should I also update A2C to use `torch.compile` + `cudagraphs` + `GradScaler` and 8 envs instead of 50? This way the comparison would be on almost equal grounds.

No, both changes would be counterproductive:

**`torch.compile` + `cudagraphs` + `GradScaler`** -- These optimize GPU throughput, but A2C's bottleneck is **CPU-side env stepping**, not GPU computation. The RTX 4090 is already mostly idle waiting for envs. The speedup would be negligible for added complexity.

**8 envs instead of 50** -- This would actually **hurt** A2C. The two algorithms use different mechanisms for the same purpose (decorrelating training data):

- DQN uses a **replay buffer** (100K transitions, sampled randomly)
- A2C uses **parallel envs** (many diverse states per step)

Reducing A2C to 8 envs removes its primary variance reduction tool. Gradients would be noisier and learning would be *slower*, not faster.

**A fair comparison is really about sample efficiency** (reward vs. environment frames), not wall-clock time. The algorithms have fundamentally different compute profiles. DQN will always win on wall-clock for Pong because replay is just more efficient for this type of problem.

---

## Q: What metrics should I use to compare sample efficiency?

The key metric is **mean reward (reward_100) vs. total environment frames**.

Environment frames normalizes for the different number of envs and step rates. To track this in A2C, you'd want to log `step_idx * NUM_ENVS` (since each exp_source step advances all 50 envs) rather than raw `step_idx`. The DQN script already tracks `frame_idx` which counts `+= N_ENVS` per step.

Then plot both curves on the same x-axis (total env frames) and compare:

- **Frames to threshold** -- how many frames to reach, say, reward 0 or reward 15
- **Area under the reward curve** -- overall learning speed

DQN will almost certainly win on sample efficiency for Pong due to replay. A2C's strengths show more in continuous control or when you need an explicit policy.
