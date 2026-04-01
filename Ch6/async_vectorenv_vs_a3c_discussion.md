# AsyncVectorEnv vs A3C: Understanding "Asynchronous" in RL

## Context

In `02_dqn_pong_double_nstep.py`, we use `gym.vector.AsyncVectorEnv` to run multiple Pong environments in parallel. This raised the question: is this the same "asynchronous" concept from the paper *Asynchronous Methods for Deep Reinforcement Learning* (Mnih et al., 2016, arXiv:1602.01783)?

**Answer: No.** The word "asynchronous" means different things in these two contexts.

## The Distinction

### `gym.vector.AsyncVectorEnv`

"Async" here means the **environments step in parallel** (in separate processes). Instead of stepping env1, then env2, then env3 sequentially, they all step at the same time. This is just **data collection parallelism**. Both the DQN code here and A2C use it for the same reason: gather experience faster.

### A3C (Mnih et al., 2016)

"Asynchronous" refers to **asynchronous gradient updates**. Multiple worker threads each have their **own copy of the network**, independently collect experience, compute gradients, and push those gradients to a shared parameter server **without waiting for each other**. Workers are unsynchronized with respect to the learning step -- that is the core idea.

### Summary Table

|                          | DQN (this code) / A2C       | A3C                                  |
|--------------------------|-----------------------------|--------------------------------------|
| Multiple envs in parallel| Yes (`AsyncVectorEnv`)      | Yes (one per worker)                 |
| Number of networks       | **1**                       | **Many** (one per worker)            |
| Gradient updates         | **1 centralized**, synchronized | **Many**, asynchronous & lock-free |

## Why A2C also uses `AsyncVectorEnv`

The book *Deep Reinforcement Learning: Hands-On* uses `gym.vector.AsyncVectorEnv` for the data-parallel version of A2C. This is because A2C is the **synchronous** version of A3C -- it collects data from parallel envs but does a **single, synchronized** gradient update. The parallelism is only at the environment/data level, not at the learning/gradient level.

## Relevant Papers

1. **Mnih et al., 2016** -- *Asynchronous Methods for Deep Reinforcement Learning* (arXiv:1602.01783). The A3C paper. Describes asynchronous gradient updates across multiple workers, each with their own network copy.

2. **Clemente et al., 2017** -- *Efficient Parallel Methods for Deep Reinforcement Learning* (arXiv:1705.04862). Explicitly discusses batched/synchronous data collection as an alternative to A3C's asynchronous updates and compares the approaches.

3. **Espeholt et al., 2018** -- *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures* (arXiv:1802.01561). Describes a centralized learner with many parallel actors, and explicitly contrasts this with A3C's approach.

4. **Horgan et al., 2018** -- *Distributed Prioritized Experience Replay* (Ape-X) (arXiv:1803.00933). Uses distributed actors feeding into a shared replay buffer with a single centralized learner -- architecturally the closest to what `02_dqn_pong_double_nstep.py` does, scaled up.

Note: A2C as a named algorithm does not have its own paper. It emerged from OpenAI's Baselines implementation as the synchronous variant of A3C.
