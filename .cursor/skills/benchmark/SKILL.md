---
name: benchmark
description: Run Random, MCTS, and DQN on Umamusime, print each method's final state and reward, and give a one-sentence summary of the chosen strategy. Use when benchmarking agents, comparing Random, MCTS, and DQN, or asking which training strategy each method chose.
---

# Benchmark Random, MCTS, and DQN

Compare OpenSpiel's uniform random bot, MCTS search, and DQN on a full 72-turn (3-year) Umamusime game.

## Instructions

1. From the repository root, run:

```bash
uv run python -m umamusime.compare
```

2. The command **always retrains DQN from scratch** (it does not reuse `dqn_checkpoint.pt`), plays **3** games for each of Random, MCTS, and greedy DQN, and prints:
   - How long **one** MCTS run took
   - How long DQN **training** took
   - The reward of every run, and which run was best
   - Final state of the **best** (highest-reward) run
   - Reward of that best run
   - Actions of that best run, in order, named with `state.action_to_string`

3. For each method, write **one sentence** summarizing the chosen rest/training strategy from that **best** run's action sequence. Do not count actions programmatically or reuse a canned template. Do not summarize the discarded runs.

4. Show the user the MCTS one-run time, the DQN training time, each method's run rewards, the best run's final state and reward, and your one-sentence summary. Do not re-run the play loops.

Do not reimplement the bots. `umamusime.compare` calls `umamusime.random_bot.play` (OpenSpiel `UniformRandomBot`), `umamusime.mcts.play`, `umamusime.dqn.train`, and `umamusime.dqn.play`.
