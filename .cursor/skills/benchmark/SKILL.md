---
name: benchmark
description: Run Random, truncated MCTS, and DQN on Umamusime, print each method's final state and reward, and give a one-sentence summary of the chosen strategy. Use when benchmarking agents, comparing Random, MCTS, and DQN, or asking which training strategy each method chose.
---

# Benchmark Random, MCTS, and DQN

Compare OpenSpiel's uniform random bot, truncated MCTS search, and DQN on Mihono Bourbon's 78-turn Umamusime career (3 years plus Finale 1–6). Scheduled career races replace training turns and soft-fail if speed or stamina is below that race's minimums. Training can also fail mid-career when energy after cost is low; those fails cost the turn and (for speed/stamina/power/guts) −10 to the trained stat.

## Instructions

1. From the repository root, run:

```bash
uv run python -m umamusime.compare
```

2. The command **always retrains DQN from scratch** (it does not reuse `dqn_checkpoint.pt`), plays **3** games for each of Random, truncated MCTS, and greedy DQN, and prints:
   - How long **one** MCTS run took
   - How long DQN **training** took
   - The reward of every run, and which run was best
   - Final state of the **best** (highest-reward) run
   - Reward of that best run
   - Whether that best run **finished normally** or ended in a **career race soft fail**
   - How many times training **failed** in that best run, and **when** (calendar label + action) each fail happened
   - Actions of that best run, in order, named with `state.action_to_string`

3. For each method, write **one sentence** summarizing the chosen rest/training strategy from that **best** run's action sequence **and** its training-fail record. Do not count actions programmatically or reuse a canned template. Do not summarize the discarded runs. When fails matter (many fails, clusters at low energy, or almost none despite aggressive training), mention them; when the run never failed, you can say so briefly or omit if irrelevant.

4. Show the user the MCTS one-run time, the DQN training time, each method's run rewards, the best run's final state and reward, whether it finished normally or soft-failed a career race, the training-fail count and timings, and your one-sentence summary. Do not re-run the play loops.

Do not reimplement the bots. `umamusime.compare` calls `umamusime.bots.random_bot.play` (OpenSpiel `UniformRandomBot`), `umamusime.bots.mcts_truncated.play`, `umamusime.bots.dqn.train`, and `umamusime.bots.dqn.play`. The original full-rollout bot stays at `umamusime.bots.mcts.play` and is not part of this benchmark.
