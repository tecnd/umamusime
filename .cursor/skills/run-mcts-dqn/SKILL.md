---
name: run-mcts-dqn
description: Run both MCTS and DQN on Umamusime, print each method's final state and reward, and give a one-sentence summary of the chosen strategy. Use when comparing MCTS and DQN, running both agents, or asking which training strategy each method chose.
---

# Run MCTS and DQN

Compare MCTS search and DQN on a full 72-turn (3-year) Umamusime game.

## Instructions

1. From the repository root, run:

```bash
uv run python -m umamusime.compare
```

2. The command trains DQN if `dqn_checkpoint.pt` is missing (then saves it), plays **3** MCTS games and **3** greedy DQN games, and prints for each method:
   - The reward of every run, and which run was best
   - Final state of the **best** (highest-reward) run
   - Reward of that best run
   - Actions of that best run, in order, named with `state.action_to_string`

3. For each method, write **one sentence** summarizing the chosen rest/training strategy from that **best** run's action sequence. Do not count actions programmatically or reuse a canned template. Do not summarize the discarded runs.

4. Show the user each method's run rewards, the best run's final state and reward, and your one-sentence summary. Do not re-run the play loops.

Do not reimplement MCTS or DQN. `umamusime.compare` calls `umamusime.mcts.play` and `umamusime.dqn.play`.
