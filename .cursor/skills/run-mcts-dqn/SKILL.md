---
name: run-mcts-dqn
description: Run both MCTS and DQN on Umamusime, print each method's final state and reward, and give a one-sentence summary of the chosen strategy. Use when comparing MCTS and DQN, running both agents, or asking which training strategy each method chose.
---

# Run MCTS and DQN

Compare MCTS search and DQN on a full 60-turn Umamusime game.

## Instructions

1. From the repository root, run:

```bash
uv run python -m umamusime.compare
```

2. The command trains DQN if `dqn_checkpoint.pt` is missing (then saves it), plays one MCTS game and one greedy DQN game, and prints for each method:
   - Final state
   - Reward
   - One sentence summarizing the chosen rest/training strategy

3. Show that output to the user without re-running the play loops yourself.

Do not reimplement MCTS or DQN. `umamusime.compare` calls `umamusime.mcts.play` and `umamusime.dqn.play`.
