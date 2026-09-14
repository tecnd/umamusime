# AGENTS.md

## Project

Umamusime is a Python 3.11/OpenSpiel implementation of a simplified, single-player Umamusume career. `rules.md` is the detailed gameplay specification; keep it synchronized with behavior changes.

## Setup and checks

```bash
uv sync
uv run mypy src
uv run python -c "from umamusime.game import UmaGame; print(UmaGame().new_initial_state())"
```

There is currently no automated test suite. Run the type check and a focused smoke command for changed behavior. `uv run python -m umamusime.compare` is the full Random/truncated-MCTS/DQN benchmark; it is slow, always retrains DQN, and overwrites the ignored `dqn_checkpoint.pt`.

Useful entry points:

- `uv run umamusime` — interactive human game.
- `uv run python -m umamusime.bots.random_bot` — random baseline.
- `uv run python -m umamusime.bots.mcts_truncated` — benchmark MCTS.
- `uv run python -m umamusime.bots.dqn` — load or create a DQN checkpoint.

## Code map

- `src/umamusime/game.py` registers `UmaGame` and defines OpenSpiel metadata/parameters.
- `src/umamusime/state.py` owns the turn-phase state machine, action application, rewards, and terminal conditions.
- `src/umamusime/training.py`, `cards.py`, `scoring.py`, and `calendar.py` hold game data and pure calculations.
- `src/umamusime/actions.py` defines shared action/stat indices; tuple ordering is `speed, stamina, power, guts, wit, skill_points`.
- `src/umamusime/observer.py` builds the normalized RL observation.
- `src/umamusime/bots/` contains human, random, full/truncated MCTS, and DQN players; `compare.py` benchmarks selected bots.
- `typings/pyspiel.pyi` supplies local OpenSpiel types and is excluded from project checks.

## Conventions and pitfalls

- Preserve OpenSpiel's explicit phases: six support-placement chance nodes, one player decision, then an optional fail/success chance node.
- Keep stat/action tuple indices aligned through the constants in `actions.py`; skill points are the sixth stat but have no facility or growth parameter.
- Use `UmaGame(reward_model=TERMINAL)` only for MCTS metadata compatibility; the default per-turn reward model is required by DQN.
- Seed NumPy/OpenSpiel paths when reproducibility matters. Do not commit generated `*.pt`, virtual environments, or build artifacts.
- Prefer typed helpers, immutable tuples/frozen data, relative package imports, and constants for game values. Update `rules.md` when rules, defaults, scoring, or known limitations change.
