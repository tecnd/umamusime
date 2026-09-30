# AGENTS.md

## Project

Umamusime is a Python 3.11/OpenSpiel implementation of a simplified, single-player Umamusume career. The project uses `uv` for Python, dependency, environment, and command management. `docs/rules.md` is the detailed gameplay specification; keep it synchronized with behavior changes. `docs/mcts-tuning.md` records how the shipped MCTS parameters were chosen, and `docs/dqn-tuning.md` does the same for the DQN observation and hyperparameters.

## Setup and checks

```bash
uv sync
uv run ruff check src
uv run ty check
uv run python -c "from umamusime.game import UmaGame; print(UmaGame().new_initial_state())"
```

There is currently no automated test suite. Run Ruff, ty, and a focused smoke command for changed behavior. `uv run python -m umamusime.compare` is the full Random/MCTS/DQN benchmark; it is slow, always retrains DQN, and overwrites the ignored `dqn_checkpoint.pt`.

Useful entry points:

- `uv run umamusime` — interactive human game.
- `uv run python -m umamusime.bots.random_bot` — random baseline.
- `uv run python -m umamusime.bots.mcts` — benchmark MCTS.
- `uv run python -m umamusime.bots.dqn` — load or create a DQN checkpoint (about 2 minutes to train). Delete `dqn_checkpoint.pt` after changing the observation size or network shape, since `load` will not match the old weights.

## Code map

- `src/umamusime/game.py` registers `UmaGame` and defines OpenSpiel metadata/parameters.
- `src/umamusime/state.py` owns the turn-phase state machine, action application, rewards, and terminal conditions.
- `src/umamusime/training.py`, `cards.py`, `scoring.py`, and `calendar.py` hold game data and pure calculations.
- `src/umamusime/actions.py` defines shared action/stat indices; tuple ordering is `speed, stamina, power, guts, wit, skill_points`.
- `src/umamusime/observer.py` builds the normalized RL observation (115 floats, layout in `docs/dqn-tuning.md`).
- `src/umamusime/bots/` contains human, random, MCTS, and DQN players; `compare.py` benchmarks selected bots.
- `docs/rules.md` is the gameplay specification; `docs/mcts-tuning.md` and `docs/dqn-tuning.md` explain the shipped MCTS and DQN parameters.
- `typings/pyspiel.pyi` supplies local OpenSpiel types for ty and is excluded as project source.

## Conventions and pitfalls

- Preserve OpenSpiel's explicit phases: six support-placement chance nodes, one player decision, then an optional fail/success chance node.
- Keep stat/action tuple indices aligned through the constants in `actions.py`; skill points are the sixth stat but have no facility or growth parameter.
- Use `UmaGame(reward_model=TERMINAL)` only for MCTS metadata compatibility; the default per-turn reward model is required by DQN.
- Seed NumPy/OpenSpiel paths when reproducibility matters. Split each play
  seed with `SeedSequence.spawn(2)` into separate career and bot/search
  `RandomState`s (`rng.split_rngs`) so decision randomness does not reshuffle
  placements or fail rolls. Do not commit generated `*.pt`, virtual
  environments, or build artifacts.
- When generating random seeds for testing, draw 128-bit random ints with
  Python's `secrets.randbits(128)`. Pass them through `rng.split_rngs`
  (`SeedSequence` accepts any size); `np.random.RandomState`, `env.seed`, and
  `DQNConfig.seed` only take values below 2**32, so do not hand them a raw
  128-bit seed. Log the seed you used so a failure can be reproduced.
- Prefer typed helpers, immutable tuples/frozen data, relative package imports, and constants for game values. Update `docs/rules.md` when rules, defaults, scoring, or known limitations change.
- Use `uv run ruff format src` to format the codebase.

## Cursor Cloud specific instructions

Cloud Agent VMs can boot from a snapshot whose git checkout is behind GitHub. On first environment setup, pull the latest `master` before exploring or branching:

```bash
git fetch origin master
git pull --ff-only origin master
```

If the session is already on a feature branch, fetch `origin/master` without switching away from that branch, then create new work from the updated `master`.
