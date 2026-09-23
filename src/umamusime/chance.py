"""Deterministic environment chance draws keyed by career position.

Bot RNGs must not share this stream. A placement or fail/success roll depends
only on the career seed and which chance node is being resolved, so two bots
that reach the same turn see the same supports and the same fail roll even if
their earlier decisions differed.
"""

from __future__ import annotations

import numpy as np

from .state import UmaState

# Separate from the career seed so search / random-bot decisions never collide
# with environment draws when someone reuses `seed` for both.
BOT_SEED_OFFSET = 1_000_003


def bot_rng(seed: int) -> np.random.RandomState:
    return np.random.RandomState(seed + BOT_SEED_OFFSET)


def _chance_rng(
    seed: int, turn: int, channel: int, index: int
) -> np.random.RandomState:
    # SeedSequence mixing is stable across processes (unlike hash()).
    stream = np.random.SeedSequence([seed, turn, channel, index])
    return np.random.RandomState(stream.generate_state(1)[0])


def sample_chance(state: UmaState, seed: int) -> int:
    """Draw the chance outcome for `state` from the career seed alone."""
    if not state.is_chance_node():
        raise ValueError("sample_chance requires a chance node")
    outcomes, probs = zip(*state.chance_outcomes())
    turn, channel, index = state.chance_stream_key()
    rng = _chance_rng(seed, turn, channel, index)
    return int(rng.choice(outcomes, p=probs))


class SeededChanceSampler:
    """OpenSpiel `ChanceEventSampler` that uses `sample_chance`."""

    def __init__(self, seed: int | None = None):
        self.seed(seed)

    def seed(self, seed: int | None = None) -> None:
        self._seed = 0 if seed is None else int(seed)

    def __call__(self, state) -> int:
        return sample_chance(state, self._seed)
