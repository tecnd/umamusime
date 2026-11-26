"""Split one seed into independent career and bot RandomStates."""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SplitRngs:
    """Independent career and bot RandomStates spawned from one seed."""

    career_rng: np.random.RandomState
    bot_rng: np.random.RandomState


def split_rngs(seed: int) -> SplitRngs:
    """Return independent career and bot RandomStates from SeedSequence.spawn(2)."""
    env_ss, bot_ss = np.random.SeedSequence(seed).spawn(2)
    return SplitRngs(
        career_rng=np.random.RandomState(np.random.MT19937(env_ss)),
        bot_rng=np.random.RandomState(np.random.MT19937(bot_ss)),
    )
