"""Split one seed into independent career and bot RandomStates."""

import numpy as np


def split_rngs(seed: int) -> tuple[np.random.RandomState, np.random.RandomState]:
    """Return (career_rng, bot_rng) from SeedSequence.spawn(2)."""
    env_ss, bot_ss = np.random.SeedSequence(seed).spawn(2)
    return (
        np.random.RandomState(np.random.MT19937(env_ss)),
        np.random.RandomState(np.random.MT19937(bot_ss)),
    )
