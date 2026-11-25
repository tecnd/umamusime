import numpy as np
from open_spiel.python.bots.uniform_random import UniformRandomBot

from ..game import UmaGame
from ..rng import split_rngs
from ..state import UmaState
from . import base


class RandomBot:
    """Uniform random bot implementing the Umamusime Bot protocol."""

    def __init__(self, rng: np.random.RandomState | None = None) -> None:
        self._bot = UniformRandomBot(0, rng or np.random.RandomState())

    def predict(self, state: UmaState) -> int:
        return int(self._bot.step(state))


def play(
    *,
    verbose: bool = True,
    seed: int | None = 42,
    game: UmaGame | None = None,
) -> base.PlayResult:
    bot_rng = split_rngs(seed).bot_rng if seed is not None else None
    return base.play(RandomBot(bot_rng), game=game, seed=seed, verbose=verbose)


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
