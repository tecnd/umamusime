from open_spiel.python.bots.human import HumanBot as OpenSpielHumanBot

from ..game import UmaGame
from ..state import UmaState
from . import base


class HumanBot:
    """Interactive terminal player implementing the Umamusime Bot protocol."""

    def __init__(self) -> None:
        self._bot = OpenSpielHumanBot()

    def predict(self, state: UmaState) -> int:
        print(state)
        return int(self._bot.step(state))


def play(*, game: UmaGame | None = None, seed: int | None = None) -> base.PlayResult:
    result = base.play(
        HumanBot(),
        game=game,
        seed=seed,
        verbose=False,
        catch_interrupt=True,
    )
    if not result.interrupted:
        print(result.state)
        print(f"Returns: {result.state.returns()}")
    return result


def main() -> None:
    play()


if __name__ == "__main__":
    main()
