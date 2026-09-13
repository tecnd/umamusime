import numpy as np
from open_spiel.python.bots.human import HumanBot

from .calendar import calendar_label
from .game import UmaGame
from .observer import UmaObserver
from .state import UmaState

__all__ = [
    "UmaGame",
    "UmaObserver",
    "UmaState",
    "calendar_label",
    "main",
]


def main() -> None:
    game = UmaGame()
    state: UmaState = game.new_initial_state()
    bot = HumanBot()
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = np.random.choice(outcomes, p=probs)
        else:
            print(state)
            action = bot.step(state)
        state.apply_action(action)
    print(state)
    print(f"Returns: {state.returns()}")


if __name__ == "__main__":
    main()
