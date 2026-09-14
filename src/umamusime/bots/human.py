import numpy as np
from open_spiel.python.bots.human import HumanBot

from ..game import UmaGame
from ..state import UmaState


def play() -> UmaState:
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
    return state


def main() -> None:
    play()


if __name__ == "__main__":
    main()
