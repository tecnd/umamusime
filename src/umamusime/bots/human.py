import secrets

from open_spiel.python.bots.human import HumanBot

from ..chance import sample_chance
from ..game import UmaGame
from ..state import UmaState


def play(*, seed: int | None = None) -> UmaState:
    game = UmaGame()
    state: UmaState = game.new_initial_state()
    bot = HumanBot()
    # Optional seed makes placements / fail rolls reproducible; default is a
    # fresh career seed so interactive play is not locked to one sequence.
    career_seed = secrets.randbits(31) if seed is None else seed
    try:
        while not state.is_terminal():
            if state.is_chance_node():
                action = sample_chance(state, career_seed)
            else:
                print(state)
                action = bot.step(state)
            state.apply_action(action)
    except KeyboardInterrupt:
        print("\nInterrupted — exiting.")
        print(state)
        print(f"Returns so far: {state.returns()}")
        return state
    print(state)
    print(f"Returns: {state.returns()}")
    return state


def main() -> None:
    play()


if __name__ == "__main__":
    main()
