import numpy as np
from open_spiel.python.bots.uniform_random import UniformRandomBot

from .umamusime import UmaGame, UmaState


def play(*, verbose: bool = True, seed: int = 42) -> tuple[UmaState, list[int]]:
    game = UmaGame()
    state: UmaState = game.new_initial_state()
    rng = np.random.RandomState(seed)
    bot = UniformRandomBot(0, rng)
    player_actions: list[int] = []
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choice(outcomes, p=probs)
        else:
            action = bot.step(state)
            player_actions.append(int(action))
        if verbose:
            print(state.action_to_string(state.current_player(), action))
        state.apply_action(action)
        if verbose:
            print(state)
    if verbose:
        print(f"Returns: {state.returns()}")
    return state, player_actions


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
