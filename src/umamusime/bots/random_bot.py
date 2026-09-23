from open_spiel.python.bots.uniform_random import UniformRandomBot

from ..chance import bot_rng, sample_chance
from ..game import UmaGame
from ..state import UmaState


def play(*, verbose: bool = True, seed: int = 42) -> tuple[UmaState, list[int]]:
    game = UmaGame()
    state: UmaState = game.new_initial_state()
    bot = UniformRandomBot(0, bot_rng(seed))
    player_actions: list[int] = []
    while not state.is_terminal():
        if state.is_chance_node():
            action = sample_chance(state, seed)
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
