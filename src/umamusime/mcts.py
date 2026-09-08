import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from .umamusime import UmaGame, UmaState


def play(*, verbose: bool = True, seed: int = 42) -> tuple[UmaState, list[int]]:
    # MCTSBot rejects non-TERMINAL reward_model, but that is only a metadata
    # check. This game already implements returns() and the rest of the State
    # API MCTS needs; per-turn rewards stay on the default game for DQN.
    game = UmaGame(reward_model=pyspiel.GameType.RewardModel.TERMINAL)
    state: UmaState = game.new_initial_state()
    rng = np.random.RandomState(seed)
    bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=100,
        evaluator=mcts.RandomRolloutEvaluator(n_rollouts=5, random_state=rng),
        random_state=rng,
    )
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
