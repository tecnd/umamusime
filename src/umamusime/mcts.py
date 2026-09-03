import numpy as np
from open_spiel.python.algorithms import mcts

from .umamusime import UmaGame, UmaState


def main() -> None:
    game = UmaGame()
    state: UmaState = game.new_initial_state()
    rng = np.random.RandomState(42)
    bot = mcts.MCTSBot(
        game,
        uct_c=2,
        max_simulations=100,
        evaluator=mcts.RandomRolloutEvaluator(n_rollouts=5, random_state=rng),
        random_state=rng,
    )
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = rng.choice(outcomes, p=probs)
        else:
            action = bot.step(state)
        print(state.action_to_string(state.current_player(), action))
        state.apply_action(action)
        print(state)
    print(f"Returns: {state.returns()}")


if __name__ == "__main__":
    main()
