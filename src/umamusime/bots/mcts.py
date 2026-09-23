import pyspiel
from open_spiel.python.algorithms import mcts as openspiel_mcts

from ..game import UmaGame
from ..rng import split_rngs
from ..state import UmaState

# Full-career random rollouts, uct_c=200, 100 simulations, and 15 rollouts:
# on 96 paired seeds that beat a 6-turn cutoff by +1981 ± 358 (12 finished
# careers versus 1) and beat uct_c=2 by +1238 ± 452. About 42s per game
# (max 62s on those seeds). uct_c=400 did not beat 200. See docs/mcts-tuning.md.
UCT_C = 200.0
MAX_SIMULATIONS = 100
N_ROLLOUTS = 15


class FastRolloutEvaluator(openspiel_mcts.RandomRolloutEvaluator):
    """Random rollouts to a terminal state, without numpy's `choice(p=...)`.

    `choice` validates the distribution on every call, which costs ~5us per
    chance node against ~0.3us for a cumulative sum. With eight chance nodes a
    turn that dominates the search. The rollout policy is unchanged.
    """

    def evaluate(self, state):
        random_sample = self._random_state.random_sample
        total = 0.0
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            while not working_state.is_terminal():
                if working_state.is_chance_node():
                    threshold = random_sample()
                    cumulative = 0.0
                    for action, probability in working_state.chance_outcomes():
                        cumulative += probability
                        if threshold < cumulative:
                            break
                else:
                    legal_actions = working_state.legal_actions()
                    action = legal_actions[int(random_sample() * len(legal_actions))]
                working_state.apply_action(action)
            total += working_state.returns()[0]
        return [total / self.n_rollouts]


def play(*, verbose: bool = True, seed: int = 42) -> tuple[UmaState, list[int]]:
    # MCTSBot rejects non-TERMINAL reward_model, but that is only a metadata
    # check. This game already implements returns() and the rest of the State
    # API MCTS needs; per-turn rewards stay on the default game for DQN.
    game = UmaGame(reward_model=pyspiel.GameType.RewardModel.TERMINAL)
    state: UmaState = game.new_initial_state()
    env_rng, search_rng = split_rngs(seed)
    bot = openspiel_mcts.MCTSBot(
        game,
        uct_c=UCT_C,
        max_simulations=MAX_SIMULATIONS,
        evaluator=FastRolloutEvaluator(n_rollouts=N_ROLLOUTS, random_state=search_rng),
        random_state=search_rng,
    )
    player_actions: list[int] = []
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = env_rng.choice(outcomes, p=probs)
            action_label = None
        else:
            action = bot.step(state)
            player_actions.append(int(action))
            action_label = state.action_to_string(state.current_player(), action)
        state.apply_action(action)
        if verbose and action_label is not None:
            print(action_label)
            print(state)
    if verbose:
        print(f"Returns: {state.returns()}")
    return state, player_actions


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
