import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from .umamusime import _NODES_PER_TURN, UmaGame, UmaState

# Rollouts are ~99% of search time, so they are the only thing worth tuning.
# Truncating them is both faster and stronger: a rollout that plays random
# actions to the end of the career mostly measures "what does random play
# score", which barely depends on the action being evaluated. Cutting it short
# turns the leaf value into the score accumulated over the next few turns,
# which actually separates the candidate actions.
ROLLOUT_TURNS = 6
ROLLOUT_LENGTH = ROLLOUT_TURNS * _NODES_PER_TURN


class TruncatedRolloutEvaluator(mcts.RandomRolloutEvaluator):
    """Random rollouts, sampled without numpy's `choice(p=...)`.

    `choice` validates the distribution on every call, which costs ~5us per
    chance node against ~0.3us for a cumulative sum. With eight chance nodes a
    turn that dominates the search. The rollout policy is unchanged.
    """

    def evaluate(self, state):
        random_sample = self._random_state.random_sample
        total = 0.0
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            length = 0
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
                length += 1
                if self.max_length is not None and length >= self.max_length:
                    break
            total += working_state.returns()[0]
        return [total / self.n_rollouts]


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
        evaluator=TruncatedRolloutEvaluator(
            n_rollouts=5, random_state=rng, max_length=ROLLOUT_LENGTH
        ),
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
