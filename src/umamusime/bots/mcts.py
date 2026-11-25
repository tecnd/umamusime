import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts as openspiel_mcts

from ..game import UmaGame
from ..rng import split_rngs
from ..state import UmaState
from . import base

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


class MCTSBot:
    """MCTS player implementing the Umamusime Bot protocol."""

    def __init__(
        self,
        *,
        uct_c: float = UCT_C,
        max_simulations: int = MAX_SIMULATIONS,
        n_rollouts: int = N_ROLLOUTS,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        self._uct_c = uct_c
        self._max_simulations = max_simulations
        self._n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()
        self._bot: openspiel_mcts.MCTSBot | None = None
        self._game_params: dict[str, object] | None = None

    def _get_bot(self, state: UmaState) -> openspiel_mcts.MCTSBot:
        params = dict(state.get_game().get_parameters())
        if self._bot is None or params != self._game_params:
            # MCTSBot rejects non-TERMINAL reward_model, but that is only a
            # metadata check. This game already implements returns() and the
            # rest of the State API MCTS needs; per-turn rewards stay on the
            # default game for DQN.
            game = UmaGame(
                params=params,
                reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            )
            self._bot = openspiel_mcts.MCTSBot(
                game,
                uct_c=self._uct_c,
                max_simulations=self._max_simulations,
                evaluator=FastRolloutEvaluator(
                    n_rollouts=self._n_rollouts,
                    random_state=self._random_state,
                ),
                random_state=self._random_state,
            )
            self._game_params = params
        return self._bot

    def predict(self, state: UmaState) -> int:
        return int(self._get_bot(state).step(state))


def play(
    *,
    verbose: bool = True,
    seed: int | None = 42,
    game: UmaGame | None = None,
) -> base.PlayResult:
    search_rng = split_rngs(seed)[1] if seed is not None else None
    bot = MCTSBot(random_state=search_rng)
    return base.play(bot, game=game, seed=seed, verbose=verbose)


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
