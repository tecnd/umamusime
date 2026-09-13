from collections.abc import Sequence

import pyspiel

from .actions import (
    NODES_PER_TURN as _NODES_PER_TURN,
    NUM_ACTIONS as _NUM_ACTIONS,
    NUM_CARDS as _NUM_CARDS,
    NUM_PLACEMENT_OUTCOMES as _NUM_PLACEMENT_OUTCOMES,
)
from .calendar import MAX_TURNS as _MAX_TURNS
from .cards import SupportCard, cards_from_param, deck_param
from .observer import UmaObserver
from .scoring import (
    MAX_STAT as _MAX_STAT,
    SKILL_POINT_WEIGHT as _SKILL_POINT_WEIGHT,
    stat_score as _stat_score,
)
from .state import CardState, UmaState
from .training import (
    max_skill_points_per_turn as _max_skill_points_per_turn,
    placement_outcomes as _placement_outcomes,
    summed_initial_stats as _summed_initial_stats,
)

# Re-exported so existing `from .umamusime import UmaGame, UmaState` callers
# and `_MAX_TURNS` / `_NODES_PER_TURN` private imports keep working.
__all__ = [
    "CardState",
    "UmaGame",
    "UmaObserver",
    "UmaState",
    "_MAX_TURNS",
    "_NODES_PER_TURN",
]


def _make_game_type(
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
) -> pyspiel.GameType:
    return pyspiel.GameType(
        short_name="umamusime",
        long_name="Umamusime",
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        reward_model=reward_model,
        max_num_players=1,
        min_num_players=1,
        provides_information_state_string=True,
        provides_information_state_tensor=False,
        provides_observation_string=True,
        provides_observation_tensor=True,
        parameter_specification={"cards": deck_param()},
    )


# Per-turn rewards are real (used by DQN). MCTSBot only checks this metadata
# and can be given TERMINAL via UmaGame(reward_model=...).
_GAME_TYPE = _make_game_type()


def _max_utility(cards: Sequence[SupportCard]) -> float:
    """Highest possible return for this deck.

    Each of the five capped stats scores at most stat_score(1200),
    including the deck's initial grants. Skill points keep the flat 1.3
    weight times the all-cards-attend rainbow ceiling already used for
    the observation scale.
    """
    five = 5 * _stat_score(_MAX_STAT)
    skill = _SKILL_POINT_WEIGHT * _max_skill_points_per_turn(cards) * _MAX_TURNS
    return five + skill


def _game_info(cards: Sequence[SupportCard]) -> pyspiel.GameInfo:
    # Five-stat lookup of the current value cannot go below 0; skill points
    # never decrease.
    return pyspiel.GameInfo(
        num_distinct_actions=_NUM_ACTIONS,
        max_chance_outcomes=_NUM_PLACEMENT_OUTCOMES,
        num_players=1,
        min_utility=0.0,
        max_utility=_max_utility(cards),
        max_game_length=_MAX_TURNS * _NODES_PER_TURN,
    )


class UmaGame(pyspiel.Game):
    def __init__(self, params=None, *, reward_model=None):
        params = dict(params or {})
        params.setdefault("cards", deck_param())
        cards = cards_from_param(str(params["cards"]))
        if len(cards) != _NUM_CARDS:
            raise ValueError(
                f"Expected a deck of {_NUM_CARDS} support cards, "
                f"got {len(cards)}"
            )
        game_type = (
            _make_game_type(reward_model) if reward_model is not None else _GAME_TYPE
        )
        super().__init__(game_type, _game_info(cards), params)
        self.cards = cards
        self.placement_outcomes = tuple(
            _placement_outcomes(card) for card in self.cards
        )
        self.initial_stats = _summed_initial_stats(self.cards)
        self.skill_point_obs_scale = max(
            1, _max_skill_points_per_turn(self.cards) * _MAX_TURNS
        )

    def new_initial_state(self, state=None):
        return UmaState(self, state)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return UmaObserver(params)


pyspiel.register_game(_GAME_TYPE, UmaGame)
