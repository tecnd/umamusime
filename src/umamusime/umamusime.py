import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import pyspiel

from .actions import (
    CHANCE_FAIL as _CHANCE_FAIL,
    CHANCE_SUCCESS as _CHANCE_SUCCESS,
    DECISION_ACTIONS as _DECISION_ACTIONS,
    NODES_PER_TURN as _NODES_PER_TURN,
    NUM_ACTIONS as _NUM_ACTIONS,
    NUM_CARDS as _NUM_CARDS,
    NUM_PLACEMENT_OUTCOMES as _NUM_PLACEMENT_OUTCOMES,
    NUM_STATS as _NUM_STATS,
    PLACEMENT_ACTIONS as _PLACEMENT_ACTIONS,
    PLACEMENT_AWAY as _PLACEMENT_AWAY,
    REST_ACTION as _REST_ACTION,
    RESULT_ACTIONS as _RESULT_ACTIONS,
    STAT_INDEX as _STAT_INDEX,
    STAT_TRAIN_ACTIONS as _STAT_TRAIN_ACTIONS,
    TRAINING_ACTIONS as _TRAINING_ACTIONS,
    TRAINING_FOR_STAT as _TRAINING_FOR_STAT,
    WIT_ACTION as _WIT_ACTION,
    Phase as _Phase,
)
from .calendar import (
    MAX_TURNS as _MAX_TURNS,
    SUMMER_CAMP_TURNS as _SUMMER_CAMP_TURNS,
    TENNO_SHO_MIN_SPEED as _TENNO_SHO_MIN_SPEED,
    TENNO_SHO_MIN_STAMINA as _TENNO_SHO_MIN_STAMINA,
    TENNO_SHO_SPRING_TURN as _TENNO_SHO_SPRING_TURN,
    calendar_label,
)
from .cards import SupportCard, cards_from_param, deck_param
from .scoring import (
    MAX_STAT as _MAX_STAT,
    MIN_STAT as _MIN_STAT,
    SKILL_POINT_WEIGHT as _SKILL_POINT_WEIGHT,
    stat_score as _stat_score,
)
from .training import (
    FAIL_STATS as _FAIL_STATS,
    FRIENDSHIP_PER_TRAINING as _FRIENDSHIP_PER_TRAINING,
    MAX_FACILITY_LEVEL as _MAX_FACILITY_LEVEL,
    MAX_FRIENDSHIP as _MAX_FRIENDSHIP,
    RAINBOW_FRIENDSHIP as _RAINBOW_FRIENDSHIP,
    STARTING_ENERGY as _STARTING_ENERGY,
    TRAINING_ENERGY as _TRAINING_ENERGY,
    TRAINING_STATS as _TRAINING_STATS,
    UMA_GROWTH as _UMA_GROWTH,
    clip_energy as _clip_energy,
    facility_level as _facility_level,
    max_skill_points_per_turn as _max_skill_points_per_turn,
    placement_outcomes as _placement_outcomes,
    stat_train_failure_chance as _stat_train_failure_chance,
    summed_initial_stats as _summed_initial_stats,
    training_multiplier as _training_multiplier,
)

JsonDict = dict[str, Any]


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


class CardState(NamedTuple):
    """Per-career run state for one support card slot."""

    friendship: int
    placement: int = _PLACEMENT_AWAY


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


class UmaState(pyspiel.State):
    def __init__(
        self, game: UmaGame, state: pyspiel.StateStruct | JsonDict | None = None
    ):
        super().__init__(game)
        self._turn = 0

        (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
            self._skill_points,
        ) = game.initial_stats
        self._energy = _STARTING_ENERGY
        # Successful uses of speed, stamina, power, guts, wit facilities.
        self._facility_uses = (0, 0, 0, 0, 0)

        self._card_states = tuple(
            CardState(card.initial_friendship) for card in game.cards
        )
        self._placement_index = 0
        self._phase = _Phase.PLACEMENT

        self._score = float(
            sum(_stat_score(stat) for stat in game.initial_stats[:5])
        )
        self._last_reward = 0.0
        self._pending_action: int | None = None
        self._soft_failed = False

    def current_player(self):
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._phase == _Phase.DECISION:
            return 0
        return pyspiel.PlayerId.CHANCE

    def action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self._phase == _Phase.PLACEMENT:
                card = self.get_game().cards[self._placement_index]
                where = (
                    "away"
                    if action == _PLACEMENT_AWAY
                    else self.action_to_string(0, action)
                )
                return f"{card.name} -> {where}"
            match action:
                case 0:
                    return "fail"
                case 1:
                    return "success"
                case _:
                    raise ValueError(f"Invalid chance action: {action}")
        match action:
            case 0:
                return "rest"
            case 1:
                return "train_speed"
            case 2:
                return "train_stamina"
            case 3:
                return "train_power"
            case 4:
                return "train_guts"
            case 5:
                return "train_wit"
            case _:
                raise ValueError(f"Invalid action: {action}")

    def legal_actions(self, player=None):
        if self._phase == _Phase.PLACEMENT:
            return list(_PLACEMENT_ACTIONS)
        if self._phase == _Phase.RESULT:
            return list(_RESULT_ACTIONS)
        return list(_DECISION_ACTIONS)

    def _permanent_facility_level_for(self, action: int) -> int:
        return _facility_level(self._facility_uses[action - 1])

    def _facility_level_for(self, action: int) -> int:
        if self.is_summer_camp():
            return _MAX_FACILITY_LEVEL
        return self._permanent_facility_level_for(action)

    def _attending(self, action: int) -> list[int]:
        """Indices of the support cards on a training facility this turn."""
        assert action in _TRAINING_ACTIONS
        return [
            index
            for index, card_state in enumerate(self._card_states)
            if card_state.placement == action
        ]

    def _is_rainbow(self, index: int) -> bool:
        return self._card_states[index].friendship >= _RAINBOW_FRIENDSHIP

    def _wit_energy_recovery(self) -> int:
        # Only rainbowed cards standing on the wit facility recover energy.
        cards = self.get_game().cards
        return sum(
            cards[index].wit_friendship_recovery
            for index in self._attending(_WIT_ACTION)
            if self._is_rainbow(index)
        )

    def _energy_delta(self, action: int) -> int:
        level = 1 if action == _REST_ACTION else self._facility_level_for(action)
        delta = _TRAINING_ENERGY[action][level - 1]
        if action == _WIT_ACTION:
            delta += self._wit_energy_recovery()
        return delta

    def _training_gains(self, action: int) -> tuple[int, ...]:
        """Stat gains of a successful training, including support card effects."""
        if action == _REST_ACTION:
            return (0,) * _NUM_STATS
        base = _TRAINING_STATS[action][self._facility_level_for(action) - 1]
        cards = self.get_game().cards
        attending = self._attending(action)

        friendship_multiplier = 1.0
        mood_effect = 0.0
        training_effectiveness = 0.0
        stat_bonus = [0] * _NUM_STATS
        for index in attending:
            card = cards[index]
            if _TRAINING_FOR_STAT[card.main_stat] == action and self._is_rainbow(
                index
            ):
                friendship_multiplier *= 1.0 + card.friendship_bonus / 100.0
            mood_effect += card.mood_effect
            training_effectiveness += card.training_effectiveness
            for stat, amount in card.stat_bonus.items():
                stat_bonus[_STAT_INDEX[stat]] += amount

        multiplier = _training_multiplier(
            friendship_multiplier,
            mood_effect,
            training_effectiveness,
            len(attending),
        )
        # A stat bonus only applies to stats the training already grants.
        # UmaGrowth is per-stat (Bourbon: 0.2 stamina, 0.1 power).
        return tuple(
            math.floor(
                (amount + stat_bonus[index])
                * multiplier
                * (1.0 + _UMA_GROWTH[index])
            )
            if amount
            else 0
            for index, amount in enumerate(base)
        )

    def chance_outcomes(self):
        assert self.is_chance_node()
        if self._phase == _Phase.PLACEMENT:
            return self.get_game().placement_outcomes[self._placement_index]
        assert self._pending_action is not None
        energy_after = _clip_energy(
            self._energy + self._energy_delta(self._pending_action)
        )
        p_fail = _stat_train_failure_chance(energy_after)
        return [(_CHANCE_FAIL, p_fail), (_CHANCE_SUCCESS, 1.0 - p_fail)]

    def _apply_stats(self, gains: Sequence[int]) -> tuple[int, ...]:
        """Applies stat changes and returns the deltas that actually landed."""
        before = (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
        )
        after = []
        deltas = []
        for old, gain in zip(before, gains):
            new = old + gain
            if new < _MIN_STAT:
                new = _MIN_STAT
            elif new > _MAX_STAT:
                new = _MAX_STAT
            after.append(new)
            deltas.append(new - old)
        (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
        ) = after
        self._skill_points += gains[5]
        deltas.append(gains[5])
        return tuple(deltas)

    def _apply_training_result(self, action: int, success: bool) -> None:
        if success:
            gains = self._training_gains(action)
            # Energy recovery is read before friendships move this turn.
            self._energy = _clip_energy(self._energy + self._energy_delta(action))
            if action in _TRAINING_ACTIONS:
                uses = list(self._facility_uses)
                uses[action - 1] += 1
                self._facility_uses = tuple(uses)
                card_states = list(self._card_states)
                for index in self._attending(action):
                    card_state = card_states[index]
                    card_states[index] = CardState(
                        min(
                            _MAX_FRIENDSHIP,
                            card_state.friendship + _FRIENDSHIP_PER_TRAINING,
                        ),
                        card_state.placement,
                    )
                self._card_states = tuple(card_states)
        else:
            gains = _FAIL_STATS[action]

        before = (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
        )
        applied = self._apply_stats(gains)
        after = (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
        )
        self._last_reward = sum(
            _stat_score(new) - _stat_score(old) for new, old in zip(after, before)
        ) + _SKILL_POINT_WEIGHT * applied[5]
        self._score += self._last_reward

        self._turn += 1
        self._pending_action = None
        self._begin_turn()

    def _begin_turn(self) -> None:
        if self._turn >= _MAX_TURNS or self._soft_failed:
            return
        if self._turn + 1 == _TENNO_SHO_SPRING_TURN:
            self._resolve_tenno_sho_spring()
            return
        self._card_states = tuple(
            CardState(card_state.friendship, _PLACEMENT_AWAY)
            for card_state in self._card_states
        )
        self._placement_index = 0
        self._phase = _Phase.PLACEMENT

    def _resolve_tenno_sho_spring(self) -> None:
        # Year 3 Late April is the race, not a training turn.
        self._turn += 1
        if (
            self._speed < _TENNO_SHO_MIN_SPEED
            or self._stamina < _TENNO_SHO_MIN_STAMINA
        ):
            self._soft_failed = True
            return
        self._begin_turn()

    def apply_action(self, action):
        if self._phase == _Phase.PLACEMENT:
            if action not in range(_NUM_PLACEMENT_OUTCOMES):
                raise ValueError(f"Invalid placement outcome: {action}")
            index = self._placement_index
            card_states = list(self._card_states)
            card_states[index] = CardState(card_states[index].friendship, action)
            self._card_states = tuple(card_states)
            self._placement_index = index + 1
            if index + 1 == _NUM_CARDS:
                self._phase = _Phase.DECISION
            return

        if self._phase == _Phase.RESULT:
            assert self._pending_action is not None
            self._apply_training_result(
                self._pending_action, success=(action == _CHANCE_SUCCESS)
            )
            return

        if action not in range(_NUM_ACTIONS):
            raise ValueError(f"Invalid action: {action}")

        energy_after = _clip_energy(self._energy + self._energy_delta(action))
        fail_p = (
            _stat_train_failure_chance(energy_after)
            if action in _STAT_TRAIN_ACTIONS
            else 0.0
        )
        if 0.0 < fail_p < 1.0:
            self._pending_action = action
            self._phase = _Phase.RESULT
            return
        self._apply_training_result(action, success=(fail_p == 0.0))

    def is_terminal(self):
        return self._turn >= _MAX_TURNS or self._soft_failed

    def ended_by_tenno_sho_fail(self) -> bool:
        return self._soft_failed

    def _current_turn_1based(self) -> int:
        if self.is_terminal():
            return self._turn if self._turn > 0 else 1
        return self._turn + 1

    def is_summer_camp(self) -> bool:
        return _SUMMER_CAMP_TURNS[self._current_turn_1based()]

    def rewards(self):
        return [self._last_reward]

    def returns(self):
        return [self._score]

    def _friendship_string(self) -> str:
        cards = self.get_game().cards
        return ", ".join(
            f"{card.name} {card_state.friendship}"
            for card, card_state in zip(cards, self._card_states)
        )

    def _placement_string(self) -> str:
        cards = self.get_game().cards
        by_place: dict[int, list[str]] = {}
        for index in range(self._placement_index):
            by_place.setdefault(self._card_states[index].placement, []).append(
                cards[index].name
            )
        parts = []
        for place, names in sorted(by_place.items()):
            where = (
                "away" if place == _PLACEMENT_AWAY else self.action_to_string(0, place)
            )
            parts.append(f"{where}: {'/'.join(names)}")
        return ", ".join(parts) if parts else "unrolled"

    def __str__(self):
        career = (
            f"{calendar_label(self._current_turn_1based())}, "
            f"Speed: {self._speed}, Stamina: {self._stamina}, "
            f"Power: {self._power}, Guts: {self._guts}, Wit: {self._wit}, "
            f"Skill points: {self._skill_points}, Energy: {self._energy}, "
            f"Facilities: speed {self._facility_level_for(1)}, "
            f"stamina {self._facility_level_for(2)}, "
            f"power {self._facility_level_for(3)}, "
            f"guts {self._facility_level_for(4)}, "
            f"wit {self._facility_level_for(5)}"
        )
        lines = [career, f"Friendship: {self._friendship_string()}"]
        if self._soft_failed:
            lines.append(
                "Ended: Tenno Sho (Spring) soft fail "
                f"(need {_TENNO_SHO_MIN_SPEED} speed and "
                f"{_TENNO_SHO_MIN_STAMINA} stamina)"
            )
        if not self.is_terminal():
            lines.append(f"Supports: {self._placement_string()}")
        return "\n".join(lines)


class UmaObserver:
    def __init__(self, params=None):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # 13 career features, then a placement one-hot and friendship per card.
        self.tensor = np.zeros(
            13 + _NUM_CARDS * (_NUM_PLACEMENT_OUTCOMES + 1), np.float32
        )
        self.dict = {"observation": self.tensor}

    def set_from(self, state: UmaState, player):
        del player
        # Scaled to roughly [0, 1] so the values are usable as network inputs.
        self.tensor[:] = 0.0
        self.tensor[:13] = (
            state._turn / _MAX_TURNS,
            state._speed / _MAX_STAT,
            state._stamina / _MAX_STAT,
            state._power / _MAX_STAT,
            state._guts / _MAX_STAT,
            state._wit / _MAX_STAT,
            state._skill_points / state.get_game().skill_point_obs_scale,
            state._energy / _STARTING_ENERGY,
            state._facility_level_for(1) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(2) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(3) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(4) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(5) / _MAX_FACILITY_LEVEL,
        )
        offset = 13
        for index in range(_NUM_CARDS):
            card_state = state._card_states[index]
            self.tensor[offset + card_state.placement] = 1.0
            self.tensor[offset + _NUM_PLACEMENT_OUTCOMES] = (
                card_state.friendship / _MAX_FRIENDSHIP
            )
            offset += _NUM_PLACEMENT_OUTCOMES + 1

    def string_from(self, state: UmaState, player):
        del player
        return str(state)


pyspiel.register_game(_GAME_TYPE, UmaGame)
