import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyspiel

from .cards import DEFAULT_DECK, STATS, SupportCard

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
        parameter_specification={},
    )


# Per-turn rewards are real (used by DQN). MCTSBot only checks this metadata
# and can be given TERMINAL via UmaGame(reward_model=...).
_GAME_TYPE = _make_game_type()

# 3 years × 12 months × 2 half-months.
_MAX_TURNS = 72
_TURNS_PER_YEAR = 24
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_SUMMER_CAMP_YEARS = frozenset({2, 3})
_SUMMER_CAMP_MONTHS = frozenset({"July", "August"})

_NUM_STATS = len(STATS)
_STAT_INDEX = {stat: index for index, stat in enumerate(STATS)}
_NUM_CARDS = 6
# Support cards attend one of the five training facilities, or nobody's.
_PLACEMENT_AWAY = 0
_NUM_PLACEMENT_OUTCOMES = 6

# A turn is six placement rolls, the player's action, then a fail/success roll.
_NODES_PER_TURN = _NUM_CARDS + 2

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=6,
    max_chance_outcomes=_NUM_PLACEMENT_OUTCOMES,
    num_players=1,
    min_utility=-5000.0,
    max_utility=50000.0,
    max_game_length=_MAX_TURNS * _NODES_PER_TURN,
)

# Score per point of (speed, stamina, power, guts, wit, skill points) gained.
_STAT_WEIGHTS = (3.0, 1.0, 1.0, 1.0, 1.5, 0.9)

# Facility levels 1–5. Rest is not a facility; its row is unused.
# Each entry is (speed, stamina, power, guts, wit, skill points) before support
# card bonuses, on a successful training.
_TRAINING_STATS = (
    ((0, 0, 0, 0, 0, 0),) * 5,
    (
        (11, 0, 6, 0, 0, 4),
        (12, 0, 6, 0, 0, 4),
        (13, 0, 6, 0, 0, 4),
        (14, 0, 7, 0, 0, 4),
        (15, 0, 8, 0, 0, 4),
    ),
    (
        (0, 10, 0, 6, 0, 4),
        (0, 11, 0, 6, 0, 4),
        (0, 12, 0, 6, 0, 4),
        (0, 13, 0, 7, 0, 4),
        (0, 14, 0, 8, 0, 4),
    ),
    (
        (0, 6, 9, 0, 0, 4),
        (0, 6, 10, 0, 0, 4),
        (0, 6, 11, 0, 0, 4),
        (0, 7, 12, 0, 0, 4),
        (0, 8, 13, 0, 0, 4),
    ),
    (
        (5, 0, 5, 8, 0, 4),
        (5, 0, 5, 9, 0, 4),
        (5, 0, 5, 10, 0, 4),
        (6, 0, 5, 11, 0, 4),
        (6, 0, 6, 12, 0, 4),
    ),
    (
        (2, 0, 0, 0, 10, 5),
        (2, 0, 0, 0, 11, 5),
        (2, 0, 0, 0, 12, 5),
        (3, 0, 0, 0, 13, 5),
        (4, 0, 0, 0, 14, 5),
    ),
)

# Energy change at facility levels 1–5. Rest is always +50.
_TRAINING_ENERGY = (
    (50, 50, 50, 50, 50),
    (-21, -22, -23, -25, -27),
    (-19, -20, -21, -23, -25),
    (-20, -21, -22, -24, -26),
    (-22, -23, -24, -26, -28),
    (5, 5, 5, 5, 5),
)

# Stat change applied when that training fails.
_FAIL_STATS = (
    (0, 0, 0, 0, 0, 0),
    (-10, 0, 0, 0, 0, 0),
    (0, -10, 0, 0, 0, 0),
    (0, 0, -10, 0, 0, 0),
    (0, 0, 0, -10, 0, 0),
    (0, 0, 0, 0, 0, 0),
)

_REST_ACTION = 0
_WIT_ACTION = 5
_STAT_TRAIN_ACTIONS = frozenset({1, 2, 3, 4})
_TRAINING_ACTIONS = frozenset({1, 2, 3, 4, 5})
# Training action that a support card's main stat maps to.
_TRAINING_FOR_STAT = {stat: index + 1 for index, stat in enumerate(STATS[:5])}
_MIN_FACILITY_LEVEL = 1
_MAX_FACILITY_LEVEL = 5
_USES_PER_FACILITY_LEVEL = 4

# Support card placement weights, before specialty priority.
_TRAINING_WEIGHT = 100.0
_AWAY_WEIGHT = 50.0

# Friendship gauges run 0–100; 80 or more "rainbows" the card.
_MAX_FRIENDSHIP = 100
_RAINBOW_FRIENDSHIP = 80
_FRIENDSHIP_PER_TRAINING = 5

# Fixed inputs to the stat gain formula.
_BASE_MOOD = 0.2
_UMA_GROWTH = 0.0
_PER_CARD_BONUS = 0.05

_MAX_ENERGY = 100
_STARTING_ENERGY = _MAX_ENERGY
_MIN_STAT = 0
_MAX_STAT = 1200
# Skill points are uncapped; this is only used to scale the observation.
_SKILL_POINT_OBS_SCALE = 1000

_FAIL_FREE_ENERGY = 50
_FAIL_CHANCE_AT_ZERO = 0.99

_CHANCE_FAIL = 0
_CHANCE_SUCCESS = 1

_PHASE_PLACEMENT = 0
_PHASE_DECISION = 1
_PHASE_RESULT = 2


def _calendar_parts(turn_1based: int) -> tuple[int, str, str]:
    index = turn_1based - 1
    year = index // _TURNS_PER_YEAR + 1
    month = _MONTHS[(index % _TURNS_PER_YEAR) // 2]
    half = "Early" if index % 2 == 0 else "Late"
    return year, month, half


def calendar_label(turn_1based: int) -> str:
    """Calendar date for a 1-based turn in 1..72.

    Turn 1 is Year 1, Early January; turn 72 is Year 3, Late December.
    """
    year, month, half = _calendar_parts(turn_1based)
    return f"Year {year}, {half} {month}"


def _is_summer_camp_turn(turn_1based: int) -> bool:
    # Years 2–3, Early July through Late August inclusive.
    year, month, _half = _calendar_parts(turn_1based)
    return year in _SUMMER_CAMP_YEARS and month in _SUMMER_CAMP_MONTHS


def _clip_energy(energy: int) -> int:
    return max(0, min(_MAX_ENERGY, energy))


def _clip_stat(value: int) -> int:
    return max(_MIN_STAT, min(_MAX_STAT, value))


def _facility_level(uses: int) -> int:
    return min(
        _MAX_FACILITY_LEVEL,
        _MIN_FACILITY_LEVEL + uses // _USES_PER_FACILITY_LEVEL,
    )


def _stat_train_failure_chance(energy_after: int) -> float:
    # Remaining energy >= 50 never fails; 0 is 99% fail; linear in between.
    if energy_after >= _FAIL_FREE_ENERGY:
        return 0.0
    remaining = max(energy_after, 0)
    return (
        _FAIL_CHANCE_AT_ZERO
        * (_FAIL_FREE_ENERGY - remaining)
        / _FAIL_FREE_ENERGY
    )


def _placement_outcomes(card: SupportCard) -> list[tuple[int, float]]:
    """Where a card shows up this turn: away, or one of the five trainings."""
    weights = [_AWAY_WEIGHT] + [_TRAINING_WEIGHT] * len(_TRAINING_ACTIONS)
    weights[_TRAINING_FOR_STAT[card.main_stat]] += card.specialty_priority
    total = sum(weights)
    return [(outcome, weight / total) for outcome, weight in enumerate(weights)]


def _summed_initial_stats(cards: Sequence[SupportCard]) -> tuple[int, ...]:
    totals = [0] * _NUM_STATS
    for card in cards:
        for stat, amount in card.initial_stats.items():
            totals[_STAT_INDEX[stat]] += amount
    return tuple(totals)


class UmaGame(pyspiel.Game):
    def __init__(self, params=None, *, reward_model=None, cards=DEFAULT_DECK):
        game_type = (
            _make_game_type(reward_model) if reward_model is not None else _GAME_TYPE
        )
        super().__init__(game_type, _GAME_INFO, params or {})
        self.cards = tuple(cards)
        if len(self.cards) != _NUM_CARDS:
            raise ValueError(
                f"Expected a deck of {_NUM_CARDS} support cards, "
                f"got {len(self.cards)}"
            )
        self.placement_outcomes = tuple(
            _placement_outcomes(card) for card in self.cards
        )
        self.initial_stats = _summed_initial_stats(self.cards)

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

        self._friendship = tuple(card.initial_friendship for card in game.cards)
        self._placements = (_PLACEMENT_AWAY,) * _NUM_CARDS
        self._placement_index = 0
        self._phase = _PHASE_PLACEMENT

        self._score = 0.0
        self._last_reward = 0.0
        self._pending_action: int | None = None

    def current_player(self):
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._phase == _PHASE_DECISION:
            return 0
        return pyspiel.PlayerId.CHANCE

    def action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self._phase == _PHASE_PLACEMENT:
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
        if self._phase == _PHASE_PLACEMENT:
            return list(range(_NUM_PLACEMENT_OUTCOMES))
        if self._phase == _PHASE_RESULT:
            return [_CHANCE_FAIL, _CHANCE_SUCCESS]
        return [a for a in range(_GAME_INFO.num_distinct_actions)]

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
            for index, placement in enumerate(self._placements)
            if placement == action
        ]

    def _is_rainbow(self, index: int) -> bool:
        return self._friendship[index] >= _RAINBOW_FRIENDSHIP

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

        multiplier = (
            friendship_multiplier
            * (1.0 + _BASE_MOOD * (1.0 + mood_effect / 100.0))
            * (1.0 + training_effectiveness / 100.0)
            * (1.0 + _PER_CARD_BONUS * len(attending))
            * (1.0 + _UMA_GROWTH / 100.0)
        )
        # A stat bonus only applies to stats the training already grants.
        return tuple(
            math.floor((amount + stat_bonus[index]) * multiplier) if amount else 0
            for index, amount in enumerate(base)
        )

    def chance_outcomes(self):
        assert self.is_chance_node()
        if self._phase == _PHASE_PLACEMENT:
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
        after = tuple(_clip_stat(old + gain) for old, gain in zip(before, gains))
        (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
        ) = after
        self._skill_points += gains[5]
        return tuple(new - old for new, old in zip(after, before)) + (gains[5],)

    def _apply_training_result(self, action: int, success: bool) -> None:
        if success:
            gains = self._training_gains(action)
            # Energy recovery is read before friendships move this turn.
            self._energy = _clip_energy(self._energy + self._energy_delta(action))
            if action in _TRAINING_ACTIONS:
                uses = list(self._facility_uses)
                uses[action - 1] += 1
                self._facility_uses = tuple(uses)
                friendship = list(self._friendship)
                for index in self._attending(action):
                    friendship[index] = min(
                        _MAX_FRIENDSHIP,
                        friendship[index] + _FRIENDSHIP_PER_TRAINING,
                    )
                self._friendship = tuple(friendship)
        else:
            gains = _FAIL_STATS[action]

        applied = self._apply_stats(gains)
        self._last_reward = sum(
            weight * delta for weight, delta in zip(_STAT_WEIGHTS, applied)
        )
        self._score += self._last_reward

        self._turn += 1
        self._pending_action = None
        self._begin_turn()

    def _begin_turn(self) -> None:
        self._placements = (_PLACEMENT_AWAY,) * _NUM_CARDS
        self._placement_index = 0
        self._phase = _PHASE_PLACEMENT

    def apply_action(self, action):
        if self._phase == _PHASE_PLACEMENT:
            if action not in range(_NUM_PLACEMENT_OUTCOMES):
                raise ValueError(f"Invalid placement outcome: {action}")
            placements = list(self._placements)
            placements[self._placement_index] = action
            self._placements = tuple(placements)
            self._placement_index += 1
            if self._placement_index == _NUM_CARDS:
                self._phase = _PHASE_DECISION
            return

        if self._phase == _PHASE_RESULT:
            assert self._pending_action is not None
            self._apply_training_result(
                self._pending_action, success=(action == _CHANCE_SUCCESS)
            )
            return

        if action not in range(_GAME_INFO.num_distinct_actions):
            raise ValueError(f"Invalid action: {action}")

        energy_after = _clip_energy(self._energy + self._energy_delta(action))
        fail_p = (
            _stat_train_failure_chance(energy_after)
            if action in _STAT_TRAIN_ACTIONS
            else 0.0
        )
        if 0.0 < fail_p < 1.0:
            self._pending_action = action
            self._phase = _PHASE_RESULT
            return
        self._apply_training_result(action, success=(fail_p == 0.0))

    def is_terminal(self):
        return self._turn >= _MAX_TURNS

    def _current_turn_1based(self) -> int:
        if self._turn >= _MAX_TURNS:
            return _MAX_TURNS
        return self._turn + 1

    def is_summer_camp(self) -> bool:
        return _is_summer_camp_turn(self._current_turn_1based())

    def rewards(self):
        return [self._last_reward]

    def returns(self):
        return [self._score]

    def _friendship_string(self) -> str:
        cards = self.get_game().cards
        return ", ".join(
            f"{card.name} {friendship}"
            for card, friendship in zip(cards, self._friendship)
        )

    def _placement_string(self) -> str:
        cards = self.get_game().cards
        by_place: dict[int, list[str]] = {}
        for index in range(self._placement_index):
            by_place.setdefault(self._placements[index], []).append(
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
            state._skill_points / _SKILL_POINT_OBS_SCALE,
            state._energy / _STARTING_ENERGY,
            state._facility_level_for(1) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(2) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(3) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(4) / _MAX_FACILITY_LEVEL,
            state._facility_level_for(5) / _MAX_FACILITY_LEVEL,
        )
        offset = 13
        for index in range(_NUM_CARDS):
            self.tensor[offset + state._placements[index]] = 1.0
            self.tensor[offset + _NUM_PLACEMENT_OUTCOMES] = (
                state._friendship[index] / _MAX_FRIENDSHIP
            )
            offset += _NUM_PLACEMENT_OUTCOMES + 1

    def string_from(self, state: UmaState, player):
        del player
        return str(state)


pyspiel.register_game(_GAME_TYPE, UmaGame)
