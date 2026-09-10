from typing import Any

import numpy as np
import pyspiel

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

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=6,
    max_chance_outcomes=2,
    num_players=1,
    min_utility=0.0,
    max_utility=20000.0,
    max_game_length=_MAX_TURNS,
)

# Score awarded per successful action, indexed by action id.
_ACTION_REWARDS = (0.0, 3.0, 1.0, 1.0, 1.0, 1.5)

# Skill points granted on a successful action: rest, speed, stamina, power,
# guts, wit. There is no dedicated skill-point training.
_ACTION_SKILL_POINTS = (0, 2, 2, 2, 2, 4)
_SKILL_POINT_WEIGHT = 0.9

# Facility levels 1–5. Rest is not a facility; its row is unused.
# Each entry is (speed, stamina, power, guts, wit) on a successful training.
_TRAINING_STATS = (
    ((0, 0, 0, 0, 0),) * 5,
    (
        (10, 0, 5, 0, 0),
        (11, 0, 5, 0, 0),
        (12, 0, 5, 0, 0),
        (13, 0, 6, 0, 0),
        (14, 0, 7, 0, 0),
    ),
    (
        (0, 9, 0, 4, 0),
        (0, 10, 0, 4, 0),
        (0, 11, 0, 4, 0),
        (0, 12, 0, 5, 0),
        (0, 13, 0, 6, 0),
    ),
    (
        (0, 5, 8, 0, 0),
        (0, 5, 9, 0, 0),
        (0, 5, 10, 0, 0),
        (0, 6, 11, 0, 0),
        (0, 7, 12, 0, 0),
    ),
    (
        (4, 0, 4, 8, 0),
        (4, 0, 4, 9, 0),
        (4, 0, 4, 10, 0),
        (5, 0, 4, 11, 0),
        (5, 0, 5, 12, 0),
    ),
    (
        (2, 0, 0, 0, 9),
        (2, 0, 0, 0, 10),
        (2, 0, 0, 0, 11),
        (3, 0, 0, 0, 12),
        (4, 0, 0, 0, 13),
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

# (speed, stamina, power, guts, wit) applied when that training fails.
_FAIL_STATS = (
    (0, 0, 0, 0, 0),
    (-10, 0, 0, 0, 0),
    (0, -10, 0, 0, 0),
    (0, 0, -10, 0, 0),
    (0, 0, 0, -10, 0),
    (0, 0, 0, 0, 0),
)

_STAT_TRAIN_ACTIONS = frozenset({1, 2, 3, 4})
_TRAINING_ACTIONS = frozenset({1, 2, 3, 4, 5})
_MIN_FACILITY_LEVEL = 1
_MAX_FACILITY_LEVEL = 5
_USES_PER_FACILITY_LEVEL = 4

_MAX_ENERGY = 100
_STARTING_ENERGY = _MAX_ENERGY
_MIN_STAT = 0
_MAX_STAT = 1200
# Skill points are uncapped; this is only used to scale the observation.
_SKILL_POINT_OBS_SCALE = max(_ACTION_SKILL_POINTS) * _GAME_INFO.max_game_length
_FAIL_FREE_ENERGY = 50
_FAIL_CHANCE_AT_ZERO = 0.99

_CHANCE_FAIL = 0
_CHANCE_SUCCESS = 1


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


class UmaGame(pyspiel.Game):
    def __init__(self, params=None, *, reward_model=None):
        game_type = (
            _make_game_type(reward_model) if reward_model is not None else _GAME_TYPE
        )
        super().__init__(game_type, _GAME_INFO, params or {})

    def new_initial_state(self, state=None):
        return UmaState(self, state)

    def make_py_observer(self, iig_obs_type=None, params=None):
        return UmaObserver(params)


class UmaState(pyspiel.State):
    def __init__(
        self, game: UmaGame, state: pyspiel.StateStruct | JsonDict | None = None
    ):
        super().__init__(game)
        self._is_chance_node = False
        self._turn = 0

        self._speed = 0
        self._stamina = 0
        self._power = 0
        self._guts = 0
        self._wit = 0
        self._skill_points = 0
        self._energy = _STARTING_ENERGY
        # Successful uses of speed, stamina, power, guts, wit facilities.
        self._facility_uses = (0, 0, 0, 0, 0)

        self._score = 0.0
        self._last_reward = 0.0
        self._pending_action: int | None = None

    def current_player(self):
        if self._is_chance_node:
            return pyspiel.PlayerId.CHANCE
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        return 0

    def action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
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
        if self._is_chance_node:
            return [_CHANCE_FAIL, _CHANCE_SUCCESS]
        return [a for a in range(_GAME_INFO.num_distinct_actions)]

    def _permanent_facility_level_for(self, action: int) -> int:
        return _facility_level(self._facility_uses[action - 1])

    def _facility_level_for(self, action: int) -> int:
        if self.is_summer_camp():
            return _MAX_FACILITY_LEVEL
        return self._permanent_facility_level_for(action)

    def _energy_delta(self, action: int) -> int:
        level = 1 if action == 0 else self._facility_level_for(action)
        return _TRAINING_ENERGY[action][level - 1]

    def chance_outcomes(self):
        assert self.is_chance_node()
        assert self._pending_action is not None
        energy_after = _clip_energy(
            self._energy + self._energy_delta(self._pending_action)
        )
        p_fail = _stat_train_failure_chance(energy_after)
        return [(_CHANCE_FAIL, p_fail), (_CHANCE_SUCCESS, 1.0 - p_fail)]

    def _apply_training_result(self, action: int, success: bool) -> None:
        if success:
            self._energy = _clip_energy(self._energy + self._energy_delta(action))
            if action in _TRAINING_ACTIONS:
                speed, stamina, power, guts, wit = _TRAINING_STATS[action][
                    self._facility_level_for(action) - 1
                ]
                uses = list(self._facility_uses)
                uses[action - 1] += 1
                self._facility_uses = tuple(uses)
            else:
                speed, stamina, power, guts, wit = (0, 0, 0, 0, 0)
            skill_points = _ACTION_SKILL_POINTS[action]
            self._skill_points += skill_points
            self._last_reward = (
                _ACTION_REWARDS[action] + _SKILL_POINT_WEIGHT * skill_points
            )
            self._score += self._last_reward
        else:
            speed, stamina, power, guts, wit = _FAIL_STATS[action]
            self._last_reward = 0.0
        self._speed = _clip_stat(self._speed + speed)
        self._stamina = _clip_stat(self._stamina + stamina)
        self._power = _clip_stat(self._power + power)
        self._guts = _clip_stat(self._guts + guts)
        self._wit = _clip_stat(self._wit + wit)
        self._turn += 1
        self._pending_action = None
        self._is_chance_node = False

    def apply_action(self, action):
        if self._is_chance_node:
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
            self._is_chance_node = True
            return
        self._apply_training_result(action, success=(fail_p == 0.0))

    def is_terminal(self):
        return self._turn >= _GAME_INFO.max_game_length

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

    def __str__(self):
        return (
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


class UmaObserver:
    def __init__(self, params=None):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        self.tensor = np.zeros(13, np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state: UmaState, player):
        del player
        # Scaled to roughly [0, 1] so the values are usable as network inputs.
        self.tensor[:] = (
            state._turn / _GAME_INFO.max_game_length,
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

    def string_from(self, state: UmaState, player):
        del player
        return str(state)


pyspiel.register_game(_GAME_TYPE, UmaGame)
