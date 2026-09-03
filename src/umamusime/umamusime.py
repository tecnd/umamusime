from typing import Any

import numpy as np
import pyspiel

JsonDict = dict[str, Any]

_GAME_TYPE = pyspiel.GameType(
    short_name="umamusime",
    long_name="Umamusime",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=1,
    min_num_players=1,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={},
)

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=6,
    max_chance_outcomes=3,
    num_players=1,
    min_utility=0.0,
    max_utility=20000.0,
    max_game_length=60,
)

# Score awarded per action, indexed by action id.
_ACTION_REWARDS = (0.0, 3.0, 1.0, 1.0, 1.0, 1.5)

_STARTING_ENERGY = 100


class UmaGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})

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
        self._energy = _STARTING_ENERGY

        self._score = 0.0
        self._last_reward = 0.0

    def current_player(self):
        if self._is_chance_node:
            return pyspiel.PlayerId.CHANCE
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        return 0

    def action_to_string(self, player, action):
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
        return [a for a in range(_GAME_INFO.num_distinct_actions)]

    def apply_action(self, action):
        match action:
            case 0:
                self._energy += 10
            case 1:
                self._speed += 1
                self._energy = max(0, self._energy - 10)
            case 2:
                self._stamina += 1
                self._energy = max(0, self._energy - 10)
            case 3:
                self._power += 1
                self._energy = max(0, self._energy - 10)
            case 4:
                self._guts += 1
                self._energy = max(0, self._energy - 10)
            case 5:
                self._wit += 1
                self._energy = max(0, self._energy - 10)
            case _:
                raise ValueError(f"Invalid action: {action}")

        self._last_reward = _ACTION_REWARDS[action]
        self._score += self._last_reward
        self._turn += 1

    def is_terminal(self):
        return self._turn >= _GAME_INFO.max_game_length

    def rewards(self):
        return [self._last_reward]

    def returns(self):
        return [self._score]

    def __str__(self):
        return f"Turn: {self._turn}, Speed: {self._speed}, Stamina: {self._stamina}, Power: {self._power}, Guts: {self._guts}, Wit: {self._wit}, Energy: {self._energy}"


class UmaObserver:
    def __init__(self, params=None):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        self.tensor = np.zeros(7, np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state: UmaState, player):
        del player
        # Scaled to roughly [0, 1] so the values are usable as network inputs.
        max_stat = _GAME_INFO.max_game_length
        self.tensor[:] = (
            state._turn / max_stat,
            state._speed / max_stat,
            state._stamina / max_stat,
            state._power / max_stat,
            state._guts / max_stat,
            state._wit / max_stat,
            state._energy / _STARTING_ENERGY,
        )

    def string_from(self, state: UmaState, player):
        del player
        return str(state)


pyspiel.register_game(_GAME_TYPE, UmaGame)
