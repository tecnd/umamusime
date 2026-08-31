from typing import Any

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
    provides_observation_tensor=False,
    parameter_specification={})

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=6,
    max_chance_outcomes=3,
    num_players=1,
    min_utility=0.0,
    max_utility=20000.0,
    max_game_length=60)

class UmaGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or {})
    def new_initial_state(self, state=None):
        return UmaState(self, state)

class UmaState(pyspiel.State):
    def __init__(self, game: UmaGame, state:pyspiel.StateStruct | JsonDict | None=None):
        super().__init__(game)
        self._is_chance_node = False
        self._turn = 0

    def current_player(self):
        if self._is_chance_node:
            return pyspiel.PlayerId.CHANCE
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

    def apply_action(self, action): ...
    def is_terminal(self):
        return self._turn >= _GAME_INFO.max_game_length
    def returns(self): ...
    def __str__(self): ...

pyspiel.register_game(_GAME_TYPE, UmaGame)
