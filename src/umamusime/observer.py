from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .actions import NUM_CARDS, NUM_PLACEMENT_OUTCOMES, TRAINING_ACTIONS
from .calendar import MAX_TURNS
from .scoring import MAX_STAT
from .training import MAX_FACILITY_LEVEL, MAX_FRIENDSHIP, STARTING_ENERGY

if TYPE_CHECKING:
    from .state import UmaState


class UmaObserver:
    _BASE_FEATURES = 13 + NUM_CARDS * (NUM_PLACEMENT_OUTCOMES + 1)
    _RACE_FEATURES = 4
    _ACTION_FEATURES = len(TRAINING_ACTIONS) * 9
    _CARD_FEATURES = NUM_CARDS
    _RAINBOW_FEATURES = len(TRAINING_ACTIONS)
    OBSERVATION_SIZE = (
        _BASE_FEATURES
        + _RACE_FEATURES
        + _ACTION_FEATURES
        + _CARD_FEATURES
        + _RAINBOW_FEATURES
    )

    def __init__(self, params=None):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # Base career/card features, then race margins and action outcomes.
        self.tensor = np.zeros(self.OBSERVATION_SIZE, np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state: UmaState, player):
        del player
        # Scaled to roughly [0, 1] so the values are usable as network inputs.
        speed, stamina, power, guts, wit, skill_points = state.stats
        self.tensor[:] = 0.0
        self.tensor[:13] = (
            state.turn / MAX_TURNS,
            speed / MAX_STAT,
            stamina / MAX_STAT,
            power / MAX_STAT,
            guts / MAX_STAT,
            wit / MAX_STAT,
            skill_points / state.get_game().skill_point_obs_scale,
            state.energy / STARTING_ENERGY,
            state.facility_level(1) / MAX_FACILITY_LEVEL,
            state.facility_level(2) / MAX_FACILITY_LEVEL,
            state.facility_level(3) / MAX_FACILITY_LEVEL,
            state.facility_level(4) / MAX_FACILITY_LEVEL,
            state.facility_level(5) / MAX_FACILITY_LEVEL,
        )
        offset = 13
        for card_state in state.card_states:
            self.tensor[offset + card_state.placement] = 1.0
            self.tensor[offset + NUM_PLACEMENT_OUTCOMES] = (
                card_state.friendship / MAX_FRIENDSHIP
            )
            offset += NUM_PLACEMENT_OUTCOMES + 1

        races = state.get_game().races
        next_race = next((race for race in races if race.turn >= state.turn), None)
        race_features = [0.0] * self._RACE_FEATURES
        if next_race is not None:
            race_features = [
                (next_race.turn - state.turn) / MAX_TURNS,
                (speed - next_race.min_speed) / MAX_STAT,
                (stamina - next_race.min_stamina) / MAX_STAT,
                0.0,
            ]
        remaining_races = [race for race in races if race.turn >= state.turn]
        if remaining_races:
            race_features[3] = (
                stamina - max(race.min_stamina for race in remaining_races)
            ) / MAX_STAT
        self.tensor[offset : offset + self._RACE_FEATURES] = race_features
        offset += self._RACE_FEATURES

        for action in sorted(TRAINING_ACTIONS):
            gains, failure_probability, energy_delta, attending = (
                state.training_features(action)
            )
            self.tensor[offset : offset + 6] = np.asarray(gains, dtype=np.float32) / 60
            self.tensor[offset + 6 : offset + 9] = (
                failure_probability,
                energy_delta / 50,
                attending / NUM_CARDS,
            )
            offset += 9

        for index in range(NUM_CARDS):
            self.tensor[offset + index] = float(state.is_rainbow(index))
        offset += NUM_CARDS
        for action in sorted(TRAINING_ACTIONS):
            self.tensor[offset + action - 1] = state.rainbow_count(action) / NUM_CARDS

    def string_from(self, state: UmaState, player):
        del player
        return str(state)
