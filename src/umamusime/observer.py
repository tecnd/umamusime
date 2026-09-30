from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .actions import (
    NUM_CARDS,
    NUM_PLACEMENT_OUTCOMES,
    NUM_STATS,
    TRAINING_ACTIONS,
    TRAINING_FOR_STAT,
)
from .calendar import MAX_TURNS
from .scoring import MAX_STAT
from .training import MAX_FACILITY_LEVEL, MAX_FRIENDSHIP, STARTING_ENERGY

if TYPE_CHECKING:
    from .state import UmaState

_CAREER_FEATURES = 13
_BASE_FEATURES = _CAREER_FEATURES + NUM_CARDS * (NUM_PLACEMENT_OUTCOMES + 1)

_SORTED_TRAININGS = tuple(sorted(TRAINING_ACTIONS))
# Next-race turns-until, speed margin, stamina margin, then the stamina margin
# against the hardest remaining race.
_RACE_FEATURES = 4
# Per training: six gains, fail chance, energy delta, share of cards attending.
_TRAINING_FEATURES = NUM_STATS + 3
# Per card rainbow bit, then per training the share of cards that are rainbowed
# on their own specialty.
_RAINBOW_FEATURES = NUM_CARDS + len(_SORTED_TRAININGS)
_OBSERVATION_SIZE = (
    _BASE_FEATURES
    + _RACE_FEATURES
    + len(_SORTED_TRAININGS) * _TRAINING_FEATURES
    + _RAINBOW_FEATURES
)

# Largest single-training gain is well under this (level 5 speed with a full
# rainbow deck is about 45 speed and 20 skill points).
_GAIN_SCALE = 60.0
_ENERGY_SCALE = float(STARTING_ENERGY // 2)


class UmaObserver:
    def __init__(self, params=None):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # Base career features, a placement one-hot and friendship per card,
        # then features derived from the rules that the network would
        # otherwise have to learn: race gates, this turn's training outcomes,
        # and rainbow status.
        self.tensor = np.zeros(_OBSERVATION_SIZE, np.float32)
        self.dict = {"observation": self.tensor}

    def set_from(self, state: UmaState, player):
        del player
        self.tensor[:] = 0.0
        self._set_base(state)
        offset = _BASE_FEATURES
        self._set_race_margins(state, offset)
        offset += _RACE_FEATURES
        for action in _SORTED_TRAININGS:
            self._set_training(state, action, offset)
            offset += _TRAINING_FEATURES
        self._set_rainbows(state, offset)

    def _set_base(self, state: UmaState) -> None:
        # Scaled to roughly [0, 1] so the values are usable as network inputs.
        speed, stamina, power, guts, wit, skill_points = state.stats
        self.tensor[:_CAREER_FEATURES] = (
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
        offset = _CAREER_FEATURES
        for card_state in state.card_states:
            self.tensor[offset + card_state.placement] = 1.0
            self.tensor[offset + NUM_PLACEMENT_OUTCOMES] = (
                card_state.friendship / MAX_FRIENDSHIP
            )
            offset += NUM_PLACEMENT_OUTCOMES + 1

    def _set_race_margins(self, state: UmaState, offset: int) -> None:
        speed, stamina = state.stats[:2]
        upcoming = [race for race in state.get_game().races if race.turn >= state.turn]
        if not upcoming:
            return
        race = upcoming[0]
        hardest_stamina = max(upcoming, key=lambda r: r.min_stamina).min_stamina
        self.tensor[offset : offset + _RACE_FEATURES] = (
            (race.turn - state.turn) / MAX_TURNS,
            (speed - race.min_speed) / MAX_STAT,
            (stamina - race.min_stamina) / MAX_STAT,
            (stamina - hardest_stamina) / MAX_STAT,
        )

    def _set_training(self, state: UmaState, action: int, offset: int) -> None:
        gains = state.training_gains(action)
        for index, gain in enumerate(gains):
            self.tensor[offset + index] = gain / _GAIN_SCALE
        self.tensor[offset + NUM_STATS] = state.failure_probability(action)
        self.tensor[offset + NUM_STATS + 1] = state.energy_delta(action) / _ENERGY_SCALE
        self.tensor[offset + NUM_STATS + 2] = len(state.attending(action)) / NUM_CARDS

    def _set_rainbows(self, state: UmaState, offset: int) -> None:
        cards = state.get_game().cards
        for index in range(NUM_CARDS):
            if state.is_rainbow(index):
                self.tensor[offset + index] = 1.0
        offset += NUM_CARDS
        for slot, action in enumerate(_SORTED_TRAININGS):
            self.tensor[offset + slot] = (
                sum(
                    1
                    for index in state.attending(action)
                    if TRAINING_FOR_STAT[cards[index].main_stat] == action
                    and state.is_rainbow(index)
                )
                / NUM_CARDS
            )

    def string_from(self, state: UmaState, player):
        del player
        return str(state)
