from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .actions import NUM_CARDS, NUM_PLACEMENT_OUTCOMES
from .calendar import MAX_TURNS
from .scoring import MAX_STAT
from .training import MAX_FACILITY_LEVEL, MAX_FRIENDSHIP, STARTING_ENERGY

if TYPE_CHECKING:
    from .state import UmaState


class UmaObserver:
    def __init__(self, params=None):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        # 13 career features, then a placement one-hot and friendship per card.
        self.tensor = np.zeros(
            13 + NUM_CARDS * (NUM_PLACEMENT_OUTCOMES + 1), np.float32
        )
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

    def string_from(self, state: UmaState, player):
        del player
        return str(state)
