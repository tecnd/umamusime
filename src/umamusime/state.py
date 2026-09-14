from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import pyspiel

from . import training
from .actions import (
    ACTION_NAMES,
    CHANCE_FAIL,
    CHANCE_SUCCESS,
    DECISION_ACTIONS,
    NUM_ACTIONS,
    NUM_CARDS,
    NUM_PLACEMENT_OUTCOMES,
    NUM_STATS,
    PLACEMENT_ACTIONS,
    PLACEMENT_AWAY,
    REST_ACTION,
    RESULT_ACTIONS,
    RESULT_NAMES,
    STAT_INDEX,
    STAT_TRAIN_ACTIONS,
    TRAINING_ACTIONS,
    TRAINING_FOR_STAT,
    WIT_ACTION,
    Phase,
)
from .calendar import (
    MAX_TURNS,
    SUMMER_CAMP_TURNS,
    TENNO_SHO_MIN_SPEED,
    TENNO_SHO_MIN_STAMINA,
    TENNO_SHO_SPRING_TURN,
    calendar_label,
)
from .scoring import MAX_STAT, MIN_STAT, SKILL_POINT_WEIGHT, stat_score
from .training import (
    FAIL_STATS,
    FRIENDSHIP_PER_TRAINING,
    MAX_FACILITY_LEVEL,
    MAX_FRIENDSHIP,
    RAINBOW_FRIENDSHIP,
    STARTING_ENERGY,
    TRAINING_ENERGY,
    TRAINING_STATS,
    clip_energy,
    stat_train_failure_chance,
    training_multiplier,
)

if TYPE_CHECKING:
    from .game import UmaGame

JsonDict = dict[str, Any]


class CardState(NamedTuple):
    """Per-career run state for one support card slot."""

    friendship: int
    placement: int = PLACEMENT_AWAY


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
        self._energy = STARTING_ENERGY
        # Successful uses of speed, stamina, power, guts, wit facilities.
        self._facility_uses: tuple[int, ...] = (0, 0, 0, 0, 0)

        self._card_states = tuple(
            CardState(card.initial_friendship) for card in game.cards
        )
        self._placement_index = 0
        self._phase = Phase.PLACEMENT

        self._score = float(
            sum(stat_score(stat) for stat in game.initial_stats[:5])
        )
        self._last_reward = 0.0
        self._pending_action: int | None = None
        self._soft_failed = False

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def stats(self) -> tuple[int, int, int, int, int, int]:
        return (
            self._speed,
            self._stamina,
            self._power,
            self._guts,
            self._wit,
            self._skill_points,
        )

    @property
    def energy(self) -> int:
        return self._energy

    @property
    def card_states(self) -> tuple[CardState, ...]:
        return self._card_states

    def current_player(self):
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._phase == Phase.DECISION:
            return 0
        return pyspiel.PlayerId.CHANCE

    def action_to_string(self, player, action):
        if player == pyspiel.PlayerId.CHANCE:
            if self._phase == Phase.PLACEMENT:
                card = self.get_game().cards[self._placement_index]
                where = (
                    "away"
                    if action == PLACEMENT_AWAY
                    else self.action_to_string(0, action)
                )
                return f"{card.name} -> {where}"
            if action not in range(len(RESULT_NAMES)):
                raise ValueError(f"Invalid chance action: {action}")
            return RESULT_NAMES[action]
        if action not in range(len(ACTION_NAMES)):
            raise ValueError(f"Invalid action: {action}")
        return ACTION_NAMES[action]

    def legal_actions(self, player=None):
        if self._phase == Phase.PLACEMENT:
            return list(PLACEMENT_ACTIONS)
        if self._phase == Phase.RESULT:
            return list(RESULT_ACTIONS)
        return list(DECISION_ACTIONS)

    def _permanent_facility_level_for(self, action: int) -> int:
        return training.facility_level(self._facility_uses[action - 1])

    def facility_level(self, action: int) -> int:
        """Current facility level for a training action, including summer camp."""
        if self.is_summer_camp():
            return MAX_FACILITY_LEVEL
        return self._permanent_facility_level_for(action)

    def _attending(self, action: int) -> list[int]:
        """Indices of the support cards on a training facility this turn."""
        assert action in TRAINING_ACTIONS
        return [
            index
            for index, card_state in enumerate(self._card_states)
            if card_state.placement == action
        ]

    def _is_rainbow(self, index: int) -> bool:
        return self._card_states[index].friendship >= RAINBOW_FRIENDSHIP

    def _wit_energy_recovery(self) -> int:
        # Only rainbowed cards standing on the wit facility recover energy.
        cards = self.get_game().cards
        return sum(
            cards[index].wit_friendship_recovery
            for index in self._attending(WIT_ACTION)
            if self._is_rainbow(index)
        )

    def _energy_delta(self, action: int) -> int:
        level = 1 if action == REST_ACTION else self.facility_level(action)
        delta = TRAINING_ENERGY[action][level - 1]
        if action == WIT_ACTION:
            delta += self._wit_energy_recovery()
        return delta

    def _training_gains(self, action: int) -> tuple[int, ...]:
        """Stat gains of a successful training, including support card effects."""
        if action == REST_ACTION:
            return (0,) * NUM_STATS
        base = TRAINING_STATS[action][self.facility_level(action) - 1]
        cards = self.get_game().cards
        attending = self._attending(action)

        friendship_multiplier = 1.0
        mood_effect = 0.0
        training_effectiveness = 0.0
        stat_bonus = [0] * NUM_STATS
        for index in attending:
            card = cards[index]
            if TRAINING_FOR_STAT[card.main_stat] == action and self._is_rainbow(
                index
            ):
                friendship_multiplier *= 1.0 + card.friendship_bonus / 100.0
            mood_effect += card.mood_effect
            training_effectiveness += card.training_effectiveness
            for stat, amount in card.stat_bonus.items():
                stat_bonus[STAT_INDEX[stat]] += amount

        multiplier = training_multiplier(
            friendship_multiplier,
            mood_effect,
            training_effectiveness,
            len(attending),
        )
        # A stat bonus only applies to stats the training already grants.
        # UmaGrowth is per-stat (defaults: 0.2 stamina, 0.1 power);
        # skill points always have 0 growth.
        return tuple(
            math.floor(
                (amount + stat_bonus[index])
                * multiplier
                * (1.0 + self.get_game().uma_growth[index])
            )
            if amount
            else 0
            for index, amount in enumerate(base)
        )

    def chance_outcomes(self):
        assert self.is_chance_node()
        if self._phase == Phase.PLACEMENT:
            return self.get_game().placement_outcomes[self._placement_index]
        assert self._pending_action is not None
        energy_after = clip_energy(
            self._energy + self._energy_delta(self._pending_action)
        )
        p_fail = stat_train_failure_chance(energy_after)
        return [(CHANCE_FAIL, p_fail), (CHANCE_SUCCESS, 1.0 - p_fail)]

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
            if new < MIN_STAT:
                new = MIN_STAT
            elif new > MAX_STAT:
                new = MAX_STAT
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
            self._energy = clip_energy(self._energy + self._energy_delta(action))
            if action in TRAINING_ACTIONS:
                uses = list(self._facility_uses)
                uses[action - 1] += 1
                self._facility_uses = tuple(uses)
                card_states = list(self._card_states)
                for index in self._attending(action):
                    card_state = card_states[index]
                    card_states[index] = CardState(
                        min(
                            MAX_FRIENDSHIP,
                            card_state.friendship + FRIENDSHIP_PER_TRAINING,
                        ),
                        card_state.placement,
                    )
                self._card_states = tuple(card_states)
        else:
            gains = FAIL_STATS[action]

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
            stat_score(new) - stat_score(old) for new, old in zip(after, before)
        ) + SKILL_POINT_WEIGHT * applied[5]
        self._score += self._last_reward

        self._turn += 1
        self._pending_action = None
        self._begin_turn()

    def _begin_turn(self) -> None:
        if self._turn >= MAX_TURNS or self._soft_failed:
            return
        if self._turn + 1 == TENNO_SHO_SPRING_TURN:
            self._resolve_tenno_sho_spring()
            return
        self._card_states = tuple(
            CardState(card_state.friendship, PLACEMENT_AWAY)
            for card_state in self._card_states
        )
        self._placement_index = 0
        self._phase = Phase.PLACEMENT

    def _resolve_tenno_sho_spring(self) -> None:
        # Year 3 Late April is the race, not a training turn.
        self._turn += 1
        if (
            self._speed < TENNO_SHO_MIN_SPEED
            or self._stamina < TENNO_SHO_MIN_STAMINA
        ):
            self._soft_failed = True
            return
        self._begin_turn()

    def apply_action(self, action):
        if self._phase == Phase.PLACEMENT:
            if action not in range(NUM_PLACEMENT_OUTCOMES):
                raise ValueError(f"Invalid placement outcome: {action}")
            index = self._placement_index
            card_states = list(self._card_states)
            card_states[index] = CardState(card_states[index].friendship, action)
            self._card_states = tuple(card_states)
            self._placement_index = index + 1
            if index + 1 == NUM_CARDS:
                self._phase = Phase.DECISION
            return

        if self._phase == Phase.RESULT:
            assert self._pending_action is not None
            self._apply_training_result(
                self._pending_action, success=(action == CHANCE_SUCCESS)
            )
            return

        if action not in range(NUM_ACTIONS):
            raise ValueError(f"Invalid action: {action}")

        energy_after = clip_energy(self._energy + self._energy_delta(action))
        fail_p = (
            stat_train_failure_chance(energy_after)
            if action in STAT_TRAIN_ACTIONS
            else 0.0
        )
        if 0.0 < fail_p < 1.0:
            self._pending_action = action
            self._phase = Phase.RESULT
            return
        self._apply_training_result(action, success=(fail_p == 0.0))

    def is_terminal(self):
        return self._turn >= MAX_TURNS or self._soft_failed

    def ended_by_tenno_sho_fail(self) -> bool:
        return self._soft_failed

    def _current_turn_1based(self) -> int:
        if self.is_terminal():
            return self._turn if self._turn > 0 else 1
        return self._turn + 1

    def is_summer_camp(self) -> bool:
        return SUMMER_CAMP_TURNS[self._current_turn_1based()]

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
            where = "away" if place == PLACEMENT_AWAY else ACTION_NAMES[place]
            parts.append(f"{where}: {'/'.join(names)}")
        return ", ".join(parts) if parts else "unrolled"

    def __str__(self):
        career = (
            f"{calendar_label(self._current_turn_1based())}, "
            f"Speed: {self._speed}, Stamina: {self._stamina}, "
            f"Power: {self._power}, Guts: {self._guts}, Wit: {self._wit}, "
            f"Skill points: {self._skill_points}, Energy: {self._energy}, "
            f"Facilities: speed {self.facility_level(1)}, "
            f"stamina {self.facility_level(2)}, "
            f"power {self.facility_level(3)}, "
            f"guts {self.facility_level(4)}, "
            f"wit {self.facility_level(5)}"
        )
        lines = [career, f"Friendship: {self._friendship_string()}"]
        if self._soft_failed:
            lines.append(
                "Ended: Tenno Sho (Spring) soft fail "
                f"(need {TENNO_SHO_MIN_SPEED} speed and "
                f"{TENNO_SHO_MIN_STAMINA} stamina)"
            )
        if not self.is_terminal():
            lines.append(f"Supports: {self._placement_string()}")
        return "\n".join(lines)
