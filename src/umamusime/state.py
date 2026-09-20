from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import pyspiel
from tabulate import tabulate

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
    TRAINING_ACTIONS,
    TRAINING_FOR_STAT,
    WIT_ACTION,
    Phase,
)
from .calendar import (
    MAX_TURNS,
    SUMMER_CAMP_TURNS,
    calendar_label,
)
from .races import CareerRace
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
    train_failure_chance,
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

        self._score = float(sum(stat_score(stat) for stat in game.initial_stats[:5]))
        self._last_reward = 0.0
        self._pending_action: int | None = None
        self._failed_race: CareerRace | None = None
        self._last_training_failed = False
        # (turn, action) for each failed training in this career.
        self._training_failures: list[tuple[int, int]] = []
        self._start_current_turn()
        # Turn-0 event score is part of the opening total, not a step reward.
        self._last_reward = 0.0

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def training_failures(self) -> tuple[tuple[int, int], ...]:
        """Failed trainings as (0-based turn, action) in chronological order."""
        return tuple(self._training_failures)

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

    def _failure_probability(self, action: int) -> float:
        energy_after = self._energy + self._energy_delta(action)
        return train_failure_chance(action, energy_after)

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
            if TRAINING_FOR_STAT[card.main_stat] == action and self._is_rainbow(index):
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
        p_fail = self._failure_probability(self._pending_action)
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
        self._last_training_failed = not success
        if not success:
            self._training_failures.append((self._turn, action))

        if success or action == WIT_ACTION:
            # Energy recovery is read before friendships move this turn.
            self._energy = clip_energy(self._energy + self._energy_delta(action))

        gains = self._training_gains(action) if success else FAIL_STATS[action]
        if success and action in TRAINING_ACTIONS:
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

        before = self.stats[:5]
        applied = self._apply_stats(gains)
        after = self.stats[:5]
        self._last_reward = self._stat_score_delta(before, after, applied[5])
        self._score += self._last_reward

        self._turn += 1
        self._pending_action = None
        self._start_current_turn()

    def _stat_score_delta(
        self, before: Sequence[int], after: Sequence[int], skill_delta: int
    ) -> float:
        return (
            sum(stat_score(new) - stat_score(old) for new, old in zip(after, before))
            + SKILL_POINT_WEIGHT * skill_delta
        )

    def _apply_pre_turn_effects(self) -> None:
        event = self.get_game().events_by_turn.get(self._turn)
        if event is None:
            return
        if event.energy:
            self._energy = clip_energy(self._energy + event.energy)
        if not event.inspiration:
            return
        game = self.get_game()
        gains = (game.inspiration_speed, game.inspiration_stamina, 0, 0, 0, 0)
        before = self.stats[:5]
        applied = self._apply_stats(gains)
        after = self.stats[:5]
        reward = self._stat_score_delta(before, after, applied[5])
        self._score += reward
        self._last_reward += reward

    def _start_current_turn(self) -> None:
        if self._turn >= MAX_TURNS or self._failed_race is not None:
            return
        self._apply_pre_turn_effects()
        race = self.get_game().races_by_turn.get(self._turn)
        if race is not None:
            self._resolve_race(race)
            return
        self._card_states = tuple(
            CardState(card_state.friendship, PLACEMENT_AWAY)
            for card_state in self._card_states
        )
        self._placement_index = 0
        self._phase = Phase.PLACEMENT

    def _resolve_race(self, race: CareerRace) -> None:
        self._turn += 1
        if self._speed < race.min_speed or self._stamina < race.min_stamina:
            self._failed_race = race
            return
        self._start_current_turn()

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

        p_fail = self._failure_probability(action)
        if 0.0 < p_fail < 1.0:
            self._pending_action = action
            self._phase = Phase.RESULT
            return
        self._apply_training_result(action, success=(p_fail == 0.0))

    def is_terminal(self):
        return self._turn >= MAX_TURNS or self._failed_race is not None

    def ended_by_race_fail(self) -> bool:
        return self._failed_race is not None

    def failed_race(self) -> CareerRace | None:
        return self._failed_race

    def _display_turn(self) -> int:
        """0-based turn shown in labels; terminal states show the last consumed turn."""
        if self.is_terminal():
            return max(self._turn - 1, 0)
        return self._turn

    def is_summer_camp(self) -> bool:
        return SUMMER_CAMP_TURNS[self._display_turn()]

    def rewards(self):
        return [self._last_reward]

    def returns(self):
        return [self._score]

    def _support_lines_for(self, action: int) -> list[str]:
        cards = self.get_game().cards
        lines = []
        for index in self._attending(action):
            card = cards[index]
            card_state = self._card_states[index]
            prefix = (
                "* "
                if TRAINING_FOR_STAT[card.main_stat] == action
                and self._is_rainbow(index)
                else ""
            )
            if card_state.friendship >= RAINBOW_FRIENDSHIP:
                label = f"{card.name} (R)"
            else:
                label = f"{card.name} {card_state.friendship}"
            lines.append(f"{prefix}{label}")
        return lines

    def _next_race(self) -> CareerRace | None:
        """Soonest scheduled race on or after the displayed turn."""
        turn = self._display_turn()
        for race in self.get_game().races:
            if race.turn >= turn:
                return race
        return None

    def _event_message(self) -> str | None:
        event = self.get_game().events_by_turn.get(self._display_turn())
        if event is None:
            return None
        parts: list[str] = []
        if event.energy:
            parts.append(f"+{event.energy} energy")
        if event.inspiration:
            game = self.get_game()
            parts.append(
                "Inspiration "
                f"(+{game.inspiration_speed} speed, "
                f"+{game.inspiration_stamina} stamina)"
            )
        return f"Event: {'; '.join(parts)}" if parts else None

    def __str__(self):
        lines = []
        if self._last_training_failed:
            lines.append("Failed!")
        lines.append(f"{calendar_label(self._display_turn())}")
        event_message = self._event_message()
        if event_message is not None:
            lines.append(event_message)
        lines.append(f"Energy: {self._energy}")
        if not self.is_terminal():
            race = self._next_race()
            if race is not None:
                turns_until = race.turn - self._display_turn()
                if turns_until == 0:
                    when = "now"
                elif turns_until == 1:
                    when = "in 1 turn"
                else:
                    when = f"in {turns_until} turns"
                lines.append(
                    f"Next race: {race.name} {when} "
                    f"(need {race.min_speed} speed, {race.min_stamina} stamina)"
                )
        show_next_training = not self.is_terminal()
        training_cols = []
        for action, stat in enumerate(self.stats[:5], start=1):
            col = [f"Lv. {self.facility_level(action)}", stat]
            if show_next_training:
                col.append(self._training_gains(action))
                col.append(f"{self._failure_probability(action) * 100:.0f}% fail")
                col.extend(self._support_lines_for(action))
            training_cols.append(col)
        lines.append(
            tabulate(
                {
                    "Speed": training_cols[0],
                    "Stamina": training_cols[1],
                    "Power": training_cols[2],
                    "Guts": training_cols[3],
                    "Wit": training_cols[4],
                },
                headers="keys",
                tablefmt="github",
            )
        )
        lines.append(f"Skill points: {self._skill_points}")
        if self._failed_race is not None:
            race = self._failed_race
            lines.append(
                f"Ended: {race.name} soft fail "
                f"(need {race.min_speed} speed and {race.min_stamina} stamina)"
            )
        return "\n".join(lines)
