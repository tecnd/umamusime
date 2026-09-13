import math
from collections.abc import Sequence

from .actions import (
    NUM_STATS,
    STAT_INDEX,
    TRAINING_ACTIONS,
    TRAINING_FOR_STAT,
)
from .cards import SupportCard

# Facility levels 1–5. Rest is not a facility; its row is unused.
# Each entry is (speed, stamina, power, guts, wit, skill points) before support
# card bonuses, on a successful training.
TRAINING_STATS = (
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
TRAINING_ENERGY = (
    (50, 50, 50, 50, 50),
    (-21, -22, -23, -25, -27),
    (-19, -20, -21, -23, -25),
    (-20, -21, -22, -24, -26),
    (-22, -23, -24, -26, -28),
    (5, 5, 5, 5, 5),
)

# Stat change applied when that training fails.
FAIL_STATS = (
    (0, 0, 0, 0, 0, 0),
    (-10, 0, 0, 0, 0, 0),
    (0, -10, 0, 0, 0, 0),
    (0, 0, -10, 0, 0, 0),
    (0, 0, 0, -10, 0, 0),
    (0, 0, 0, 0, 0, 0),
)

MIN_FACILITY_LEVEL = 1
MAX_FACILITY_LEVEL = 5
USES_PER_FACILITY_LEVEL = 4

# Support card placement weights, before specialty priority.
TRAINING_WEIGHT = 100.0
AWAY_WEIGHT = 50.0

# Friendship gauges run 0–100; 80 or more "rainbows" the card.
MAX_FRIENDSHIP = 100
RAINBOW_FRIENDSHIP = 80
FRIENDSHIP_PER_TRAINING = 5

# Fixed inputs to the stat gain formula. UmaGrowth is Mihono Bourbon's
# in-game rate: +20% stamina, +10% power, nothing else.
BASE_MOOD = 0.2
UMA_GROWTH = (0.0, 0.2, 0.1, 0.0, 0.0, 0.0)
PER_CARD_BONUS = 0.05

MAX_ENERGY = 100
STARTING_ENERGY = MAX_ENERGY

_FAIL_FREE_ENERGY = 50
_FAIL_CHANCE_AT_ZERO = 0.99


def clip_energy(energy: int) -> int:
    return max(0, min(MAX_ENERGY, energy))


def facility_level(uses: int) -> int:
    return min(
        MAX_FACILITY_LEVEL,
        MIN_FACILITY_LEVEL + uses // USES_PER_FACILITY_LEVEL,
    )


def stat_train_failure_chance(energy_after: int) -> float:
    # Remaining energy >= 50 never fails; 0 is 99% fail; linear in between.
    if energy_after >= _FAIL_FREE_ENERGY:
        return 0.0
    remaining = max(energy_after, 0)
    return (
        _FAIL_CHANCE_AT_ZERO
        * (_FAIL_FREE_ENERGY - remaining)
        / _FAIL_FREE_ENERGY
    )


def placement_outcomes(card: SupportCard) -> list[tuple[int, float]]:
    """Where a card shows up this turn: away, or one of the five trainings."""
    weights = [AWAY_WEIGHT] + [TRAINING_WEIGHT] * len(TRAINING_ACTIONS)
    weights[TRAINING_FOR_STAT[card.main_stat]] += card.specialty_priority
    total = sum(weights)
    return [(outcome, weight / total) for outcome, weight in enumerate(weights)]


def summed_initial_stats(cards: Sequence[SupportCard]) -> tuple[int, ...]:
    totals = [0] * NUM_STATS
    for card in cards:
        for stat, amount in card.initial_stats.items():
            totals[STAT_INDEX[stat]] += amount
    return tuple(totals)


def training_multiplier(
    friendship_multiplier: float,
    mood_effect: float,
    training_effectiveness: float,
    num_characters: int,
) -> float:
    return (
        friendship_multiplier
        * (1.0 + BASE_MOOD * (1.0 + mood_effect / 100.0))
        * (1.0 + training_effectiveness / 100.0)
        * (1.0 + PER_CARD_BONUS * num_characters)
    )


def max_skill_points_per_turn(cards: Sequence[SupportCard]) -> int:
    """Upper bound on skill points from one successful training.

    Assumes every card in the deck attends and every matching card is
    rainbowed — enough to keep the observation at most 1.
    """
    skill = STAT_INDEX["skill_points"]
    bonus = sum(card.stat_bonus.get("skill_points", 0) for card in cards)
    mood_effect = sum(card.mood_effect for card in cards)
    training_effectiveness = sum(card.training_effectiveness for card in cards)
    best = 0
    for action in TRAINING_ACTIONS:
        friendship_multiplier = 1.0
        for card in cards:
            if TRAINING_FOR_STAT[card.main_stat] == action:
                friendship_multiplier *= 1.0 + card.friendship_bonus / 100.0
        # Skill-point columns are constant across facility levels.
        base = TRAINING_STATS[action][0][skill]
        gain = math.floor(
            (base + bonus)
            * training_multiplier(
                friendship_multiplier,
                mood_effect,
                training_effectiveness,
                len(cards),
            )
        )
        best = max(best, gain)
    return best
