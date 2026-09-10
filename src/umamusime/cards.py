from collections.abc import Mapping
from dataclasses import dataclass, field

# Stat order used by every stat tuple in the game. Skill points are a stat:
# they take stat bonuses and the training multiplier like the rest.
STATS = ("speed", "stamina", "power", "guts", "wit", "skill_points")

# Stats that have a training facility of their own.
TRAINABLE_STATS = STATS[:5]


@dataclass(frozen=True)
class SupportCard:
    """A support card, limited to the stats this game models."""

    name: str
    # Training this card gives a friendship bonus to, and the training its
    # specialty priority is added to.
    main_stat: str
    friendship_bonus: float = 0.0
    mood_effect: float = 0.0
    training_effectiveness: float = 0.0
    initial_friendship: int = 0
    specialty_priority: float = 0.0
    wit_friendship_recovery: int = 0
    # One-time stat grants applied at the start of the career.
    initial_stats: Mapping[str, int] = field(default_factory=dict)
    # Added to a training's base gain for that stat, when this card attends.
    stat_bonus: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.main_stat not in TRAINABLE_STATS:
            raise ValueError(f"Unknown main stat: {self.main_stat}")
        for mapping in (self.initial_stats, self.stat_bonus):
            unknown = set(mapping) - set(STATS)
            if unknown:
                raise ValueError(f"Unknown stats: {sorted(unknown)}")


KITASAN_BLACK = SupportCard(
    name="Kitasan Black",
    main_stat="speed",
    friendship_bonus=25,
    mood_effect=30,
    training_effectiveness=15,
    initial_friendship=35,
    specialty_priority=100,
    stat_bonus={"power": 1},
)

TOKAI_TEIO = SupportCard(
    name="Tokai Teio",
    main_stat="speed",
    friendship_bonus=32,
    mood_effect=60,
    initial_friendship=25,
    specialty_priority=35,
    initial_stats={"speed": 20},
    stat_bonus={"power": 1},
)

SWEEP_TOSHO = SupportCard(
    name="Sweep Tosho",
    main_stat="speed",
    friendship_bonus=30,
    mood_effect=40,
    training_effectiveness=10,
    initial_friendship=25,
    specialty_priority=50,
    stat_bonus={"skill_points": 1},
)

FINE_MOTION = SupportCard(
    name="Fine Motion",
    main_stat="wit",
    friendship_bonus=37.5,
    mood_effect=30,
    training_effectiveness=15,
    initial_friendship=15,
    specialty_priority=35,
    wit_friendship_recovery=5,
    initial_stats={"wit": 35},
    stat_bonus={"wit": 1},
)

SUPER_CREEK = SupportCard(
    name="Super Creek",
    main_stat="stamina",
    friendship_bonus=37.5,
    training_effectiveness=15,
    initial_friendship=30,
    specialty_priority=55,
    initial_stats={"stamina": 35},
    stat_bonus={"stamina": 1},
)

AGNES_TACHYON = SupportCard(
    name="Agnes Tachyon",
    main_stat="wit",
    friendship_bonus=20,
    mood_effect=40,
    training_effectiveness=5,
    initial_friendship=25,
    specialty_priority=50,
    wit_friendship_recovery=4,
    initial_stats={"wit": 20},
    stat_bonus={"wit": 1, "skill_points": 1},
)

DEFAULT_DECK = (
    KITASAN_BLACK,
    TOKAI_TEIO,
    SWEEP_TOSHO,
    FINE_MOTION,
    SUPER_CREEK,
    AGNES_TACHYON,
)
