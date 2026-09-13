from enum import IntEnum, auto

from .cards import STATS

NUM_STATS = len(STATS)
STAT_INDEX = {stat: index for index, stat in enumerate(STATS)}
NUM_CARDS = 6
# Support cards attend one of the five training facilities, or nobody's.
PLACEMENT_AWAY = 0
NUM_PLACEMENT_OUTCOMES = 6

# A turn is six placement rolls, the player's action, then a fail/success roll.
NODES_PER_TURN = NUM_CARDS + 2
NUM_ACTIONS = 6

REST_ACTION = 0
WIT_ACTION = 5
STAT_TRAIN_ACTIONS = frozenset({1, 2, 3, 4})
TRAINING_ACTIONS = frozenset({1, 2, 3, 4, 5})
# Training action that a support card's main stat maps to.
TRAINING_FOR_STAT = {stat: index + 1 for index, stat in enumerate(STATS[:5])}

CHANCE_FAIL = 0
CHANCE_SUCCESS = 1

PLACEMENT_ACTIONS = tuple(range(NUM_PLACEMENT_OUTCOMES))
RESULT_ACTIONS = (CHANCE_FAIL, CHANCE_SUCCESS)
DECISION_ACTIONS = tuple(range(NUM_ACTIONS))

ACTION_NAMES = (
    "rest",
    "train_speed",
    "train_stamina",
    "train_power",
    "train_guts",
    "train_wit",
)
RESULT_NAMES = ("fail", "success")


class Phase(IntEnum):
    PLACEMENT = auto()
    DECISION = auto()
    RESULT = auto()
