# 3 years × 12 months × 2 half-months.
MAX_TURNS = 72
_TURNS_PER_YEAR = 24
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_SUMMER_CAMP_YEARS = frozenset({2, 3})
_SUMMER_CAMP_MONTHS = frozenset({"July", "August"})
# Year 3 Late April: Tenno Sho (Spring). Not a training turn.
TENNO_SHO_SPRING_TURN = 56
TENNO_SHO_MIN_SPEED = 400
TENNO_SHO_MIN_STAMINA = 400


def _calendar_parts(turn_1based: int) -> tuple[int, str, str]:
    index = turn_1based - 1
    year = index // _TURNS_PER_YEAR + 1
    month = _MONTHS[(index % _TURNS_PER_YEAR) // 2]
    half = "Early" if index % 2 == 0 else "Late"
    return year, month, half


def calendar_label(turn_1based: int) -> str:
    """Calendar date for a 1-based turn in 1..72.

    Turn 1 is Year 1, Early January; turn 72 is Year 3, Late December.
    """
    year, month, half = _calendar_parts(turn_1based)
    return f"Year {year}, {half} {month}"


assert calendar_label(TENNO_SHO_SPRING_TURN) == "Year 3, Late April"


def _is_summer_camp_turn(turn_1based: int) -> bool:
    # Years 2–3, Early July through Late August inclusive.
    year, month, _half = _calendar_parts(turn_1based)
    return year in _SUMMER_CAMP_YEARS and month in _SUMMER_CAMP_MONTHS


# Indexed by 1-based turn; index 0 is unused. Read on every training, so it is
# worth keeping out of the calendar arithmetic above.
SUMMER_CAMP_TURNS = (False,) + tuple(
    _is_summer_camp_turn(turn) for turn in range(1, MAX_TURNS + 1)
)
