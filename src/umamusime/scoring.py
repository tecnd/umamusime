MIN_STAT = 0
MAX_STAT = 1200

# Skill points are temporarily unweighted: wit training's extra skill-point
# column was a reason to spam wit and walk into the Tenno Sho (Spring) fail,
# and this isolates whether the five-stat lookup alone still prefers that.
# The five training stats use the UmaTools / umakonga lookup: raw per-point
# rates in 50-point blocks, accumulated, then round(raw / 10). We only need
# 0-1200 while the game cap stays there.
# https://daftuyda.moe/guides/rating-system#2-stat-scoring
SKILL_POINT_WEIGHT = 0.0
_STAT_RAW_RATES = (
    5,
    8,
    10,
    13,
    16,
    18,
    21,
    24,
    26,
    28,
    29,
    30,
    31,
    33,
    34,
    35,
    39,
    41,
    42,
    43,
    52,
    55,
    66,
    68,
    68,
)


def _js_round(value: float) -> int:
    """JavaScript Math.round: halves away from zero, not banker's rounding."""
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


def _build_stat_scores() -> tuple[int, ...]:
    scores = [0]
    raw = 0
    index = 0
    for stat in range(1, MAX_STAT + 1):
        if stat <= 49:
            index = 0
        elif stat <= 99:
            index = 1
        elif stat % 50 == 0:
            index += 1
        raw += _STAT_RAW_RATES[index]
        scores.append(_js_round(raw / 10.0))
    return tuple(scores)


_STAT_SCORES = _build_stat_scores()


def stat_score(value: int) -> int:
    if value <= MIN_STAT:
        return _STAT_SCORES[0]
    if value >= MAX_STAT:
        return _STAT_SCORES[MAX_STAT]
    return _STAT_SCORES[value]
