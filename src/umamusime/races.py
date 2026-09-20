from collections.abc import Sequence
from typing import NamedTuple


class CareerRace(NamedTuple):
    """A career race that replaces the training turn and soft-fails on miss."""

    turn: int
    name: str
    min_speed: int
    min_stamina: int


class TurnEvent(NamedTuple):
    """Pre-turn effects applied before placement or race resolution."""

    turn: int
    energy: int = 0
    inspiration: bool = False


DEFAULT_RACES: tuple[CareerRace, ...] = (
    CareerRace(11, "Debut", 111, 74),
    CareerRace(22, "Asahi Hai", 179, 154),
    CareerRace(29, "Spring Stakes", 220, 294),
    CareerRace(30, "Satsuki Sho", 224, 365),
    CareerRace(33, "Japanese Derby", 299, 314),
    CareerRace(43, "Kikuka Sho", 285, 556),
    CareerRace(55, "Tenno Sho (Spring)", 367, 815),
    CareerRace(69, "Japan Cup", 503, 556),
    CareerRace(71, "Arima Kinen", 523, 724),
    CareerRace(73, "URA Finale Qualifier", 522, 519),
    CareerRace(75, "URA Finale Semifinal", 539, 614),
    CareerRace(77, "URA Finale Finals", 580, 670),
)

DEFAULT_EVENTS: tuple[TurnEvent, ...] = (
    TurnEvent(0, inspiration=True),
    TurnEvent(24, energy=20),
    TurnEvent(30, inspiration=True),
    TurnEvent(38, energy=30),
    TurnEvent(48, energy=30),
    TurnEvent(49, energy=20),
    TurnEvent(54, inspiration=True),
    TurnEvent(62, energy=30),
)

DEFAULT_INSPIRATION_SPEED = 50
DEFAULT_INSPIRATION_STAMINA = 50


def races_param(races: Sequence[CareerRace] = DEFAULT_RACES) -> str:
    return ",".join(
        f"{race.turn}:{race.name}:{race.min_speed}:{race.min_stamina}" for race in races
    )


def races_from_param(value: str) -> tuple[CareerRace, ...]:
    """Parse `turn:name:min_speed:min_stamina` entries separated by commas."""
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    races: list[CareerRace] = []
    seen_turns: set[int] = set()
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 4:
            raise ValueError(f"Expected turn:name:min_speed:min_stamina, got {entry!r}")
        turn_text, name, speed_text, stamina_text = parts
        name = name.strip()
        if not name:
            raise ValueError(f"Race name is empty in {entry!r}")
        try:
            turn = int(turn_text)
            min_speed = int(speed_text)
            min_stamina = int(stamina_text)
        except ValueError as exc:
            raise ValueError(f"Invalid race entry {entry!r}") from exc
        if turn < 0:
            raise ValueError(f"Race turn must be non-negative, got {turn}")
        if turn in seen_turns:
            raise ValueError(f"Duplicate race turn {turn}")
        seen_turns.add(turn)
        races.append(CareerRace(turn, name, min_speed, min_stamina))
    return tuple(sorted(races, key=lambda race: race.turn))
