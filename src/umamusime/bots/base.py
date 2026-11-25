from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np

from ..game import UmaGame
from ..rng import split_rngs
from ..state import UmaState


class Bot(Protocol):
    def predict(self, state: UmaState) -> int: ...


@dataclasses.dataclass(frozen=True)
class PlayResult:
    state: UmaState
    actions: tuple[int, ...]
    interrupted: bool = False


def advance_chance(state: UmaState, rng: np.random.RandomState) -> None:
    """Resolve chance nodes in place until a decision or terminal node is reached."""
    while not state.is_terminal() and state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(int(rng.choice(outcomes, p=probs)))


def play(
    bot: Bot,
    *,
    game: UmaGame | None = None,
    seed: int | None = None,
    verbose: bool = False,
    catch_interrupt: bool = False,
) -> PlayResult:
    game = game or UmaGame()
    state: UmaState = game.new_initial_state()
    env_rng = split_rngs(seed)[0] if seed is not None else np.random.RandomState()
    actions: list[int] = []

    try:
        advance_chance(state, env_rng)
        while not state.is_terminal():
            action = bot.predict(state.clone())
            if action not in state.legal_actions():
                raise ValueError(
                    f"Illegal action {action}; legal: {state.legal_actions()}"
                )
            actions.append(action)
            if verbose:
                print(state.action_to_string(0, action))
            state.apply_action(action)
            advance_chance(state, env_rng)
            if verbose:
                print(state)
    except KeyboardInterrupt:
        if not catch_interrupt:
            raise
        print("\nInterrupted — exiting.")
        print(state)
        print(f"Returns so far: {state.returns()}")
        return PlayResult(state=state, actions=tuple(actions), interrupted=True)

    if verbose:
        print(f"Returns: {state.returns()}")
    return PlayResult(state=state, actions=tuple(actions))
