from collections.abc import Callable

from . import dqn, mcts
from .umamusime import UmaGame, UmaState

_RUNS = 3
_SEED = 42

PlayFn = Callable[[int], tuple[UmaState, list[int]]]


def _best_of(play: PlayFn) -> tuple[UmaState, list[int], list[float], int]:
    results: list[tuple[float, UmaState, list[int]]] = []
    for index in range(_RUNS):
        state, actions = play(index)
        results.append((state.returns()[0], state, actions))
    scores = [score for score, _, _ in results]
    best_index = max(range(_RUNS), key=lambda index: results[index][0])
    _, state, actions = results[best_index]
    return state, actions, scores, best_index


def _report(
    name: str,
    state: UmaState,
    actions: list[int],
    scores: list[float],
    best_index: int,
) -> None:
    print(f"=== {name} ===")
    print(
        "Runs: "
        + ", ".join(f"{score:.1f}" for score in scores)
        + f"; using run {best_index + 1}"
    )
    print(f"Final state:\n{state}")
    print(f"Reward: {state.returns()[0]:.1f}")
    print(
        "Actions: "
        + ", ".join(state.action_to_string(0, action) for action in actions)
    )
    print()


def main() -> None:
    print("Deck: " + ", ".join(card.name for card in UmaGame().cards) + "\n")
    mcts_state, mcts_actions, mcts_scores, mcts_best = _best_of(
        lambda index: mcts.play(verbose=False, seed=_SEED + index)
    )
    _report("MCTS", mcts_state, mcts_actions, mcts_scores, mcts_best)
    dqn_state, dqn_actions, dqn_scores, dqn_best = _best_of(
        lambda index: dqn.play(verbose=False, seed=_SEED + index)
    )
    _report("DQN", dqn_state, dqn_actions, dqn_scores, dqn_best)


if __name__ == "__main__":
    main()
