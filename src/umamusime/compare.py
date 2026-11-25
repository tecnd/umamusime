import time
from collections.abc import Callable

from .bots import PlayResult, dqn, mcts, random_bot
from .calendar import calendar_label
from .game import UmaGame
from .state import UmaState

_RUNS = 3
_SEED = 42

PlayFn = Callable[[int], PlayResult]


def _best_of(play: PlayFn) -> tuple[PlayResult, list[float], int]:
    results = [play(index) for index in range(_RUNS)]
    scores = [result.state.returns()[0] for result in results]
    best_index = max(range(_RUNS), key=scores.__getitem__)
    return results[best_index], scores, best_index


def _format_training_fails(state: UmaState) -> str:
    fails = state.training_failures
    if not fails:
        return "Training fails: none"
    parts = [
        f"{calendar_label(turn)} {state.action_to_string(0, action)}"
        for turn, action in fails
    ]
    return f"Training fails: {len(fails)} ({', '.join(parts)})"


def _report(
    name: str,
    result: PlayResult,
    scores: list[float],
    best_index: int,
) -> None:
    state = result.state
    print(f"=== {name} ===")
    print(
        "Runs: "
        + ", ".join(f"{score:.1f}" for score in scores)
        + f"; using run {best_index + 1}"
    )
    print(f"Final state:\n{state}")
    print(f"Reward: {state.returns()[0]:.1f}")
    if state.ended_by_race_fail():
        race = state.failed_race()
        assert race is not None
        print(f"Ended: {race.name} soft fail")
    else:
        print("Ended: finished normally")
    print(_format_training_fails(state))
    print(
        "Actions: "
        + ", ".join(state.action_to_string(0, action) for action in result.actions)
    )
    print()


def main() -> None:
    print("Deck: " + ", ".join(card.name for card in UmaGame().cards) + "\n")

    random_result, random_scores, random_best = _best_of(
        lambda index: random_bot.play(verbose=False, seed=_SEED + index)
    )
    _report("Random", random_result, random_scores, random_best)

    mcts_run_s: list[float] = []

    def play_mcts(index: int) -> PlayResult:
        started = time.perf_counter()
        result = mcts.play(verbose=False, seed=_SEED + index)
        if index == 0:
            mcts_run_s.append(time.perf_counter() - started)
        return result

    mcts_result, mcts_scores, mcts_best = _best_of(play_mcts)
    print(f"MCTS one run: {mcts_run_s[0]:.1f}s")
    _report("MCTS", mcts_result, mcts_scores, mcts_best)

    train_s = dqn.train(verbose=False)
    print(f"DQN train time: {train_s:.1f}s")
    dqn_result, dqn_scores, dqn_best = _best_of(
        lambda index: dqn.play(verbose=False, seed=_SEED + index)
    )
    _report("DQN", dqn_result, dqn_scores, dqn_best)


if __name__ == "__main__":
    main()
