from collections import Counter

from . import dqn, mcts
from .umamusime import UmaState

_PLAYER_ACTIONS = (
    "rest",
    "speed",
    "stamina",
    "power",
    "guts",
    "wit",
)


def summarize_strategy(actions: list[int]) -> str:
    counts = Counter(actions)
    rest = counts.get(0, 0)
    training = [(i, counts[i]) for i in range(1, 6) if counts[i]]
    training.sort(key=lambda item: item[1], reverse=True)
    if not training:
        return "The policy rested every turn and never trained."
    top_id, top_n = training[0]
    top_name = _PLAYER_ACTIONS[top_id]
    extras = training[1:]
    if extras:
        extra_txt = ", ".join(
            f"{_PLAYER_ACTIONS[i]} {n} time{'s' if n != 1 else ''}"
            for i, n in extras
        )
        train_txt = (
            f"focused on {top_name} training ({top_n} turns), also using {extra_txt}"
        )
    else:
        train_txt = f"focused exclusively on {top_name} training ({top_n} turns)"
    if rest:
        rest_word = "time" if rest == 1 else "times"
        return f"The policy {train_txt}, resting {rest} {rest_word} to recover energy."
    return f"The policy {train_txt} without resting."


def _report(name: str, state: UmaState, actions: list[int]) -> None:
    print(f"=== {name} ===")
    print(f"Final state: {state}")
    print(f"Reward: {state.returns()[0]}")
    print(f"Strategy: {summarize_strategy(actions)}")
    print()


def main() -> None:
    mcts_state, mcts_actions = mcts.play(verbose=False)
    _report("MCTS", mcts_state, mcts_actions)
    dqn_state, dqn_actions = dqn.play(verbose=False)
    _report("DQN", dqn_state, dqn_actions)


if __name__ == "__main__":
    main()
