from . import dqn, mcts
from .umamusime import UmaState


def _report(name: str, state: UmaState, actions: list[int]) -> None:
    print(f"=== {name} ===")
    print(f"Final state: {state}")
    print(f"Reward: {state.returns()[0]:.1f}")
    print(
        "Actions: "
        + ", ".join(state.action_to_string(0, action) for action in actions)
    )
    print()


def main() -> None:
    mcts_state, mcts_actions = mcts.play(verbose=False)
    _report("MCTS", mcts_state, mcts_actions)
    dqn_state, dqn_actions = dqn.play(verbose=False)
    _report("DQN", dqn_state, dqn_actions)


if __name__ == "__main__":
    main()
