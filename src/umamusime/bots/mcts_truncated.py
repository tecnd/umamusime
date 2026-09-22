from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts

from ..actions import NODES_PER_TURN
from ..game import UmaGame
from ..state import UmaState

# Rollouts return the absolute career score, so their length is how much of
# the remaining career, including race soft-fails, search can see. A 6-turn
# cutoff never reaches the next stamina gate. Full-career rollouts do: on 96
# paired seeds they scored +1981 ± 358 against that cutoff at the old
# uct_c=2, finished 12 careers instead of 1, and survived +19 turns.
# uct_c=200 beat uct_c=2 by +1238 ± 452 on the same full-horizon seeds;
# uct_c=400 did not beat 200. 100 simulations and 15 rollouts take about
# 42s per game, and the longest of those 96 was 62s. A cutoff of 48 turns
# is identical to no cutoff, because a random rollout ends at a race
# before then.
ROLLOUT_TURNS = None
DEFAULT_UCT_C = 200.0
DEFAULT_MAX_SIMULATIONS = 100
DEFAULT_N_ROLLOUTS = 15
# Career chance nodes and the search consume different streams. Paired sweeps
# then share one environment seed without the bot's internals shifting luck.
BOT_SEED_OFFSET = 1_000_003

ChildSelection = Callable[[mcts.SearchNode, int, float], float]


@dataclass
class PlayTrace:
    """Optional diagnostics filled in by `play`."""

    snapshots: dict[int, tuple[int, int]] = field(default_factory=dict)
    decisions: int = 0


class TruncatedRolloutEvaluator(mcts.RandomRolloutEvaluator):
    """Random rollouts, sampled without numpy's `choice(p=...)`.

    `choice` validates the distribution on every call, which costs ~5us per
    chance node against ~0.3us for a cumulative sum. With eight chance nodes a
    turn that dominates the search. The rollout policy is unchanged.
    """

    def evaluate(self, state):
        random_sample = self._random_state.random_sample
        total = 0.0
        for _ in range(self.n_rollouts):
            working_state = state.clone()
            length = 0
            while not working_state.is_terminal():
                if working_state.is_chance_node():
                    threshold = random_sample()
                    cumulative = 0.0
                    for action, probability in working_state.chance_outcomes():
                        cumulative += probability
                        if threshold < cumulative:
                            break
                else:
                    legal_actions = working_state.legal_actions()
                    action = legal_actions[int(random_sample() * len(legal_actions))]
                working_state.apply_action(action)
                length += 1
                if self.max_length is not None and length >= self.max_length:
                    break
            total += working_state.returns()[0]
        return [total / self.n_rollouts]


def _rollout_length(rollout_turns: int | None) -> int | None:
    if rollout_turns is None:
        return None
    return rollout_turns * NODES_PER_TURN


def _record_passed_turns(
    snapshots: dict[int, tuple[int, int]],
    record_turns: tuple[int, ...],
    before: int,
    state: UmaState,
) -> None:
    """Stats on arrival at `record_turns`, including race turns consumed inside one action."""
    after = state.turn
    for turn in record_turns:
        if turn not in snapshots and before < turn <= after:
            snapshots[turn] = (state.stats[0], state.stats[1])


def play(
    *,
    verbose: bool = True,
    seed: int = 42,
    uct_c: float = DEFAULT_UCT_C,
    max_simulations: int = DEFAULT_MAX_SIMULATIONS,
    n_rollouts: int = DEFAULT_N_ROLLOUTS,
    rollout_turns: int | None = ROLLOUT_TURNS,
    solve: bool = True,
    dont_return_chance_node: bool = False,
    child_selection_fn: ChildSelection | None = None,
    record_turns: tuple[int, ...] = (),
    decision_limit: int | None = None,
    trace: PlayTrace | None = None,
) -> tuple[UmaState, list[int]]:
    """Play one career.

    `rollout_turns=None` rolls out until the career ends. `decision_limit`
    stops after that many player decisions so a sweep can time search without
    finishing the career; scored games leave it unset.
    """
    # MCTSBot rejects non-TERMINAL reward_model, but that is only a metadata
    # check. This game already implements returns() and the rest of the State
    # API MCTS needs; per-turn rewards stay on the default game for DQN.
    game = UmaGame(reward_model=pyspiel.GameType.RewardModel.TERMINAL)
    state: UmaState = game.new_initial_state()
    env_rng = np.random.RandomState(seed)
    bot_rng = np.random.RandomState(seed + BOT_SEED_OFFSET)
    bot_kwargs: dict[str, object] = {}
    if child_selection_fn is not None:
        bot_kwargs["child_selection_fn"] = child_selection_fn
    bot = mcts.MCTSBot(
        game,
        uct_c=uct_c,
        max_simulations=max_simulations,
        evaluator=TruncatedRolloutEvaluator(
            n_rollouts=n_rollouts,
            random_state=bot_rng,
            max_length=_rollout_length(rollout_turns),
        ),
        solve=solve,
        random_state=bot_rng,
        dont_return_chance_node=dont_return_chance_node,
        **bot_kwargs,
    )
    player_actions: list[int] = []
    snapshots: dict[int, tuple[int, int]] = {}
    while not state.is_terminal():
        if (
            decision_limit is not None
            and len(player_actions) >= decision_limit
            and not state.is_chance_node()
        ):
            break
        before = state.turn
        if state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            action = env_rng.choice(outcomes, p=probs)
            action_label = None
        else:
            action = bot.step(state)
            player_actions.append(int(action))
            action_label = state.action_to_string(state.current_player(), action)
        state.apply_action(action)
        if record_turns:
            _record_passed_turns(snapshots, record_turns, before, state)
        if verbose and action_label is not None:
            print(action_label)
            print(state)
    if verbose:
        print(f"Returns: {state.returns()}")
    if trace is not None:
        trace.snapshots = snapshots
        trace.decisions = len(player_actions)
    return state, player_actions


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
