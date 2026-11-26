import copy
import dataclasses
import pathlib
import statistics
import time
from collections.abc import Sequence

import numpy as np
import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

from ..game import UmaGame
from ..state import UmaState
from . import base

_CHECKPOINT = pathlib.Path("dqn_checkpoint.pt")
# Agent steps between greedy evaluation log lines during training.
_EVAL_EVERY_STEPS = 34_000
# Greedy evaluation careers use fixed seeds on a separate career runner so
# logging never perturbs the training chance stream.
_EVAL_SEED = 1000
_LOG_EVAL_CAREERS = 10


@dataclasses.dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters; see docs/dqn-tuning.md for how they were chosen."""

    # One per DQN.step call during training, including each episode's terminal step.
    total_steps: int = 400_000
    hidden_layers: tuple[int, ...] = (64, 64)
    learning_rate: float = 0.01
    replay_buffer_capacity: int = 100_000
    batch_size: int = 128
    learn_every: int = 10
    update_target_network_every: int = 250
    min_buffer_size_to_learn: int = 1000
    epsilon_end: float = 0.05
    # Fraction of training steps over which epsilon falls linearly to its
    # floor. OpenSpiel's default "exp" schedule never gets below
    # end + (1 - end) / e (~0.43 here), so the policy would stay mostly random.
    epsilon_decay_fraction: float = 0.3
    # Once epsilon reaches its floor, score the greedy policy on this many
    # held-out careers every snapshot_eval_every_steps and restore the best
    # weights at the end; the final weights alone sometimes collapse. 0 keeps
    # the final weights.
    snapshot_eval_careers: int = 50
    snapshot_eval_every_steps: int = 20_000
    seed: int = 42

    @property
    def epsilon_decay_duration(self) -> int:
        return int(self.epsilon_decay_fraction * self.total_steps)


CONFIG = DQNConfig()


@dataclasses.dataclass(frozen=True)
class EvalResult:
    mean: float
    stdev: float
    median: float
    finished: int
    careers: int

    def __str__(self) -> str:
        return (
            f"mean {self.mean:.0f} sd {self.stdev:.0f} median {self.median:.0f} "
            f"finished {self.finished}/{self.careers}"
        )


def _save(agent: dqn.DQN, path: pathlib.Path) -> None:
    # DQN.save writes keys that DQN.load does not read, so write the names
    # load() expects instead of calling it.
    torch.save(
        {
            "iteration": agent._iteration,
            "last_loss_value": agent._last_loss_value,
            "model_state_dict": agent._q_network.state_dict(),
            "optimizer_state_dict": agent._optimizer.state_dict(),
        },
        path,
    )


def _make_env_and_agent(
    config: DQNConfig = CONFIG,
    *,
    game: UmaGame | None = None,
) -> tuple[rl_environment.Environment, dqn.DQN]:
    env = rl_environment.Environment(game or UmaGame())
    agent = dqn.DQN(
        player_id=0,
        state_representation_size=env.observation_spec()["info_state"][0],
        num_actions=env.action_spec()["num_actions"],
        hidden_layers_sizes=list(config.hidden_layers),
        replay_buffer_capacity=config.replay_buffer_capacity,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        learn_every=config.learn_every,
        update_target_network_every=config.update_target_network_every,
        min_buffer_size_to_learn=config.min_buffer_size_to_learn,
        optimizer_str=dqn.Optimiser.ADAM,
        epsilon_end=config.epsilon_end,
        epsilon_decay_schedule_str=dqn.EpsilonDecaySchedule.LINEAR,
        epsilon_decay_duration=config.epsilon_decay_duration,
        seed=config.seed,
    )
    return env, agent


class DQNBot:
    """Greedy DQN player implementing the Umamusime Bot protocol."""

    def __init__(self, agent: dqn.DQN) -> None:
        self._agent = agent

    def predict(self, state: UmaState) -> int:
        with torch.inference_mode():
            action, _ = self._agent.act_epsilon_greedy(
                state.observation_tensor(),
                state.legal_actions(),
                0.0,
            )
        return int(action)


def evaluate(
    agent: dqn.DQN,
    *,
    careers: int = 30,
    first_seed: int = _EVAL_SEED,
    game: UmaGame | None = None,
    seeds: Sequence[int] | None = None,
) -> EvalResult:
    """Greedy careers on seeds first_seed..first_seed + careers - 1, or on seeds."""
    game = game or UmaGame()
    bot = DQNBot(agent)
    chosen = (
        tuple(seeds)
        if seeds is not None
        else tuple(range(first_seed, first_seed + careers))
    )
    scores: list[float] = []
    finished = 0
    for seed in chosen:
        result = base.play(bot, game=game, seed=seed, verbose=False)
        scores.append(result.state.returns()[0])
        finished += not result.state.ended_by_race_fail()
    return EvalResult(
        mean=statistics.fmean(scores),
        stdev=statistics.pstdev(scores),
        median=statistics.median(scores),
        finished=finished,
        careers=len(chosen),
    )


def _snapshot_career_seeds(training_seed: int, count: int) -> tuple[int, ...]:
    """Distinct random 32-bit career seeds, kept off any consecutive eval block.

    split_rngs uses the first two children of SeedSequence(seed).spawn, so the
    snapshot stream is the third child.
    """
    stream = np.random.SeedSequence(training_seed).spawn(3)[2]
    draw = np.random.default_rng(stream)
    chosen: dict[int, None] = {}
    while len(chosen) < count:
        chosen[int(draw.integers(0, 2**32))] = None
    return tuple(chosen)


def _restore_snapshot(agent: dqn.DQN, weights: dict[str, torch.Tensor]) -> None:
    agent._q_network.load_state_dict(weights)
    agent._target_q_network.load_state_dict(weights)


def _train(
    env: rl_environment.Environment,
    agent: dqn.DQN,
    config: DQNConfig = CONFIG,
    *,
    log: bool,
) -> int | None:
    """Train until the step budget. Return the step of the restored snapshot.

    None means snapshot selection was off, or epsilon never reached its floor.
    """
    env.seed(config.seed)
    eval_game: UmaGame = env.game
    episode = 0
    next_log_step = _EVAL_EVERY_STEPS
    snapshot_seeds: tuple[int, ...] = ()
    next_snapshot = config.epsilon_decay_duration
    best_mean: float | None = None
    best_step: int | None = None
    best_weights: dict[str, torch.Tensor] | None = None
    if config.snapshot_eval_careers > 0:
        if config.snapshot_eval_every_steps <= 0:
            raise ValueError("snapshot_eval_every_steps must be positive")
        snapshot_seeds = _snapshot_career_seeds(
            config.seed, config.snapshot_eval_careers
        )
    while agent.step_counter < config.total_steps:
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step)
            time_step = env.step([agent_output.action])
        agent.step(time_step)
        episode += 1
        if log and agent.step_counter >= next_log_step:
            result = evaluate(agent, careers=_LOG_EVAL_CAREERS, game=eval_game)
            print(
                f"Episode {episode}, step {agent.step_counter}, "
                f"loss {agent.loss}, greedy {result}"
            )
            while next_log_step <= agent.step_counter:
                next_log_step += _EVAL_EVERY_STEPS
        if (
            snapshot_seeds
            and agent.step_counter >= config.epsilon_decay_duration
            and agent.step_counter >= next_snapshot
        ):
            scored = evaluate(agent, seeds=snapshot_seeds, game=eval_game)
            if best_mean is None or scored.mean > best_mean:
                best_mean = scored.mean
                best_step = int(agent.step_counter)
                best_weights = copy.deepcopy(agent._q_network.state_dict())
            while next_snapshot <= agent.step_counter:
                next_snapshot += config.snapshot_eval_every_steps
    if best_weights is not None:
        _restore_snapshot(agent, best_weights)
        if log:
            print(
                f"Restored snapshot from step {best_step}, held-out mean {best_mean:.0f}"
            )
    return best_step


def train(
    *,
    verbose: bool = True,
    checkpoint: pathlib.Path = _CHECKPOINT,
    config: DQNConfig = CONFIG,
    game: UmaGame | None = None,
) -> float:
    """Train a new agent from scratch, overwrite the checkpoint, return seconds."""
    env, agent = _make_env_and_agent(config, game=game)
    started = time.perf_counter()
    _train(env, agent, config, log=verbose)
    elapsed = time.perf_counter() - started
    _save(agent, checkpoint)
    if verbose:
        print(f"Trained {checkpoint} in {elapsed:.1f}s")
    return elapsed


def play(
    *,
    verbose: bool = True,
    checkpoint: pathlib.Path = _CHECKPOINT,
    seed: int | None = None,
    retrain: bool = False,
    config: DQNConfig = CONFIG,
    game: UmaGame | None = None,
) -> base.PlayResult:
    """Play one greedy career, training and saving a checkpoint first if needed.

    The checkpoint does not record which game it was trained on, so pass
    retrain=True when game differs from the one that produced it.
    """
    game = game or UmaGame()
    env, agent = _make_env_and_agent(config, game=game)
    if retrain or not checkpoint.exists():
        _train(env, agent, config, log=verbose)
        _save(agent, checkpoint)
        if verbose:
            print(f"Saved {checkpoint}")
    else:
        agent.load(checkpoint)
        if verbose:
            print(f"Loaded {checkpoint}, skipping training")

    return base.play(DQNBot(agent), game=game, seed=seed, verbose=verbose)


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
