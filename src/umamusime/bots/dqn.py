import dataclasses
import pathlib
import statistics
import time

import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

from ..calendar import MAX_TURNS
from ..game import UmaGame
from ..state import UmaState

_CHECKPOINT = pathlib.Path("dqn_checkpoint.pt")
_EVAL_EVERY = 1000
# Greedy evaluation careers use fixed seeds on a separate environment so
# logging never perturbs the training chance stream.
_EVAL_SEED = 1000
_LOG_EVAL_CAREERS = 10


@dataclasses.dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters; see docs/dqn-tuning.md for how they were chosen."""

    episodes: int = 6000
    hidden_layers: tuple[int, ...] = (64, 64)
    learning_rate: float = 0.01
    replay_buffer_capacity: int = 50_000
    batch_size: int = 128
    learn_every: int = 10
    update_target_network_every: int = 1000
    min_buffer_size_to_learn: int = 1000
    epsilon_end: float = 0.05
    # Fraction of training steps over which epsilon falls linearly to its
    # floor. OpenSpiel's default "exp" schedule never gets below
    # end + (1 - end) / e (~0.43 here), so the policy would stay mostly random.
    epsilon_decay_fraction: float = 0.6
    seed: int = 42

    @property
    def epsilon_decay_duration(self) -> int:
        return int(self.epsilon_decay_fraction * self.episodes * MAX_TURNS)


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
) -> tuple[rl_environment.Environment, dqn.DQN]:
    env = rl_environment.Environment(UmaGame())
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


def evaluate(
    agent: dqn.DQN,
    *,
    careers: int = 30,
    first_seed: int = _EVAL_SEED,
    env: rl_environment.Environment | None = None,
) -> EvalResult:
    """Greedy careers on seeds first_seed..first_seed + careers - 1."""
    env = env or rl_environment.Environment(UmaGame())
    scores: list[float] = []
    finished = 0
    for seed in range(first_seed, first_seed + careers):
        env.seed(seed)
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
        state: UmaState = env.get_state
        scores.append(state.returns()[0])
        finished += not state.ended_by_race_fail()
    return EvalResult(
        mean=statistics.fmean(scores),
        stdev=statistics.pstdev(scores),
        median=statistics.median(scores),
        finished=finished,
        careers=careers,
    )


def _train(
    env: rl_environment.Environment,
    agent: dqn.DQN,
    config: DQNConfig = CONFIG,
    *,
    log: bool,
) -> None:
    env.seed(config.seed)
    eval_env = rl_environment.Environment(UmaGame()) if log else None
    for episode in range(config.episodes):
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step)
            time_step = env.step([agent_output.action])
        agent.step(time_step)
        if log and (episode + 1) % _EVAL_EVERY == 0:
            result = evaluate(agent, careers=_LOG_EVAL_CAREERS, env=eval_env)
            print(f"Episode {episode + 1}, loss {agent.loss}, greedy {result}")


def train(
    *,
    verbose: bool = True,
    checkpoint: pathlib.Path = _CHECKPOINT,
    config: DQNConfig = CONFIG,
) -> float:
    """Train a new agent from scratch, overwrite the checkpoint, return seconds."""
    env, agent = _make_env_and_agent(config)
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
) -> tuple[UmaState, list[int]]:
    env, agent = _make_env_and_agent(config)
    if retrain or not checkpoint.exists():
        _train(env, agent, config, log=verbose)
        _save(agent, checkpoint)
        if verbose:
            print(f"Saved {checkpoint}")
    else:
        agent.load(checkpoint)
        if verbose:
            print(f"Loaded {checkpoint}, skipping training")

    # Seed only the evaluation episode so training stays independent of it.
    if seed is not None:
        env.seed(seed)
    time_step = env.reset()
    actions: list[int] = []
    while not time_step.last():
        agent_output = agent.step(time_step, is_evaluation=True)
        action = int(agent_output.action)
        actions.append(action)
        if verbose:
            print(env.get_state.action_to_string(0, action))
        time_step = env.step([action])
    state: UmaState = env.get_state
    if verbose:
        print(state)
        print(f"Returns: {state.returns()}")
    return state, actions


def main() -> None:
    play(verbose=True)


if __name__ == "__main__":
    main()
