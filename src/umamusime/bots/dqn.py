import pathlib
import statistics
import time

import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

from ..calendar import MAX_TURNS
from ..game import UmaGame
from ..state import UmaState

# Rich observations, linear exploration, and a larger replay buffer reached
# about 6476 mean score on 30 held-out seeds in a 99-second probe run.
_TRAINING_EPISODES = 6000
_TRAINING_SEED = 42
_EVAL_EVERY = 200
_CHECKPOINT = pathlib.Path("dqn_checkpoint.pt")


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


def _make_env_and_agent() -> tuple[rl_environment.Environment, dqn.DQN]:
    env = rl_environment.Environment(UmaGame())
    env.seed(_TRAINING_SEED)
    agent = dqn.DQN(
        player_id=0,
        state_representation_size=env.observation_spec()["info_state"][0],
        num_actions=env.action_spec()["num_actions"],
        hidden_layers_sizes=[64, 64],
        replay_buffer_capacity=50000,
        batch_size=128,
        learning_rate=0.01,
        optimizer_str=dqn.Optimiser.ADAM,
        learn_every=4,
        update_target_network_every=500,
        epsilon_end=0.05,
        epsilon_decay_schedule_str=dqn.EpsilonDecaySchedule.LINEAR,
        epsilon_decay_duration=int(_TRAINING_EPISODES * MAX_TURNS * 0.6),
        seed=_TRAINING_SEED,
    )
    return env, agent


def _evaluate(
    env: rl_environment.Environment,
    agent: dqn.DQN,
    *,
    first_seed: int = 1000,
    num_seeds: int = 30,
) -> tuple[float, float, float, int]:
    """Return mean, population SD, median score, and finish count."""
    scores: list[float] = []
    finished = 0
    for seed in range(first_seed, first_seed + num_seeds):
        env.seed(seed)
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step, is_evaluation=True)
            time_step = env.step([agent_output.action])
        state = env.get_state
        scores.append(state.returns()[0])
        finished += not state.ended_by_race_fail()
    return (
        statistics.fmean(scores),
        statistics.pstdev(scores),
        statistics.median(scores),
        finished,
    )


def _evaluation_summary(env: rl_environment.Environment, agent: dqn.DQN) -> str:
    mean, standard_deviation, median, finished = _evaluate(env, agent)
    return (
        f"greedy mean {mean:.0f}, median {median:.0f}, "
        f"sd {standard_deviation:.0f}, finished {finished}/30"
    )


def _train(env: rl_environment.Environment, agent: dqn.DQN, *, log: bool) -> None:
    for episode in range(_TRAINING_EPISODES):
        time_step = env.reset()
        while not time_step.last():
            agent_output = agent.step(time_step)
            time_step = env.step([agent_output.action])
        agent.step(time_step)
        if log and (episode + 1) % _EVAL_EVERY == 0:
            print(
                f"Episode {episode + 1}, loss {agent.loss}, "
                f"{_evaluation_summary(env, agent)}"
            )


def train(*, verbose: bool = True, checkpoint: pathlib.Path = _CHECKPOINT) -> float:
    """Train a new agent from scratch, overwrite the checkpoint, return seconds."""
    env, agent = _make_env_and_agent()
    started = time.perf_counter()
    _train(env, agent, log=verbose)
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
) -> tuple[UmaState, list[int]]:
    env, agent = _make_env_and_agent()
    if retrain or not checkpoint.exists():
        _train(env, agent, log=verbose)
        _save(agent, checkpoint)
        if verbose:
            print(f"Saved {checkpoint}")
    else:
        try:
            agent.load(checkpoint)
        except (KeyError, RuntimeError):
            if verbose:
                print(f"Checkpoint {checkpoint} is incompatible; retraining")
            _train(env, agent, log=verbose)
            _save(agent, checkpoint)
        else:
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
