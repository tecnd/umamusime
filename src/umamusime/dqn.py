import pathlib

import torch
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

from .umamusime import UmaGame

_TRAINING_EPISODES = 600
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


def _eval_return(env: rl_environment.Environment, agent: dqn.DQN) -> float:
    time_step = env.reset()
    total = 0.0
    while not time_step.last():
        agent_output = agent.step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
        total += time_step.rewards[0]
    return total


def main() -> None:
    env = rl_environment.Environment(UmaGame())
    agent = dqn.DQN(
        player_id=0,
        state_representation_size=env.observation_spec()["info_state"][0],
        num_actions=env.action_spec()["num_actions"],
        hidden_layers_sizes=[64, 64],
        replay_buffer_capacity=10000,
        batch_size=128,
        learning_rate=0.01,
        optimizer_str=dqn.Optimiser.ADAM,
        epsilon_decay_duration=_TRAINING_EPISODES * 60 // 2,
    )

    if _CHECKPOINT.exists():
        agent.load(_CHECKPOINT)
        print(f"Loaded {_CHECKPOINT}, skipping training")
    else:
        for episode in range(_TRAINING_EPISODES):
            time_step = env.reset()
            while not time_step.last():
                agent_output = agent.step(time_step)
                time_step = env.step([agent_output.action])
            agent.step(time_step)
            if (episode + 1) % _EVAL_EVERY == 0:
                print(
                    f"Episode {episode + 1}, loss {agent.loss}, "
                    f"greedy return {_eval_return(env, agent)}"
                )
        _save(agent, _CHECKPOINT)
        print(f"Saved {_CHECKPOINT}")

    time_step = env.reset()
    while not time_step.last():
        agent_output = agent.step(time_step, is_evaluation=True)
        print(env.get_state.action_to_string(0, agent_output.action))
        time_step = env.step([agent_output.action])
    print(env.get_state)
    print(f"Returns: {env.get_state.returns()}")


if __name__ == "__main__":
    main()
