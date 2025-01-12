import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv
from typing import List

from agents.sac import SACAgent

from copy import deepcopy

from utils.replay_buffer import ReplayBuffer

from utils.plotting import plot_single_agent, plot_all_agents

from typing import Tuple


def set_seed(seed: int = 0) -> None:
    """Set a seed used in the simulation."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_environment(
    central_agent: bool = True,
    SEED: int = 0,
    path: str = "data/citylearn_challenge_2023_phase_1",
) -> CityLearnEnv:
    """Create a CityLearn environment using the provided path, seed, and agent type."""
    set_seed(SEED)
    root_directory = Path(path)

    if not root_directory.exists():
        raise ValueError(f"Path {path} does not exist")

    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_directory,
        random_seed=SEED,
        central_agent=central_agent,
    )

    return env


def create_agents(
    env: CityLearnEnv,
    central_agent: bool = False,
    hidden_dim: int = 256,
    buffer_size: int = 100000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.01,
    alpha: float = 0.05,
    batch_size: int = 256,
    action_space: list = None,
    observation_space_dim: int = 0,
    action_space_dim: int = 0,
    num_buildings: int = 0,
) -> List[SACAgent]:
    """
    Creates the agents with the given specification.
    Args:
        env: The CityLearn environment.
        central_agent: Whether to create a central agent.
        hidden_dim: The hidden dimension of the network for the SAC agent.
        buffer_size: The replay buffer size.
        learning_rate: The learning rate for the SAC agent.
        gamma: The discount factor for the SAC agent.
        tau: The target network update rate for the SAC agent.
        alpha: The temperature parameter for the SAC agent.
        batch_size: The batch size used for the SAC agent.
    Returns:
        Agent or a list of agents
    """

    agents = []
    if central_agent:
        num_agents = 1
    else:
        num_agents = num_buildings

    for _ in range(num_agents):
        agents.append(
            SACAgent(
                observation_space_dim=observation_space_dim,
                action_space_dim=action_space_dim,
                hidden_dim=hidden_dim,
                buffer_size=buffer_size,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                batch_size=batch_size,
                action_space=action_space,
            )
        )
    return agents


def train_centralized_agent(
    env: CityLearnEnv,
    agent: SACAgent,
    episodes: int = 100,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the central agent.
    Args:
        env: The CityLearn environment.
        agent: The central agent.
        episodes: The number of episodes to train the agent.
    Returns:
        The list of rewards
    """

    reward_list = []
    day_rewards = []
    episode_returns = []

    for episode in range(episodes):
        observation = env.reset()

        episode_return = 0
        current_daily_reward = 0

        while not env.done:
            flat_observation = (
                np.concatenate(observation)
                if isinstance(observation, list)
                else observation
            )

            # The CityLearn environment expects a list of actions even for the centralized agent
            actions = [agent.select_action(flat_observation).tolist()]

            agent.total_steps += 1

            next_observation, reward, info, done = env.step(actions)

            reward_list.append(np.sum(reward))
            current_daily_reward += np.sum(reward)

            if agent.total_steps % 24 == 0:
                day_rewards.append(np.mean(current_daily_reward))
                current_daily_reward = 0

            flat_next_observation = (
                np.concatenate(next_observation)
                if isinstance(next_observation, list)
                else next_observation
            )

            episode_return += np.sum(reward)

            agent.replay_buffer.push(
                flat_observation,
                actions,
                np.sum(reward),
                flat_next_observation,
                len(done),
            )
            
            agent.train()

            observation = next_observation

        episode_returns.append(episode_return)

        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_return}")

        plot_single_agent(day_rewards, agent_type="centralized", plot_folder="plots/")

    return reward_list, episode_returns, day_rewards


def train_decentralized_agent(
    env: CityLearnEnv,
    agents: list[SACAgent],
    episodes: int = 100,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the central agent.
    Args:
        env: The CityLearn environment.
        agents: A list of agents.
        episodes: The number of episodes to train the agents.
    Returns:
        The list of rewards
    """

    reward_list = []
    day_rewards = []
    episode_returns = []

    for episode in range(episodes):
        observation = env.reset()

        episode_return = 0
        current_daily_reward = 0

        while not env.done:
            actions = [0 for _ in range(len(agents))]

            for i in range(len(agents)):
                actions[i] = agents[i].select_action(np.array(observation[i])).tolist()

            for agent in agents:
                agent.total_steps += 1

            next_observation, reward, info, done = env.step(actions)

            reward_list.append(reward)
            current_daily_reward += np.sum(reward)
            
            if agents[0].total_steps % 24 == 0:
                daily_rewards.append(np.mean(current_daily_reward))
                current_daily_reward = 0
            
            episode_return += np.sum(reward)

            for i in range(len(agents)):
                agents[i].replay_buffer.push(
                    observation[i],
                    actions[i],
                    reward[i],
                    next_observation[i],
                    len(done),
                )
                
            for agent in agents:
                agent.train()

            observation = next_observation

        episode_returns.append(episode_return)

        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_return}")

        plot_single_agent(day_rewards, agent_type="decentralized", plot_folder="plots/")

    return reward_list, episode_returns, day_rewards


def train_maml_agent(
    env: CityLearnEnv,
    base_agent: list[SACAgent],
    episodes: int = 100,
    building_count: int = 1,
    learning_rate: float = 3e-4,
    k_shots: int = 3,
) -> Tuple[List[float], List[float], List[float]]:
    """Train the MAML agent.
    Args:
        env: The CityLearn environment.
        agents: A list of agents.
        episodes: The number of episodes to train the agents.
    Returns:
        The list of rewards
    """
    reward_list = []
    day_rewards = []
    episode_returns = []

    actor_optimizer = torch.optim.Adam(base_agent.actor.parameters(), lr=learning_rate)

    critic_optimizer = torch.optim.Adam(
        base_agent.critic.parameters(), lr=learning_rate
    )

    copied_agents = [deepcopy(base_agent) for _ in range(building_count)]

    building_buffers = [ReplayBuffer(capacity=100000) for _ in range(building_count)]

    for episode in range(episodes):
        observation = env.reset()

        episode_return = 0
        current_daily_reward = 0

        while not env.done:
            if base_agent.total_steps % k_shots == 0:
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

                for copied_agent in copied_agents:
                    for param, copied_param in zip(
                        base_agent.actor.parameters(), copied_agent.actor.parameters()
                    ):
                        if copied_param.grad is not None:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            param.grad += copied_param.grad / building_count

                    for param, copied_param in zip(
                        base_agent.critic.parameters(), copied_agent.critic.parameters()
                    ):
                        if copied_param.grad is not None:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)
                            param.grad += copied_param.grad / building_count

                actor_optimizer.step()
                critic_optimizer.step()

                copied_agents = [deepcopy(base_agent) for _ in range(building_count)]

            actions = [0 for _ in range(building_count)]
            for i in range(building_count):
                actions[i] = (
                    copied_agents[i].select_action(np.array(observation[i])).tolist()
                )

            base_agent.total_steps += 1

            next_observation, reward, info, done = env.step(actions)

            reward_list.append(reward)
            current_daily_reward += np.sum(reward)

            if base_agent.total_steps % 24 == 0:
                day_rewards.append(np.mean(current_daily_reward))
                current_daily_reward = 0

            episode_return += np.sum(reward)

            for building_buffer in building_buffers:
                building_buffer.push(
                    observation[i],
                    actions[i],
                    reward[i],
                    next_observation[i],
                    len(done),
                )

            total_agent_loss = 0
            total_critic_loss = 0
            for i in range(building_count):
                agent_loss, critic_loss = copied_agents[i].train(
                    update=True, custom_buffer=building_buffers[i]
                )
                total_agent_loss += agent_loss
                total_critic_loss += critic_loss

            observation = next_observation

        episode_returns.append(episode_return)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_return}")

        plot_single_agent(day_rewards, agent_type="maml", plot_folder="plots/")

    return reward_list, episode_returns, day_rewards


def setup_single_agent(
    agent_type: str = "centralized",
    seed: int = 0,
    hyperparameters_dict: dict = {},
    episodes: int = 100,
) -> List[float]:
    set_seed(seed)

    centralized = True if agent_type == "centralized" else False

    environment = create_environment(
        central_agent=centralized,
        SEED=seed,
        path="data/citylearn_challenge_2023_phase_1",
    )

    num_buildings = len(environment.buildings)
    observation_space_dim = environment.observation_space[0].shape[0]
    action_space_dim = environment.action_space[0].shape[0]

    hidden_dim = hyperparameters_dict.get("hidden_dim", 256)
    buffer_size = hyperparameters_dict.get("buffer_size", 100000)
    learning_rate = hyperparameters_dict.get("learning_rate", 3e-4)
    gamma = hyperparameters_dict.get("gamma", 0.99)
    tau = hyperparameters_dict.get("tau", 0.01)
    alpha = hyperparameters_dict.get("alpha", 0.05)
    batch_size = hyperparameters_dict.get("batch_size", 256)
    k_shots = hyperparameters_dict.get("k_shots", 3)

    agents = create_agents(
        env=environment,
        central_agent=centralized,
        hidden_dim=hidden_dim,
        buffer_size=buffer_size,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        batch_size=batch_size,
        action_space=environment.action_space,
        observation_space_dim=observation_space_dim,
        action_space_dim=action_space_dim,
        num_buildings=num_buildings,
    )
    print("-" * 50)
    print(f"Agent type: {agent_type}")
    print(f"Number of buildings: {num_buildings}")
    print(f"Observation space dimension: {observation_space_dim}")
    print(f"Action space dimension: {action_space_dim}")
    print(f"Number of agents: {len(agents)}")
    print("Agent hyperparameters:")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Buffer size: {buffer_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Tau: {tau}")
    print(f"Alpha: {alpha}")
    print(f"Batch size: {batch_size}")
    print(f"K-shots: {k_shots} (only used for MAML)")
    print("-" * 50)

    if agent_type == "centralized":
        return train_centralized_agent(
            env=environment,
            agent=agents[0],
            episodes=episodes,
        )

    elif agent_type == "decentralized":
        return train_decentralized_agent(
            env=environment,
            agents=agents,
            episodes=episodes,
        )

    elif agent_type == "maml":
        return train_maml_agent(
            env=environment,
            base_agent=agents[0],
            episodes=episodes,
            building_count=num_buildings,
            k_shots=k_shots,
        )


def setup_all_agents(
    seed: int = 0, episodes: int = 100, hyperparameters_dict: dict = {}
) -> None:
    _, episode_returns_centralized, daily_rewards_centralized = setup_single_agent(
        agent_type="centralized",
        seed=seed,
        episodes=episodes,
        hyperparameters_dict=hyperparameters_dict,
    )
    _, episode_returns_decentralized, daily_rewards_decentralized = setup_single_agent(
        agent_type="decentralized",
        seed=seed,
        episodes=episodes,
        hyperparameters_dict=hyperparameters_dict,
    )
    _, episode_returns_maml, daily_rewards_maml = setup_single_agent(
        agent_type="maml",
        seed=seed,
        episodes=episodes,
        hyperparameters_dict=hyperparameters_dict,
    )

    rewards_dict = {
        "centralized": daily_rewards_centralized,
        "decentralized": daily_rewards_decentralized,
        "maml": daily_rewards_maml,
    }

    plot_all_agents(rewards_dict, plot_folder="plots/")
