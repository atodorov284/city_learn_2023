import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from matplotlib import pyplot as plt
from typing import List

from copy import deepcopy
from utils.replay_buffer import ReplayBuffer

from agents.sac import SACAgent
from agents.autoencoder import SACEncoder

# VARIABLE DEFINITIONS--------------------
SEED = 0

CENTRALIZED_OBSERVATION_DIMENSION = 77
DECENTRALIZED_OBSERVATION_DIMENSION = 39

CENTRALIZED_ACTION_DIMENSION = 18
DECENTRALIZED_ACTION_DIMENSION = 6

ENCODER_HIDDEN_DIMENSION = 65
ENCODER_OUTPUT_DIMENSION = 50  # should be smaller than the observation dimension

TRAINING_EPISODES = 100
# ---------------------------------------


def train_sac_agent(
    env: CityLearnEnv,
    agents: list[SACAgent],
    episodes: int = 100,
) -> None:
    """Train SAC agent in the environment"""

    reward_list = []    # List to store rewards
    day_rewards = []    # List to store daily rewards
    episode_rewards = []  # List to store episode rewards
    
    building_count = 3
    
    base_agent = agents[0]
    
    actor_optimizer = torch.optim.Adam(base_agent.actor.parameters(), lr=3e-4)
    
    critic_optimizer = torch.optim.Adam(base_agent.critic.parameters(), lr=3e-4 * 2)

    copied_agents = [deepcopy(base_agent) for _ in range(building_count)]
    
    building_buffers = [ReplayBuffer(capacity=1) for _ in range(building_count)]


    for episode in range(episodes):
        # Reset environment and get initial observation
        observation = env.reset()

        episode_reward = 0
        curr_day_reward = 0


        while not env.done:
            
            if base_agent.total_steps % 3 == 0:
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

                # Accumulate gradients from each copied agent
                for copied_agent in copied_agents:
                    for param, copied_param in zip(base_agent.actor.parameters(), copied_agent.actor.parameters()):
                        if copied_param.grad is not None:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)  # Initialize if needed
                            param.grad += copied_param.grad/building_count
                            
                    for param, copied_param in zip(base_agent.critic.parameters(), copied_agent.critic.parameters()):
                        if copied_param.grad is not None:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param)  # Initialize if needed
                            param.grad += copied_param.grad/building_count
                            # print(param.grad)
                            
                # Use the optimizer to update parameters with accumulated gradients
                actor_optimizer.step()
                critic_optimizer.step()

                copied_agents = [deepcopy(base_agent) for _ in range(building_count)]
                
                
            # select actions based on different paradigms
            actions = [0 for _ in range(building_count)]
            for i in range(building_count):
                # agent_actions is used for the replay buffer
                actions[i] = copied_agents[i].select_action(observation[i]).tolist()

            # print(f"actions: {actions}") # action is a list of lists (one for each agent) of actions)
            for agent in agents:
                agent.total_steps += 1

            # take a step
            next_observation, reward, info, done = env.step(actions)

            # encode the next observation
            with torch.no_grad():
                # next_observation = encoder.encode(next_observation)
                pass

            reward_list.append(np.sum(reward))
            curr_day_reward += np.sum(reward)

            if base_agent.total_steps % 24 == 0:  # 168 for weekly, 1 for hourly
                day_rewards.append(np.mean(curr_day_reward))
                curr_day_reward = 0

            episode_reward += np.sum(reward)

            # store the transition in the replay buffer
            for building_buffer in building_buffers:
                building_buffer.push(
                    observation[i],
                    actions[i],
                    reward[i],
                    next_observation[i],
                    len(done),
                )

            # train the agents if enough timesteps have passed
            
            total_agent_loss = 0
            total_critic_loss = 0
            for i in range(building_count):
                agent_loss, critic_loss = copied_agents[0].train(update=True, custom_buffer=building_buffers[i])
                total_agent_loss += agent_loss
                total_critic_loss += critic_loss
                # print(f"Building {i} Loss: {agent_loss}, Critic Loss: {critic_loss}")

            observation = next_observation

        episode_rewards.append(episode_reward)

        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward}")

        plot_rewards(day_rewards, agent_type="decentralized", plot_folder="plots/")
       

    # print(day_rewards)
    return reward_list, episode_rewards, day_rewards


def set_seed(seed: int = 0) -> None:
    """Set a seed used in the simulation."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_environment(
    central_agent: bool = True,
    SEED=0,
    path: str = "data/citylearn_challenge_2023_phase_1",
):
    """
    Creates the CityLearn environment.
    Args:

    """

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
    buffer_size: int = 1,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.01,
    alpha: float = 0.05,
    batch_size: int = 256,
    exploration_timesteps: int = 0,
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
        exploration_timesteps: The number of exploration timesteps for the SAC agent.
    Returns:
        Agent or a list of agents
    """
    observation_space_dim = DECENTRALIZED_OBSERVATION_DIMENSION
    action_space_dim = DECENTRALIZED_ACTION_DIMENSION
    building_number = 3

    agents = []
    agents.append(
        SACAgent(
            observation_space_dim=observation_space_dim,
            action_space_dim=action_space_dim,
            hidden_dim=hidden_dim,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            action_space=env.action_space,
            exploration_timesteps=exploration_timesteps,
        )
    )
    return agents


def plot_rewards(
    rewards: list[float], agent_type: str = "centralized", plot_folder: str = "plots/"
) -> None:
    """
    Plots the rewards for different agent types.

    Args:
        rewards: List of rewards to plot
        agent_type: Type of agent ("centralized", "decentralized", or "both")
    """
    valid_types = ["centralized", "decentralized", "both"]
    if agent_type.lower() not in valid_types:
        raise ValueError(f"agent_type must be one of {valid_types}")

    rewards = np.array(rewards)
    steps = range(1, len(rewards) + 1)

    plt.figure(figsize=(12, 6))

    # Plot raw rewards
    plt.plot(steps, rewards, alpha=0.3, color="blue", label="Raw Rewards")

    # Add rolling average
    window_size = 15
    rolling_mean = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
    plt.plot(
        steps,
        rolling_mean,
        color="red",
        linewidth=2,
        label=f"{window_size}-step Moving Average",
    )

    title_prefix = {
        "centralized": "Centralized",
        "decentralized": "Decentralized",
        "both": "Centralized vs Decentralized",
    }
    plt.title(
        f"{title_prefix[agent_type.lower()]} SAC Agent Rewards Over Time",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

    plt.tight_layout()
    # save in directory plots
    save_path = plot_folder + f"step_rewards_{agent_type.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()

def evaluate_agent_performance(env: CityLearnEnv) -> None:
    """ 
    Evaluates the performance of the agent in an environment.   
    Uses premade functionality used in the challenge.
    """    
    kpis = env.evaluate() # Obtain the Key Performance Metrics
    kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
    kpis = kpis.dropna(how='all')
    kpis_reset = kpis.reset_index().rename(columns={"index": "metric"})
    print(kpis_reset)

if __name__ == "__main__":
    # Create the environments
    # centralized_env = create_environment(
    #     central_agent=True, SEED=SEED, path="data/citylearn_challenge_2023_phase_1"
    # )
    decentralized_env = create_environment(central_agent=False, SEED=SEED,  path="data/citylearn_challenge_2023_phase_1")

    # Create the agents
    # centralized_agent = create_agents(centralized_env, central_agent=True)
    decentralized_agent = create_agents(decentralized_env, central_agent=False)
    
    # print(centralized_agent[0].device)
    # Train the agent
    # rewards_centralized, episode_rewards_centralized, daily_rewards_centralized = (
    #     train_sac_agent(
    #         centralized_env, centralized_agent, episodes=TRAINING_EPISODES, central_agent=True
    #     )
    # )
    rewards_decentralized, episode_rewards_decentralized, daily_rewards_decentralized = train_sac_agent(decentralized_env, decentralized_agent, episodes=TRAINING_EPISODES)

    # Plot the rewards
    #plot_rewards(daily_rewards_centralized, agent_type="centralized", plot_folder="plots/")
    # plot_rewards(daily_rewards_decentralized, agent_type="decentralized", plot_folder="plots/")

    # Evaluate the agent
    # evaluate_agent_performance(decentralized_env)
