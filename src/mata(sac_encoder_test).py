import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from matplotlib import pyplot as plt
from typing import List

from agents.sac import SACAgent
from agents.autoencoder import SACEncoder

def train_sac_agent(
    env: CityLearnEnv, 
    agents: list[SACAgent], 
    episodes: int = 100, 
    central_agent: bool = False,
    use_random_encoder: bool = True
) -> None:
    """Train SAC agent in the environment"""
    encoder = SACEncoder(observation_space_dim=77, output_dim=30, hidden_dim=50)

    if not use_random_encoder:
        # Load the state_dict and set the model to evaluation mode
        loaded_state_dict = torch.load("current_sac_encoder.pth")
        encoder.load_state_dict(loaded_state_dict)
        # encoder.eval()

    reward_list = []
    day_rewards = []
    episode_rewards = []  # List to store episode rewards

    for episode in range(episodes):
        # Reset environment and get initial observation
        observation = env.reset()

        episode_reward = 0
        curr_day_reward = 0

        with torch.no_grad():
            observation = encoder.encode(observation)
            
        while not env.done:
            if central_agent:
                flat_observation = np.concatenate(observation) if isinstance(observation, list) else observation

            
            # select actions based on different paradigms
            if not central_agent:
                actions = [0 for _ in range(len(agents))]
                
                for i in range(len(agents)):
                    # agent_actions is used for the replay buffer
                    actions[i] = agents[i].select_action(observation[i]).tolist()
                    # ADD THIS TO REPLAY BUFFER SAMPLING ASW
            else:
                actions = [agents[0].select_action(flat_observation).tolist()]
            
            # print(f"actions: {actions}") # action is a list of lists (one for each agent) of actions)
            for agent in agents:
                agent.total_steps += 1
                    
            next_observation, reward, info, done = env.step(actions)

            with torch.no_grad():
                next_observation = encoder.encode(next_observation)
            
            reward_list.append(np.sum(reward))
            curr_day_reward += np.sum(reward)
            
            if agent.total_steps % 24 == 0:
                day_rewards.append(curr_day_reward)
                curr_day_reward = 0
            

            if central_agent:      
                flat_next_observation = np.concatenate(next_observation) if isinstance(next_observation, list) else next_observation
            
            episode_reward += np.sum(reward)

            if not central_agent:
                for i in range(len(agents)):
                    agents[i].replay_buffer.push(
                        observation[i], 
                        actions[i], 
                        reward[i],
                        next_observation[i], 
                        len(done)
                    )
            else:               
                agents[0].replay_buffer.push(
                    flat_observation, 
                    actions, 
                    np.sum(reward),
                    flat_next_observation, 
                    len(done)
                )
            
            # train the agents if enough timesteps have passed
            if agents[0].total_steps >= agents[0].exploration_timesteps:
                for agent in agents:
                    agent.train()
            
            
            observation = next_observation

        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward}")
    #print(day_rewards)
    return reward_list, episode_rewards, day_rewards


def set_seed(seed: int = 0) -> None:
    """ Set a seed used in the simulation. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

def create_environment(central_agent: bool = True, SEED = 0, path: str = "data/citylearn_challenge_2023_phase_1"):
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

def create_agents(env: CityLearnEnv, central_agent: bool = False,
                  hidden_dim: int = 256, buffer_size:int = 100000,
                  learning_rate: float = 3e-4, gamma: float = 0.99,
                  tau: float = 0.01, alpha: float = 0.05,
                  batch_size: int = 256,
                  exploration_timesteps: int = 0) -> SACAgent | List:
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
    if central_agent:
        observation_space_dim = 30 #77 TEST WITH SOME LARGER NUMBERS
        action_space_dim = 18
        building_number = 1
    else:
        observation_space_dim = 39
        action_space_dim = 6 
        building_number = 3
    
    agents = []
    for _ in range(building_number):
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
                exploration_timesteps = exploration_timesteps,
            )
        )
    return agents
    
    
def plot_rewards(rewards: list[float], agent_type: str = "centralized", plot_folder: str = "plots/") -> None:
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
    plt.plot(steps, rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Add rolling average
    window_size = 100
    rolling_mean = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
    plt.plot(steps, rolling_mean, color='red', linewidth=2, label=f'{window_size}-step Moving Average')
    
    title_prefix = {
        "centralized": "Centralized",
        "decentralized": "Decentralized",
        "both": "Centralized vs Decentralized"
    }
    plt.title(f"{title_prefix[agent_type.lower()]} SAC Agent Rewards Over Time", 
              fontsize=16, fontweight='bold')
    
    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    #save in directory plots
    save_path = plot_folder + f"step_rewards_{agent_type.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    
    # Create the environments
    centralized_env = create_environment(central_agent=True, SEED=0,  path="data/citylearn_challenge_2023_phase_1")
    #decentralized_env = create_environment(central_agent=False, SEED=0,  path="data/citylearn_challenge_2023_phase_1")

    # Create the agents
    centralized_agent = create_agents(centralized_env, central_agent=True)
    #decentralized_agent = create_agents(decentralized_env, central_agent=False)
    
    # Train the agent
    rewards_centralized, episode_rewards_centralized, daily_rewards_centralized = train_sac_agent(centralized_env, centralized_agent, episodes=100, central_agent=True)
    #rewards_decentralized, episode_rewards_decentralized, daily_rewards_decentralized = train_sac_agent(decentralized_env, decentralized_agent, episodes=15, central_agent=False)
    
    # Plot the rewards
    plot_rewards(daily_rewards_centralized, agent_type="centralized", plot_folder="plots/")
    #plot_rewards(daily_rewards_decentralized, agent_type="decentralized", plot_folder="plots/")