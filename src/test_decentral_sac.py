import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from matplotlib import pyplot as plt

from agents.sac import SACAgent

def train_sac_agent(
    env: CityLearnEnv, 
    agents: list[SACAgent], 
    episodes: int = 100, 
    decentralized = False
) -> None:
    """Train SAC agent in the environment"""
    total_reward = 0
    
    reward_list = []
    episode_rewards = []  # List to store episode rewards
    
    for episode in range(episodes):
        # Reset environment and get initial observation
        observation = env.reset()
        print(len(observation[0]))

        episode_reward = 0
        
        while not env.done:
            if not decentralized:
                flat_observation = np.concatenate(observation) if isinstance(observation, list) else observation

            
            # select actions based on different paradigms
            if decentralized:
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
            
            reward_list.append(np.sum(reward))

            if not decentralized:       
                flat_next_observation = np.concatenate(next_observation) if isinstance(next_observation, list) else next_observation
            
            episode_reward += np.sum(reward)

            if decentralized:
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

        total_reward += episode_reward
        episode_rewards.append(episode_reward)
        print(total_reward)
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward}")
    
    return reward_list, episode_rewards


def set_seed(seed: int = 0) -> None:
    """ Set seed. """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

def create_environment(central_agent: bool = True, SEED = 0, path: str = "data/citylearn_challenge_2023_phase_1"):
    """ Creates the CityLearn environment. """
    
    set_seed(SEED)
    root_directory = Path(path)
    
    if not root_directory.exists():
        raise ValueError(f"Path {path} does not exist")
    
    schema_path = root_directory / "schema.json"
    schema_path = path
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
                  exploration_timesteps: int = 0):
    """
    Creates the agents with the given specification.
    Args:
        
    
    """
    if central_agent:
        observation_space_dim = 49
        action_space_dim = 18
        building_number = 1
    else:
        observation_space_dim = 29
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
    
if __name__ == "__main__":
    
    # Create the environments
    centralized_env = create_environment(central_agent=True, SEED=0,  path="data/citylearn_challenge_2023_phase_1")
    decentralized_env = create_environment(central_agent=False, SEED=0,  path="data/citylearn_challenge_2023_phase_1")
    
    # Create the agents
    centralized_agent = create_agents(centralized_env, central_agent=True)
    decentralized_agent = create_agents(decentralized_env, central_agent=False)
    
    # Train the agent
    rewards, episode_rewards = train_sac_agent(decentralized_env, decentralized_agent, episodes=6, decentralized=True)
    
    # TODO: CONVERT TO FUNCTION
    # Convert rewards list to numpy array for easier manipulation
    summed_rewards = np.array(rewards)  # rewards is your list of reward lists
    steps = range(1, len(summed_rewards) + 1)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot raw rewards
    plt.plot(steps, summed_rewards, alpha=0.3, color='blue', label='Raw Rewards')

    # Add rolling average
    window_size = 100  # Adjust this value based on your needs
    rolling_mean = pd.Series(summed_rewards).rolling(window=window_size, min_periods=1).mean()
    plt.plot(steps, rolling_mean, color='red', linewidth=2, label=f'{window_size}-step Moving Average')

    # Customize the plot
    plt.title("Decentralized SAC Agent Rewards Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Optional: Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Optional: Use scientific notation for large numbers on x-axis
    plt.ticklabel_format(axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig("step_rewards_decentralized.png", dpi=300, bbox_inches='tight')
    plt.show()