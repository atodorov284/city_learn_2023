import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d

from agents.base_agent import Agent
from agents.random_agent import RandomAgent
from agents.sac import SACAgent

def train_sac_agent(
    env: CityLearnEnv, 
    agent: SACAgent, 
    episodes: int = 100, 
) -> None:
    """Train SAC agent in the environment"""
    episode_rewards = []  # List to store episode rewards
    
    for episode in range(episodes):
        # Reset environment and get initial observation
        observation = env.reset()
        episode_reward = 0
        
        while not env.done:
            flat_observation = np.concatenate(observation) if isinstance(observation, list) else observation
            
            action = [agent.select_action(flat_observation).tolist()]
            
            agent.total_steps += 1
                    
            next_observation, reward, info, done = env.step(action)
            
            flat_next_observation = np.concatenate(next_observation) if isinstance(next_observation, list) else next_observation
            
            episode_reward += np.sum(reward)  # Accumulate reward for the episode
                        
            agent.replay_buffer.push(
                flat_observation, 
                action, 
                np.sum(reward),
                flat_next_observation, 
                len(done)
            )
            
            if agent.total_steps >= agent.exploration_timesteps:
                agent.train()
            
            observation = next_observation
        
        episode_rewards.append(episode_reward)  # Store the episode's total reward
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward}")
    
    return episode_rewards

def centralized_interact_with_env(
    env: CityLearnEnv, agent: Agent = RandomAgent, episodes: int = 100
) -> None:
    """Interact with environment using agent"""
    episode_rewards = []
    for episode in range(episodes):
        observation = env.reset()
        episode_reward = 0
        while not env.done:
            action = agent.select_action(observation)
            observation, reward, info, done = env.step(action)
            episode_reward += np.sum(reward)
        episode_rewards.append(episode_reward)
    
    return episode_rewards

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
root_directory = Path("data/citylearn_challenge_2023_phase_1")

if __name__ == "__main__":
    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_directory,
        random_seed=SEED,
        central_agent=True,
    )

    observation_space_dim = 49
    action_space_dim = 18 # set to 18 and turn on other actions
        
    # Initialize SAC Agent
    sac_agent = SACAgent(
        observation_space_dim=observation_space_dim, 
        action_space_dim=action_space_dim,
        hidden_dim=291,
        buffer_size=123593,
        batch_size=446,
        learning_rate=6e-4,
        gamma=0.9252841056268958,
        tau=0.09474412328105289,
        alpha=0.1537358441036939,
        action_space=env.action_space,
        exploration_timesteps = 0
    )
        
    # Train the agent
    episode_rewards = train_sac_agent(env, sac_agent, episodes=10)

    # Plot the episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, 
             label='Episode Rewards', color='green', marker='o')
    
    # Add rolling average
    window_size = 5
    if len(episode_rewards) >= window_size:
        rolling_mean = pd.Series(episode_rewards).rolling(window=window_size).mean()

    plt.title("Centralized SAC Episode Rewards", fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Total Episode Reward", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)

    plt.savefig("episode_rewards.png", dpi=300, bbox_inches='tight')
    plt.show()