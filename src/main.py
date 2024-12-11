import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv

from agents.base_agent import Agent
from agents.random_agent import RandomAgent
from agents.sac import SACAgent


def train_sac_agent(
    env: CityLearnEnv, 
    agent: SACAgent, 
    episodes: int = 100, 
) -> None:
    """Train SAC agent in the environment"""
    total_reward = 0
    
    for episode in range(episodes):
        # Reset environment and get initial observation
        observation = env.reset()
        episode_reward = 0
        
        while not env.done:
            flat_observation = np.concatenate(observation) if isinstance(observation, list) else observation
            
            action = [agent.select_action(flat_observation).tolist()]
            
            print(action)
            
                    
            next_observation, reward, info, done = env.step(action)
                        
            flat_next_observation = np.concatenate(next_observation) if isinstance(next_observation, list) else next_observation
            
            
            episode_reward += np.sum(reward)
                        
            agent.replay_buffer.push(
                flat_observation, 
                action, 
                np.sum(reward), 
                flat_next_observation, 
                len(done)
            )
            
            agent.train()
            
            observation = next_observation
        
        total_reward += episode_reward
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward}")
    
    return agent


def centralized_interact_with_env(
    env: CityLearnEnv, agent: Agent = RandomAgent, episodes: int = 100
) -> None:
    """Interact with environment using agent"""
    for episode in range(episodes):
        observation = env.reset()
        while not env.done:
            action = agent.select_action(observation)
            print(action)
            observation, reward, info, done = env.step(action)


if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    root_directory = Path("data/citylearn_challenge_2023_phase_2_local_evaluation")
    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_directory,
        random_seed=SEED,
        central_agent=True,
    )
    random_agent = RandomAgent(env.observation_space, env.action_space)

    centralized_interact_with_env(env, random_agent)
    
    
    observation_space_dim = 52
    action_space_dim = 9
    
    
    # Initialize SAC Agent
    sac_agent = SACAgent(
        observation_space_dim=observation_space_dim, 
        action_space_dim=action_space_dim,
        hidden_dim=256,
        buffer_size=100000,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2
    )
    
    # Train the agent
    # train_sac_agent(env, sac_agent, episodes=100)
