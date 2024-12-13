import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.sac import SAC as SACAgent
from matplotlib import pyplot as plt

from agents.base_agent import Agent
from agents.random_agent import RandomAgent



def train_sac_agent(
    env: CityLearnEnv, 
    agent: SACAgent, 
    episodes: int = 100, 
) -> None:
    """Train SAC agent in the environment"""
    total_reward = 0
    
    reward_list = []
    
    for episode in range(episodes):
        # Reset environment and get initial observation
        observation = env.reset()
        print(len(observation[0])) # 3 arrays of seperate observations
        episode_reward = 0
        
        while not env.done: # terminated
            flat_observation = np.concatenate(observation) if isinstance(observation, list) else observation
            
            action = agent.predict(flat_observation)
            
            print("action:",action)
          
                    
            next_observation, reward, info, done = env.step(action)
            print("reward:",reward)
            
            reward_list.append(reward)
                        
            flat_next_observation = np.concatenate(next_observation) if isinstance(next_observation, list) else next_observation
            
            episode_reward += np.sum(reward)
                        
            
            
            observation = next_observation
            exit(0)
        
        total_reward += episode_reward
        print(total_reward)
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {episode_reward}")
    
    plt.plot(reward_list, '.k')
    plt.show()
    return agent


if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    root_directory = Path("data/citylearn_challenge_2023_phase_2_online_evaluation_2")
    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_directory,
        random_seed=SEED,
        central_agent=False,
    )
    
    sac_agent = SACAgent(env)
    
    # Train the agent
    train_sac_agent(env, sac_agent, episodes=1)
