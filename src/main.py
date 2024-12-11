import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv

from agents.base_agent import Agent
from agents.random_agent import RandomAgent


def centralized_interact_with_env(env: CityLearnEnv, agent: Agent = RandomAgent, episodes: int = 100) -> None:
    """Interact with environment using agent"""
    for episode in range(episodes):
        observation = env.reset()
        while not env.done:
            action = agent.select_action(observation)
            observation, reward, info, done = env.step(action)
        
    

if __name__ == "__main__":
    SEED = 4242
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    root_directory = Path("data/citylearn_challenge_2023_phase_2_local_evaluation")
    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path, root_directory=root_directory, random_seed=SEED, central_agent=True
    )
    random_agent = RandomAgent(env.observation_space, env.action_space)
    
    centralized_interact_with_env(env, random_agent)
    



