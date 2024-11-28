from pathlib import Path

from citylearn.citylearn import CityLearnEnv
from agents.random_agent import RandomAgent
# from citylearn.agents.base import Agent as RandomAgent
from utils import format_evaluation
import numpy as np
import random
import torch


if __name__ == "__main__":
    SEED = 4242
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    root_directory = Path("data/citylearn_challenge_2023_phase_1")
    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(schema=schema_path, root_directory=root_directory, random_seed=SEED)
    model = RandomAgent(env)
    
    observations = env.reset()

    while not env.done:
        actions = model.predict(observations)
        observations, reward, info, done = env.step(actions)

    random_agent_data = model.env.evaluate_citylearn_challenge()
    
    print(format_evaluation(random_agent_data))

