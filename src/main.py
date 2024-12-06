import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.agents.sac import SAC

from agents.random_agent import RandomAgent

if __name__ == "__main__":
    SEED = 4242
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    root_directory = Path("data/citylearn_challenge_2023_phase_2_local_evaluation")
    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path, root_directory=root_directory, random_seed=SEED
    )
    #model = RandomAgent(env)
    model = SAC(env)

    model.learn(episodes=20)

    random_agent_data = model.evaluate_agent()

    print(random_agent_data)
