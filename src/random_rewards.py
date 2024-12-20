from citylearn.agents.base import Agent as RandomAgent
from citylearn.citylearn import CityLearnEnv
from pathlib import Path

root_directory = Path("data/citylearn_challenge_2023_phase_1")
schema_path = root_directory / "schema.json"
env = CityLearnEnv(schema=schema_path, root_directory=root_directory, central_agent=True)

import numpy as np
random_model = RandomAgent(env)

rewards = []
for i in range (100):
    observations = env.reset()

    while not env.done:
        actions = random_model.predict(observations)
        observations, reward, info, done = env.step(actions)
        rewards.append(reward)

    print(min(rewards), max(rewards), np.mean(rewards))