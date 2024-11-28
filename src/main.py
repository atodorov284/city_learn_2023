from citylearn.agents.base import Agent
from citylearn.citylearn import CityLearnEnv
from pathlib import Path
import pandas as pd


root_directory = Path('data/citylearn_challenge_2023_phase_1')
schema_path = root_directory / 'schema.json'
env = CityLearnEnv(schema=schema_path, root_directory=root_directory)
model = Agent(env)

observations = env.reset()

while not env.done:
    actions = model.predict(observations)
    observations, reward, info, done = env.step(actions)
    
print(pd.DataFrame(model.env.evaluate_citylearn_challenge()))