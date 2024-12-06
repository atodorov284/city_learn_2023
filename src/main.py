import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv
from stable_baselines3.sac import SAC as Agent  # remove later
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper # same


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

    # EXTRA CODE FOR FRIDAY: REMOVE LATER -------------------------------------------
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)
    model = Agent('MlpPolicy', env)

    # train
    episodes = 2
    model.learn(total_timesteps=env.unwrapped.time_steps*episodes)

    # test
    observations, _ = env.reset()

    rewards = []
    while not env.unwrapped.terminated:
        actions, _ = model.predict(observations, deterministic=True)
        observations, reward, _, _, _ = env.step(actions)
        rewards.append(reward)

    # plot rewards
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.show()

    # evaluate
    kpis = env.unwrapped.evaluate()
    kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
    kpis = kpis.dropna(how='all')
    print(kpis)

    exit(0)
    # END OF EXTRA CODE --------------------------------------------------------------

    model = RandomAgent(env)
    model.learn(episodes=20)

    random_agent_data = model.evaluate_agent()

    print(random_agent_data)



