from citylearn.agents.base import Agent as RandomAgent
from citylearn.citylearn import CityLearnEnv
import matplotlib.pyplot as plt
from pathlib import Path

root_directory = Path("data/citylearn_challenge_2023_phase_1")
schema_path = root_directory / "schema.json"

CENTRAL = True

env = CityLearnEnv(
    schema=schema_path, root_directory=root_directory, central_agent=CENTRAL
)

import numpy as np

random_model = RandomAgent(env)

# MAKE FOR BOTH CENTRALIZED AND DECENTRALIZED TO COMPARE

low_reward_dict = {}
# round all rewards to 3 decimals to make them more readable
for i in range(20):
    observations = env.reset()
    rewards = []
    while not env.done:
        actions = random_model.predict(observations)
        observations, reward, info, done = env.step(actions)
        rewards.append(round(np.sum(reward), 3))
    print(f"episode {i} Total reward: {np.sum(rewards)}")
    # print(min(rewards), max(rewards), np.mean(rewards))

q25 = np.quantile(rewards, 0.25)  # 25th percentile (1st quartile)
q50 = np.quantile(rewards, 0.50)  # 50th percentile (median)
q75 = np.quantile(rewards, 0.75)  # 75th percentile (3rd quartile)

print(f"25th percentile: {q25}")
print(f"50th percentile: {q50}")
print(f"75th percentile: {q75}")


# plot and save rewards as a histogram with a title "Random Agent REWARD-TYPE Rewards"
plt.hist(rewards, bins=100)
plt.title("Random Agent Combined Scaled Rewards")
plt.xlabel("Reward")
plt.xlim(-100, 0)  # ADD ONLY TO SCALED VERSIONS OF REWARD !!!!!!!!!!!!!!!!!!!!
plt.ylabel("Frequency")
if CENTRAL:
    plt.savefig("random_rewards_central.png")
else:
    plt.savefig("random_rewards_decentral.png")

print(min(rewards), max(rewards), np.mean(rewards))
