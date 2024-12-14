import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from citylearn.citylearn import CityLearnEnv
from matplotlib import pyplot as plt

from citylearn.agents.sac import SAC as SACAgent


if __name__ == "__main__":
    SEED = 0
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    # Make sure Schema is for decentralized setting
    root_directory = Path("data/citylearn_challenge_2023_phase_2_local_evaluation")
    schema_path = root_directory / "schema.json"
    centralized_env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_directory,
        random_seed=SEED,
        central_agent=True,
    )
    
    decentralized_observation_space_dim = 29
    decentralized_action_space_dim = 6 
    building_number = 3
    
    #################################################################
    
    agent = SACAgent(centralized_env)
    
    # Train the agent
    observations = centralized_env.reset()
    rewards = []

    while not centralized_env.done:
        actions = agent.predict(observations)
        observations, reward, info, done = centralized_env.step(actions)
        rewards.append(reward)

    # Convert rewards list to numpy array for easier manipulation
    rewards_array = np.array(rewards)  # rewards is your list of reward lists
    steps = range(1, len(rewards_array) + 1)

    # Calculate the sum of rewards at each step across all agents
    summed_rewards = np.sum(rewards_array, axis=1)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot raw rewards
    plt.plot(steps, summed_rewards, alpha=0.3, color='blue', label='Raw Rewards')

    # Add rolling average
    window_size = 100  # Adjust this value based on your needs
    rolling_mean = pd.Series(summed_rewards).rolling(window=window_size, min_periods=1).mean()
    plt.plot(steps, rolling_mean, color='red', linewidth=2, label=f'{window_size}-step Moving Average')

    # Customize the plot
    plt.title("SAC Agent Rewards Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # Optional: Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Optional: Use scientific notation for large numbers on x-axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig("step_rewards_premade_centralized.png", dpi=300, bbox_inches='tight')
    plt.show()