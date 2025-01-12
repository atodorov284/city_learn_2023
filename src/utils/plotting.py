import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_single_agent(
    rewards: list[float], agent_type: str = "centralized", plot_folder: str = "plots/", window_size: int = 15
) -> None:
    """
    Plots the rewards for different agent types.

    Args:
        rewards: List of rewards to plot
        agent_type: Type of agent ("centralized", "decentralized", "maml")
    """
    valid_types = ["centralized", "decentralized", "maml"]
    if agent_type.lower() not in valid_types:
        raise ValueError(f"agent_type must be one of {valid_types}")

    rewards = np.array(rewards)
    steps = range(1, len(rewards) + 1)

    plt.figure(figsize=(12, 6))

    # Plot raw rewards
    plt.plot(steps, rewards, alpha=0.3, color="blue", label="Raw Rewards")

    # Add rolling average
    window_size = 15
    rolling_mean = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
    plt.plot(
        steps,
        rolling_mean,
        color="red",
        linewidth=2,
        label=f"{window_size}-step Moving Average",
    )

    title_prefix = {
        "centralized": "Centralized",
        "decentralized": "Decentralized",
        "maml": "MAML",
    }
    plt.title(
        f"{title_prefix[agent_type.lower()]} SAC Agent Rewards Over Time",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

    plt.tight_layout()
    save_path = plot_folder + f"step_rewards_{agent_type.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")