import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_single_agent(
    rewards: list[float],
    agent_type: str = "centralized",
    plot_folder: str = "plots/",
    window_size: int = 15,
    experiment_id: str = None,
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
    save_path = plot_folder + f"step_rewards_{agent_type.lower()}_{experiment_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_all_agents(
    rewards_dict: dict[str, list[float]],
    plot_folder: str = "plots/",
    experiment_id: str = None,
) -> None:
    """
    Plots the rewards for different agent types on the same plot.

    Args:
        rewards_dict: Dictionary containing rewards for each agent type
                     Format: {'centralized': [...], 'decentralized': [...], 'maml': [...]}
        plot_folder: Folder to save the plots in
    """
    plt.figure(figsize=(12, 6))

    colors = {"centralized": "blue", "decentralized": "red", "maml": "green"}

    # Plot each agent type
    for agent_type, rewards in rewards_dict.items():
        rewards = np.array(rewards)
        steps = range(1, len(rewards) + 1)

        # Plot raw rewards with low alpha
        # plt.plot(steps, rewards, alpha=0.2, color=colors[agent_type])

        # Add rolling average
        window_size = 24
        # Calculate rolling mean and SEM
        rolling_data = pd.Series(rewards).rolling(window=window_size, min_periods=1)
        rolling_mean = rolling_data.mean()
        rolling_sem = rolling_data.std() / np.sqrt(window_size)
        plt.plot(
            steps,
            rolling_mean,
            color=colors[agent_type],
            linewidth=2,
            label=f"{agent_type.capitalize()}",
        )
        # Add SEM bands
        plt.fill_between(
            steps,
            rolling_mean - rolling_sem,
            rolling_mean + rolling_sem,
            color=colors[agent_type],
            alpha=0.1,
        )

    plt.title(
        "Agent Performance Comparison",
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
    save_path = plot_folder + f"comparison_rewards_{experiment_id}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
