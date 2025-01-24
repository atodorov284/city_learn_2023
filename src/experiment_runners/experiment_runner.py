import random
from pathlib import Path

import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv
from typing import List

import time

from agents.wrappers.centralized import CentralizedSACAgent
from agents.wrappers.decentralized import DecentralizedSACAgent
from agents.wrappers.maml import MAMLSACAgent
from agents.wrappers.citylearn_wrapper import CityLearnWrapperAgent

from concurrent.futures import ThreadPoolExecutor, as_completed


from utils.plotting import plot_single_agent, plot_all_agents


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def set_seed(seed: int = 0) -> None:
    """Set a seed used in the simulation."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_environment(
    central_agent: bool = True,
    SEED: int = 0,
    path: str = "data/citylearn_challenge_2023_phase_1",
) -> CityLearnEnv:
    """Create a CityLearn environment using the provided path, seed, and agent type."""
    set_seed(SEED)
    root_directory = Path(path)

    if not root_directory.exists():
        raise ValueError(f"Path {path} does not exist")

    schema_path = root_directory / "schema.json"
    env = CityLearnEnv(
        schema=schema_path,
        root_directory=root_directory,
        random_seed=SEED,
        central_agent=central_agent,
    )

    return env


def train_citylearn_agent(
    env: CityLearnEnv,
    agent: CityLearnWrapperAgent,
    episodes: int = 100,
    experiment_id: str = None,
    eval_mode: bool = False,
    agent_type: str = "centralized",
) -> List[float]:
    start_time = time.time()

    agent.reset()

    day_rewards = {
        "total_reward": [],
        "mean_reward": [],
        "sem_reward": []
    }
    
    episode_rewards = {
        "total_reward": [],
        "mean_reward": [],
        "sem_reward": []
    }
    
    for episode in range(episodes):
        observation = env.reset()
        
        raw_rewards_daily = []
        raw_rewards_episode = []

        while not env.done:
            actions = agent.select_action(observation)

            next_observation, reward, info, done = env.step(actions)
            
            # Calculate the step reward
            step_reward = np.sum(reward)

            raw_rewards_episode.append(step_reward)
            raw_rewards_daily.append(step_reward)

            if agent.total_steps % 24 == 0:
                day_rewards["total_reward"].append(np.sum(raw_rewards_daily))
                day_rewards["mean_reward"].append(np.mean(raw_rewards_daily))
                day_rewards["sem_reward"].append(np.std(raw_rewards_daily))

            agent.add_to_buffer(observation, actions, reward, next_observation, done)

            agent.train(eval_mode=eval_mode)

            observation = next_observation

        episode_rewards["total_reward"].append(np.sum(raw_rewards_episode))
        episode_rewards["mean_reward"].append(np.mean(raw_rewards_episode))
        episode_rewards["sem_reward"].append(np.std(raw_rewards_episode))
        
        print(f"Agent: {agent_type}, Episode: {episode+1}/{episodes}, Eval_mode: {eval_mode}.")

        plot_single_agent(
            day_rewards,
            agent_type=agent_type,
            plot_folder="plots/",
            experiment_id=f"{experiment_id}_daily",
        )
        
        plot_single_agent(
            episode_rewards,
            agent_type=agent_type,
            plot_folder="plots/",
            experiment_id=f"{experiment_id}_episode",
        )
        

    print(f"Wall Time for {agent_type}: {time.time() - start_time:.2f} seconds, Eval_mode: {eval_mode}")
    return day_rewards, episode_rewards


def setup_single_agent(
    agent_type: str = "centralized",
    seed: int = 0,
    hyperparameters_dict: dict = {},
    episodes: int = 100,
    experiment_id: str = None,
) -> List[float]:
    set_seed(seed)

    centralized = True if agent_type == "centralized" else False

    training_env = create_environment(
        central_agent=centralized,
        SEED=seed,
        path="data/citylearn_challenge_2023_phase_1",
    )

    eval_env = create_environment(
        central_agent=centralized,
        SEED=seed,
        path="data/citylearn_challenge_2023_phase_3_1",
    )

    num_buildings = len(training_env.buildings)
    observation_space_dim = training_env.observation_space[0].shape[0]
    action_space_dim = training_env.action_space[0].shape[0]

    hidden_dim = hyperparameters_dict.get("hidden_dim", 256)
    buffer_size = hyperparameters_dict.get("buffer_size", 100000)
    learning_rate = hyperparameters_dict.get("learning_rate", 3e-4)
    gamma = hyperparameters_dict.get("gamma", 0.99)
    tau = hyperparameters_dict.get("tau", 0.01)
    alpha = hyperparameters_dict.get("alpha", 0.05)
    batch_size = hyperparameters_dict.get("batch_size", 256)
    k_shots = hyperparameters_dict.get("k_shots", 3)
    
    print("-" * 50)
    print(f"Experiment ID: {experiment_id}")
    print(f"Episodes (months): {episodes}")
    print(f"Agent type: {agent_type}")
    print(f"Number of buildings: {num_buildings}")
    print(f"Observation space dimension: {observation_space_dim}")
    print(f"Action space dimension: {action_space_dim}")
    print("Agent hyperparameters:")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Buffer size: {buffer_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Tau: {tau}")
    print(f"Alpha: {alpha}")
    print(f"Batch size: {batch_size}")
    print(f"K-shots: {k_shots} (only used for MAML)")
    print("-" * 50)

    if agent_type == "centralized":
        agent = CentralizedSACAgent(
            env=training_env,
            central_agent=centralized,
            hidden_dim=hidden_dim,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            batch_size=batch_size,
            action_space=training_env.action_space,
            observation_space_dim=observation_space_dim,
            action_space_dim=action_space_dim,
            num_buildings=num_buildings,
        )

    elif agent_type == "decentralized":
        agent = DecentralizedSACAgent(
            env=training_env,
            central_agent=centralized,
            hidden_dim=hidden_dim,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            batch_size=batch_size,
            action_space=training_env.action_space,
            observation_space_dim=observation_space_dim,
            action_space_dim=action_space_dim,
            num_buildings=num_buildings,
        )

    elif agent_type == "maml":
        agent = MAMLSACAgent(
            env=training_env,
            central_agent=centralized,
            hidden_dim=hidden_dim,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            batch_size=batch_size,
            action_space=training_env.action_space,
            observation_space_dim=observation_space_dim,
            action_space_dim=action_space_dim,
            num_buildings=num_buildings,
            k_shots=k_shots,
        )

    daily_rewards_training, episode_rewards_training = train_citylearn_agent(
        agent=agent,
        env=training_env,
        episodes=episodes,
        experiment_id=f"{experiment_id}_train",
        agent_type=agent_type,
    )

    daily_rewards_eval, episode_rewards_eval = train_citylearn_agent(
        agent=agent,
        env=eval_env,
        episodes=1, # Only one episode for evaluation
        experiment_id=f"{experiment_id}_eval",
        agent_type=agent_type,
        eval_mode=True,
    )

    return daily_rewards_training, daily_rewards_eval, episode_rewards_training, episode_rewards_eval


def setup_all_agents(
    seed: int = 0,
    episodes: int = 100,
    hyperparameters_dict: dict = {},
    experiment_id: str = None,
) -> None:
    agent_types = ["centralized", "decentralized", "maml"]
    training_results_daily = {}
    training_results_episode = {}
    eval_results_daily = {}
    eval_results_episode = {}

    def setup_agent(agent_type):
        return setup_single_agent(
            agent_type=agent_type,
            seed=seed,
            episodes=episodes,
            hyperparameters_dict=hyperparameters_dict,
            experiment_id=experiment_id,
        )

    with ThreadPoolExecutor(max_workers=len(agent_types)) as executor:
        future_to_agent_type = {
            executor.submit(setup_agent, agent): agent for agent in agent_types
        }

        for future in as_completed(future_to_agent_type):
            agent_type = future_to_agent_type[future]
            try:
                daily_rewards_train, daily_rewards_eval, episode_rewards_train, episode_rewards_eval = future.result()
                training_results_daily[agent_type] = daily_rewards_train
                eval_results_daily[agent_type] = daily_rewards_eval
                training_results_episode[agent_type] = episode_rewards_train
                eval_results_episode[agent_type] = episode_rewards_eval
            except Exception as e:
                print(f"Error processing agent {agent_type}: {e}")

    plot_all_agents(
        training_results_daily, plot_folder="plots/", experiment_id=f"{experiment_id}_train_daily"
    )
    
    plot_all_agents(
        training_results_episode, plot_folder="plots/", experiment_id=f"{experiment_id}_train_episode"
    )
    
    plot_all_agents(
        eval_results_daily, plot_folder="plots/", experiment_id=f"{experiment_id}_eval_daily"
    )
    
    print(f"Evaluation results episode: {eval_results_episode}")
