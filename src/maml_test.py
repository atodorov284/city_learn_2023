import random
from pathlib import Path

import numpy as np
import torch
from typing import List

from agents.sac import SACAgent
from citylearn.citylearn import CityLearnEnv
from agents.autoencoder import SACEncoder

from agents.maml import MAMLAgent


SEED = 0

CENTRALIZED_OBSERVATION_DIMENSION = 77
DECENTRALIZED_OBSERVATION_DIMENSION = 39

CENTRALIZED_ACTION_DIMENSION = 18
DECENTRALIZED_ACTION_DIMENSION = 6

ENCODER_HIDDEN_DIMENSION = 65
ENCODER_OUTPUT_DIMENSION = 50  # should be smaller than the observation dimension

TRAINING_EPISODES = 100

def train_maml_agent_citylearn(env: CityLearnEnv, maml_agent: MAMLAgent, episodes: int = 100):
    """
    Train MAML agents for CityLearn with a decentralized approach for each building.
    Args:
        env: CityLearn environment.
        maml_agent: MAMLAgent instance.
        episodes: Number of episodes for training.
    """
    for episode in range(episodes):
        observation = env.reset()
        
        while not env.done:
            maml_agent.inner_adaptation(observation, env)
            meta_losses = maml_agent.compute_meta_losses(observation, env)
            maml_agent.outer_update(meta_losses)
            
        print(f"Episode {episode+1}/{episodes} complete.")
        
def set_seed(seed: int = 0) -> None:
    """Set a seed used in the simulation."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def create_environment(
    central_agent: bool = True,
    SEED=0,
    path: str = "data/citylearn_challenge_2023_phase_1",
):
    """
    Creates the CityLearn environment.
    Args:

    """

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

def create_agents(
    env: CityLearnEnv,
    central_agent: bool = False,
    hidden_dim: int = 64,
    buffer_size: int = 1000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.01,
    alpha: float = 0.05,
    batch_size: int = 256,
    exploration_timesteps: int = 0,
) -> List[SACAgent]:
    """
    Creates the agents with the given specification.
    Args:
        env: The CityLearn environment.
        central_agent: Whether to create a central agent.
        hidden_dim: The hidden dimension of the network for the SAC agent.
        buffer_size: The replay buffer size.
        learning_rate: The learning rate for the SAC agent.
        gamma: The discount factor for the SAC agent.
        tau: The target network update rate for the SAC agent.
        alpha: The temperature parameter for the SAC agent.
        batch_size: The batch size used for the SAC agent.
        exploration_timesteps: The number of exploration timesteps for the SAC agent.
    Returns:
        Agent or a list of agents
    """
    if central_agent:
        observation_space_dim = CENTRALIZED_OBSERVATION_DIMENSION 
        action_space_dim = CENTRALIZED_ACTION_DIMENSION
        building_number = 1
    else:
        observation_space_dim = DECENTRALIZED_OBSERVATION_DIMENSION
        action_space_dim = DECENTRALIZED_ACTION_DIMENSION
        building_number = 3

    agents = []
    for _ in range(1):
        agents.append(
            SACAgent(
                observation_space_dim=observation_space_dim,
                action_space_dim=action_space_dim,
                hidden_dim=hidden_dim,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                action_space=env.action_space,
                exploration_timesteps=exploration_timesteps,
            )
        )
    return agents

if __name__ == "__main__":
    env = create_environment(central_agent=False, SEED=SEED, path="data/citylearn_challenge_2023_phase_1")
    sac_agents = create_agents(env, central_agent=False)
    maml_agent = MAMLAgent(sac_agents[0], inner_lr=0.01, outer_lr=0.001, n_inner_steps=1, meta_batch_size=3)
    train_maml_agent_citylearn(env, maml_agent, episodes=TRAINING_EPISODES)