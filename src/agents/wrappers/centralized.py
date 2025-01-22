from agents.wrappers.citylearn_wrapper import CityLearnWrapperAgent

from citylearn.citylearn import CityLearnEnv

import numpy as np


class CentralizedSACAgent(CityLearnWrapperAgent):
    def __init__(
        self,
        env: CityLearnEnv,
        central_agent: bool = True,
        hidden_dim: int = 256,
        buffer_size: int = 100000,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        alpha: float = 0.05,
        batch_size: int = 256,
        action_space: list = None,
        observation_space_dim: int = 0,
        action_space_dim: int = 0,
        num_buildings: int = 0,
    ) -> None:
        super().__init__(
            env,
            central_agent,
            hidden_dim,
            buffer_size,
            learning_rate,
            gamma,
            tau,
            alpha,
            batch_size,
            action_space,
            observation_space_dim,
            action_space_dim,
            num_buildings,
        )
        self.agent = self.agents[0]

    def select_action(self, observation):
        flat_observation = (
            np.concatenate(observation)
            if isinstance(observation, list)
            else observation
        )
        actions = [self.agent.select_action(flat_observation).tolist()]
        self.agent.total_steps += 1
        return actions

    @property
    def total_steps(self):
        return self.agent.total_steps

    def add_to_buffer(
        self, observation, actions, reward, next_observation, done
    ) -> None:
        flat_observation = (
            np.concatenate(observation)
            if isinstance(observation, list)
            else observation
        )

        flat_next_observation = (
            np.concatenate(next_observation)
            if isinstance(next_observation, list)
            else next_observation
        )

        self.agent.replay_buffer.push(
            flat_observation,
            actions,
            np.sum(reward),
            flat_next_observation,
            len(done),
        )

    def train(self, eval_mode=False) -> None:
        # Centralized doesn't care about eval_mode, this is only for MAML but required so the abstraction works
        self.agent.train()
