from agents.wrappers.citylearn_wrapper import CityLearnWrapperAgent

from citylearn.citylearn import CityLearnEnv

import numpy as np


class DecentralizedSACAgent(CityLearnWrapperAgent):
    def __init__(
        self,
        env: CityLearnEnv,
        central_agent: bool = False,
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

    def select_action(self, observation: list) -> list:
        actions = [0 for _ in range(self.num_buildings)]

        for i in range(self.num_buildings):
            actions[i] = self.agents[i].select_action(np.array(observation[i])).tolist()

        for agent in self.agents:
            agent.total_steps += 1

        return actions

    @property
    def total_steps(self):
        return self.agents[0].total_steps

    def add_to_buffer(
        self, observation, actions, reward, next_observation, done
    ) -> None:
        for i in range(self.num_buildings):
            self.agents[i].replay_buffer.push(
                observation[i],
                actions[i],
                reward[i],
                next_observation[i],
                len(done),
            )

    def train(self, eval_mode=False) -> None:
        for agent in self.agents:
            agent.train()
