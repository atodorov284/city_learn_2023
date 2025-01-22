from agents.wrappers.citylearn_wrapper import CityLearnWrapperAgent

from citylearn.citylearn import CityLearnEnv

import numpy as np

from copy import deepcopy

from utils.replay_buffer import ReplayBuffer


import torch


class MAMLSACAgent(CityLearnWrapperAgent):
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
        k_shots: int = 3,
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

        self.k_shots = k_shots

        self.base_agent = self.agents[0]

        self.actor_optimizer = torch.optim.Adam(
            self.base_agent.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.base_agent.critic.parameters(), lr=learning_rate
        )

        self._copy_agents()

        self._initialize_custom_buffers(buffer_size)

    def _copy_agents(self) -> None:
        self.copied_agents = [
            deepcopy(self.base_agent) for _ in range(self.num_buildings)
        ]

    def _outer_loop(self) -> None:
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        for copied_agent in self.copied_agents:
            for param, copied_param in zip(
                self.base_agent.actor.parameters(), copied_agent.actor.parameters()
            ):
                if copied_param.grad is not None:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += copied_param.grad / self.num_buildings

            for param, copied_param in zip(
                self.base_agent.critic.parameters(), copied_agent.critic.parameters()
            ):
                if copied_param.grad is not None:
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    param.grad += copied_param.grad / self.num_buildings

            torch.nn.utils.clip_grad_norm_(
                self.base_agent.actor.parameters(), max_norm=1.0
            )
            torch.nn.utils.clip_grad_norm_(
                self.base_agent.critic.parameters(), max_norm=1.0
            )

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self._copy_agents()

    def select_action(self, observation):
        actions = [0 for _ in range(self.num_buildings)]

        for i in range(self.num_buildings):
            actions[i] = (
                self.copied_agents[i].select_action(np.array(observation[i])).tolist()
            )

        self.base_agent.total_steps += 1

        return actions

    @property
    def total_steps(self):
        return self.base_agent.total_steps

    def add_to_buffer(
        self, observation, actions, reward, next_observation, done
    ) -> None:
        for i, building_buffer in enumerate(self.building_buffers):
            building_buffer.push(
                observation[i],
                actions[i],
                reward[i],
                next_observation[i],
                len(done),
            )

    def _initialize_custom_buffers(self, capacity: int) -> None:
        self.building_buffers = [
            ReplayBuffer(capacity=capacity) for _ in range(self.num_buildings)
        ]

    def train(self, eval_mode=False) -> None:
        if self.base_agent.total_steps % self.k_shots == 0 and not eval_mode:
            self._outer_loop()

        for i in range(self.num_buildings):
            self.copied_agents[i].train(custom_buffer=self.building_buffers[i])
