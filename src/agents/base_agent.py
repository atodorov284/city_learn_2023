"""
This code has been adapted from the citylearn library version 2.1b9
https://www.citylearn.net/citylearn_challenge/2023.html
"""

from typing import Any, List, Mapping

import numpy as np
from citylearn.base import Environment
from citylearn.citylearn import CityLearnEnv
from gym import spaces


class BaseAgent(Environment):
    """Base agent class.

    Parameters
    ----------
    env : CityLearnEnv
        CityLearn environment.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments used to initialize super class.
    """

    def __init__(self, env: CityLearnEnv, **kwargs: Any) -> None:
        """
        Initialize the environment and set up the agent.
        """
        self.env = env
        self.observation_names = self.env.observation_names
        self.action_names = self.env.action_names
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.episode_time_steps = self.env.time_steps
        self.building_metadata = self.env.get_metadata()["buildings"]
        super().__init__(
            seconds_per_time_step=self.env.seconds_per_time_step,
            random_seed=self.env.random_seed,
            episode_tracker=env.episode_tracker,
        )
        self.reset()

    @property
    def observation_names(self) -> List[List[str]]:
        """Names of active observations that can be used to map observation values."""

        return self.__observation_names

    @property
    def action_names(self) -> List[List[str]]:
        """Names of active actions that can be used to map action values."""

        return self.__action_names

    @property
    def observation_space(self) -> List[spaces.Box]:
        """Format of valid observations."""

        return self.__observation_space

    @property
    def action_space(self) -> List[spaces.Box]:
        """Format of valid actions."""
        return self.__action_space

    @property
    def episode_time_steps(self) -> int:
        """
        Return number of time steps in one episode
        """
        return self.__episode_time_steps

    @property
    def building_metadata(self) -> List[Mapping[str, Any]]:
        """Building(s) metadata."""
        return self.__building_metadata

    @property
    def action_dimension(self) -> List[int]:
        """Number of returned actions."""
        return [s.shape[0] for s in self.action_space]

    @property
    def actions(self) -> List[List[List[Any]]]:
        """Action history/time series."""
        return self.__actions

    @observation_names.setter
    def observation_names(self, observation_names: List[List[str]]) -> None:
        """
        Set observation names
        """
        self.__observation_names = observation_names

    @action_names.setter
    def action_names(self, action_names: List[List[str]]) -> None:
        """
        Set action names
        """
        self.__action_names = action_names

    @observation_space.setter
    def observation_space(self, observation_space: List[spaces.Box]) -> None:
        """
        Set observation space
        """
        self.__observation_space = observation_space

    @action_space.setter
    def action_space(self, action_space: List[spaces.Box]) -> None:
        """
        Set action space
        """
        self.__action_space = action_space

    @episode_time_steps.setter
    def episode_time_steps(self, episode_time_steps: int) -> None:
        """Number of time steps in one episode."""

        self.__episode_time_steps = episode_time_steps

    @building_metadata.setter
    def building_metadata(self, building_metadata: List[Mapping[str, Any]]) -> None:
        """
        Set building metadata
        """
        self.__building_metadata = building_metadata

    @actions.setter
    def actions(self, actions: List[List[Any]]) -> None:
        """
        Set actions for the current time step.
        """
        for i in range(len(self.action_space)):
            self.__actions[i][self.time_step] = actions[i]

    def learn(
        self,
        episodes: int = None,
        deterministic: bool = None,
        deterministic_finish: bool = None,
    ) -> None:
        """Train agent.

        Parameters
        ----------
        episodes: int, default: 1
            Number of training episode >= 1.
        deterministic: bool, default: False
            Indicator to take deterministic actions i.e. strictly exploit the learned policy.
        deterministic_finish: bool, default: False
            Indicator to take deterministic actions in the final episode.
        """

        episodes = 1 if episodes is None else episodes
        deterministic_finish = (
            False if deterministic_finish is None else deterministic_finish
        )
        deterministic = False if deterministic is None else deterministic

        for episode in range(episodes):
            deterministic = deterministic or (
                deterministic_finish and episode >= episodes - 1
            )
            observations = self.env.reset()
            self.episode_time_steps = self.episode_tracker.episode_time_steps
            done = False
            time_step = 0
            rewards_list = []

            while not done:
                actions = self.predict(observations, deterministic=deterministic)

                # apply actions to citylearn_env
                next_observations, rewards, done, _ = self.env.step(actions)
                rewards_list.append(rewards)

                # update
                if not deterministic:
                    self.update(
                        observations, actions, rewards, next_observations, done=done
                    )
                else:
                    pass

                observations = [o for o in next_observations]

                time_step += 1

            rewards = np.array(rewards_list, dtype="float")

    def predict(
        self, observations: List[List[float]], deterministic: bool = None
    ) -> List[List[float]]:
        """Provide actions for current time step.
        Overwrite this method to provide actions for the current time step.
        """

        pass

    def update(self, *args, **kwargs) -> None:
        """Update replay buffer and networks.

        Notes
        -----
        This implementation does nothing but is kept to keep the API for all agents similar during simulation.
        """

        pass

    def next_time_step(self) -> None:
        """
        Update time step.
        """
        super().next_time_step()

        for i in range(len(self.action_space)):
            self.__actions[i].append([])

    def reset(self) -> None:
        """
        Reset agent.
        """
        super().reset()
        self.__actions = [[[]] for _ in self.action_space]
