from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, observation_space: list, action_space: list) -> None:
        """Initialize the base agent agent"""
        self._observation_space = observation_space
        self._action_space = action_space

    @abstractmethod
    def select_action(self, observation: list) -> list:
        """Select an action using the base agent"""
        pass
