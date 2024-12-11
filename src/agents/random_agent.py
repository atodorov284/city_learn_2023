from agents.base_agent import Agent


class RandomAgent(Agent):
    def __init__(self, observation_space: list, action_space: list) -> None:
        """Initialize agent"""
        self._observation_space = observation_space
        self._action_space = action_space
    
    def select_action(self, observation: list) -> list:
        """Select a random action"""
        return [list(s.sample()) for s in self._action_space]