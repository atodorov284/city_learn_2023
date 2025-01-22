from agents.base_agent import Agent


class RandomAgent(Agent):
    def __init__(self, observation_space: list, action_space: list) -> None:
        """Initialize the random agent"""
        super().__init__(observation_space, action_space)

    def select_action(self, observation: list) -> list:
        """Select a random action"""
        return [list(s.sample()) for s in self._action_space]
