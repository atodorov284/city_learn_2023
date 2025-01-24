from abc import ABC, abstractmethod
from agents.base_models.sac import SACAgent

from citylearn.citylearn import CityLearnEnv

from typing import List


class CityLearnWrapperAgent(ABC):
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
        self.env = env
        self.central_agent = central_agent
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.action_space = action_space
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.num_buildings = num_buildings
        self.agents = []
        self.create_agents()

    @abstractmethod
    def select_action(self, observation: List[List[float]]) -> List[List[float]]:
        """
        Selects an action for each entity in the CityLearn environment.
        """

        pass

    @abstractmethod
    def add_to_buffer(self, state: List[List[float]], action: List[List[float]], reward: List[float], next_state: List[List[float]], done: int):
        """
        Adds a transition to the replay buffer of each building agent.
        """
        pass

    @abstractmethod
    def train(self, eval_mode: bool=False):
        """
        Trains the agent using the specified mode.
        """
        pass

    def reset(self) -> None:
        """
        Resets the agents' total steps to 0.
        """
        for agent in self.agents:
            agent.total_steps = 0

    def create_agents(self) -> None:
        """
        Creates the agents, either a single central agent or a list of agents
        one for each building.
        """

        if self.central_agent:
            self.num_agents = 1
        else:
            self.num_agents = self.num_buildings

        for _ in range(self.num_agents):
            self.agents.append(
                SACAgent(
                    observation_space_dim=self.observation_space_dim,
                    action_space_dim=self.action_space_dim,
                    hidden_dim=self.hidden_dim,
                    buffer_size=self.buffer_size,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    tau=self.tau,
                    alpha=self.alpha,
                    batch_size=self.batch_size,
                    action_space=self.action_space,
                )
            )
