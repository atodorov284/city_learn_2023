from abc import ABC, abstractmethod
from agents.base_models.sac import SACAgent

from citylearn.citylearn import CityLearnEnv


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
    def select_action(self, observation):
        pass

    @abstractmethod
    def add_to_buffer(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def train(self, eval_mode=False):
        pass

    def reset(self) -> None:
        for agent in self.agents:
            agent.total_steps = 0

    def create_agents(self) -> None:
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
