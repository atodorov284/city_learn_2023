import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agents.base_agent import Agent
from utils.replay_buffer import ReplayBuffer

class SACEncoder(nn.Module):
    def __init__(
        self, observation_space_dim: int, output_dim: int, hidden_dim: int
    ) -> None:
        """Initialize the encoder network"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(SACEncoder, self).__init__()

        self.fc1 = nn.Linear(observation_space_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()
        
        self.to(self.device)

    def _initialize_weights(self):
        """
        Initialize the weights of the network layers using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Compute a compressed representation of a state

        Args:
            state (torch.Tensor): Input state

        Returns:
            Compressed state representation
        """
        q1 = F.relu(self.fc1(observation))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1