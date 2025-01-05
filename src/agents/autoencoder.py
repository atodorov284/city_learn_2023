import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from typing import List

from agents.base_agent import Agent
from utils.replay_buffer import ReplayBuffer

class SACEncoder(nn.Module):
    def __init__(
        self, observation_space_dim: int, output_dim: int, hidden_dim: int, use_hidden_layers: bool = True
    ) -> None:
        """Initialize the encoder network"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(SACEncoder, self).__init__()

        #encoder
        if use_hidden_layers:
            self.fc1 = nn.Linear(observation_space_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc1 = nn.Linear(observation_space_dim, output_dim)
        
        self.use_hidden_layers = use_hidden_layers

        #decoder
        self.dc1 = nn.Linear(output_dim, hidden_dim)
        self.dc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dc3 = nn.Linear(hidden_dim, observation_space_dim)

        self._initialize_weights()

        self.to(self.device)

    def _initialize_weights(self):
        """
        Initialize the weights of the network layers using Xavier initialization.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.use_hidden_layers:
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        nn.init.xavier_uniform_(self.dc1.weight)
        nn.init.xavier_uniform_(self.dc2.weight)
        nn.init.xavier_uniform_(self.dc3.weight)



    def forward(self, observation: List) -> np.ndarray:
        """
        Forward pass of the autoencoder.
        """
        # change obs to tensor 
        observation = np.array(observation)
        observation = torch.from_numpy(observation).float().to(self.device)

        #encode
        q1 = F.relu(self.fc1(observation))
        if self.use_hidden_layers:
            q1 = F.relu(self.fc2(q1))
            q1 = self.fc3(q1)

        enc_repr = q1

        #decode
        q1 = F.relu(self.dc1(enc_repr))
        q1 = F.relu(self.dc2(q1))
        q1 = self.dc3(q1)

        return q1  

    def encode(self, observation: List) -> np.ndarray:  
        """
        Use the encode-only part after training.
        OR use this as a random encoder.
        """ 
        # change obs to tensor 
        observation = np.array(observation)
        observation = torch.from_numpy(observation).float().to(self.device)

        q1 = F.relu(self.fc1(observation))
        if self.use_hidden_layers:
            q1 = F.relu(self.fc2(q1))
            q1 = self.fc3(q1)

        return list(q1.detach().cpu().numpy())
