import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agents.base_agent import Agent
from utils.replay_buffer import ReplayBuffer

class SACEncoder(nn.Module):
    pass