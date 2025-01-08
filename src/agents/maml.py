from typing import List
from agents.sac import SACAgent
from torch.optim import Adam
import torch

class MAMLAgent:
    def __init__(self, base_agents: List[SACAgent], inner_lr=0.01, outer_lr=0.001, n_inner_steps=1, meta_batch_size=3):
        """
        Initializes MAMLAgent for multiple agents in a multi-building setting.
        Args:
            base_agents: List of SACAgent instances, one for each building.
        """
        self.base_agents = base_agents  # One SAC agent per building
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.meta_batch_size = meta_batch_size
        self.outer_optimizers = [Adam(agent.actor.parameters(), lr=outer_lr) for agent in base_agents]
        self.inner_optimizers = [Adam(agent.actor.parameters(), lr=inner_lr) for agent in base_agents]

    def inner_adaptation(self, task_trajectories):
        """
        Inner-loop adaptation for each building using its trajectory.
        Args:
            task_trajectories: List of trajectories for each building.
        """
        for i, agent in enumerate(self.base_agents):
            for _ in range(self.n_inner_steps):
                losses = []
                for obs, action, reward, next_obs, log_prob in task_trajectories[i]:
                    loss = agent.compute_loss(obs, action, reward, next_obs, log_prob)
                    losses.append(loss)
                inner_loss = torch.stack(losses).mean()
                self.inner_optimizers[i].zero_grad()
                inner_loss.backward()
                self.inner_optimizers[i].step()

    def compute_meta_losses(self, task_trajectories):
        """
        Compute meta-losses for each building after inner adaptation.
        Args:
            task_trajectories: List of trajectories for each building.
        Returns:
            List of meta-losses for outer update.
        """
        meta_losses = []
        for i, agent in enumerate(self.base_agents):
            losses = []
            for obs, action, reward, next_obs, log_prob in task_trajectories[i]:
                loss = agent.compute_loss(obs, action, reward, next_obs, log_prob)
                losses.append(loss)
            meta_losses.append(torch.stack(losses).mean())
        return meta_losses

    def outer_update(self, meta_losses):
        """
        Outer-loop update using meta-losses for each building.
        Args:
            meta_losses: List of meta-losses from different buildings.
        """
        for i, loss in enumerate(meta_losses):
            self.outer_optimizers[i].zero_grad()
            loss.backward()
            self.outer_optimizers[i].step()
