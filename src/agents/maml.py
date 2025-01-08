from typing import List
from agents.sac import SACAgent
from torch.optim import Adam
import torch
import numpy as np
from copy import deepcopy

class MAMLAgent:
    def __init__(self, agent: SACAgent, inner_lr=0.01, outer_lr=0.001, n_inner_steps=1, meta_batch_size=3):
        """
        Initializes MAMLAgent for multiple agents in a multi-building setting.
        Args:
            base_agents: List of SACAgent instances, one for each building.
        """
        self.base_agents = agent  # One SAC agent per building
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.meta_batch_size = meta_batch_size
        self.outer_optimizer = Adam(agent.actor.parameters(), lr=outer_lr)

    def inner_adaptation(self, current_state, env, update=True):
        """
        Inner-loop adaptation for each building.
        """
        current_state = torch.tensor(current_state, dtype=torch.float32)
        inner_optimizers = []
        for _ in range (3):
            copied_agent = deepcopy(self.base_agents)
            inner_optimizers.append(Adam(copied_agent.actor.parameters(), lr=self.inner_lr))
        
        actions_with_log_probs = []
        for i in range(3):
            actions_with_log_probs.append(self.base_agents.actor.sample_action(current_state[i].unsqueeze(0))) #change this potentially
        
        actions = [action[0][0].squeeze(0).detach() for action in actions_with_log_probs]
        log_probs = [action[1] for action in actions_with_log_probs]
        print(np.array(actions).shape)
        print(actions)
        if env.done:
            print("gg")
            exit(0)
        next_observations, rewards, _, done = env.step(actions)
        
        rewards = np.array(rewards)
        current_state = np.array(current_state)
        actions = np.array(actions)
        next_observations = np.array(next_observations)
        log_probs = np.array(log_probs)
        done = np.array(done)

        losses = []
        for i, agent in enumerate(self.base_agents):    
            inner_loss = agent.compute_loss(rewards[i], current_state[i], actions[i], next_observations[i], log_probs[i], done, self.n_inner_steps)

            if update:
                self.inner_optimizers[i].zero_grad()
                inner_loss.backward()
                self.inner_optimizers[i].step()
            else:
                losses.append(inner_loss)

        if not update:
            return losses


    def compute_meta_losses(self,  current_state, env):
        """
        Compute meta-losses for each building after inner adaptation.
        Args:
            task_trajectories: List of trajectories for each building.
        Returns:
            List of meta-losses for outer update.
        """
        meta_losses = self.inner_adaptation(current_state, env, update=False)
        return meta_losses

    def outer_update(self, meta_losses):
        """
        Outer-loop update using meta-losses for each building.
        Args:
            meta_losses: List of meta-losses from different buildings.
        """
        # sum the losses
        loss_sum = sum(meta_losses)
        self.outer_optimizer.zero_grad()
        loss_sum.backward()
        self.outer_optimizer.step()
