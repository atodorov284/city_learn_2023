import torch
import torch.optim as optim
import torch.nn.functional as F
from agents.sac import SACAgent

class MAML(SACAgent):
    def __init__(self, observation_space_dim: int, action_space_dim: int, inner_lr=0.01, outer_lr=0.001, num_inner_updates=1, action_space=None):
        """
        Initialize MAML for decentralized learning.
        """
        super().__init__(observation_space_dim, action_space_dim, action_space=action_space)
        self.inner_lr = inner_lr  # Inner loop learning rate
        self.outer_lr = outer_lr  # Outer loop learning rate
        self.num_inner_updates = num_inner_updates  # Number of gradient updates in the inner loop
        
        # Ensure you add the parameters of the networks that will be optimized
        self.actor = self.actor
        self.critic = self.critic
        
        # Optimizer for the outer loop (meta-optimizer)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.outer_lr
        )

    def inner_update(self, state, action, reward, next_state):
        """
        Perform a single gradient update step for a specific building using the SAC update.
        """
        # Compute the loss using the SACAgent's `compute_loss`
        loss = self.compute_loss(reward, state, action, next_state, done=torch.zeros_like(reward), log_probs=torch.zeros_like(reward))
        
        # Compute gradients and apply the inner update
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.data -= self.inner_lr * param.grad  # Update with inner learning rate
        
        return loss

    def update(self, state, action, reward, next_state):
        """
        Meta-update for all buildings using the MAML algorithm.
        This includes both the inner and outer loop updates.
        """
        # Perform inner loop updates for each building
        building_losses = []
        for i in range(len(state)):  # Loop over each building (assumes state, action, etc. are lists for each building)
            building_state = state[i]
            building_action = action[i]
            building_reward = reward[i]
            building_next_state = next_state[i]
            
            self.inner_update(building_state, building_action, building_reward, building_next_state)
            building_losses.append(self.compute_loss(building_reward, building_state, building_action, building_next_state, done=torch.zeros_like(building_reward), log_probs=torch.zeros_like(building_reward)))
        
        # Outer loop: gradient update based on the building losses
        self.optimizer.zero_grad()
        total_loss = sum(building_losses)  # Sum losses for all buildings
        total_loss.backward()
        self.optimizer.step()

    def sample_action(self, state, deterministic=False):
        """
        Sample action for each building using the SAC policy.
        """
        actions = []
        for s in state:
            action = super().select_action(s, deterministic)
            actions.append(action)
        return actions 