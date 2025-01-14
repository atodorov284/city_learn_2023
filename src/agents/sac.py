import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from agents.base_agent import Agent
from utils.replay_buffer import ReplayBuffer

LOG_STD_MIN = -1
LOG_STD_MAX = 1


class SACAgent(Agent):
    def __init__(
        self,
        observation_space_dim: int,
        action_space_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.05,
        buffer_size: int = 100000,
        batch_size: int = 32,
        action_space=None,
    ) -> None:
        """Initialize the Soft-Actor Critic agent"""
        super().__init__(observation_space_dim, action_space_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_space_dim = action_space_dim

        self.actor = Actor(
            observation_space_dim,
            hidden_dim,
            action_space_dim,
            action_space=action_space,
        )
        self.critic = Critic(observation_space_dim, action_space_dim, hidden_dim)
        self.critic_target = Critic(observation_space_dim, action_space_dim, hidden_dim)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate * 2
        )

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Entropy temperature (can be made learnable)
        self.target_entropy = -action_space_dim  # -dim(A) is a good heuristic
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        # Tracking
        self.total_steps = 0

        # Initialize statistics for logging
        self.running_reward_mean = 0
        self.running_reward_std = 1

        self.reward_scale = 0.1

    def select_action(self, state: np.array, deterministic: bool = False) -> np.array:
        """
        Select an action from the policy

        Args:
            state (np.array): Input state
            deterministic (bool): Whether to use deterministic policy (after training)

        Returns:
            Numpy array of selected action
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Sample action from policy
        with torch.no_grad():
            action, _ = self.actor.sample_action(state, deterministic)

        return action.detach().cpu().numpy()[0]

    def compute_loss(self, states, actions, rewards, next_states, done):
        """
        Train the maml xd
        """
        scaled_rewards = rewards * self.reward_scale
        print(rewards)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(scaled_rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_action(next_states)

            # Target Q-values
            q1_next, q2_next = self.critic_target(next_states, next_actions)

            q1_next = q1_next.squeeze()
            q2_next = q2_next.squeeze()

            min_q_next = torch.min(q1_next, q2_next)

            # Compute target Q-values with entropy
            target_q = rewards + (1 - done) * self.gamma * (
                min_q_next - self.alpha * next_log_probs.squeeze()
            )

        # Current Q-values
        # Go from tensor of size [1,1,9] to [1,9]
        q1_current, q2_current = self.critic(states, actions.squeeze(1))

        q1_loss = F.mse_loss(q1_current.squeeze(), target_q.squeeze())
        q2_loss = F.mse_loss(q2_current.squeeze(), target_q.squeeze())
        critic_loss = q1_loss + q2_loss

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        sampled_actions, log_probs = self.actor.sample_action(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss with entropy
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Optimize alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()

        # Soft update of target network
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def train(self, update=True, custom_buffer=None) -> None:
        """
        Train the SAC agent using a batch from replay buffer
        """
        if custom_buffer is not None:
            self.replay_buffer = custom_buffer

        states, actions, rewards, next_states, done = self.replay_buffer.sample(
            self.batch_size
        )

        # Update running statistics
        self.running_reward_mean = 0.99 * self.running_reward_mean + 0.01 * np.mean(
            rewards
        )
        self.running_reward_std = 0.99 * self.running_reward_std + 0.01 * np.std(
            rewards
        )

        # Scale rewards for training
        scaled_rewards = rewards * self.reward_scale

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(scaled_rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample_action(next_states)

            # Target Q-values
            q1_next, q2_next = self.critic_target(next_states, next_actions)

            q1_next = q1_next.squeeze()
            q2_next = q2_next.squeeze()

            min_q_next = torch.min(q1_next, q2_next)

            # Compute target Q-values with entropy
            target_q = rewards + (1 - done) * self.gamma * (
                min_q_next - self.alpha * next_log_probs.squeeze()
            )

        # Current Q-values
        # Go from tensor of size [1,1,9] to [1,9]
        q1_current, q2_current = self.critic(states, actions.squeeze(1))

        q1_loss = F.mse_loss(q1_current.squeeze(), target_q.squeeze())
        q2_loss = F.mse_loss(q2_current.squeeze(), target_q.squeeze())
        critic_loss = q1_loss + q2_loss

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if update:
            self.critic_optimizer.step()

        sampled_actions, log_probs = self.actor.sample_action(states)
        q1_pi, q2_pi = self.critic(states, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Actor loss with entropy
        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if update:
            self.actor_optimizer.step()

        # Add alpha optimization
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        if update:
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Soft update of target network
        self._soft_update(self.critic, self.critic_target)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        """
        Soft update of target network parameters

        Args:
            local_model (nn.Module): Source network
            target_model (nn.Module): Target network to be updated
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """
        Save the trained agent's parameters to a file

        Args:
            path (str): Path where to save the agent's parameters
        """
        torch.save(
            {
                # Model architectures and weights
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                # Optimizer states
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                # Important parameters
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": self.alpha,
                "reward_scale": self.reward_scale,
                # Running statistics
                "running_reward_mean": self.running_reward_mean,
                "running_reward_std": self.running_reward_std,
                # Training progress
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load a trained agent's parameters from a file

        Args:
            path (str): Path to the saved agent's parameters
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model architectures and weights
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])

        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        # Load parameters
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.alpha = checkpoint["alpha"]
        self.reward_scale = checkpoint["reward_scale"]

        # Load running statistics
        self.running_reward_mean = checkpoint["running_reward_mean"]
        self.running_reward_std = checkpoint["running_reward_std"]

        # Load training progress
        self.total_steps = checkpoint["total_steps"]


class Critic(nn.Module):
    def __init__(
        self, observation_space_dim: int, action_space_dim: int, hidden_dim: int
    ) -> None:
        """Initialize the actor network"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._output_dim = 1

        super(Critic, self).__init__()

        # First Q-network
        self.fc1 = nn.Linear(observation_space_dim + action_space_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Second Q-network
        self.fc4 = nn.Linear(observation_space_dim + action_space_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for a state-action pair

        Args:
            state (torch.Tensor): Input state
            action (torch.Tensor): Input action

        Returns:
            Two Q-value estimates
        """
        x = torch.cat([observation, action], 1)

        # First Q-network
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Second Q-network
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2


class Actor(nn.Module):
    def __init__(
        self,
        observation_space_dim: int,
        hidden_dim: int,
        action_space_dim: int,
        action_space,
    ) -> None:
        """Initialize the actor network"""
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(observation_space_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_space_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_space_dim)

        init_w = 0.003
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

        self.fc_logstd.weight.data.uniform_(-init_w, init_w)
        self.fc_logstd.bias.data.uniform_(-init_w, init_w)

        action_space = action_space[0]
        self.action_scale = torch.FloatTensor(
            (action_space.high - action_space.low) / 2.0
        )
        self.action_bias = torch.FloatTensor(
            (action_space.high + action_space.low) / 2.0
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get action distribution parameters

        Args:
            observation (torch.Tensor): Input state

        Returns:
            mean, log_std of the action distribution
        """
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # mean = torch.clamp(mean, 0, 1)

        return mean, log_std

    def sample_action(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample an action from the policy

        Args:
            observation (torch.Tensor): Input state
            deterministic (bool): Whether to use deterministic policy

        Returns:
            Sampled action, log probability of the action
        """
        # Get mean and log std
        mean, log_std = self(observation)

        if deterministic:
            # Use deterministic action with probability 1
            return torch.tanh(mean), 0

        std = torch.exp(log_std)
        normal_dist = torch.distributions.Normal(mean, std)

        z = normal_dist.rsample()

        action = torch.tanh(z)

        action = self.action_scale * action + self.action_bias

        # Log probability correction for tanh
        log_prob = normal_dist.log_prob(z)
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


if __name__ == "__main__":
    critic = Critic(10, 10, 10)
    print(critic.forward(torch.ones(10), torch.ones(10)))

    actor = Actor(10, 10, 10)
    print(actor.sample_action(torch.ones(10)))
