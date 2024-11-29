import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from src.models.base_model import BaseRLModel

class A2CModel(BaseRLModel):
    """Advantage Actor-Critic (A2C) model"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001,
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        super().__init__(state_dim, action_dim, learning_rate)
    
    def _build_network(self) -> None:
        """Build the actor-critic network architecture"""
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(self.state_dim)
        
        # Shared layers with batch normalization
        self.shared = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Actor mean network
        self.actor_mean = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Tanh()  # Bound mean to [-1, 1]
        )
        
        # Actor std network with careful initialization
        self.actor_std = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Softplus()  # Ensure positive std
        )
        
        # Initialize the last layers with small weights
        for layer in [self.actor_mean[-2], self.actor_std[-2]]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(layer.bias, -3e-3, 3e-3)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize critic's last layer with small weights
        self.critic[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.critic[-1].bias.data.uniform_(-3e-3, 3e-3)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def _get_policy_dist(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy distribution parameters"""
        mu = self.actor_mean(features)
        std = self.actor_std(features)
        
        # Ensure std is positive and reasonably bounded
        std = torch.clamp(std, min=1e-3, max=1.0)
        
        # Add small epsilon to prevent exactly zero std
        std = std + 1e-6
        
        return mu, std
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action given state"""
        self.eval()
        with torch.no_grad():
            # Convert state to tensor and normalize
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Debug state
            print(f"A2C Raw state - min: {torch.min(state_tensor).item()}, max: {torch.max(state_tensor).item()}")
            print(f"A2C State contains NaN: {torch.isnan(state_tensor).any().item()}")
            
            # Apply input normalization
            normalized_state = self.input_bn(state_tensor)
            
            # Debug normalized state
            print(f"A2C Normalized state - min: {torch.min(normalized_state).item()}, max: {torch.max(normalized_state).item()}")
            print(f"A2C Normalized state contains NaN: {torch.isnan(normalized_state).any().item()}")
            
            # Forward pass
            features = self.shared(normalized_state)
            mu, std = self._get_policy_dist(features)
            
            # Debug network outputs
            print(f"A2C mu shape: {mu.shape}, std shape: {std.shape}")
            print(f"A2C std min: {torch.min(std).item()}, std max: {torch.max(std).item()}")
            print(f"A2C mu min: {torch.min(mu).item()}, mu max: {torch.max(mu).item()}")
            
            # Sample from normal distribution with validated parameters
            dist = torch.normal(mu, std)
            action = torch.clamp(dist, -1, 1)
            
        return action.numpy().squeeze()
    
    def train_step(self, states: np.ndarray, actions: np.ndarray,
              advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Train the A2C model on a batch of data"""
        super().train(True)  # Set model to training mode
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Forward pass
        normalized_states = self.input_bn(states)
        features = self.shared(normalized_states)
        mu, std = self._get_policy_dist(features)
        values = self.critic(features).squeeze()
        
        # Calculate log probabilities
        log_prob = -0.5 * (
            torch.log(2.0 * np.pi)
            + 2.0 * torch.log(std)
            + torch.square(actions - mu) / torch.square(std)
        )
        logp = torch.sum(log_prob, dim=1)
        
        # Calculate entropy
        entropy = torch.sum(torch.log(std) + 0.5 * torch.log(2.0 * np.pi * np.e), dim=-1)
        entropy = torch.mean(entropy)
        
        # A2C policy loss
        policy_loss = -torch.mean(logp * advantages)
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.item())
        }
    
    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save(self.input_bn.state_dict(), f"{path}/a2c_input_bn")
        torch.save(self.shared.state_dict(), f"{path}/a2c_shared")
        torch.save(self.actor_mean.state_dict(), f"{path}/a2c_actor_mean")
        torch.save(self.actor_std.state_dict(), f"{path}/a2c_actor_std")
        torch.save(self.critic.state_dict(), f"{path}/a2c_critic")
    
    def load(self, path: str) -> None:
        """Load model weights"""
        self.input_bn.load_state_dict(torch.load(f"{path}/a2c_input_bn"))
        self.shared.load_state_dict(torch.load(f"{path}/a2c_shared"))
        self.actor_mean.load_state_dict(torch.load(f"{path}/a2c_actor_mean"))
        self.actor_std.load_state_dict(torch.load(f"{path}/a2c_actor_std"))
        self.critic.load_state_dict(torch.load(f"{path}/a2c_critic"))
