import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from src.models.base_model import BaseRLModel

class DDPGModel(BaseRLModel):
    """Deep Deterministic Policy Gradient (DDPG) model"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 actor_lr: float = 0.0001, critic_lr: float = 0.001,
                 tau: float = 0.001, gamma: float = 0.99,
                 buffer_size: int = 100000, batch_size: int = 64):
        self.tau = tau  # Target network update rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Initialize replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = []
        
        super().__init__(state_dim, action_dim, actor_lr)
    
    def _build_network(self) -> None:
        """Build the actor and critic networks"""
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        
        # Target actor network
        self.target_actor = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        
        # Critic network (state + action -> Q-value)
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Target critic network
        self.target_critic = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Copy weights to target networks
        self._update_target_networks(tau=1.0)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
    def _update_target_networks(self, tau: float = None) -> None:
        """Update target networks using polyak averaging"""
        if tau is None:
            tau = self.tau
            
        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action given state"""
        self.eval()
        with torch.no_grad():
            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)
            
            state = torch.FloatTensor(state)
            action = self.actor(state)
            return action.cpu().numpy()
    
    def train_step(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, next_states: np.ndarray,
              dones: np.ndarray) -> Dict[str, float]:
        """Train the DDPG model on a batch of data"""
        super().train(True)  # Set model to training mode
        
        # Store transition in replay buffer
        for i in range(len(states)):
            self.replay_buffer.append({
                'state': states[i],
                'action': actions[i],
                'reward': rewards[i],
                'next_state': next_states[i],
                'done': dones[i]
            })
        
        # Sample random minibatch from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)
        states = np.array([x['state'] for x in batch])
        actions = np.array([x['action'] for x in batch])
        rewards = np.array([x['reward'] for x in batch])
        next_states = np.array([x['next_state'] for x in batch])
        dones = np.array([x['done'] for x in batch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get target actions and Q-values
        target_actions = self.target_actor(next_states)
        target_q = self.target_critic(torch.cat([next_states, target_actions], dim=1)).squeeze()
        
        # Compute target values (Bellman equation)
        targets = rewards + self.gamma * (1 - dones) * target_q
        
        # Update critic
        current_q = self.critic(torch.cat([states, actions], dim=1)).squeeze()
        critic_loss = F.mse_loss(current_q, targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target_networks()
        
        return {
            'actor_loss': float(actor_loss.item()),
            'critic_loss': float(critic_loss.item())
        }
    
    def save(self, path: str) -> None:
        """Save model weights"""
        torch.save(self.actor.state_dict(), f"{path}/ddpg_actor")
        torch.save(self.critic.state_dict(), f"{path}/ddpg_critic")
        torch.save(self.target_actor.state_dict(), f"{path}/ddpg_target_actor")
        torch.save(self.target_critic.state_dict(), f"{path}/ddpg_target_critic")
    
    def load(self, path: str) -> None:
        """Load model weights"""
        self.actor.load_state_dict(torch.load(f"{path}/ddpg_actor"))
        self.critic.load_state_dict(torch.load(f"{path}/ddpg_critic"))
        self.target_actor.load_state_dict(torch.load(f"{path}/ddpg_target_actor"))
        self.target_critic.load_state_dict(torch.load(f"{path}/ddpg_target_critic"))
