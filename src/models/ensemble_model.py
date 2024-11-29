import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from src.models.base_model import BaseRLModel
from src.models.ppo_model import PPOModel
from src.models.a2c_model import A2CModel
from src.models.ddpg_model import DDPGModel

class EnsembleModel:
    """Ensemble model combining PPO, A2C, and DDPG"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 model_configs: Dict[str, Dict[str, Any]] = None):
        """
        Initialize ensemble model
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            model_configs: Configuration for each model
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Default configurations
        if model_configs is None:
            model_configs = {
                'ppo': {'learning_rate': 0.001, 'clip_ratio': 0.2},
                'a2c': {'learning_rate': 0.001, 'value_coef': 0.5},
                'ddpg': {'actor_lr': 0.0001, 'critic_lr': 0.001}
            }
        
        # Initialize models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {
            'ppo': PPOModel(state_dim, action_dim, **model_configs['ppo']),
            'a2c': A2CModel(state_dim, action_dim, **model_configs['a2c']),
            'ddpg': DDPGModel(state_dim, action_dim, **model_configs['ddpg'])
        }
        
        # Initialize performance metrics
        self.performance_window = 100
        self.model_returns = {name: [] for name in self.models.keys()}
        self.model_weights = self._initialize_weights()
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize ensemble weights"""
        n_models = len(self.models)
        return {name: 1.0 / n_models for name in self.models.keys()}
    
    def _update_weights(self) -> None:
        """Update ensemble weights based on recent performance"""
        if all(len(returns) >= self.performance_window for returns in self.model_returns.values()):
            # Calculate average returns for each model
            avg_returns = {
                name: np.mean(returns[-self.performance_window:])
                for name, returns in self.model_returns.items()
            }
            
            # Convert to softmax weights
            returns = np.array(list(avg_returns.values()))
            weights = np.exp(returns - np.max(returns))
            weights = weights / np.sum(weights)
            
            # Update weights
            for name, weight in zip(self.models.keys(), weights):
                self.model_weights[name] = float(weight)
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action using ensemble of models"""
        # Get predictions from each model
        predictions = {
            name: model.predict(state) for name, model in self.models.items()
        }
        
        # Debug predictions
        for name, pred in predictions.items():
            print(f"\n{name} prediction:")
            print(f"Shape: {pred.shape if hasattr(pred, 'shape') else 'scalar'}")
            print(f"Min: {np.min(pred)}, Max: {np.max(pred)}")
            
            # Ensure prediction is 2D array with shape (1, action_dim)
            if isinstance(pred, (float, np.float32, np.float64)):
                predictions[name] = np.array([[pred]])
            elif pred.ndim == 0:
                predictions[name] = np.array([[pred.item()]])
            elif pred.ndim == 1:
                predictions[name] = pred.reshape(1, -1)
            elif pred.ndim == 2 and pred.shape[0] != 1:
                predictions[name] = pred.reshape(1, -1)
        
        # Initialize ensemble prediction
        ensemble_prediction = np.zeros((1, self.action_dim))
        
        # Weighted sum of predictions
        for name, pred in predictions.items():
            ensemble_prediction += self.model_weights[name] * pred
            
        # Debug ensemble prediction
        print("\nEnsemble prediction:")
        print(f"Shape: {ensemble_prediction.shape}")
        print(f"Min: {np.min(ensemble_prediction)}, Max: {np.max(ensemble_prediction)}")
        
        return ensemble_prediction.squeeze()
    
    def _calculate_returns_advantages(self, rewards: List[float], values: List[float],
                                   next_values: List[float], dones: List[bool],
                                   gamma: float = 0.99, lambda_: float = 0.95
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate returns and advantages using GAE"""
        rewards = np.array(rewards)
        values = np.array(values)
        next_values = np.array(next_values)
        dones = np.array(dones)
        
        # Calculate TD errors and advantages
        deltas = rewards + gamma * next_values * (1 - dones) - values
        advantages = np.zeros_like(deltas)
        
        # GAE calculation
        last_advantage = 0
        for t in reversed(range(len(deltas))):
            advantages[t] = deltas[t] + gamma * lambda_ * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        # Calculate returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step(self, states: np.ndarray, actions: np.ndarray,
                  rewards: np.ndarray, next_states: np.ndarray,
                  dones: np.ndarray) -> Dict[str, float]:
        """Train all models for one step"""
        metrics = {}
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate values using PPO's critic for advantage estimation
        with torch.no_grad():
            values = self.models['ppo'].critic(self.models['ppo'].shared(states)).squeeze().cpu().numpy()
            next_values = self.models['ppo'].critic(self.models['ppo'].shared(next_states)).squeeze().cpu().numpy()
        
        # Calculate advantages and returns
        advantages, returns = self._calculate_returns_advantages(
            rewards.cpu().numpy(), values, next_values, dones.cpu().numpy()
        )
        
        # Train each model
        for name, model in self.models.items():
            if name == 'ppo':
                # Calculate old log probabilities for PPO
                with torch.no_grad():
                    features = model.shared(states)
                    mu, std = model._get_policy_dist(features)
                    log_prob = -0.5 * (
                        torch.log(2.0 * np.pi)
                        + 2.0 * torch.log(std)
                        + torch.square(actions - mu) / torch.square(std)
                    )
                    old_logp = torch.sum(log_prob, dim=1)
                
                model_metrics = model.train_step(
                    states.cpu().numpy(),
                    actions.cpu().numpy(),
                    advantages,
                    returns,
                    old_logp.cpu().numpy()
                )
            elif name == 'a2c':
                model_metrics = model.train_step(
                    states.cpu().numpy(),
                    actions.cpu().numpy(),
                    advantages,
                    returns
                )
            elif name == 'ddpg':
                model_metrics = model.train_step(
                    states.cpu().numpy(),
                    actions.cpu().numpy(),
                    rewards.cpu().numpy(),
                    next_states.cpu().numpy(),
                    dones.cpu().numpy()
                )
            
            # Add model prefix to metrics
            metrics.update({f"{name}_{k}": v for k, v in model_metrics.items()})
        
        # Update ensemble weights based on recent performance
        self._update_weights()
        
        return metrics
    
    def save_models(self, path: str) -> None:
        """Save all models"""
        for name, model in self.models.items():
            model.save(f"{path}/{name}")
    
    def load_models(self, path: str) -> None:
        """Load all models"""
        for name, model in self.models.items():
            model.load(f"{path}/{name}")
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights"""
        return self.model_weights.copy()
