from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os
from src.env.trading_env import TradingEnvironment

class BaseRLModel(nn.Module, ABC):
    """Base class for all RL models"""
    
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        """
        Initialize base RL model
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build networks
        self._build_network()
        self.to(self.device)
    
    @abstractmethod
    def _build_network(self) -> None:
        """Build the neural network architecture"""
        pass
    
    @abstractmethod
    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict action given current state"""
        pass
    
    @abstractmethod
    def train_step(self, *args, **kwargs) -> Dict[str, float]:
        """Train the model on a batch of data"""
        pass
    
    def save(self, path: str) -> None:
        """Save model state dict"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
    
    def load(self, path: str) -> None:
        """Load model state dict"""
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=self.device)
        self.load_state_dict(state_dict)
    
    def _build_mlp(self, x: torch.Tensor, hidden_sizes: List[int], output_size: int,
                   activation: str = 'relu', output_activation: Optional[str] = None) -> torch.Tensor:
        """Helper function to build a multi-layer perceptron"""
        for h in hidden_sizes:
            x = nn.ReLU()(nn.Linear(x.shape[-1], h)(x))
        return nn.Linear(x.shape[-1], output_size)(x)
