import sys
import os
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.ensemble_model import EnsembleModel
from src.models.ppo_model import PPOModel
from src.models.a2c_model import A2CModel
from src.models.ddpg_model import DDPGModel

def test_ensemble_prediction():
    """Test ensemble prediction functionality"""
    # Create mock state
    state_dim = 30  # Typical state dimension for our trading env
    action_dim = 1  # Single continuous action for portfolio allocation
    state = np.random.normal(0, 1, (state_dim,))
    
    # Model configurations
    model_configs = {
        'ppo': {
            'learning_rate': 0.001,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'device': str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        },
        'a2c': {
            'learning_rate': 0.001,
            'value_coef': 0.5,
            'entropy_coef': 0.01
        },
        'ddpg': {
            'actor_lr': 0.0001,
            'critic_lr': 0.001,
            'tau': 0.001,
            'gamma': 0.99
        }
    }
    
    # Create ensemble model
    ensemble = EnsembleModel(state_dim, action_dim, model_configs)
    
    # Test prediction
    print("\nTesting ensemble prediction...")
    prediction = ensemble.predict(state)
    print(f"\nFinal ensemble prediction shape: {prediction.shape}")
    print(f"Final ensemble prediction: {prediction}")
    print(f"Min value: {np.min(prediction)}")
    print(f"Max value: {np.max(prediction)}")

if __name__ == "__main__":
    test_ensemble_prediction()
