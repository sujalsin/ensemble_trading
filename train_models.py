import numpy as np
import pandas as pd
from datetime import datetime
from src.data.data_processor import DataProcessor
from src.env.trading_env import TradingEnvironment
from src.models.ensemble_model import EnsembleModel
from src.evaluation.backtest import BacktestEngine

def train_models(symbols=['AAPL', 'MSFT', 'GOOGL'], 
                start_date='2020-01-01',
                end_date='2024-01-01',
                initial_balance=100000.0,
                num_episodes=100,
                batch_size=32,
                eval_interval=10):
    """Train the ensemble trading model"""
    
    print("\nTraining configuration:")
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Initial balance: ${initial_balance:,.2f}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Evaluation interval: {eval_interval}")
    
    # Process data
    print("\nProcessing data...")
    data_processor = DataProcessor(symbols, start_date, end_date)
    train_data, test_data, scaler = data_processor.process_pipeline()
    
    # Create environment
    env = TradingEnvironment(train_data, initial_balance=initial_balance)
    eval_env = TradingEnvironment(test_data, initial_balance=initial_balance)
    
    # Initialize model
    model = EnsembleModel(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        model_configs={
            'ppo': {
                'learning_rate': 0.0003,
                'clip_ratio': 0.2
            },
            'a2c': {
                'learning_rate': 0.0003,
                'value_coef': 0.5
            },
            'ddpg': {
                'actor_lr': 0.0001,
                'critic_lr': 0.001
            }
        }
    )
    
    # Training loop
    best_sharpe = float('-inf')
    episode_returns = []
    running_reward = 0
    
    print("\nStarting training...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        transitions = []
        
        # Collect episode transitions
        while not done:
            action = model.predict(state)
            next_state, reward, done, info = env.step(action)
            
            transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            state = next_state
            episode_reward += reward
        
        # Process transitions and train models
        states = np.array([t['state'] for t in transitions])
        actions = np.array([t['action'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions])
        next_states = np.array([t['next_state'] for t in transitions])
        dones = np.array([t['done'] for t in transitions])
        
        # Train in batches
        batch_metrics = []
        for i in range(0, len(transitions), batch_size):
            batch_states = states[i:i+batch_size]
            batch_actions = actions[i:i+batch_size]
            batch_rewards = rewards[i:i+batch_size]
            batch_next_states = next_states[i:i+batch_size]
            batch_dones = dones[i:i+batch_size]
            
            # Train all models
            metrics = model.train_step(
                batch_states, batch_actions, batch_rewards,
                batch_next_states, batch_dones
            )
            batch_metrics.append(metrics)
        
        # Update running reward
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        episode_returns.append(episode_reward)
        
        # Print episode summary
        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Running reward: {running_reward:.2f}")
        
        # Print latest batch metrics
        if batch_metrics:
            print("\nTraining metrics (last batch):")
            for k, v in batch_metrics[-1].items():
                print(f"{k}: {v:.4f}")
        
        # Evaluate on test set periodically
        if (episode + 1) % eval_interval == 0:
            backtest = BacktestEngine(model, eval_env)
            results = backtest.run_backtest()
            metrics = backtest.calculate_metrics()
            
            print(f"\nEvaluation metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save best model based on Sharpe ratio
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                model.save_weights('best_model_weights')
                print(f"New best model saved! Sharpe ratio: {best_sharpe:.4f}")
    
    return model, metrics

if __name__ == "__main__":
    trained_model, final_metrics = train_models()
    print("\nTraining completed!")
    print("\nFinal evaluation metrics:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")
