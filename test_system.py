import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.data_processor import DataProcessor
from src.env.trading_env import TradingEnvironment
from src.models.ensemble_model import EnsembleModel
from src.evaluation.backtest import BacktestEngine

def test_system():
    # Test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    initial_balance = 100000.0
    
    print("1. Testing Data Processing...")
    data_processor = DataProcessor(symbols, start_date, end_date)
    train_data, test_data, scaler = data_processor.process_pipeline()
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    print("\n2. Testing Trading Environment...")
    env = TradingEnvironment(train_data, initial_balance=initial_balance)
    observation = env.reset()
    print(f"Observation shape: {observation.shape}")
    print(f"Action shape: {env.action_space.shape}")
    
    # Test one step
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    print(f"Step result - Reward: {reward:.4f}, Portfolio Value: {info['portfolio_value']:.2f}")
    
    print("\n3. Testing Ensemble Model...")
    model = EnsembleModel(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    # Test prediction
    state = env.reset()
    action = model.predict(state)
    print(f"Model prediction shape: {action.shape}")
    
    print("\n4. Testing Backtesting Engine...")
    backtest = BacktestEngine(model, env, initial_balance=initial_balance)
    results = backtest.run_backtest()
    metrics = backtest.calculate_metrics()
    
    print("\nBacktest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_system()
