import argparse
import os
from datetime import datetime, timedelta
import pandas as pd

from src.data.data_processor import DataProcessor
from src.env.trading_env import TradingEnvironment
from src.models.ensemble_model import EnsembleModel
from src.evaluation.backtest import BacktestEngine

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Trading System')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                       help='List of stock symbols to trade')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                       help='Start date for data collection')
    parser.add_argument('--end_date', type=str, 
                       default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                       help='End date for data collection')
    parser.add_argument('--initial_balance', type=float, default=100000.0,
                       help='Initial portfolio balance')
    parser.add_argument('--train_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--model_path', type=str, default='models',
                       help='Path to save/load models')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode: train new models or test existing ones')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize data processor and load data
    print("Loading and processing data...")
    data_processor = DataProcessor(args.symbols, args.start_date, args.end_date)
    train_data, test_data, scaler = data_processor.process_pipeline()
    
    # Create training environment
    train_env = TradingEnvironment(train_data, initial_balance=args.initial_balance)
    test_env = TradingEnvironment(test_data, initial_balance=args.initial_balance)
    
    # Initialize ensemble model
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    model = EnsembleModel(state_dim, action_dim)
    
    if args.mode == 'train':
        print(f"Training ensemble model for {args.train_episodes} episodes...")
        training_history = model.train(train_env, episodes=args.train_episodes)
        
        # Save models
        print("Saving trained models...")
        model.save_models(args.model_path)
        
        # Save training history
        pd.DataFrame(training_history).to_csv('results/training_history.csv')
    else:
        print("Loading pre-trained models...")
        model.load_models(args.model_path)
    
    # Run backtesting
    print("Running backtesting...")
    backtest = BacktestEngine(model, test_env, initial_balance=args.initial_balance)
    results = backtest.run_backtest()
    
    # Calculate and display metrics
    metrics = backtest.calculate_metrics()
    print("\nBacktest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    results.to_csv('results/backtest_results.csv')
    
    # Plot results
    print("\nGenerating plots...")
    backtest.plot_results(save_path='results/backtest_plots.png')
    
    # Get model weights
    weights = model.get_model_weights()
    print("\nFinal Model Weights:")
    for model_name, weight in weights.items():
        print(f"{model_name}: {weight:.4f}")

if __name__ == '__main__':
    main()
