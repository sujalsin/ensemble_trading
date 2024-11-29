import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from src.models.ensemble_model import EnsembleModel
from src.env.trading_env import TradingEnvironment

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, model: EnsembleModel, env: TradingEnvironment,
                 initial_balance: float = 100000.0):
        """
        Initialize backtesting engine
        
        Args:
            model: Trained ensemble model
            env: Trading environment
            initial_balance: Initial portfolio balance
        """
        self.model = model
        self.env = env
        self.initial_balance = initial_balance
        self.results = None
    
    def run_backtest(self) -> pd.DataFrame:
        """Run backtest simulation"""
        state = self.env.reset()
        done = False
        
        # Initialize results tracking
        results = []
        current_step = 0
        
        while not done:
            # Get action from model
            action = self.model.predict(state)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Record results
            results.append({
                'step': current_step,
                'portfolio_value': info['portfolio_value'],
                'balance': info['balance'],
                'positions': info['positions'],
                'reward': reward
            })
            
            state = next_state
            current_step += 1
        
        # Convert results to DataFrame
        self.results = pd.DataFrame(results)
        return self.results
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if self.results is None:
            raise ValueError("Must run backtest before calculating metrics")
        
        portfolio_values = self.results['portfolio_value'].values
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        annual_return = total_return * (252 / len(returns))  # Annualized return
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        winning_days = np.sum(returns > 0)
        win_rate = winning_days / len(returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def get_position_analysis(self) -> pd.DataFrame:
        """Analyze position changes and trading activity"""
        if self.results is None:
            raise ValueError("Must run backtest before analyzing positions")
        
        # Convert positions to DataFrame
        positions = pd.DataFrame(self.results['positions'].tolist(),
                               index=self.results.index)
        
        # Calculate position changes
        position_changes = positions.diff()
        
        # Calculate trading activity
        trades = position_changes.abs()
        trading_volume = trades.sum(axis=1)
        
        return pd.DataFrame({
            'positions': positions.values.tolist(),
            'position_changes': position_changes.values.tolist(),
            'trading_volume': trading_volume
        }, index=self.results.index)
    
    def plot_results(self, save_path: str = None) -> None:
        """Plot backtest results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.results is None:
            raise ValueError("Must run backtest before plotting results")
        
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot portfolio value
        ax1.plot(self.results['portfolio_value'], label='Portfolio Value')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        
        # Plot returns distribution
        returns = np.diff(self.results['portfolio_value']) / self.results['portfolio_value'][:-1]
        sns.histplot(returns, kde=True, ax=ax2)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        
        # Plot drawdown
        portfolio_values = self.results['portfolio_value'].values
        cumulative_returns = (1 + np.diff(portfolio_values) / portfolio_values[:-1]).cumprod()
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        
        ax3.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Drawdown')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
