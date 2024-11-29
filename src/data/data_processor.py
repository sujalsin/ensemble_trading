import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple
import ta
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Data collection and preprocessing for stock trading"""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        Initialize DataProcessor
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = StandardScaler()
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        dfs = []
        for symbol in self.symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                # Validate data
                if df.empty:
                    raise ValueError(f"No data received for {symbol}")
                
                # Select and rename columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = [f"{symbol}_{col.lower()}" for col in df.columns]
                
                # Handle missing values
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Validate no NaN values remain
                if df.isnull().any().any():
                    raise ValueError(f"NaN values remain in {symbol} data after filling")
                
                dfs.append(df)
                print(f"Successfully fetched data for {symbol}")
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                raise
        
        self.data = pd.concat(dfs, axis=1)
        print(f"Combined data shape: {self.data.shape}")
        return self.data
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add technical indicators to the dataset"""
        print("\nCalculating technical indicators...")
        
        # Ensure data exists and is clean
        if self.data is None or self.data.empty:
            raise ValueError("No data available for technical analysis")
        
        # Forward fill any NaN values in the raw data first
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        for symbol in self.symbols:
            print(f"\nProcessing {symbol}:")
            
            # Get price data
            close_prices = self.data[f"{symbol}_close"]
            high_prices = self.data[f"{symbol}_high"]
            low_prices = self.data[f"{symbol}_low"]
            volume = self.data[f"{symbol}_volume"]
            
            # Validate price data
            if any(s.isnull().any() for s in [close_prices, high_prices, low_prices, volume]):
                raise ValueError(f"Missing price data for {symbol}")
            
            # Calculate basic price indicators
            print("- Calculating moving averages...")
            self.data[f"{symbol}_sma_20"] = close_prices.rolling(window=20, min_periods=1).mean()
            self.data[f"{symbol}_sma_50"] = close_prices.rolling(window=50, min_periods=1).mean()
            
            # Calculate MACD
            print("- Calculating MACD...")
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            self.data[f"{symbol}_macd"] = macd - signal
            
            # Calculate RSI
            print("- Calculating RSI...")
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            self.data[f"{symbol}_rsi"] = 100 - (100 / (1 + rs))
            
            # Calculate Stochastic
            print("- Calculating Stochastic...")
            low_min = low_prices.rolling(window=14, min_periods=1).min()
            high_max = high_prices.rolling(window=14, min_periods=1).max()
            self.data[f"{symbol}_stoch"] = 100 * (close_prices - low_min) / (high_max - low_min)
            
            # Calculate Bollinger Bands
            print("- Calculating Bollinger Bands...")
            sma = close_prices.rolling(window=20, min_periods=1).mean()
            std = close_prices.rolling(window=20, min_periods=1).std()
            self.data[f"{symbol}_bbands_upper"] = sma + (2 * std)
            self.data[f"{symbol}_bbands_lower"] = sma - (2 * std)
            
            # Calculate OBV
            print("- Calculating OBV...")
            obv = (np.sign(close_prices.diff()) * volume).fillna(0).cumsum()
            self.data[f"{symbol}_obv"] = obv
        
        # Handle any remaining NaN values
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')
        
        # Verify calculations
        nan_check = self.data.isnull().sum()
        if nan_check.any():
            print("\nWarning: NaN values found in columns:")
            print(nan_check[nan_check > 0])
            self.data = self.data.fillna(0)
        
        print("\nTechnical indicators calculation complete.")
        return self.data
    
    def preprocess_data(self) -> Tuple[pd.DataFrame, StandardScaler]:
        """Preprocess the data for model training"""
        print("\nPreprocessing data...")
        
        # Initial data validation
        if self.data is None or self.data.empty:
            raise ValueError("No data available for preprocessing")
        
        # Step 1: Handle missing values
        print("1. Handling missing values...")
        self.data = self.data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Step 2: Calculate returns
        print("2. Calculating returns...")
        for symbol in self.symbols:
            close_prices = self.data[f"{symbol}_close"]
            returns = close_prices.pct_change().fillna(0)
            # Clip extreme returns
            returns = np.clip(returns, -1.0, 1.0)
            self.data[f"{symbol}_returns"] = returns
            print(f"   {symbol} returns - min: {returns.min():.4f}, max: {returns.max():.4f}")
        
        # Step 3: Handle infinite values
        print("3. Handling infinite values...")
        self.data = self.data.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Step 4: Scale features
        print("4. Scaling features...")
        feature_columns = [col for col in self.data.columns if not col.endswith('_returns')]
        
        # Fit scaler on training portion only
        train_size = int(len(self.data) * 0.8)
        train_data = self.data[feature_columns][:train_size]
        
        # Remove any remaining invalid values before scaling
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.fillna(train_data.mean())
        
        self.scaler.fit(train_data)
        
        # Transform all data
        all_data = self.data[feature_columns].replace([np.inf, -np.inf], np.nan)
        all_data = all_data.fillna(all_data.mean())
        self.data[feature_columns] = self.scaler.transform(all_data)
        
        # Final validation
        print("\n5. Final validation...")
        nan_check = self.data.isnull().sum()
        inf_check = np.isinf(self.data.values).sum()
        
        if nan_check.any():
            print("Warning: NaN values found. Replacing with 0...")
            self.data = self.data.fillna(0)
        
        if inf_check > 0:
            print("Warning: Infinite values found. Clipping values...")
            self.data = self.data.clip(-1e6, 1e6)
        
        print("\nPreprocessing complete!")
        print(f"Final data shape: {self.data.shape}")
        return self.data, self.scaler
    
    def prepare_train_test_split(self, test_size: float = 0.2
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the data into training and testing sets"""
        train_size = int(len(self.data) * (1 - test_size))
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        
        return train_data, test_data
    
    def process_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Run the complete data processing pipeline"""
        print("\nFetching data...")
        self.fetch_data()
        print(f"Initial data shape: {self.data.shape}")
        print(f"NaN values after fetch: {self.data.isnull().sum().sum()}")
        
        print("\nAdding technical indicators...")
        self.add_technical_indicators()
        print(f"Data shape after indicators: {self.data.shape}")
        print(f"NaN values after indicators: {self.data.isnull().sum().sum()}")
        
        print("\nPreprocessing data...")
        self.preprocess_data()
        print(f"Data shape after preprocessing: {self.data.shape}")
        print(f"NaN values after preprocessing: {self.data.isnull().sum().sum()}")
        print(f"Inf values after preprocessing: {np.isinf(self.data.values).sum()}")
        
        print("\nSplitting data...")
        train_data, test_data = self.prepare_train_test_split()
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        return train_data, test_data, self.scaler
