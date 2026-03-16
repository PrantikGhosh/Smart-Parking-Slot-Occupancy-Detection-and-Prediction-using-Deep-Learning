"""
Feature Engineering for Time Series Prediction
Extracts and transforms features from occupancy data for model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import get_database


class FeatureEngineer:
    """Handles feature extraction and preprocessing for time series models"""
    
    def __init__(self, db=None):
        """Initialize feature engineer"""
        self.db = db or get_database()
    
    def load_occupancy_data(self, parking_lot_id: int, slot_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load occupancy events from database as DataFrame
        
        Args:
            parking_lot_id: Parking lot ID
            slot_id: Optional specific slot ID
            
        Returns:
            DataFrame with timestamp, slot_id, status, confidence
        """
        events = self.db.get_occupancy_events(parking_lot_id, slot_id=slot_id)
        
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        return df
    
    def create_binary_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert status to binary (1 = empty, 0 = occupied)
        
        Args:
            df: DataFrame with 'status' column
            
        Returns:
            DataFrame with 'occupancy' column (0 or 1)
        """
        df = df.copy()
        df['occupancy'] = df['status'].map({
            'empty': 1,
            'occupied': 0,
            'unknown': np.nan
        })
        # Forward fill unknown values then backward fill any remaining
        df['occupancy'] = df['occupancy'].ffill().bfill()
        
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from timestamp
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['minute'] = df['timestamp'].dt.minute
        
        # Derived features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        
        # Cyclical encoding for better continuity4 (23:00 is close to 00:00)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def calculate_rolling_features(self, df: pd.DataFrame, 
                                   windows: List[str] = ['1H', '3H', '6H', '24H']) -> pd.DataFrame:
        """
        Calculate rolling statistics
        
        Args: 
            df: DataFrame with 'timestamp' and 'occupancy'
            windows: List of window sizes (pandas offset strings)
            
        Returns:
            DataFrame with rolling feature columns
        """
        df = df.copy()
        df = df.set_index('timestamp')
        
        for window in windows:
            # Rolling mean occupancy
            df[f'rolling_mean_{window}'] = df['occupancy'].rolling(window, min_periods=1).mean()
            # Rolling std deviation
            df[f'rolling_std_{window}'] = df['occupancy'].rolling(window, min_periods=1).std().fillna(0)
            # Rolling sum (total time empty)
            df[f'rolling_sum_{window}'] = df['occupancy'].rolling(window, min_periods=1).sum()
        
        df = df.reset_index()
        return df
    
    def calculate_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12]) -> pd.DataFrame:
        """
        Create lagged occupancy features
        
        Args:
            df: DataFrame with 'occupancy'
            lags: List of lag periods
            
        Returns:
            DataFrame with lag columns
        """
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}'] = df['occupancy'].shift(lag)
        
        # Fill NaN values from shifting (use bfill instead of deprecated method)
        df = df.bfill()
        
        return df
    
    def prepare_features_for_ml(self, df: pd.DataFrame, 
                                feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare final feature matrix and target for ML models
        
        Args:
            df: DataFrame with all features
            feature_cols: Optional list of feature column names to use
            
        Returns:
            Tuple of (X, y) arrays
        """
        df = df.copy()
        
        if feature_cols is None:
            # Default feature set
            feature_cols = [
                'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'rolling_mean_1H', 'rolling_mean_3H', 'rolling_std_1H'
            ]
            
            # Add lag features if they exist
            lag_cols = [col for col in df.columns if col.startswith('lag_')]
            feature_cols.extend(lag_cols)
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].values
        y = df['occupancy'].values if 'occupancy' in df.columns else None
        
        return X, y
    
    def create_sequences_for_lstm(self, X: np.ndarray, y: np.ndarray, 
                                  sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        
        Args:
            X: Feature array (samples, features)
            y: Target array (samples,)
            sequence_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def prepare_for_prophet(self, df: pd.DataFrame, slot_id: str) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns)
        
        Args:
            df: DataFrame with 'timestamp' and 'occupancy'
            slot_id: Slot ID for filtering
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        df = df.copy()
        
        if 'slot_id' in df.columns:
            df = df[df['slot_id'] == slot_id]
        
        prophet_df = pd.DataFrame({
            'ds': df['timestamp'],
            'y': df['occupancy']
        })
        
        return prophet_df
    
    def full_pipeline(self, parking_lot_id: int, slot_id: str) -> Dict:
        """
        Run complete feature engineering pipeline
        
        Args:
            parking_lot_id: Parking lot ID
            slot_id: Slot ID
            
        Returns:
            Dictionary with processed data and metadata
        """
        # Load data
        df = self.load_occupancy_data(parking_lot_id, slot_id)
        
        if df.empty:
            return {'error': 'No data found'}
        
        # Create binary target
        df = self.create_binary_target(df)
        
        # Extract time features
        df = self.extract_time_features(df)
        
        # Calculate rolling features
        df = self.calculate_rolling_features(df)
        
        # Calculate lag features
        df = self.calculate_lag_features(df)
        
        # Prepare for Prophet
        prophet_df = self.prepare_for_prophet(df, slot_id)
        
        # Prepare for ML
        X, y = self.prepare_features_for_ml(df)
        
        return {
            'dataframe': df,
            'prophet_data': prophet_df,
            'X': X,
            'y': y,
            'slot_id': slot_id,
            'parking_lot_id': parking_lot_id,
            'total_samples': len(df),
            'date_range': (df['timestamp'].min(), df['timestamp'].max())
        }
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2) -> Tuple:
        """
        Split data into train and test sets (temporal split)
        
        Args:
            X: Feature array
            y: Target array
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Test on parking lot 1, slot A1
    result = engineer.full_pipeline(parking_lot_id=1, slot_id='A1')
    
    if 'error' not in result:
        print(f"Successfully processed {result['total_samples']} samples")
        print(f"Date range: {result['date_range']}")
        print(f"Feature shape: {result['X'].shape}")
        print(f"Target shape: {result['y'].shape}")
    else:
        print(f"Error: {result['error']}")
