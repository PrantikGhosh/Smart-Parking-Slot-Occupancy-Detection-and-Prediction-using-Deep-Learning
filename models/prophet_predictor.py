"""
Facebook Prophet Model for Parking Slot Availability Prediction
Handles seasonality and trend detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import pickle
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")


class ProphetPredictor:
    """Facebook Prophet predictor for parking occupancy"""
    
    def __init__(self):
        """Initialize Prophet predictor"""
        self.model = None
        self.forecast = None
    
    def build_model(self, daily_seasonality: bool = True,
                   weekly_seasonality: bool = True,
                   yearly_seasonality: bool = False,
                   seasonality_mode: str = 'multiplicative') -> None:
        """
        Build Prophet model with custom configuration
        
        Args:
            daily_seasonality: Enable daily patterns
            weekly_seasonality: Enable weekly patterns
            yearly_seasonality: Enable yearly patterns
            seasonality_mode: 'additive' or 'multiplicative'
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required. Install with: pip install prophet")
        
        # Suppress verbose logging
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)
        logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        
        self.model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.05,  # Flexibility in trend changes
            interval_width=0.95  # 95% confidence intervals
        )
        
        # Workaround for stan_backend attribute error
        # Set this immediately after model creation
        if not hasattr(self.model, 'stan_backend'):
            self.model.stan_backend = {'type': 'CMDSTANPY'}

    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data in Prophet format
        
        Args:
            df: DataFrame with 'timestamp' and 'occupancy' columns
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['timestamp']),
            'y': df['occupancy'].astype(float)
        })
        
        return prophet_df
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the Prophet model on prepared data
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Suppress output during training
        import logging
        logger = logging.getLogger('cmdstanpy')
        old_level = logger.level
        logger.setLevel(logging.ERROR)
        
        try:
            self.model.fit(df)
        finally:
            logger.setLevel(old_level)
    
    def predict(self, periods: int = 24, freq: str = '1H') -> pd.DataFrame:
        """
        Make future predictions
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency ('H' for hourly, 'D' for daily, etc.)
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        self.forecast = self.model.predict(future)
        
        return self.forecast
    
    def get_forecast_summary(self, n_future: int = None) -> pd.DataFrame:
        """
        Get forecast summary with key columns
        
        Args:
            n_future: Number of future points to return (None = all)
            
        Returns:
            DataFrame with ds, yhat, yhat_lower, yhat_upper
        """
        if self.forecast is None:
            raise ValueError("No forecast available. Call predict() first.")
        
        summary = self.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        if n_future:
            summary = summary.tail(n_future)
        
        # Clip predictions to [0, 1] range
        summary['yhat'] = summary['yhat'].clip(0, 1)
        summary['yhat_lower'] = summary['yhat_lower'].clip(0, 1)
        summary['yhat_upper'] = summary['yhat_upper'].clip(0, 1)
        
        return summary
    
    def cross_validate(self, df: pd.DataFrame, 
                      initial: str = '7 days',
                      period: str = '1 days',
                      horizon: str = '3 hours') -> pd.DataFrame:
        """
        Perform cross-validation
        
        Args:
            df: Training dataframe
            initial: Initial training period
            period: Spacing between cutoff dates
            horizon: Forecast horizon
            
        Returns:
            DataFrame with cross-validation results
        """
        from prophet.diagnostics import cross_validation, performance_metrics
        
        if self.model is None:
            self.build_model()
            self.train(df)
        
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        df_metrics = performance_metrics(df_cv)
        
        return df_metrics
    
    def get_components(self) -> Dict:
        """
        Get model components (trend, seasonality)
        
        Returns:
            Dictionary with component dataframes
        """
        if self.forecast is None:
            raise ValueError("No forecast available")
        
        components = {
            'trend': self.forecast[['ds', 'trend']],
            'weekly': self.forecast[['ds', 'weekly']] if 'weekly' in self.forecast.columns else None,
            'daily': self.forecast[['ds', 'daily']] if 'daily' in self.forecast.columns else None,
            'hourly': self.forecast[['ds', 'hourly']] if 'hourly' in self.forecast.columns else None
        }
        
        return {k: v for k, v in components.items() if v is not None}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
    
    def evaluate(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """
        Evaluate predictions
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
        r2 = r2_score(actual, predicted)
        
        # Binary classification metrics (threshold at 0.5)
        actual_binary = (actual > 0.5).astype(int)
        predicted_binary = (predicted > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        accuracy = accuracy_score(actual_binary, predicted_binary)
        precision = precision_score(actual_binary, predicted_binary, zero_division=0)
        recall = recall_score(actual_binary, predicted_binary, zero_division=0)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }


if __name__ == "__main__":
    # Example usage
    if PROPHET_AVAILABLE:
        # Generate dummy time series data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='H')
        
        # Simulate occupancy pattern
        hour_pattern = np.sin(2 * np.pi * dates.hour / 24) * 0.3 + 0.5
        day_pattern = np.sin(2 * np.pi * dates.dayofweek / 7) * 0.2
        noise = np.random.normal(0, 0.1, len(dates))
        
        occupancy = (hour_pattern + day_pattern + noise).clip(0, 1)
        
        df = pd.DataFrame({
            'ds': dates,
            'y': occupancy
        })
        
        # Train model
        predictor = ProphetPredictor()
        predictor.build_model()
        result = predictor.train(df)
        
        print(f"Trained on {result['training_samples']} samples")
        print(f"Date range: {result['date_range']}")
        
        # Make predictions
        forecast = predictor.predict(periods=24, freq='H')
        summary = predictor.get_forecast_summary(n_future=24)
        
        print(f"\nNext 24 hours forecast:")
        print(summary.head(10))
        
        # Evaluate on training data (just for demo)
        train_pred = forecast.head(len(df))['yhat']
        metrics = predictor.evaluate(df['y'], train_pred)
        
        print(f"\nMetrics:")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
    else:
        print("Prophet not available. Cannot run example.")
