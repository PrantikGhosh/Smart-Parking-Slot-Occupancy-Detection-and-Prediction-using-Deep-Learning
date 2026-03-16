"""
Statistical Models for Parking Slot Availability Prediction
Simple, effective models that work with limited data
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta


class MovingAveragePredictor:
    """Simple moving average predictor"""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize moving average predictor
        
        Args:
            window_size: Number of recent observations to average
        """
        self.window_size = window_size
    
    def predict(self, occupancy_history: np.ndarray) -> float:
        """
        Predict occupancy probability using moving average
        
        Args:
            occupancy_history: Array of recent occupancy values (1=occupied, 0=empty)
            
        Returns:
            Probability of being occupied (0-1)
        """
        if len(occupancy_history) < 1:
            return 0.5  # No data, assume 50/50
        
        # Use last N observations
        recent = occupancy_history[-self.window_size:]
        return np.mean(recent)


class ExponentialSmoothingPredictor:
    """Exponential smoothing predictor - recent data matters more"""
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize exponential smoothing predictor
        
        Args:
            alpha: Smoothing parameter (0-1). Higher = more weight on recent data
        """
        self.alpha = alpha
    
    def predict(self, occupancy_history: np.ndarray) -> float:
        """
        Predict using exponential smoothing
        
        Args:
            occupancy_history: Array of occupancy values
            
        Returns:
            Probability of being occupied
        """
        if len(occupancy_history) < 1:
            return 0.5
        
        # Apply exponential smoothing
        smoothed = occupancy_history[0]
        for value in occupancy_history[1:]:
            smoothed = self.alpha * value + (1 - self.alpha) * smoothed
        
        return smoothed


class TimeOfDayPredictor:
    """Enhanced time-of-day pattern predictor"""
    
    def __init__(self, time_window_minutes: int = 30):
        """
        Initialize time-of-day predictor
        
        Args:
            time_window_minutes: Time window for finding similar times
        """
        self.time_window_minutes = time_window_minutes
    
    def predict(self, df: pd.DataFrame, target_time: datetime) -> Optional[float]:
        """
        Predict based on historical patterns at similar times
        
        Args:
            df: DataFrame with 'timestamp' and 'is_occupied' columns
            target_time: Time to predict for
            
        Returns:
            Probability of being occupied, or None if insufficient data
        """
        if len(df) < 10:
            return None
        
        # Convert timestamps to datetime if needed
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Target hour and day of week
        target_hour = target_time.hour
        target_dow = target_time.weekday()
        
        # Extract hour and day of week from historical data
        df['hour'] = df['datetime'].dt.hour
        df['dow'] = df['datetime'].dt.weekday()
        
        # Find similar times (same hour +/- 1, same day of week)
        similar_times = df[
            (df['hour'].between(target_hour - 1, target_hour + 1)) &
            (df['dow'] == target_dow)
        ]
        
        # Fallback: if not enough data, just use same hour on any day
        if len(similar_times) < 5:
            similar_times = df[df['hour'].between(target_hour - 1, target_hour + 1)]
        
        # Still not enough? Use all data
        if len(similar_times) < 3:
            similar_times = df
        
        # Calculate probability
        if len(similar_times) > 0:
            return similar_times['is_occupied'].mean()
        
        return None


class TrendPredictor:
    """Simple trend-based predictor using recent slope"""
    
    def __init__(self, window_size: int = 20):
        """
        Initialize trend predictor
        
        Args:
            window_size: Number of recent observations to calculate trend
        """
        self.window_size = window_size
    
    def predict(self, occupancy_history: np.ndarray, steps_ahead: int = 1) -> float:
        """
        Predict based on recent trend
        
        Args:
            occupancy_history: Array of occupancy values
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            Probability of being occupied
        """
        if len(occupancy_history) < 5:
            return 0.5
        
        # Use recent window
        recent = occupancy_history[-self.window_size:]
        
        # Calculate simple linear trend
        x = np.arange(len(recent))
        y = recent
        
        # Simple linear regression (slope)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0
        
        # Predict future value
        current_value = recent[-1]
        predicted = current_value + slope * steps_ahead
        
        # Clip to [0, 1]
        return np.clip(predicted, 0, 1)


class EnsembleStatisticalPredictor:
    """Ensemble of statistical models with automatic weighting"""
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.moving_avg = MovingAveragePredictor(window_size=10)
        self.exp_smoothing = ExponentialSmoothingPredictor(alpha=0.3)
        self.time_of_day = TimeOfDayPredictor(time_window_minutes=30)
        self.trend = TrendPredictor(window_size=20)
    
    def predict(self, df: pd.DataFrame, target_time: datetime, 
                steps_ahead: int = 1) -> Dict:
        """
        Predict using ensemble of models
        
        Args:
            df: DataFrame with timestamp and occupancy data
            target_time: Time to predict for
            steps_ahead: Steps ahead (for trend model)
            
        Returns:
            Dictionary with predictions from all models and ensemble result
        """
        if len(df) < 5:
            return {
                'ensemble_prediction': 0.5,
                'confidence': 0.0,
                'models_used': [],
                'individual_predictions': {}
            }
        
        occupancy_array = df['is_occupied'].values
        predictions = {}
        
        # Moving average
        try:
            predictions['moving_average'] = self.moving_avg.predict(occupancy_array)
        except:
            pass
        
        # Exponential smoothing
        try:
            predictions['exp_smoothing'] = self.exp_smoothing.predict(occupancy_array)
        except:
            pass
        
        # Time of day
        try:
            tod_pred = self.time_of_day.predict(df, target_time)
            if tod_pred is not None:
                predictions['time_of_day'] = tod_pred
        except:
            pass
        
        # Trend
        try:
            predictions['trend'] = self.trend.predict(occupancy_array, steps_ahead)
        except:
            pass
        
        # Calculate ensemble prediction (simple average)
        if predictions:
            ensemble_pred = np.mean(list(predictions.values()))
            
            # Calculate confidence based on agreement between models
            pred_values = list(predictions.values())
            if len(pred_values) > 1:
                std = np.std(pred_values)
                confidence = 1 - std  # Lower std = higher confidence
            else:
                confidence = 0.5
        else:
            ensemble_pred = 0.5
            confidence = 0.0
        
        return {
            'ensemble_prediction': float(ensemble_pred),
            'confidence': float(np.clip(confidence, 0, 1)),
            'models_used': list(predictions.keys()),
            'individual_predictions': predictions,
            'data_points': len(df)
        }


def predict_with_statistics(df: pd.DataFrame, target_time: datetime,
                           steps_ahead: int = 1) -> Dict:
    """
    Convenience function for statistical prediction
    
    Args:
        df: DataFrame with timestamp and is_occupied columns
        target_time: Time to predict for
        steps_ahead: Number of steps ahead
        
    Returns:
        Prediction dictionary
    """
    ensemble = EnsembleStatisticalPredictor()
    return ensemble.predict(df, target_time, steps_ahead)


if __name__ == "__main__":
    # Example usage
    print("Testing Statistical Models...")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    
    # Simulate occupancy pattern (higher during day, lower at night)
    hour_pattern = np.sin(2 * np.pi * dates.hour / 24) * 0.3 + 0.5
    noise = np.random.normal(0, 0.1, len(dates))
    occupancy = (hour_pattern + noise).clip(0, 1)
    occupancy_binary = (occupancy > 0.5).astype(int)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'is_occupied': occupancy_binary
    })
    
    # Test prediction
    target = dates[-1] + timedelta(hours=1)
    result = predict_with_statistics(df, target, steps_ahead=1)
    
    print(f"\nPrediction Results:")
    print(f"Ensemble Prediction: {result['ensemble_prediction']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Models Used: {', '.join(result['models_used'])}")
    print(f"Individual Predictions:")
    for model, pred in result['individual_predictions'].items():
        print(f"  {model}: {pred:.3f}")
