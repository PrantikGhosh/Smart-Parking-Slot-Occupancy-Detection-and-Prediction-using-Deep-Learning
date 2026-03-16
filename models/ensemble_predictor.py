"""
Ensemble Predictor combining LSTM and Prophet models
Provides weighted predictions for improved accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from .lstm_predictor import LSTMPredictor, TENSORFLOW_AVAILABLE
from .prophet_predictor import ProphetPredictor, PROPHET_AVAILABLE


class EnsemblePredictor:
    """Ensemble model combining LSTM and Prophet predictions"""
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.lstm_model = None
        self.prophet_model = None
        self.weights = {'lstm': 0.5, 'prophet': 0.5}  # Default equal weighting
        self.models_trained = {'lstm': False, 'prophet': False}
    
    def set_weights(self, lstm_weight: float, prophet_weight: float) -> None:
        """
        Set custom weights for ensemble
        
        Args:
            lstm_weight: Weight for LSTM predictions
            prophet_weight: Weight for Prophet predictions
        """
        total = lstm_weight + prophet_weight
        self.weights = {
            'lstm': lstm_weight / total,
            'prophet': prophet_weight / total
        }
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None,
                  **kwargs) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            **kwargs: Additional arguments for LSTM training
            
        Returns:
            Training result dictionary
        """
        if not TENSORFLOW_AVAILABLE:
            print("Warning: TensorFlow not available. Skipping LSTM training.")
            return {'error': 'TensorFlow not available'}
        
        self.lstm_model = LSTMPredictor(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2]
        )
        
        result = self.lstm_model.train(X_train, y_train, X_val, y_val, **kwargs)
        self.models_trained['lstm'] = True
        
        return result
    
    def train_prophet(self, df: pd.DataFrame, **kwargs) -> Dict:
        """
        Train Prophet model
        
        Args:
            df: DataFrame with 'ds' and 'y' columns
            **kwargs: Additional arguments for Prophet training
            
        Returns:
            Training result dictionary
        """
        if not PROPHET_AVAILABLE:
            print("Warning: Prophet not available. Skipping Prophet training.")
            return {'error': 'Prophet not available'}
        
        self.prophet_model = ProphetPredictor()
        self.prophet_model.build_model(**kwargs)
        
        result = self.prophet_model.train(df)
        self.models_trained['prophet'] = True
        
        return result
    
    def predict_lstm(self, X: np.ndarray) -> np.ndarray:
        """Get LSTM predictions"""
        if not self.models_trained['lstm']:
            raise ValueError("LSTM model not trained")
        
        return self.lstm_model.predict(X)
    
    def predict_prophet(self, periods: int, freq: str = 'H') -> pd.DataFrame:
        """Get Prophet predictions"""
        if not self.models_trained['prophet']:
            raise ValueError("Prophet model not trained")
        
        return self.prophet_model.predict(periods=periods, freq=freq)
    
    def predict_ensemble(self, lstm_predictions: np.ndarray,
                        prophet_predictions: np.ndarray) -> np.ndarray:
        """
        Combine predictions using weighted average
        
        Args:
            lstm_predictions: LSTM model predictions
            prophet_predictions: Prophet model predictions
            
        Returns:
            Ensemble predictions
        """
        ensemble = (
            self.weights['lstm'] * lstm_predictions +
            self.weights['prophet'] * prophet_predictions
        )
        
        return ensemble.clip(0, 1)
    
    def predict(self, X_lstm: Optional[np.ndarray] = None,
               prophet_periods: Optional[int] = None,
               prophet_freq: str = 'H') -> Dict:
        """
        Make ensemble predictions
        
        Args:
            X_lstm: Input for LSTM (if available)
            prophet_periods: Periods to forecast with Prophet
            prophet_freq: Frequency for Prophet
            
        Returns:
            Dictionary with all predictions
        """
        results = {}
        
        # LSTM predictions
        if X_lstm is not None and self.models_trained['lstm']:
            results['lstm'] = self.predict_lstm(X_lstm)
        
        # Prophet predictions
        if prophet_periods and self.models_trained['prophet']:
            forecast = self.predict_prophet(prophet_periods, prophet_freq)
            results['prophet'] = forecast['yhat'].values[-prophet_periods:]
            results['prophet_lower'] = forecast['yhat_lower'].values[-prophet_periods:]
            results['prophet_upper'] = forecast['yhat_upper'].values[-prophet_periods:]
        
        # Ensemble (if both available)
        if 'lstm' in results and 'prophet' in results:
            # Match lengths
            min_len = min(len(results['lstm']), len(results['prophet']))
            results['ensemble'] = self.predict_ensemble(
                results['lstm'][:min_len],
                results['prophet'][:min_len]
            )
        
        return results
    
    def evaluate_ensemble(self, y_true: np.ndarray,
                         lstm_pred: np.ndarray,
                         prophet_pred: np.ndarray) -> Dict:
        """
        Evaluate ensemble performance
        
        Args:
            y_true: True values
            lstm_pred: LSTM predictions
            prophet_pred: Prophet predictions
            
        Returns:
            Dictionary with metrics for all models
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Ensemble predictions
        ensemble_pred = self.predict_ensemble(lstm_pred, prophet_pred)
        
        # Binary conversion
        y_true_bin = (y_true > 0.5).astype(int)
        lstm_bin = (lstm_pred > 0.5).astype(int)
        prophet_bin = (prophet_pred > 0.5).astype(int)
        ensemble_bin = (ensemble_pred > 0.5).astype(int)
        
        def get_metrics(y_true, y_pred):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
        
        return {
            'lstm': get_metrics(y_true_bin, lstm_bin),
            'prophet': get_metrics(y_true_bin, prophet_bin),
            'ensemble': get_metrics(y_true_bin, ensemble_bin)
        }
    
    def optimize_weights(self, y_true: np.ndarray,
                        lstm_pred: np.ndarray,
                        prophet_pred: np.ndarray,
                        metric: str = 'f1_score') -> Dict:
        """
        Find optimal weights by grid search
        
        Args:
            y_true: True values
            lstm_pred: LSTM predictions
            prophet_pred: Prophet predictions
            metric: Metric to optimize ('accuracy', 'f1_score', etc.)
            
        Returns:
            Dictionary with optimal weights and score
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        metric_func = f1_score if metric == 'f1_score' else accuracy_score
        
        best_score = 0
        best_weights = (0.5, 0.5)
        
        y_true_bin = (y_true > 0.5).astype(int)
        
        # Grid search over weights
        for lstm_w in np.linspace(0, 1, 21):  # 0.0, 0.05, 0.10, ..., 1.0
            prophet_w = 1 - lstm_w
            
            ensemble = (lstm_w * lstm_pred + prophet_w * prophet_pred).clip(0, 1)
            ensemble_bin = (ensemble > 0.5).astype(int)
            
            score = metric_func(y_true_bin, ensemble_bin, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_weights = (lstm_w, prophet_w)
        
        # Set optimal weights
        self.set_weights(best_weights[0], best_weights[1])
        
        return {
            'optimal_lstm_weight': best_weights[0],
            'optimal_prophet_weight': best_weights[1],
            f'best_{metric}': best_score
        }
    
    def save_models(self, lstm_path: str, prophet_path: str) -> None:
        """Save both models"""
        if self.models_trained['lstm']:
            self.lstm_model.save_model(lstm_path)
        
        if self.models_trained['prophet']:
            self.prophet_model.save_model(prophet_path)
    
    def load_models(self, lstm_path: str, prophet_path: str) -> None:
        """Load both models"""
        # LSTM
        if Path(lstm_path).exists() and TENSORFLOW_AVAILABLE:
            self.lstm_model = LSTMPredictor()
            self.lstm_model.load_model(lstm_path)
            self.models_trained['lstm'] = True
        
        # Prophet
        if Path(prophet_path).exists() and PROPHET_AVAILABLE:
            self.prophet_model = ProphetPredictor()
            self.prophet_model.load_model(prophet_path)
            self.models_trained['prophet'] = True


if __name__ == "__main__":
    print("Ensemble Predictor - Combining LSTM and Prophet models")
    print(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
    print(f"Prophet available: {PROPHET_AVAILABLE}")
    
    if TENSORFLOW_AVAILABLE and PROPHET_AVAILABLE:
        print("\nBoth models available. Ensemble ready for training!")
    else:
        print("\nSome models unavailable. Install missing dependencies:")
        if not TENSORFLOW_AVAILABLE:
            print("  - pip install tensorflow")
        if not PROPHET_AVAILABLE:
            print("  - pip install prophet")
