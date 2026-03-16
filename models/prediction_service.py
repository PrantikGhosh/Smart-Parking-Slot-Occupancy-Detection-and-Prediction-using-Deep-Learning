"""
Prediction Service - Integration layer between database and prediction models
Handles data preparation, model selection, and prediction orchestration
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.statistical_models import EnsembleStatisticalPredictor
from models.feature_engineering import FeatureEngineer

# Try to import ML models
try:
    from models.lstm_predictor import LSTMPredictor, TENSORFLOW_AVAILABLE, prepare_data_for_lstm
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from models.prophet_predictor import ProphetPredictor, PROPHET_AVAILABLE
except ImportError:
    PROPHET_AVAILABLE = False


class PredictionService:
    """Main prediction service that orchestrates all models"""
    
    # Data requirements for each model type
    MIN_DATA_STATISTICAL = 10
    MIN_DATA_PROPHET = 50
    MIN_DATA_LSTM = 100
    
    def __init__(self, db):
        """
        Initialize prediction service
        
        Args:
            db: Database instance
        """
        self.db = db
        self.feature_engineer = FeatureEngineer()
        self.statistical_predictor = EnsembleStatisticalPredictor()
    
    def predict_future_availability(self, lot_id: int, slot_id: str, 
                                   minutes_ahead: int) -> Dict:
        """
        Main prediction method - tries all available models
        
        Args:
            lot_id: Parking lot ID
            slot_id: Slot ID
            minutes_ahead: How many minutes into the future to predict
            
        Returns:
            Dictionary with predictions and metadata
        """
        # Get historical data
        events = self.db.get_occupancy_events(lot_id, slot_id)
        
        if len(events) < 5:
            return {
                'success': False,
                'error': 'Insufficient data (need at least 5 events)',
                'data_points': len(events)
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(events)
        df['is_occupied'] = (df['status'] == 'occupied').astype(int)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate target time
        latest_time = df['timestamp'].max()
        target_time = latest_time + timedelta(minutes=minutes_ahead)
        
        # Check data volume and capabilities
        data_info = self._check_data_volume(len(df))
        
        results = {
            'success': True,
            'target_time': target_time.isoformat(),
            'minutes_ahead': minutes_ahead,
            'data_points': len(df),
            'data_info': data_info,
            'predictions': {},
            'recommended_prediction': None,
            'model_errors': {}  # Track why models failed
        }
        
        # Always try statistical models (work with limited data)
        if data_info['can_use_statistical']:
            stat_result = self._predict_with_statistical(df, target_time, minutes_ahead)
            if stat_result:
                results['predictions']['statistical'] = stat_result
            else:
                results['model_errors']['statistical'] = "Failed to generate prediction"
        
        # Try Prophet if enough data
        if data_info['can_use_prophet']:
            if not PROPHET_AVAILABLE:
                results['model_errors']['prophet'] = "Prophet library not installed"
                print(f"[DEBUG] Prophet library not available - install with: pip install prophet")
            else:
                print(f"[DEBUG] Attempting Prophet prediction with {len(df)} events...")
                prophet_result = self._predict_with_prophet(df, target_time, minutes_ahead)
                if prophet_result:
                    print(f"[DEBUG] Prophet prediction successful!")
                    results['predictions']['prophet'] = prophet_result
                else:
                    results['model_errors']['prophet'] = "Model training failed (check terminal for details)"
                    print(f"[DEBUG] Prophet prediction returned None")
        
        # Try LSTM if enough data
        if data_info['can_use_lstm']:
            if not TENSORFLOW_AVAILABLE:
                results['model_errors']['lstm'] = "TensorFlow library not installed"
                print(f"[DEBUG] TensorFlow not available - install with: pip install tensorflow")
            else:
                print(f"[DEBUG] Attempting LSTM prediction with {len(df)} events...")
                lstm_result = self._predict_with_lstm(df, target_time, minutes_ahead)
                if lstm_result:
                    print(f"[DEBUG] LSTM prediction successful!")
                    results['predictions']['lstm'] = lstm_result
                else:
                    results['model_errors']['lstm'] = "Model training failed (check terminal for details)"
                    print(f"[DEBUG] LSTM prediction returned None")
        
        # Determine recommended prediction (priority: LSTM > Prophet > Statistical)
        if 'lstm' in results['predictions']:
            results['recommended_prediction'] = results['predictions']['lstm']['probability_occupied']
            results['recommended_model'] = 'LSTM'
            results['confidence'] = results['predictions']['lstm'].get('confidence', 0.7)
        elif 'prophet' in results['predictions']:
            results['recommended_prediction'] = results['predictions']['prophet']['probability_occupied']
            results['recommended_model'] = 'Prophet'
            results['confidence'] = results['predictions']['prophet'].get('confidence', 0.65)
        elif 'statistical' in results['predictions']:
            results['recommended_prediction'] = results['predictions']['statistical']['probability_occupied']
            results['recommended_model'] = 'Statistical Ensemble'
            results['confidence'] = results['predictions']['statistical'].get('confidence', 0.6)
        
        return results
    
    def _check_data_volume(self, num_events: int) -> Dict:
        """
        Check what models can be used based on data volume
        
        Args:
            num_events: Number of historical events
            
        Returns:
            Dictionary with capability flags
        """
        return {
            'can_use_statistical': num_events >= self.MIN_DATA_STATISTICAL,
            'can_use_prophet': num_events >= self.MIN_DATA_PROPHET,
            'can_use_lstm': num_events >= self.MIN_DATA_LSTM,
            'recommended_models': self._get_recommended_models(num_events)
        }
    
    def _get_recommended_models(self, num_events: int) -> List[str]:
        """Get list of recommended models based on data volume"""
        models = []
        
        if num_events >= self.MIN_DATA_STATISTICAL:
            models.append('Statistical')
        if num_events >= self.MIN_DATA_PROPHET and PROPHET_AVAILABLE:
            models.append('Prophet')
        if num_events >= self.MIN_DATA_LSTM and TENSORFLOW_AVAILABLE:
            models.append('LSTM')
        
        return models
    
    def _predict_with_statistical(self, df: pd.DataFrame, target_time: datetime,
                                 minutes_ahead: int) -> Optional[Dict]:
        """
        Predict using statistical ensemble
        
        Args:
            df: Historical data DataFrame
            target_time: Time to predict for
            minutes_ahead: Minutes ahead
            
        Returns:
            Prediction dictionary or None
        """
        try:
            # Calculate steps ahead (assuming 5-min intervals on average)
            steps_ahead = max(1, minutes_ahead // 5)
            
            result = self.statistical_predictor.predict(df, target_time, steps_ahead)
            
            return {
                'probability_occupied': result['ensemble_prediction'],
                'probability_empty': 1 - result['ensemble_prediction'],
                'confidence': result['confidence'],
                'models_used': result['models_used'],
                'individual_predictions': result['individual_predictions']
            }
        except Exception as e:
            print(f"Statistical prediction error: {e}")
            return None
    
    def _predict_with_prophet(self, df: pd.DataFrame, target_time: datetime,
                            minutes_ahead: int) -> Optional[Dict]:
        """
        Predict using Prophet model
        
        Args:
            df: Historical data DataFrame
            target_time: Time to predict for
            minutes_ahead: Minutes ahead
            
        Returns:
            Prediction dictionary or None
        """
        if not PROPHET_AVAILABLE:
            return None
        
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df['timestamp'],
                'y': df['is_occupied'].astype(float)
            })
            
            # Create and train model
            predictor = ProphetPredictor()
            predictor.build_model(
                daily_seasonality=True,
                weekly_seasonality=len(df) > 14,  # Only if we have 2+ weeks
                yearly_seasonality=False
            )
            predictor.train(prophet_df)
            
            # Make prediction
            periods = max(1, minutes_ahead // 5)  # Predict in 5-min intervals
            forecast = predictor.predict(periods=periods, freq='5T')
            
            # Get prediction for target time
            last_prediction = forecast.iloc[-1]
            prob_occupied = float(np.clip(last_prediction['yhat'], 0, 1))
            
            # Calculate confidence from prediction interval width
            # Normalize interval width relative to full range [0, 1]
            interval_width = last_prediction['yhat_upper'] - last_prediction['yhat_lower']
            # Confidence is inversely proportional to interval width
            # Wide interval (1.0) = low confidence (0.5), Narrow interval (0.0) = high confidence (1.0)
            confidence = float(np.clip(1 - (interval_width / 2), 0.5, 1.0))
            
            return {
                'probability_occupied': prob_occupied,
                'probability_empty': 1 - prob_occupied,
                'confidence': confidence,
                'prediction_interval': {
                    'lower': float(np.clip(last_prediction['yhat_lower'], 0, 1)),
                    'upper': float(np.clip(last_prediction['yhat_upper'], 0, 1))
                }
            }
        except Exception as e:
            print(f"Prophet prediction error: {e}")
            return None
    
    def _predict_with_lstm(self, df: pd.DataFrame, target_time: datetime,
                          minutes_ahead: int) -> Optional[Dict]:
        """
        Predict using LSTM model
        
        Args:
            df: Historical data DataFrame
            target_time: Time to predict for
            minutes_ahead: Minutes ahead
            
        Returns:
            Prediction dictionary or None
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            # Need enough data for meaningful sequences
            if len(df) < 100:
                return None
            
            # Create features using FeatureEngineer pipeline
            df_features = df.copy()
            
            # Add binary target
            df_features['occupancy'] = df_features['is_occupied'].astype(float)
            
            # Extract time features
            df_features = self.feature_engineer.extract_time_features(df_features)
            
            # Calculate rolling features (simplified for speed)
            df_features = self.feature_engineer.calculate_rolling_features(
                df_features, 
                windows=['1H', '3H']
            )
            
            # Calculate lag features
            df_features = self.feature_engineer.calculate_lag_features(
                df_features,
                lags=[1, 2, 3]
            )
            
            # Prepare features
            X, y = self.feature_engineer.prepare_features_for_ml(df_features)
            
            if X is None or y is None or len(X) < 50:
                return None
            
            # Prepare sequences for LSTM
            sequence_length = min(30, len(df) // 4)
            data = prepare_data_for_lstm(X, y, sequence_length=sequence_length, test_size=0.2)
            
            if len(data['X_train']) < 10:
                return None
            
            # Build and train LSTM
            predictor = LSTMPredictor(
                sequence_length=sequence_length,
                n_features=X.shape[1]
            )
            predictor.build_model(lstm_units=(32, 16), dropout_rate=0.2)
            
            # Quick training (just a few epochs for real-time prediction)
            predictor.train(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                epochs=10,
                batch_size=16
            )
            
            # Get last sequence and predict
            last_sequence = X[-sequence_length:]
            prediction = predictor.predict(last_sequence.reshape(1, sequence_length, -1))[0]
            
            return {
                'probability_occupied': float(prediction),
                'probability_empty': float(1 - prediction),
                'confidence': 0.75,  # LSTM typically more confident
                'training_samples': len(data['X_train'])
            }
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None


def get_prediction_service(db):
    """
    Factory function to create prediction service
    
    Args:
        db: Database instance
        
    Returns:
        PredictionService instance
    """
    return PredictionService(db)


if __name__ == "__main__":
    print("Prediction Service Module")
    print(f"TensorFlow Available: {TENSORFLOW_AVAILABLE}")
    print(f"Prophet Available: {PROPHET_AVAILABLE}")
    print(f"\nData Requirements:")
    print(f"  Statistical Models: {PredictionService.MIN_DATA_STATISTICAL}+ events")
    print(f"  Prophet Model: {PredictionService.MIN_DATA_PROPHET}+ events")
    print(f"  LSTM Model: {PredictionService.MIN_DATA_LSTM}+ events")
