"""
LSTM Model for Parking Slot Availability Prediction
Deep learning model for temporal pattern recognition
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import pickle
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM model will not work.")


class LSTMPredictor:
    """LSTM-based time series predictor for parking occupancy"""
    
    def __init__(self, sequence_length: int = 30, n_features: int = 11):
        """
        Initialize LSTM predictor
        
        Args:
            sequence_length: Number of time steps in input sequence
            n_features: Number of features per time step
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.scaler = None
    
    def build_model(self, lstm_units: Tuple[int, int] = (64, 32), 
                   dropout_rate: float = 0.2) -> None:
        """
        Build LSTM model architecture
        
        Args:
            lstm_units: Tuple of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        self.model = Sequential([
            # First LSTM layer
            LSTM(lstm_units[0], 
                 return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(dropout_rate),
            
            # Second LSTM layer
            LSTM(lstm_units[1], return_sequences=False),
            Dropout(dropout_rate),
            
            # Dense layers
            Dense(16, activation='relu'),
            Dropout(dropout_rate),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = 50, batch_size: int = 32,
             save_path: str = None, verbose: int = 1) -> Dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training targets (samples,)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            save_path: Optional path to save best model
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    save_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True
                )
            )
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return {
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_accuracy': self.history.history['accuracy'][-1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            
        Returns:
            Predicted probabilities (samples,)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_next_steps(self, last_sequence: np.ndarray, 
                          n_steps: int = 5) -> np.ndarray:
        """
        Predict multiple future steps
        
        Args:
            last_sequence: Last known sequence (sequence_length, features)
            n_steps: Number of future steps to predict
            
        Returns:
            Array of predictions
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next step
            pred = self.predict(current_sequence.reshape(1, self.sequence_length, -1))[0]
            predictions.append(pred)
            
            # Update sequence (shift and add prediction)
            # Note: This is simplified - in practice, you'd update all features
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred  # Update first feature with prediction
        
        return np.array(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        predictions_binary = (predictions > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions_binary),
            'precision': precision_score(y_test, predictions_binary, zero_division=0),
            'recall': recall_score(y_test, predictions_binary, zero_division=0),
            'f1_score': f1_score(y_test, predictions_binary, zero_division=0),
            'predictions': predictions,
            'predictions_binary': predictions_binary
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)


def prepare_data_for_lstm(X: np.ndarray, y: np.ndarray, 
                         sequence_length: int = 30,
                         test_size: float = 0.2) -> Dict:
    """
    Helper function to prepare data for LSTM training
    
    Args:
        X: Feature array (samples, features)
        y: Target array (samples,)
        sequence_length: Length of input sequences
        test_size: Fraction for testing
        
    Returns:
        Dictionary with prepared datasets
    """
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Temporal split
    split_idx = int(len(X_sequences) * (1 - test_size))
    
    X_train = X_sequences[:split_idx]
    X_test = X_sequences[split_idx:]
    y_train = y_sequences[:split_idx]
    y_test = y_sequences[split_idx:]
    
    # Further split train into train/val
    val_split = int(len(X_train) * 0.8)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'sequence_length': sequence_length,
        'n_features': X.shape[1]
    }


if __name__ == "__main__":
    # Example usage
    if TENSORFLOW_AVAILABLE:
        # Generate dummy data for demonstration
        n_samples = 1000
        n_features = 11
        sequence_length = 30
        
        X_dummy = np.random.rand(n_samples, n_features)
        y_dummy = np.random.randint(0, 2, n_samples)
        
        # Prepare data
        data = prepare_data_for_lstm(X_dummy, y_dummy, sequence_length=sequence_length)
        
        # Create and train model
        predictor = LSTMPredictor(
            sequence_length=sequence_length,
            n_features=n_features
        )
        
        print("Building LSTM model...")
        predictor.build_model()
        print(predictor.get_summary())
        
        print("\nTraining model...")
        result = predictor.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            epochs=5,  # Small number for demo
            batch_size=32
        )
        
        print(f"\nTraining complete!")
        print(f"Final accuracy: {result['final_accuracy']:.4f}")
        
        # Evaluate
        metrics = predictor.evaluate(data['X_test'], data['y_test'])
        print(f"\nTest set metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
    else:
        print("TensorFlow not available. Cannot run LSTM example.")
