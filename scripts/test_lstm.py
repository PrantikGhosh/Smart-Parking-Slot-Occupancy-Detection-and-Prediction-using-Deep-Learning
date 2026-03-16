import sys
import os
# Force CPU usage to avoid Metal/GPU hangs on Mac
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["tf_metal_device_name"] = "" 

from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import get_database
from models import FeatureEngineer, LSTMPredictor
from models.lstm_predictor import prepare_data_for_lstm

def main():
    print("Initializing Database...")
    db = get_database()
    
    # Get available parking lots
    lots = db.get_all_parking_lots()
    if not lots:
        print("Error: No parking lots found in database. Cannot run test.")
        return

    parking_lot = lots[0]
    parking_lot_id = parking_lot['id']
    print(f"Using Parking Lot: {parking_lot['name']} (ID: {parking_lot_id})")

    # Get available slots
    slots = db.get_slot_annotations(parking_lot_id)
    if not slots:
        print(f"Error: No slot annotations found for parking lot {parking_lot_id}.")
        return

    slot_id = slots[0]['slot_id']
    print(f"Using Slot ID: {slot_id}")

    # Feature Engineering
    print("\nRunning Feature Engineering Pipeline...")
    engineer = FeatureEngineer(db)
    
    # Load data
    df = engineer.load_occupancy_data(parking_lot_id, slot_id=slot_id)
    
    if df.empty:
        print("Error: No occupancy data found for this slot.")
        return
        
    print(f"Loaded {len(df)} records.")
    
    # Prepare features
    print("Extracting features...")
    df = engineer.create_binary_target(df)
    df = engineer.extract_time_features(df)
    df = engineer.calculate_rolling_features(df)
    df = engineer.calculate_lag_features(df)
    
    # Drop rows with NaNs created by lag/rolling
    df = df.dropna()
    
    if len(df) < 50:
         print(f"Warning: Only {len(df)} samples remaining after feature engineering. Training might differ from expected.")

    X, y = engineer.prepare_features_for_ml(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # LSTM Preparation
    print("\nPreparing LSTM Sequences...")
    sequence_length = 10 # Shorter sequence for testing if data is scarce
    if len(X) < sequence_length + 10:
        print("Error: Not enough data points for sequence creation.")
        return

    data = prepare_data_for_lstm(X, y, sequence_length=sequence_length)
    
    print(f"Train samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")

    # Training
    print("\nBuilding and Training LSTM Model...")
    predictor = LSTMPredictor(sequence_length=sequence_length, n_features=X.shape[1])
    predictor.build_model()
    
    history = predictor.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    print("\nTraining Complete.")
    print(f"Final Loss: {history['final_loss']:.4f}")
    print(f"Final Accuracy: {history['final_accuracy']:.4f}")

    # Evaluation
    print("\nEvaluating on Test Set...")
    metrics = predictor.evaluate(data['X_test'], data['y_test'])
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    main()