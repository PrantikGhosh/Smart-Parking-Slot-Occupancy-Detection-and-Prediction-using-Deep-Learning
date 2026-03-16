from ultralytics import YOLO
import torch
from pathlib import Path
import os


def check_gpu():
    """Check if GPU is available and print device information"""
    print("=" * 60)
    print("GPU Availability Check")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ CUDA is not available")
        print("✗ Training will proceed on CPU (slower)")
        return False


def train_yolo_model():
    """Train YOLOv8 model on PKLot dataset"""
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Configuration
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    

    
    # Path setup
    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / 'models'
    
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    # Check if local model exists, else use name to download
    model_name = 'yolov8n.pt'
    model_path = models_dir / model_name
    model_to_load = str(model_path) if model_path.exists() else model_name
    
    data_yaml = root_dir / 'data' / 'dataset_yolo' / 'data.yaml'
    
    epochs = 100
    imgsz = 640
    batch = 16  # Adjust based on GPU memory
    device = 0 if has_gpu else 'cpu'
    
    print(f"Model: {model_to_load}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    
    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing YOLOv8 Model")
    print("=" * 60)
    
    model = YOLO(model_to_load)
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print("\nThis may take several hours depending on your hardware...")
    print("Training progress will be displayed below:\n")
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(root_dir / 'runs' / 'detect'),
            name='train',
            exist_ok=True,
            patience=20,  # Early stopping patience
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            cache=False,  # Set to True if you have enough RAM
            workers=4,  # Number of dataloader workers
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=True,  # Cosine learning rate scheduler
            close_mosaic=10,  # Disable mosaic augmentation for last 10 epochs
            amp=True,  # Automatic Mixed Precision training
            fraction=1.0,  # Use 100% of dataset
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,  # Validate during training
            plots=True  # Generate plots
        )
        
        print("\n" + "=" * 60)
        print("✓ Training Completed Successfully!")
        print("=" * 60)
        
        # Print results location
        save_dir = Path(results.save_dir) if hasattr(results, 'save_dir') else (root_dir / 'runs' / 'detect' / 'train')
        print(f"\n✓ Results saved to: {save_dir}")
        print(f"✓ Best weights: {save_dir / 'weights' / 'best.pt'}")
        print(f"✓ Last weights: {save_dir / 'weights' / 'last.pt'}")
        print(f"✓ Metrics: {save_dir / 'results.csv'}")
        print(f"✓ Confusion matrix: {save_dir / 'confusion_matrix.png'}")
        
        # Validate the best model
        print("\n" + "=" * 60)
        print("Validating Best Model")
        print("=" * 60)
        
        best_model = YOLO(str(save_dir / 'weights' / 'best.pt'))
        metrics = best_model.val()
        
        print(f"\n✓ Validation Results:")
        print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
        print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"  - Precision: {metrics.box.mp:.4f}")
        print(f"  - Recall: {metrics.box.mr:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("YOLOv8 Parking Lot Detection - Model Training")
    print("=" * 60)
    print()
    
    success = train_yolo_model()
    
    if success:
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Review training metrics in runs/detect/train/results.csv")
        print("2. Check confusion matrix and other plots")
        print("3. Test the model on new data")
    else:
        print("\n✗ Training failed. Please check the error messages above.")
