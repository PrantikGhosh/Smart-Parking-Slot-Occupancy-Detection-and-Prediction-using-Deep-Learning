from ultralytics import YOLO
from ultralytics.data import YOLODataset
import yaml

# Load config
with open('data.yaml', 'r') as f:
    data_cfg = yaml.safe_load(f)

print("Testing YOLO dataset loading...")
print(f"Config: {data_cfg}")

# Try to load the dataset
try:
    # Initialize model
    model = YOLO('yolov8n.pt')
    
    # Try training for 1 epoch to see if data loads
    results = model.train(
        data='data.yaml',
        epochs=1,
        imgsz=640,
        batch=4,
        verbose=True,
        cache=False  # Don't use cache
    )
    
    print("\n✓ Dataset loaded successfully!")
    print(f"Results: {results}")
    
except Exception as e:
    print(f"\n✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
