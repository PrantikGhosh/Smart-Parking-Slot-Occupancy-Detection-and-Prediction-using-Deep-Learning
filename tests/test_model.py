from ultralytics import YOLO
from PIL import Image

# Load the latest trained model
model_path = r"e:\Final Year Project\runs\detect\train3\weights\best.pt"
model = YOLO(model_path)

# Test image
test_image = r"e:\Final Year Project\PKLot\test\2012-09-11_15_53_00_jpg.rf.8282544a640a23df05bd245a9210e663.jpg"

# Run prediction
results = model(test_image, conf=0.25)

# Print results
print(f"Model loaded from: {model_path}")
print(f"Testing on: {test_image}")
print(f"\nDetections found: {len(results[0].boxes) if results[0].boxes is not None else 0}")

if results[0].boxes is not None and len(results[0].boxes) > 0:
    boxes = results[0].boxes
    print(f"\nDetection details:")
    for i, box in enumerate(boxes[:10]):  # Show first 10 detections
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_names = {0: 'Space', 1: 'Empty', 2: 'Occupied'}
        print(f"  {i+1}. Class: {class_names.get(cls, 'Unknown')} ({cls}) - Confidence: {conf:.3f}")
else:
    print("\nNo detections found!")

# Save results
results[0].save(filename="test_detection_result.jpg")
print(f"\nResult saved to: test_detection_result.jpg")
