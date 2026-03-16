from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Load the newly trained model
model_path = r"e:\Final Year Project\runs\detect\parking_test\weights\best.pt"
model = YOLO(model_path)

# Test on a sample image from the test set
test_image = r"e:\Final Year Project\PKLot\test\2012-09-11_15_53_00_jpg.rf.8282544a640a23df05bd245a9210e663.jpg"

print(f"Testing model: {model_path}")
print(f"On image: {test_image}")

# Run prediction
results = model(test_image, conf=0.25)

# Get detection counts by class
class_names = {0: 'spaces', 1: 'space-empty', 2: 'space-occupied'}
detections_by_class = {0: 0, 1: 0, 2: 0}

if results[0].boxes is not None and len(results[0].boxes) > 0:
    boxes = results[0].boxes
    print(f"\n✅ Total Detections: {len(boxes)}")
    
    for box in boxes:
        cls = int(box.cls[0])
        detections_by_class[cls] += 1
    
    print(f"\nDetection Summary:")
    for cls_id, count in detections_by_class.items():
        print(f"  {class_names[cls_id]}: {count}")
    
    # Calculate parking statistics
    empty = detections_by_class[1]
    occupied = detections_by_class[2]
    total_spots = empty + occupied
    
    if total_spots > 0:
        occupancy_rate = (occupied / total_spots) * 100
        print(f"\n📊 Parking Statistics:")
        print(f"  Total Parking Spots: {total_spots}")
        print(f"  Empty Spots: {empty}")
        print(f"  Occupied Spots: {occupied}")
        print(f"  Occupancy Rate: {occupancy_rate:.1f}%")
    
    # Save annotated image
    results[0].save(filename="test_detection_result.jpg")
    print(f"\n✅ Result saved to: test_detection_result.jpg")
    
else:
    print("\n❌ No detections found!")

print("\n" + "="*60)
print("Model is ready to use in the dashboard!")
print(f"Model path: {model_path}")
print("="*60)
