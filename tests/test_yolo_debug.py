"""
Test YOLO Detection on Video Frame
This script helps diagnose YOLO detection issues
"""

import cv2
from ultralytics import YOLO
import sys

def test_yolo_detection(video_path, model_path="yolov8n.pt"):
    """Test YOLO on first frame of video"""
    
    print(f"Testing YOLO detection on: {video_path}")
    print(f"Using model: {model_path}")
    print("-" * 50)
    
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Cannot open video!")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame!")
        return
    
    print(f"Frame size: {frame.shape}")
    
    # Run detection with different confidence levels
    for conf in [0.05, 0.10, 0.15, 0.25, 0.50]:
        print(f"\n=== Testing with confidence={conf} ===")
        results = model(frame, conf=conf, verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                print(f"Total detections: {len(boxes)}")
                
                # COCO classes
                class_names = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                    4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'
                }
                
                vehicle_classes = {2, 3, 5, 7}
                vehicles = []
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    class_name = class_names.get(cls, f'class_{cls}')
                    
                    if cls in vehicle_classes:
                        vehicles.append((class_name, conf_val, (x1, y1, x2, y2)))
                        print(f"  ✓ VEHICLE: {class_name} (conf={conf_val:.2f}) at ({x1},{y1},{x2},{y2})")
                    else:
                        print(f"    OTHER: {class_name} (conf={conf_val:.2f})")
                
                print(f"Vehicles found: {len(vehicles)}")
            else:
                print("No boxes detected")
        else:
            print("No results")
    
    cap.release()
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_yolo_debug.py <video_path> [model_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
    
    test_yolo_detection(video_path, model_path)
