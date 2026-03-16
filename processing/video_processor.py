"""
Video Processing Pipeline
Processes annotated videos to extract occupancy time series data
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st


class VideoProcessor:
    """Processes videos to extract parking occupancy data"""
    
    def __init__(self, model_path: str, db):
        """
        Initialize video processor
        
        Args:
            model_path: Path to YOLO model
            db: ParkingDatabase instance
        """
        self.model = YOLO(model_path)
        self.db = db
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU score (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def match_detection_to_slot(self, detections, slot_bbox: Tuple,
                                iou_threshold: float = 0.1) -> Tuple[str, float]:
        """
        Match YOLO detection to annotated slot using car detection
        
        Args:
            detections: YOLO detection results
            slot_bbox: Slot bounding box (x1, y1, x2, y2)
            iou_threshold: Minimum IoU for match (lowered to 0.1 for better detection)
            
        Returns:
            Tuple of (status, confidence)
            status: 'empty', 'occupied', or 'unknown'
        """
        if detections is None or len(detections) == 0:
            return 'empty', 0.95  # No detections = empty
        
        boxes = detections[0].boxes
        if boxes is None or len(boxes) == 0:
            return 'empty', 0.95  # No detections = empty
        
        # COCO dataset vehicle classes
        VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        
        best_iou = 0.0
        best_confidence = 0.0
        vehicle_detected = False
        
        # Calculate slot center for additional checking
        slot_center_x = (slot_bbox[0] + slot_bbox[2]) / 2
        slot_center_y = (slot_bbox[1] + slot_bbox[3]) / 2
        
        for box in boxes:
            # Get detection box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection_bbox = (x1, y1, x2, y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Only check vehicle classes
            if cls not in VEHICLE_CLASSES:
                continue
            
            # Calculate IoU
            iou = self.calculate_iou(slot_bbox, detection_bbox)
            
            # Also check if detection center is inside slot
            det_center_x = (x1 + x2) / 2
            det_center_y = (y1 + y2) / 2
            
            center_in_slot = (
                slot_bbox[0] <= det_center_x <= slot_bbox[2] and
                slot_bbox[1] <= det_center_y <= slot_bbox[3]
            )
            
            # Check if slot center is inside detection
            slot_in_detection = (
                x1 <= slot_center_x <= x2 and
                y1 <= slot_center_y <= y2
            )
            
            # Match if IoU exceeds threshold OR centers overlap
            if iou > iou_threshold or center_in_slot or slot_in_detection:
                vehicle_detected = True
                if iou > best_iou or conf > best_confidence:
                    best_iou = max(best_iou, iou)
                    best_confidence = max(best_confidence, conf)
        
        if vehicle_detected:
            return 'occupied', best_confidence
        else:
            return 'empty', 0.95  # No vehicle detected = empty
    
    def process_video(self, video_path: str, parking_lot_id: int,
                     sampling_rate: int = 5, conf_threshold: float = 0.15,
                     progress_callback=None) -> Dict:
        """
        Process video to extract occupancy data
        
        Args:
            video_path: Path to video file
            parking_lot_id: Parking lot database ID
            sampling_rate: Extract data every N seconds
            conf_threshold: YOLO confidence threshold (lowered to 0.15 for better detection)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        # Get annotations
        annotations = self.db.get_slot_annotations(parking_lot_id)
        if not annotations:
            return {'error': 'No annotations found', 'success': False}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Failed to open video', 'success': False}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sampling_rate)
        
        # Statistics
        processed_frames = 0
        total_events = 0
        events_batch = []
        
        # Debug: Track detections
        detection_stats = {'total_detections': 0, 'vehicles_found': 0, 'frames_with_vehicles': 0}
        
        # Base timestamp (start of video as reference)
        base_time = datetime.now()
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process at intervals
                if frame_number % frame_interval == 0:
                    # Run YOLO detection with lower confidence
                    results = self.model(frame, conf=conf_threshold, verbose=False)
                    
                    # Debug: Count detections
                    if results and len(results) > 0 and results[0].boxes is not None:
                        detection_stats['total_detections'] += len(results[0].boxes)
                        vehicles = sum(1 for box in results[0].boxes if int(box.cls[0]) in {2, 3, 5, 7})
                        if vehicles > 0:
                            detection_stats['vehicles_found'] += vehicles
                            detection_stats['frames_with_vehicles'] += 1
                    
                    # Calculate timestamp
                    elapsed_seconds = frame_number / fps
                    timestamp = base_time + timedelta(seconds=elapsed_seconds)
                    
                    # Match each slot
                    for slot_ann in annotations:
                        slot_bbox = (slot_ann['x1'], slot_ann['y1'], 
                                   slot_ann['x2'], slot_ann['y2'])
                        
                        status, confidence = self.match_detection_to_slot(
                            results, slot_bbox, iou_threshold=0.05  # Even lower threshold
                        )
                        
                        # Store event
                        event = {
                            'parking_lot_id': parking_lot_id,
                            'slot_id': slot_ann['slot_id'],
                            'timestamp': timestamp,
                            'frame_number': frame_number,
                            'status': status,
                            'confidence': confidence,
                            'detected_class': f'vehicle-{status}' if status != 'empty' else 'no-vehicle'
                        }
                        
                        events_batch.append(event)
                        total_events += 1
                    
                    processed_frames += 1
                    
                    # Save batch to database (every 100 events)
                    if len(events_batch) >= 100:
                        self.db.save_occupancy_events_batch(events_batch)
                        events_batch = []
                    
                    # Update progress
                    if progress_callback:
                        progress = (frame_number / total_frames) * 100
                        status_msg = f"Processed {processed_frames} frames | Vehicles: {detection_stats['vehicles_found']}"
                        progress_callback(progress, status_msg)
                
                frame_number += 1
            
            # Save remaining events
            if events_batch:
                self.db.save_occupancy_events_batch(events_batch)
            
        finally:
            cap.release()
        
        return {
            'success': True,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_events': total_events,
            'slots_tracked': len(annotations),
            'sampling_rate': sampling_rate,
            'detection_stats': detection_stats  # Return debug stats
        }

        """
        Process video to extract occupancy data
        
        Args:
            video_path: Path to video file
            parking_lot_id: Parking lot database ID
            sampling_rate: Extract data every N seconds
            conf_threshold: YOLO confidence threshold
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with processing statistics
        """
        # Get annotations
        annotations = self.db.get_slot_annotations(parking_lot_id)
        if not annotations:
            return {'error': 'No annotations found'}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Failed to open video'}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * sampling_rate)
        
        # Statistics
        processed_frames = 0
        total_events = 0
        events_batch = []
        
        # Base timestamp (start of video as reference)
        base_time = datetime.now()
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process at intervals
                if frame_number % frame_interval == 0:
                    # Run YOLO detection
                    results = self.model(frame, conf=conf_threshold, verbose=False)
                    
                    # Calculate timestamp
                    elapsed_seconds = frame_number / fps
                    timestamp = base_time + timedelta(seconds=elapsed_seconds)
                    
                    # Match each slot
                    for slot_ann in annotations:
                        slot_bbox = (slot_ann['x1'], slot_ann['y1'], 
                                   slot_ann['x2'], slot_ann['y2'])
                        
                        status, confidence = self.match_detection_to_slot(
                            results, slot_bbox
                        )
                        
                        # Store event
                        event = {
                            'parking_lot_id': parking_lot_id,
                            'slot_id': slot_ann['slot_id'],
                            'timestamp': timestamp,
                            'frame_number': frame_number,
                            'status': status,
                            'confidence': confidence,
                            'detected_class': f'space-{status}' if status != 'unknown' else None
                        }
                        
                        events_batch.append(event)
                        total_events += 1
                    
                    processed_frames += 1
                    
                    # Save batch to database (every 10 frames to improve performance)
                    if len(events_batch) >= 100:
                        self.db.save_occupancy_events_batch(events_batch)
                        events_batch = []
                    
                    # Update progress
                    if progress_callback:
                        progress = (frame_number / total_frames) * 100
                        progress_callback(progress, processed_frames)
                
                frame_number += 1
            
            # Save remaining events
            if events_batch:
                self.db.save_occupancy_events_batch(events_batch)
            
        finally:
            cap.release()
        
        return {
            'success': True,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_events': total_events,
            'slots_tracked': len(annotations),
            'sampling_rate': sampling_rate
        }
    
    def generate_annotated_video(self, video_path: str, output_path: str,
                                parking_lot_id: int, fps_reduction: int = 5):
        """
        Generate video with detection annotations overlaid
        
        Args:
            video_path: Input video path
            output_path: Output video path
            parking_lot_id: Parking lot ID
            fps_reduction: Process every Nth frame for speed
        """
        annotations = self.db.get_slot_annotations(parking_lot_id)
        if not annotations:
            return False
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps/fps_reduction, (width, height))
        
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % fps_reduction == 0:
                    # Run detection
                    results = self.model(frame, verbose=False)
                    
                    # Draw annotations
                    for slot_ann in annotations:
                        slot_bbox = (slot_ann['x1'], slot_ann['y1'],
                                   slot_ann['x2'], slot_ann['y2'])
                        
                        status, conf = self.match_detection_to_slot(results, slot_bbox)
                        
                        # Color based on status
                        if status == 'empty':
                            color = (0, 255, 0)  # Green
                        elif status == 'occupied':
                            color = (0, 0, 255)  # Red
                        else:
                            color = (255, 165, 0)  # Orange
                        
                        # Draw box
                        cv2.rectangle(frame,
                                    (slot_ann['x1'], slot_ann['y1']),
                                    (slot_ann['x2'], slot_ann['y2']),
                                    color, 3)
                        
                        # Draw label
                        label = f"{slot_ann['slot_id']}: {status} ({conf:.2f})"
                        cv2.putText(frame, label,
                                  (slot_ann['x1'], slot_ann['y1'] - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.6, color, 2)
                    
                    out.write(frame)
                
                frame_number += 1
        
        finally:
            cap.release()
            out.release()
        
        return True
