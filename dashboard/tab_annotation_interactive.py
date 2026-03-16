"""
Enhanced Mouse-Click Annotation System
4-point polygon selection, click-to-delete, and selection highlighting
"""

import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import tempfile
import os
import time

from database import get_database
from components import render_section_header, render_status_banner, Colors, VideoAnnotator
from processing import VideoProcessor
import cv2
import hashlib
from pathlib import Path
from typing import Optional, Dict


# ==================== Helper Functions ====================

def extract_frame_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Extract first frame from uploaded video without saving permanently.
    Creates a temp file just for extraction, then deletes it immediately.
    """
    try:
        # Create temp file just for frame extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Reset file pointer for future reads
        uploaded_file.seek(0)
        
        try:
            # Extract frame
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        finally:
            # Always delete temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        st.error(f"Error extracting frame: {str(e)}")
        return None


def get_video_metadata_from_upload(uploaded_file) -> Dict:
    """
    Get video metadata from uploaded file without saving permanently.
    Creates a temp file just for metadata extraction, then deletes it immediately.
    """
    try:
        # Create temp file just for metadata extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # Reset file pointer for future reads
        uploaded_file.seek(0)
        
        try:
            # Get metadata
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height
            }
        finally:
            # Always delete temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        st.error(f"Error getting video metadata: {str(e)}")
        return {}


def compute_hash_from_upload(uploaded_file) -> str:
    """
    Compute video hash from uploaded file without saving.
    """
    try:
        hasher = hashlib.sha256()
        
        # Read file content
        content = uploaded_file.read()
        hasher.update(content)
        
        # Reset file pointer for future reads
        uploaded_file.seek(0)
        
        return hasher.hexdigest()
    except Exception as e:
        st.error(f"Error computing hash: {str(e)}")
        return ""


def save_video_permanently(uploaded_file, video_hash: str) -> str:
    """
    Save uploaded video to permanent videos/ directory.
    Returns the permanent file path.
    """
    try:
        # Create videos directory if it doesn't exist
        videos_dir = Path("videos")
        videos_dir.mkdir(exist_ok=True)
        
        # Save with hash as filename
        video_path = videos_dir / f"{video_hash}.mp4"
        
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        return str(video_path)
    except Exception as e:
        st.error(f"Error saving video: {str(e)}")
        return ""



def point_in_box(px, py, box, tolerance=5):
    """Check if point is inside or near a box - reduced tolerance to avoid conflicts"""
    return (box['x1'] - tolerance <= px <= box['x2'] + tolerance and 
            box['y1'] - tolerance <= py <= box['y2'] + tolerance)


def draw_boxes_on_image(image_array, boxes, selected_slot_id=None, current_points=None):
    """Draw boxes with highlighting and current points"""
    pil_image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(pil_image)
    
    # Load font once at the beginning
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw existing boxes
    for box in boxes:
        is_selected = (selected_slot_id == box['slot_id'])
        
        # Choose color and width based on selection
        color = 'yellow' if is_selected else 'lime'
        width = 5 if is_selected else 3
        
        # Draw rectangle
        draw.rectangle(
            [box['x1'], box['y1'], box['x2'], box['y2']],
            outline=color,
            width=width
        )
        
        # Draw corners as dots if selected
        if is_selected:
            for x, y in [(box['x1'], box['y1']), (box['x2'], box['y1']), 
                         (box['x2'], box['y2']), (box['x1'], box['y2'])]:
                draw.ellipse([x-5, y-5, x+5, y+5], fill='yellow')
        
        # Draw label
        label = box['slot_id']
        bbox = draw.textbbox((box['x1'], box['y1'] - 25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((box['x1'], box['y1'] - 25), label, fill='black', font=font)
    
    # Draw current points being clicked
    if current_points:
        for i, (x, y) in enumerate(current_points):
            # Draw point
            draw.ellipse([x-6, y-6, x+6, y+6], fill='red', outline='white', width=2)
            # Draw point number
            draw.text((x+10, y-10), f"P{i+1}", fill='red', font=font)
        
        # Draw lines connecting points
        if len(current_points) > 1:
            for i in range(len(current_points) - 1):
                draw.line([current_points[i], current_points[i+1]], fill='red', width=2)
            
            # If we have 4 points, connect P4 back to P1 to complete the polygon
            if len(current_points) == 4:
                draw.line([current_points[3], current_points[0]], fill='red', width=2)
    
    return np.array(pil_image)



def tab_annotation_with_clicks():
    """Enhanced 4-point click annotation with deletion and highlighting"""
    
    db = get_database()
    render_section_header("Advanced Click Annotation", "🖱️")
    
    st.markdown("""
    ### 🎯 4-Point Annotation
    - **Click 4 times** to define slot corners (clockwise from top-left recommended)
    - **Click on slot** to highlight/select it
    - **Delete button** appears when slot is selected
    - **Undo** removes last added slot
    - **Clear All** removes everything
    """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'defined_slots' not in st.session_state:
        st.session_state.defined_slots = []
    if 'current_points' not in st.session_state:
        st.session_state.current_points = []
    if 'slot_counter' not in st.session_state:
        st.session_state.slot_counter = 1
    if 'selected_slot' not in st.session_state:
        st.session_state.selected_slot = None
    if 'last_click' not in st.session_state:
        st.session_state.last_click = None
    
    # Upload Video
    st.markdown("### 📤 Upload Video")
    uploaded_video = st.file_uploader("Choose parking lot video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_video is None:
        st.info("👆 Upload a video to start")
        return
    
    # Extract video info WITHOUT saving to temp file
    # Video stays in memory during annotation
    first_frame = extract_frame_from_upload(uploaded_video)
    metadata = get_video_metadata_from_upload(uploaded_video)
    video_hash = compute_hash_from_upload(uploaded_video)
    
    if first_frame is None:
        st.error("Failed to extract frame")
        return
    
    # Check if already processed
    existing_lot = db.get_parking_lot_by_hash(video_hash)
    if existing_lot:
        st.success(f"✅ Already processed as '{existing_lot['name']}'")
        
        # Show stats
        stats = db.get_statistics(existing_lot['id'])
        annotations = db.get_slot_annotations(existing_lot['id'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Slots Defined", len(annotations))
        with col2:
            st.metric("Occupancy Events", f"{stats.get('total_events', 0):,}")
        with col3:
            st.metric("Video Duration", f"{stats.get('video_duration', 0):.0f}s")
        
        
        # Check if needs reprocessing
        if stats.get('total_events', 0) == 0:
            st.warning("⚠️ This video has no occupancy data. It may have been processed with errors.")
        
        st.markdown("---")
        
        # Delete parking lot option  
        with st.expander("🗑️ Delete Parking Lot", expanded=False):
            st.error("⚠️ **Danger Zone**: This will permanently delete ALL data for this parking lot.")
            st.markdown("""
            **This will remove:**
            - All slot annotations
            - All occupancy events
            - All predictions
            - Parking lot metadata
            
            You'll need to re-upload and re-annotate the video from scratch.
            """)
            
            if st.button("⚠️ DELETE EVERYTHING", type="secondary"):
                # Delete all data
                with db.get_connection() as conn:
                    conn.execute("DELETE FROM predictions WHERE parking_lot_id = ?", (existing_lot['id'],))
                    conn.execute("DELETE FROM occupancy_events WHERE parking_lot_id = ?", (existing_lot['id'],))
                    conn.execute("DELETE FROM slot_annotations WHERE parking_lot_id = ?", (existing_lot['id'],))
                    conn.execute("DELETE FROM parking_lots WHERE id = ?", (existing_lot['id'],))
                    conn.commit()
                
                st.success("✅ Parking lot deleted! Upload a new video to start fresh.")
                time.sleep(2)
                st.rerun()
        
        st.markdown("---")
        
        st.markdown("---")
        
        # Check if needs reprocessing
        if stats.get('total_events', 0) == 0:
            st.warning("⚠️ This video has no occupancy data. It may have been processed with errors.")
        elif stats.get('total_events', 0) > 0:
            st.success(f"✅ Video processed successfully with {stats['total_events']:,} occupancy events")
        
        # Always show reprocess option
        st.markdown("### 🔄 Reprocess Video with YOLO")
        st.info("💡 You can reprocess this video to update occupancy data with a different model or settings.")
        
        # Model selection
        st.markdown("#### 🎯 Select YOLO Model")
        
        # Auto-detect custom models from models/ directory
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        models_dir = current_dir / "models"
        
        model_options = {}
        
        # Add all .pt files found in models/
        if models_dir.exists():
            for model_file in sorted(models_dir.glob("*.pt")):
                # Use filename as label, absolute path as value
                label = f"🎯 {model_file.name}"
                model_options[label] = str(model_file)
        
        # Smart default
        default_index = 0
        options_list = list(model_options.keys())
        
        # Prefer yolov8_parking_custom if available
        for i, opt in enumerate(options_list):
            if "yolov8_parking_custom" in opt:
                default_index = i
                break

        selected_model_name = st.selectbox(
            "Choose Model",
            options=options_list,
            index=default_index,
            key="reprocess_model_select",
            help="Select a standard YOLO model or one of your custom trained models"
        )
        
        if selected_model_name:
            model_path = model_options[selected_model_name]
        else:
            model_path = None
        
        st.caption(f"📁 Using: `{model_path}`")
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="Lower = more detections (but more false positives)"
        )
        
        if st.button("🚀 Reprocess Video with YOLO", type="primary"):
                # Delete old events
                db.delete_occupancy_events(existing_lot['id'])
                
                st.info("🎬 Starting video processing...")
                
                # Use the stored video path from database (no new temp file needed!)
                stored_video_path = existing_lot['video_path']
                
                # Verify file exists
                if not os.path.exists(stored_video_path):
                    st.error(f"❌ Video file not found: {stored_video_path}")
                    st.warning("The video may have been moved or deleted. Please re-upload.")
                    return
                
                st.info(f"📁 Using stored video: {stored_video_path}")
                
                # Process with YOLO
                processor = VideoProcessor(model_path, db)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                result = processor.process_video(
                    video_path=stored_video_path,
                    parking_lot_id=existing_lot['id'],
                    sampling_rate=5,
                    conf_threshold=conf_threshold,
                    progress_callback=lambda p, s: (progress_bar.progress(p/100), status_text.text(s))
                )
                
                # Display detection stats
                if 'detection_stats' in result:
                    st.markdown("### 🔍 Detection Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        st.metric("Total YOLO Detections", result['detection_stats']['total_detections'])
                    with stats_col2:
                        st.metric("Vehicles Found", result['detection_stats']['vehicles_found'])
                    with stats_col3:
                        st.metric("Frames with Vehicles", result['detection_stats']['frames_with_vehicles'])
                    
                    if result['detection_stats']['vehicles_found'] == 0:
                        st.error("⚠️ **No vehicles detected!** YOLO may not be finding cars. Try:")
                        st.markdown("""
                        - Check if video shows cars clearly
                        - Use a different YOLO model
                        - Lower confidence threshold further
                        """)
                
                if result.get('success'):
                    st.success(f"""
                    ✅ **Reprocessing Complete!**
                    
                    - Events: {result.get('total_events', 0):,}
                    - Frames: {result.get('processed_frames', 0):,}
                    
                    **Go to Tab 3 for predictions!** 🔮
                    """)
                    
                    st.rerun()
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown')}")
        else:
            st.info("✅ Video has occupancy data. Go to Tab 3 for predictions!")
        
        return
    
    # Video details
    col1, col2 = st.columns(2)
    with col1:
        video_name = st.text_input("Parking Lot Name", value=uploaded_video.name.split('.')[0])
    with col2:
        camera_angle = st.selectbox("Camera Angle", ["top_down", "angled", "side_view", "other"])
    
    # Slot settings
    col1, col2 = st.columns(2)
    with col1:
        slot_prefix = st.selectbox("Slot Prefix", ["A", "B", "C", "D", "E", "F"])
    with col2:
        slot_type = st.selectbox("Slot Type", ["regular", "handicap", "reserved", "other"])
    
    st.markdown("---")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.defined_slots = []
            st.session_state.current_points = []
            st.session_state.slot_counter = 1
            st.session_state.selected_slot = None
            st.session_state.last_click = None
            st.rerun()
    
    with col2:
        if st.button("⏮️ Undo Last", use_container_width=True):
            if st.session_state.defined_slots:
                st.session_state.defined_slots.pop()
                st.session_state.slot_counter -= 1
                st.session_state.selected_slot = None
                st.rerun()
    
    with col3:
        if st.button("↩️ Reset Points", use_container_width=True):
            st.session_state.current_points = []
            st.session_state.last_click = None
            st.rerun()
    
    with col4:
        # Delete selected slot
        if st.session_state.selected_slot:
            if st.button(f"❌ Delete {st.session_state.selected_slot}", type="primary", use_container_width=True):
                st.session_state.defined_slots = [
                    s for s in st.session_state.defined_slots 
                    if s['slot_id'] != st.session_state.selected_slot
                ]
                st.session_state.selected_slot = None
                st.success(f"Deleted slot!")
                st.rerun()
    
    st.markdown("---")
    
    # Instructions
    st.markdown("### 🖱️ Click on Image")
    
    points_needed = 4 - len(st.session_state.current_points)
    if points_needed > 0:
        st.info(f"👆 Click **{points_needed} more point(s)** to complete slot (Total: 4 points)")
    else:
        st.success("✅ 4 points collected! Click 'Create Slot' below")
    
    # Draw image with annotations
    annotated_frame = draw_boxes_on_image(
        first_frame.copy(),
        st.session_state.defined_slots,
        st.session_state.selected_slot,
        st.session_state.current_points
    )
    
    # Display clickable image
    clicked_coords = streamlit_image_coordinates(
        Image.fromarray(annotated_frame),
        key="image_click"
    )
    
    # Process clicks - only if it's a NEW click
    if clicked_coords is not None:
        x, y = clicked_coords["x"], clicked_coords["y"]
        current_click = (x, y)
        
        # Only process if this is a different click than last time
        if st.session_state.last_click != current_click:
            st.session_state.last_click = current_click
            
            # If we're currently defining a new slot (have 1-3 points), prioritize adding points
            if 0 < len(st.session_state.current_points) < 4:
                # Add the point regardless of proximity to existing slots
                st.session_state.current_points.append((x, y))
                st.success(f"✅ Point {len(st.session_state.current_points)}/4 at ({x}, {y})")
                st.session_state.selected_slot = None
                st.rerun()
            
            # Otherwise, check for slot selection or start new slot
            else:
                # Check if clicked on existing slot (for selection/highlighting)
                clicked_on_slot = False
                for slot in st.session_state.defined_slots:
                    if point_in_box(x, y, slot):
                        st.session_state.selected_slot = slot['slot_id']
                        st.success(f"✅ Selected slot: {slot['slot_id']}")
                        clicked_on_slot = True
                        st.rerun()
                        break
                
                # If not clicked on slot and we have 0 points, start new slot
                if not clicked_on_slot and len(st.session_state.current_points) == 0:
                    st.session_state.current_points.append((x, y))
                    st.success(f"✅ Point 1/4 at ({x}, {y})")
                    st.session_state.selected_slot = None
                    st.rerun()

    
    # Show current points
    if st.session_state.current_points:
        st.markdown("#### Current Points:")
        cols = st.columns(4)
        for i, (x, y) in enumerate(st.session_state.current_points):
            with cols[i]:
                st.write(f"**P{i+1}:** ({x}, {y})")
        
        # Create slot button
        if len(st.session_state.current_points) == 4:
            if st.button("✅ Create Slot from 4 Points", type="primary", use_container_width=True):
                # Calculate bounding box from 4 points
                xs = [p[0] for p in st.session_state.current_points]
                ys = [p[1] for p in st.session_state.current_points]
                
                new_slot = {
                    'slot_id': f"{slot_prefix}{st.session_state.slot_counter}",
                    'x1': int(min(xs)),
                    'y1': int(min(ys)),
                    'x2': int(max(xs)),
                    'y2': int(max(ys)),
                    'slot_type': slot_type,
                    'points': st.session_state.current_points  # Store original 4 points
                }
                
                st.session_state.defined_slots.append(new_slot)
                st.session_state.slot_counter += 1
                st.session_state.current_points = []
                
                st.balloons()
                st.success(f"🎉 Created {new_slot['slot_id']}!")
                st.rerun()
    
    # Show defined slots
    if st.session_state.defined_slots:
        st.markdown("---")
        st.markdown(f"### ✅ Defined Slots ({len(st.session_state.defined_slots)})")
        
        cols = st.columns(6)
        for idx, slot in enumerate(st.session_state.defined_slots):
            with cols[idx % 6]:
                is_selected = (st.session_state.selected_slot == slot['slot_id'])
                bg_color = "#f59e0b" if is_selected else "#7c3aed"
                
                st.markdown(f"""
                <div style="background: {bg_color};
                     color: white; padding: 10px; border-radius: 10px; 
                     text-align: center; margin: 4px;
                     border: {'3px solid yellow' if is_selected else 'none'};">
                    <strong>{slot['slot_id']}</strong><br>
                    <small>{slot['slot_type']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Slots Summary - Better Layout
        st.markdown("---")
        
        # Use 3 columns for better space utilization
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.metric("📊 Total Slots", len(st.session_state.defined_slots), delta=None, help="Number of parking slots annotated")
        
        with col2:
            # Count by slot type
            slot_types = {}
            for slot in st.session_state.defined_slots:
                slot_type = slot.get('slot_type', 'regular')
                slot_types[slot_type] = slot_types.get(slot_type, 0) + 1
            most_common = max(slot_types.items(), key=lambda x: x[1]) if slot_types else ('N/A', 0)
            st.metric("🏷️ Slot Types", f"{len(slot_types)} types", delta=f"{most_common[0]}: {most_common[1]}", help="Distribution of slot types")
        
        with col3:
            # Compact download button
            json_data = json.dumps(st.session_state.defined_slots, indent=2)
            st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)  # Align with metrics
            st.download_button(
                "💾",
                data=json_data,
                file_name=f"{video_name}_slots.json",
                mime="application/json",
                use_container_width=True,
                help="Download slot annotations as JSON"
            )

        
        st.markdown("---")
        
        # Process button
        st.markdown("### 🤖 Process with YOLO")
        st.info("YOLO will automatically detect cars in annotated slots")
        
        # Model selection
        st.markdown("#### 🎯 Select YOLO Model")
        
        # Auto-detect custom models from models/ directory
        from pathlib import Path
        current_dir = Path(__file__).parent.parent
        models_dir = current_dir / "models"
        
        model_options = {}
        
        # Add all .pt files found in models/
        if models_dir.exists():
            for model_file in sorted(models_dir.glob("*.pt")):
                label = f"🎯 {model_file.name}"
                model_options[label] = str(model_file)
        
        # Smart default
        default_index = 0
        options_list = list(model_options.keys())
        
        for i, opt in enumerate(options_list):
            if "yolov8_parking_custom" in opt:
                default_index = i
                break
            
        selected_model_name = st.selectbox(
            "Choose Model",
            options=options_list,
            index=default_index,
            key="new_process_model_select",
            help="Select a standard YOLO model or one of your custom trained models"
        )
        
        if selected_model_name:
            model_path = model_options[selected_model_name]
        else:
            model_path = None
        
        st.caption(f"📁 Using: `{model_path}`")
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="Lower = more detections (but more false positives)"
        )
        
        if st.button("🚀 Save & Process Video", type="primary", use_container_width=True):
            # NOW we save the video permanently (only when processing!)
            st.info("💾 Saving video to permanent storage...")
            permanent_video_path = save_video_permanently(uploaded_video, video_hash)
            
            if not permanent_video_path:
                st.error("❌ Failed to save video. Please try again.")
                return
            
            st.success(f"✅ Video saved to: {permanent_video_path}")
            
            # Save to database with permanent path
            parking_lot_id = db.save_parking_lot(
                name=video_name,
                video_path=permanent_video_path,  # Use permanent path!
                video_hash=video_hash,
                camera_angle=camera_angle,
                fps=metadata['fps'],
                duration=metadata['duration'],
                width=metadata['width'],
                height=metadata['height']
            )
            
            db.save_slot_annotations_batch(parking_lot_id, st.session_state.defined_slots)
            db.update_parking_lot_slots(parking_lot_id, len(st.session_state.defined_slots))
            
            st.success(f"✅ Saved {len(st.session_state.defined_slots)} slots")
            
            # Process video
            st.markdown("#### 🎬 Processing...")
            
            # Initialize VideoProcessor with model_path and db
            processor = VideoProcessor(model_path, db)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result = processor.process_video(
                video_path=permanent_video_path,  # Use permanent path!
                parking_lot_id=parking_lot_id,
                sampling_rate=5,
                conf_threshold=conf_threshold,  # Use selected value
                progress_callback=lambda p, s: (progress_bar.progress(p/100), status_text.text(s))
            )
            
            if result.get('success'):
                st.success(f"""
                ✅ **Complete!**
                
                - Events: {result.get('total_events', 0):,}
                - Frames: {result.get('processed_frames', 0):,}
                - Slots Tracked: {result.get('slots_tracked', 0)}
                
                **Go to Tab 3 to view predictions!** 🔮
                """)
            elif 'error' in result:
                st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            else:
                st.warning("Processing completed but success flag not set")


if __name__ == "__main__":
    tab_annotation_with_clicks()
