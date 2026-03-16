"""
Simplified Video Annotation Interface
Click-based coordinate input system without canvas dependency
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import json


class VideoAnnotator:
    """Handles video annotation interface with simple coordinate input"""
    
    def __init__(self, db):
        """Initialize video annotator"""
        self.db = db
    
    @staticmethod
    def extract_first_frame(video_path: str) -> Optional[np.ndarray]:
        """Extract first frame from video"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            st.error(f"Error extracting frame: {str(e)}")
            return None
    
    @staticmethod
    def get_video_metadata(video_path: str) -> Dict:
        """Get video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
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
        except Exception as e:
            st.error(f"Error getting video metadata: {str(e)}")
            return {}
    
    def render_annotation_interface(self, video_path: str, 
                                   parking_lot_id: Optional[int] = None) -> Dict:
        """Render simplified annotation interface"""
        st.markdown("### 📐 Define Parking Slot Boundaries")
        
        # Extract first frame
        first_frame = self.extract_first_frame(video_path)
        if first_frame is None:
            st.error("Failed to extract video frame")
            return {}
        
        # Get video metadata
        metadata = self.get_video_metadata(video_path)
        
        # Display video info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}")
        with col2:
            st.metric("FPS", f"{metadata.get('fps', 0):.1f}")
        with col3:
            st.metric("Duration", f"{metadata.get('duration', 0):.1f}s")
        with col4:
            st.metric("Frames", f"{metadata.get('frame_count', 0):,}")
        
        st.markdown("---")
        
        # Display frame
        st.markdown("### 🖼️ Video Frame")
        pil_image = Image.fromarray(first_frame)
        
        # Resize for display
        max_width = 800
        if pil_image.width > max_width:
            ratio = max_width / pil_image.width
            new_height = int(pil_image.height * ratio)
            display_image = pil_image.resize((max_width, new_height))
        else:
            display_image = pil_image
        
        st.image(display_image, caption="First frame - Use this as reference for coordinates")
        
        st.markdown("---")
        st.markdown("### ✍️ Manual Slot Definition")
        
        st.info("""
        **Instructions:**
        1. Look at the video frame above
        2. For each parking slot, enter the bounding box coordinates
        3. X1, Y1 = Top-left corner
        4. X2, Y2 = Bottom-right corner
        5. Click 'Add Slot' to save each slot
        """)
        
        # Initialize session state for annotations
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []
        
        # Load existing annotations if available
        if parking_lot_id and not st.session_state.annotations:
            existing = self.db.get_slot_annotations(parking_lot_id)
            if existing:
                st.session_state.annotations = existing
        
        # Input form for new slot
        with st.form("add_slot_form"):
            st.markdown("#### Add New Parking Slot")
            
            col1, col2 = st.columns(2)
            
            with col1:
                slot_id = st.text_input(
                    "Slot ID",
                    value=f"A{len(st.session_state.annotations) + 1}",
                    help="Unique identifier (e.g., A1, B2)"
                )
                
                x1 = st.number_input(
                    "X1 (Left)",
                    min_value=0,
                    max_value=metadata.get('width', 1920),
                    value=100,
                    step=10
                )
                
                y1 = st.number_input(
                    "Y1 (Top)",
                    min_value=0,
                    max_value=metadata.get('height', 1080),
                    value=100,
                    step=10
                )
            
            with col2:
                slot_type = st.selectbox(
                    "Slot Type",
                    ["regular", "handicap", "reserved", "other"]
                )
                
                x2 = st.number_input(
                    "X2 (Right)",
                    min_value=0,
                    max_value=metadata.get('width', 1920),
                    value=200,
                    step=10
                )
                
                y2 = st.number_input(
                    "Y2 (Bottom)",
                    min_value=0,
                    max_value=metadata.get('height', 1080),
                    value=200,
                    step=10
                )
            
            submitted = st.form_submit_button("➕ Add Slot", type="primary")
            
            if submitted:
                # Validate coordinates
                if x2 <= x1 or y2 <= y1:
                    st.error("Invalid coordinates! X2 must be > X1 and Y2 must be > Y1")
                elif any(ann['slot_id'] == slot_id for ann in st.session_state.annotations):
                    st.error(f"Slot ID '{slot_id}' already exists!")
                else:
                    new_slot = {
                        'slot_id': slot_id,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'slot_type': slot_type
                    }
                    st.session_state.annotations.append(new_slot)
                    st.success(f"✅ Added slot {slot_id}")
                    st.rerun()
        
        # Display current annotations
        if st.session_state.annotations:
            st.markdown("---")
            st.markdown(f"### ✅ Defined Slots ({len(st.session_state.annotations)})")
            
            # Create annotated image
            annotated_frame = self.visualize_annotations(
                first_frame.copy(),
                st.session_state.annotations
            )
            
            # Resize for display
            annotated_pil = Image.fromarray(annotated_frame)
            if annotated_pil.width > max_width:
                ratio = max_width / annotated_pil.width
                new_height = int(annotated_pil.height * ratio)
                annotated_pil = annotated_pil.resize((max_width, new_height))
            
            st.image(annotated_pil, caption="Annotated slots preview")
            
            # Show table of annotations
            st.markdown("#### Slot List")
            
            # Display in columns
            cols = st.columns(6)
            for idx, ann in enumerate(st.session_state.annotations):
                with cols[idx % 6]:
                    st.markdown(f"""
                    <div style="background: #2a2a2a; padding: 10px; 
                         border-radius: 8px; text-align: center; margin: 4px;
                         border: 2px solid #7c3aed;">
                        <strong style="color: #a855f7;">{ann['slot_id']}</strong><br>
                        <small style="color: #a0a0a0;">{ann['slot_type']}</small><br>
                        <small style="color: #666;">({ann['x1']},{ann['y1']})-({ann['x2']},{ann['y2']})</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Actions
            st.markdown("---")
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("🗑️ Clear All Slots", type="secondary"):
                    st.session_state.annotations = []
                    st.rerun()
            
            with action_col2:
                st.markdown("")  # Spacing
        
        return {
            'annotations': st.session_state.annotations,
            'metadata': metadata
        }
    
    def save_annotations(self, parking_lot_id: int, annotations: List[Dict]) -> bool:
        """Save annotations to database"""
        try:
            # Delete existing annotations
            self.db.delete_slot_annotations(parking_lot_id)
            
            # Save new annotations
            self.db.save_slot_annotations_batch(parking_lot_id, annotations)
            
            # Update total slots count
            self.db.update_parking_lot_slots(parking_lot_id, len(annotations))
            
            return True
        except Exception as e:
            st.error(f"Error saving annotations: {str(e)}")
            return False
    
    def visualize_annotations(self, frame: np.ndarray, 
                            annotations: List[Dict]) -> np.ndarray:
        """Draw annotations on frame"""
        img = frame.copy()
        
        for ann in annotations:
            # Draw rectangle
            cv2.rectangle(
                img,
                (ann['x1'], ann['y1']),
                (ann['x2'], ann['y2']),
                (0, 255, 0),  # Green
                3
            )
            
            # Draw label
            label = ann['slot_id']
            
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                img,
                (ann['x1'], ann['y1'] - label_h - 10),
                (ann['x1'] + label_w, ann['y1']),
                (0, 255, 0),
                -1
            )
            
            # Label text
            cv2.putText(
                img,
                label,
                (ann['x1'], ann['y1'] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),  # Black text
                2
            )
        
        return img
