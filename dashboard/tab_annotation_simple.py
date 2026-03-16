"""
Simplified Video Processing Workflow
Upload video → Export template → Import slots → Process → Extract data
"""

import streamlit as st
import tempfile
import os
from datetime import datetime
import json

from database import get_database
from components import render_section_header, render_status_banner, render_metric_card, Colors
from processing import VideoProcessor


def tab_annotation_simple():
    """Simplified video annotation and processing workflow"""
    
    db = get_database()
    render_section_header("Video Processing & Data Extraction", "📹")
    
    st.markdown("""
    ### 📝 Workflow
    1. **Upload Video** - Upload your parking lot video
    2. **Download Template** - Get JSON template with required format
    3. **Fill Template** - Add slot coordinates manually
    4. **Upload JSON** - Import your slot definitions
    5. **Process Video** - Extract occupancy data for time series
    6. **View Analytics** - Go to Tab 3 for predictions
    """)
    
    st.markdown("---")
    
    # Step 1: Video Upload
    st.markdown("### 📤 Step 1: Upload Video")
    uploaded_video = st.file_uploader(
        "Choose parking lot video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload video footage of your parking lot"
    )
    
    if uploaded_video is None:
        st.info("👆 Upload a video to get started")
        return
    
    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_video.read())
        temp_video_path = tmp_file.name
    
    # Compute video hash
    video_hash = db.compute_video_hash(temp_video_path)
    
    # Check if already processed
    existing_lot = db.get_parking_lot_by_hash(video_hash)
    
    if existing_lot:
        render_status_banner(
            f"✅ This video was already processed as '{existing_lot['name']}'",
            "success"
        )
        
        annotations = db.get_slot_annotations(existing_lot['id'])
        stats = db.get_statistics(existing_lot['id'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Slots Defined", len(annotations))
        with col2:
            st.metric("Events Recorded", f"{stats['total_events']:,}")
        with col3:
            st.metric("Duration", f"{stats['video_duration']:.0f}s")
        
        st.markdown("**Go to Tab 3 to view predictions!**")
        
        if st.button("🔄 Process Again"):
            db.delete_occupancy_events(existing_lot['id'])
            st.rerun()
        
        return
    
    # New video - get basic info
    st.success("✅ New video uploaded successfully!")
    
    # Get video metadata
    from components import VideoAnnotator
    annotator = VideoAnnotator(db)
    metadata = annotator.get_video_metadata(temp_video_path)
    
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
    
    # Step 2: Video details
    st.markdown("### 📝 Step 2: Video Details")
    
    col_name, col_angle = st.columns(2)
    with col_name:
        video_name = st.text_input(
            "Parking Lot Name",
            value=uploaded_video.name.split('.')[0],
            help="Give this parking lot a descriptive name"
        )
    
    with col_angle:
        camera_angle = st.selectbox(
            "Camera Angle",
            ["top_down", "angled", "side_view", "other"]
        )
    
    st.markdown("---")
    
    # Step 3: JSON Template
    st.markdown("### 📥 Step 3: Download & Fill Template")
    
    st.markdown("""
    **Download the JSON template below and fill in your parking slot coordinates:**
    
    Each slot needs:
    - `slot_id`: Unique identifier (e.g., "A1", "B2")
    - `x1, y1`: Top-left corner coordinates
    - `x2, y2`: Bottom-right corner coordinates
    - `slot_type`: "regular", "handicap", "reserved", or "other"
    
    **Important:** Coordinates should match your video resolution: **{}x{}**
    """.format(metadata.get('width', 0), metadata.get('height', 0)))
    
    # Template with 3 example slots
    template_data = [
        {
            "slot_id": "A1",
            "x1": 100,
            "y1": 150,
            "x2": 200,
            "y2": 250,
            "slot_type": "regular",
            "note": "Replace with your actual coordinates"
        },
        {
            "slot_id": "A2",
            "x1": 220,
            "y1": 150,
            "x2": 320,
            "y2": 250,
            "slot_type": "regular",
            "note": "Add more slots as needed"
        },
        {
            "slot_id": "A3",
            "x1": 340,
            "y1": 150,
            "x2": 440,
            "y2": 250,
            "slot_type": "handicap",
            "note": "Delete this note field before uploading"
        }
    ]
    
    template_json = json.dumps(template_data, indent=2)
    
    st.download_button(
        "📥 Download JSON Template",
        data=template_json,
        file_name=f"parking_slots_{video_name}.json",
        mime="application/json",
        type="primary"
    )
    
    st.markdown("---")
    
    # Step 4: Upload JSON
    st.markdown("### 📤 Step 4: Upload Filled JSON")
    
    json_file = st.file_uploader(
        "Upload your filled JSON file",
        type=["json"],
        help="Upload the JSON file with your parking slot coordinates"
    )
    
    if json_file is not None:
        try:
            slots_data = json.load(json_file)
            
            # Validate
            if not isinstance(slots_data, list):
                st.error("❌ JSON must be a list of slot objects")
                return
            
            if len(slots_data) == 0:
                st.error("❌ JSON file is empty")
                return
            
            # Check required fields
            required_fields = ['slot_id', 'x1', 'y1', 'x2', 'y2', 'slot_type']
            for slot in slots_data:
                missing = [f for f in required_fields if f not in slot]
                if missing:
                    st.error(f"❌ Slot '{slot.get('slot_id', 'unknown')}' is missing fields: {missing}")
                    return
            
            st.success(f"✅ Successfully loaded {len(slots_data)} parking slots!")
            
            # Preview
            st.markdown("#### Loaded Slots:")
            cols = st.columns(6)
            for idx, slot in enumerate(slots_data):
                with cols[idx % 6]:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); 
                         color: white; padding: 12px; border-radius: 10px; 
                         text-align: center; margin: 4px;">
                        <strong>{slot['slot_id']}</strong><br>
                        <small>{slot['slot_type']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Step 5: Process Video
            st.markdown("### ⚙️ Step 5: Process Video")
            
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                # Save parking lot first
                parking_lot_id = db.save_parking_lot(
                    name=video_name,
                    video_path=temp_video_path,
                    video_hash=video_hash,
                    camera_angle=camera_angle,
                    fps=metadata.get('fps'),
                    duration=metadata.get('duration'),
                    width=metadata.get('width'),
                    height=metadata.get('height')
                )
                
                # Save annotations
                db.save_slot_annotations_batch(parking_lot_id, slots_data)
                db.update_parking_lot_slots(parking_lot_id, len(slots_data))
                
                st.success(f"✅ Saved {len(slots_data)} slot annotations")
                
                # Process video
                st.markdown("#### 🎬 Processing Video...")
                
                processor = VideoProcessor(db)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                result = processor.process_video(
                    parking_lot_id=parking_lot_id,
                    video_path=temp_video_path,
                    annotations=slots_data,
                    progress_callback=lambda p, s: (
                        progress_bar.progress(p),
                        status_text.text(s)
                    )
                )
                
                if result['success']:
                    st.success(f"""
                    ✅ **Processing Complete!**
                    
                    - **Events Recorded:** {result['total_events']:,}
                    - **Frames Processed:** {result['frames_processed']:,}
                    - **Duration:** {result['processing_time']:.1f}s
                    """)
                    
                    render_status_banner(
                        "🎉 Video processed successfully! Go to Tab 3 to view predictions and analytics.",
                        "success"
                    )
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_video_path)
                    except:
                        pass
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("👆 Upload your filled JSON file to continue")


if __name__ == "__main__":
    tab_annotation_simple()
