"""
Smart Parking Prediction System - Enhanced Dashboard
Three-tab interface: Detection | Video Annotation | Slot Predictions
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import custom components
from components import (
    Colors, load_custom_css, render_header, render_metric_card,
    render_status_banner, render_section_header, create_donut_chart,
    create_occupancy_chart, create_heatmap,
    VideoAnnotator
)
from database import get_database
from processing.video_processor import VideoProcessor
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Smart Parking Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# Initialize database
db = get_database()


# ==================== Helper Functions ====================

@st.cache_resource
def load_model(model_path):
    """Load YOLO model (cached)"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def draw_detections(image, results, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: PIL Image
        results: YOLO results list
        conf_threshold: Confidence threshold for filtering
        
    Returns:
        Annotated PIL Image, stats dictionary
    """
    # Convert PIL to cv2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    stats = {
        'total_spaces': 0,
        'empty_spaces': 0,
        'occupied_spaces': 0,
        'generic_spaces': 0
    }
    
    if len(results) == 0 or results[0].boxes is None:
        return image, stats
        
    # Get model class names
    names = results[0].names
    
    # Check for abhivesh model signature (inverted labels: 0=empty, 1=occupied)
    # The user reported that for abhivesh_model (formerly shukla_model):
    # 0 ('empty') should be treated as Occupied (Red)
    # 1 ('occupied') should be treated as Empty (Green)
    is_abhivesh_model = (len(names) == 2 and names.get(0) == 'empty' and names.get(1) == 'occupied')
    
    boxes = results[0].boxes
    
    for box in boxes:
        # Get box data
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name_raw = names.get(cls, str(cls))
        
        # Determine Status and Color
        status = "Unknown"
        color = (255, 165, 0) # Default Orange
        
        if is_abhivesh_model:
            # Special handling for inverted model
            if cls == 0: 
                # User requested vice versa: empty -> Empty
                status = "Empty"
                color = (0, 255, 0) # Green
                stats['empty_spaces'] += 1
            else:
                # occupied -> Occupied
                status = "Occupied"
                color = (0, 0, 255) # Red
                stats['occupied_spaces'] += 1
        else:
            # Standard Handling
            name_lower = class_name_raw.lower()
            
            # Check for empty keywords
            if 'empty' in name_lower:
                status = "Empty"
                color = (0, 255, 0) # Green
                stats['empty_spaces'] += 1
            
            # Check for occupied/vehicle keywords
            elif any(x in name_lower for x in ['occup', 'car', 'truck', 'bus', 'motor', 'vehicle']):
                status = "Occupied"
                color = (0, 0, 255) # Red
                stats['occupied_spaces'] += 1
                
            # Default fallback (requested as 'empty' and green)
            else:
                status = "Empty"
                color = (0, 255, 0) # Green
                stats['empty_spaces'] += 1
        
        stats['total_spaces'] += 1
        
        # Draw rectangle
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        
        # Prepare label
        label = f'{status} {conf:.2f}'
        
        # Calculate label background size
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            img_cv,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_cv,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    # Convert back to PIL format
    annotated_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    return annotated_image, stats


# ==================== Tab 1: Real-Time Detection ====================

def tab_detection():
    """Real-time parking detection tab"""
    render_section_header("Real-Time Parking Detection", "🎯")
    
    # Model selection in sidebar
    with st.sidebar:
        st.markdown("### Model Settings")
        # Dynamic path resolution relative to this file
        current_dir = Path(__file__).parent.parent
        models_dir = current_dir / "models"
        
        # Get list of .pt files in models directory
        model_files = list(models_dir.glob("*.pt"))
        model_options = [m.name for m in model_files]
        
        if not model_options:
             st.warning("No models found in 'models' directory.")
             model_path = ""
        else:
            selected_model = st.selectbox(
                "Select Model",
                options=model_options,
                index=0 if "abhivesh_model.pt" not in model_options else model_options.index("abhivesh_model.pt")
            )
            model_path = str(models_dir / selected_model)
            render_status_banner(f"Model loaded: {selected_model}", "success")
        
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    # Main content
    if not model_path or not os.path.exists(model_path):
        render_status_banner("Please select a valid model!", "warning")
        return
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the path and try again.")
        return
    
    # File uploader
    st.markdown("### Upload Parking Lot Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a parking lot image for analysis"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image)
        
        # Run detection
        with st.spinner("Analyzing parking spaces..."):
            results = model(image, conf=conf_threshold)
            annotated_image, stats = draw_detections(image, results, conf_threshold)
        
        with col2:
            st.markdown("#### Detection Results")
            st.image(annotated_image)
        
        # Display statistics
        st.markdown("---")
        st.markdown("### Parking Statistics")
        
        # Metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        total = stats['empty_spaces'] + stats['occupied_spaces']
        occupancy_rate = (stats['occupied_spaces'] / total * 100) if total > 0 else 0
        
        with metric_col1:
            st.markdown(render_metric_card(
                str(total),
                "Total Spots",
                "",
                Colors.PRIMARY_GRADIENT
            ), unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(render_metric_card(
                str(stats['empty_spaces']),
                "Empty",
                "",
                Colors.SUCCESS_GRADIENT
            ), unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(render_metric_card(
                str(stats['occupied_spaces']),
                "Occupied",
                "",
                Colors.WARNING_GRADIENT
            ), unsafe_allow_html=True)
        
        with metric_col4:
            st.markdown(render_metric_card(
                f"{occupancy_rate:.1f}%",
                "Occupancy",
                "",
                Colors.INFO_GRADIENT
            ), unsafe_allow_html=True)
        
        # Chart
        if total > 0:
            st.markdown("---")
            chart_col1, chart_col2 = st.columns([1, 2])
            
            with chart_col1:
                fig = create_donut_chart(stats['empty_spaces'], stats['occupied_spaces'])
                if fig:
                    st.plotly_chart(fig, config={'displayModeBar': False})
            
            with chart_col2:
                st.markdown("#### Analysis Summary")
                if occupancy_rate > 80:
                    st.error(f"**High Occupancy Alert**: {occupancy_rate:.1f}% of spaces are occupied.")
                elif occupancy_rate > 50:
                    st.warning(f"**Moderate Occupancy**: {occupancy_rate:.1f}% of spaces are occupied.")
                else:
                    st.success(f"**Low Occupancy**: {occupancy_rate:.1f}% of spaces are occupied.")
                
                st.info(f"**Available Spaces**: {stats['empty_spaces']} out of {total} spots")


# ==================== Tab 2: Video Annotation ====================

def tab_annotation():
    """Working mouse-click annotation system"""
    from tab_annotation_interactive import tab_annotation_with_clicks
    tab_annotation_with_clicks()



# ==================== Tab 3: Slot Predictions ====================

def tab_predictions():
    """Simplified predictions tab with LSTM/Prophet integration"""
    from tab_predictions_simple import render_tab_predictions
    render_tab_predictions(db)


# ==================== Main App ====================

def main():
    """Main application entry point"""
    """Simplified predictions tab"""
    from tab_predictions_simple import render_tab_predictions
    render_tab_predictions(db)


    """Slot predictions tab with model integration"""
    render_section_header("Parking Slot Predictions", "")
    
    # Get all parking lots
    parking_lots = db.get_all_parking_lots()
    
    if not parking_lots:
        st.info("""
        No parking lots found yet!
        
        **To get started:**
        1. Go to the **Video Annotation** tab
        2. Upload and annotate a parking lot video
        3. Process the video to extract occupancy data
        4. Return here to see predictions
        """)
        return
    
    # Select parking lot
    lot_options = {lot['name']: lot['id'] for lot in parking_lots}
    selected_lot_name = st.selectbox(
        "Select Parking Lot",
        options=list(lot_options.keys())
    )
    selected_lot_id = lot_options[selected_lot_name]
    
    # Get annotations
    annotations = db.get_slot_annotations(selected_lot_id)
    
    if not annotations:
        st.warning("No slot annotations found. Please annotate the video first in Tab 2.")
        return
    
    # Import interactive layout component
    from components.interactive_parking_layout import create_interactive_parking_layout, render_slot_predictions
    
    st.markdown("### Interactive Parking Lot")
    st.markdown("**Click on any slot to viewits status and predictions**")
    
    # Create interactive layout
    selected_slot = create_interactive_parking_layout(db, selected_lot_id, annotations)
    
    # If a slot is selected, show predictions
    if selected_slot:
        st.markdown("---")
        render_slot_predictions(db, selected_lot_id, selected_slot)
        
        # Don't show the rest of the tab when a slot is selected
        return
    
    st.markdown("---")
    
    # Get statistics
    stats = db.get_statistics(selected_lot_id)
    
    if stats['total_events'] == 0:
        st.warning("""
        No occupancy data found for this parking lot.
        
        Please process the video first in the **Video Annotation** tab.
        """)
        return
    
    # Display current statistics
    st.markdown("### Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card(
            str(stats['total_slots']),
            "Total Slots",
            ""
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(render_metric_card(
            f"{stats['total_events']:,}",
            "Events Recorded",
            ""
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(render_metric_card(
            f"{stats['video_duration']:.0f}s",
            "Video Duration",
            ""
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(render_metric_card(
            f"{stats['fps']:.0f}",
            "FPS",
            ""
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Get slot annotations for selection
    annotations = db.get_slot_annotations(selected_lot_id)
    
    if not annotations:
        st.warning("No slot annotations found. Please annotate the video first.")
        return
    
    # Control panel
    st.markdown("### Prediction Controls")
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Slot selection
        slot_ids = [ann['slot_id'] for ann in annotations]
        selected_slot = st.selectbox(
            "Select Parking Slot",
            options=slot_ids,
            help="Choose a slot to analyze and predict"
        )
    
    with col_right:
        # Prediction horizon
        horizon_hours = st.slider(
            "Prediction Horizon (hours)",
            min_value=1,
            max_value=24,
            value=6,
            help="How far into the future to predict"
        )
    
    # Load occupancy data for selected slot
    from models import FeatureEngineer, LSTMPredictor, TENSORFLOW_AVAILABLE, PROPHET_AVAILABLE
    from models.lstm_predictor import prepare_data_for_lstm
    
    engineer = FeatureEngineer(db)
    
    with st.spinner(f"Loading data for slot {selected_slot}..."):
        df = engineer.load_occupancy_data(selected_lot_id, slot_id=selected_slot)
    
    if df.empty:
        st.warning(f"No data found for slot {selected_slot}")
        return
    
    # Prepare features
    df = engineer.create_binary_target(df)
    df = engineer.extract_time_features(df)
    df = engineer.calculate_rolling_features(df)
    df = engineer.calculate_lag_features(df)
    
    # Calculate current occupancy statistics
    recent_data = df.tail(100)
    current_occupancy_rate = recent_data['occupancy'].mean()
    
    # Display current status metrics
    st.markdown("---")
    st.markdown(f"### Slot {selected_slot} - Current Status")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        latest_status = "Empty" if df.iloc[-1]['occupancy'] == 1 else "Occupied"
        status_gradient = Colors.SUCCESS_GRADIENT if latest_status == "Empty" else Colors.WARNING_GRADIENT
        st.markdown(render_metric_card(
            latest_status,
            "Current Status",
            "",
            status_gradient
        ), unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(render_metric_card(
            f"{current_occupancy_rate*100:.1f}%",
            "Availability (Recent)",
            "",
            Colors.INFO_GRADIENT
        ), unsafe_allow_html=True)
    
    with metric_col3:
        # Find peak hour
        hourly_avg = df.groupby('hour')['occupancy'].mean()
        peak_hour = hourly_avg.idxmin()  # Hour with lowest availability = highest occupancy
        st.markdown(render_metric_card(
            f"{int(peak_hour):02d}:00" if pd.notna(peak_hour) else "N/A",
            "Peak Hour",
            "",
            Colors.WARNING_GRADIENT
        ), unsafe_allow_html=True)
    
    with metric_col4:
        total_samples = len(df)
        st.markdown(render_metric_card(
            f"{total_samples:,}",
            "Data Points",
            "",
            Colors.PRIMARY_GRADIENT
        ), unsafe_allow_html=True)
    
    # Historical visualization
    st.markdown("---")
    st.markdown("### Historical Occupancy Pattern")
    
    # Prepare chart data
    historical_data = []
    sample_data = df.tail(200)  # Show last 200 points
    
    for _, row in sample_data.iterrows():
        historical_data.append({
            'timestamp': row['timestamp'],
            'occupancy': row['occupancy']
        })
    
    # Create and display chart
    if historical_data:
        chart = create_occupancy_chart(historical_data)
        st.plotly_chart(chart, use_column_width=True)
    
    # Weekly heatmap
    st.markdown("---")
    st.markdown("### Weekly Usage Pattern")
    
    # Calculate hourly averages by day of week
    df_copy = df.copy()
    df_copy['day_name'] = df_copy['timestamp'].dt.day_name()
    
    heatmap_data = df_copy.groupby(['day_of_week', 'hour'])['occupancy'].mean().unstack(fill_value=0)
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    # Prepare heatmap values
    heatmap_values = []
    for day in range(7):
        if day in heatmap_data.index:
            heatmap_values.append(heatmap_data.loc[day].values.tolist())
        else:
            heatmap_values.append([0] * 24)
    
    heatmap_chart = create_heatmap({
        'days': day_names,
        'hours': hours,
        'values': heatmap_values
    })
    
    st.plotly_chart(heatmap_chart, use_column_width=True)
    
    # Prediction section (simplified - show capability)
    st.markdown("---")
    st.markdown("### Future Predictions")
    
    model_available = TENSORFLOW_AVAILABLE or PROPHET_AVAILABLE
    
    if not model_available:
        render_status_banner(
            "Prediction models require TensorFlow and/or Prophet. Install with: pip install tensorflow prophet",
            "warning"
        )
    else:
        st.info("""
        **Prediction Models Ready!**
        
        The LSTM and Prophet models are available for training. Due to the demo nature and computing requirements:
        
        - **LSTM Model**: Requires sufficient sequential data (recommended: 500+ samples)
        - **Prophet Model**: Works best with time-stamped hourly/daily data
        - **Training Time**: May take several minutes depending on data size
        
        For this demo, predictions are shown based on historical patterns.
        """)
        
        # Generate simple predictions based on historical averages
        st.markdown("#### Pattern-Based Forecast")
        
        # Use hourly averages for prediction
        next_hours = []
        current_hour = df.iloc[-1]['timestamp'].hour
        
        for i in range(1, horizon_hours + 1):
            future_hour = (current_hour + i) % 24
            if future_hour in hourly_avg.index:
                prob = hourly_avg[future_hour]
            else:
                prob = current_occupancy_rate
            
            future_time = df.iloc[-1]['timestamp'] + timedelta(hours=i)
            next_hours.append({
                'timestamp': future_time,
                'value': prob,
                'upper': min(prob + 0.15, 1.0),
                'lower': max(prob - 0.15, 0.0)
            })
        
        # Create prediction chart
        pred_chart = create_occupancy_chart(historical_data[-50:], next_hours)
        st.plotly_chart(pred_chart, use_column_width=True)
        
        # Recommendation
        st.markdown("#### Smart Recommendations")
        
        if next_hours:
            # Find best time to park (highest availability)
            best_time_idx = max(range(len(next_hours)), key=lambda i: next_hours[i]['value'])
            best_time = next_hours[best_time_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if best_time['value'] > 0.7:
                    st.success(f"""
                    **Great Availability Expected**
                    
                    Best time to visit: **{best_time['timestamp'].strftime('%I:%M %p')}**
                    
                    Predicted availability: **{best_time['value']*100:.0f}%**
                    """)
                elif best_time['value'] > 0.4:
                    st.warning(f"""
                    **Moderate Availability**
                    
                    Decent time: **{best_time['timestamp'].strftime('%I:%M %p')}**
                    
                    Predicted availability: **{best_time['value']*100:.0f}%**
                    """)
                else:
                    st.error(f"""
                    **Low Availability Expected**
                    
                    All times show high occupancy. Consider alternative slots.
                    
                    Best available: **{best_time['value']*100:.0f}%** at {best_time['timestamp'].strftime('%I:%M %p')}
                    """)
            
            with col2:
                # Show next available prediction
                next_available = next((h for h in next_hours if h['value'] > 0.6), None)
                
                if next_available:
                    time_until = (next_available['timestamp'] - df.iloc[-1]['timestamp']).seconds // 3600
                    st.info(f"""
                    **Next High Availability**
                    
                    Expected in: **{time_until} hour(s)**
                    
                    At: **{next_available['timestamp'].strftime('%I:%M %p')}**
                    """)
                else:
                    st.info("""
                    **No high availability expected in forecast window**
                    
                    Consider checking other slots or extending prediction horizon.
                    """)
    
    # Model training section (optional advanced feature)
    with st.expander("Advanced: Train Prediction Models", expanded=False):
        st.markdown("""
        Train LSTM or Prophet models on the collected data for more accurate predictions.
        
        **Note**: This requires TensorFlow and Prophet libraries installed.
        """)
        
        model_choice = st.selectbox(
            "Select Model",
            ["Pattern-Based (Default)", "LSTM (Advanced)", "Prophet (Advanced)", "Ensemble (Both)"]
        )
        
        if st.button("Train Model", disabled=not model_available):
            if model_choice == "LSTM (Advanced)" and TENSORFLOW_AVAILABLE:
                st.info(f"Training LSTM model for Slot {selected_slot}...")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # 1. Prepare Data
                    status_text.text("Preparing training data...")
                    progress_bar.progress(20)
                    
                    # Ensure we have features
                    X, y = engineer.prepare_features_for_ml(df)
                    
                    # Create sequences
                    data = prepare_data_for_lstm(X, y, sequence_length=30)
                    
                    progress_bar.progress(40)
                    
                    # 2. Build Model
                    status_text.text("Building LSTM architecture...")
                    predictor = LSTMPredictor(sequence_length=30, n_features=X.shape[1])
                    predictor.build_model()
                    
                    progress_bar.progress(50)
                    
                    # 3. Train
                    status_text.text("Training model (this may take a moment)...")
                    
                    # Create a directory for models if it doesn't exist
                    model_save_path = f"models/lstm_{selected_slot}.h5"
                    
                    result = predictor.train(
                        data['X_train'], data['y_train'],
                        data['X_val'], data['y_val'],
                        epochs=10, # Short training for demo
                        batch_size=32,
                        save_path=model_save_path
                    )
                    
                    progress_bar.progress(90)
                    
                    # 4. Evaluate
                    status_text.text("Evaluating performance...")
                    metrics = predictor.evaluate(data['X_test'], data['y_test'])
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    
                    # Display Results
                    st.success(f"Model trained successfully! Saved to {model_save_path}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
                    col2.metric("Precision", f"{metrics['precision']:.2%}")
                    col3.metric("Recall", f"{metrics['recall']:.2%}")
                    
                    # visualize training history
                    st.line_chart(result['history']['loss'])
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.exception(e)
            
            elif model_choice == "Prophet (Advanced)" and PROPHET_AVAILABLE:
                st.warning("Prophet training implementation coming soon!")
            else:
                st.info("Please select a valid model and ensure libraries are installed.")



# ==================== Main App ====================

def main():
    """Main application entry point"""
    
    # Header
    render_header(
        "Smart Parking Prediction System",
        "AI-Powered Real-Time Detection & Future Occupancy Forecasting"
    )
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Professional tab navigation
        page = st.radio(
            "",
            ["Real-Time Detection", "Video Annotation", "Slot Predictions"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # About section (minimal, no emojis)
        st.markdown("### About")
        st.markdown("""
        **Smart Parking System**
        
        Capabilities:
        - Real-time YOLO detection
        - Video annotation & processing
        - Time-series prediction (LSTM & Prophet)
        - Interactive visualizations
        """)
        
        st.markdown("---")
        
        # System info
        st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Render selected page
    if page == "Real-Time Detection":
        tab_detection()
    elif page == "Video Annotation":
        tab_annotation()
    elif page == "Slot Predictions":
        tab_predictions()


if __name__ == "__main__":
    load_custom_css()
    db = get_database()
    main()
