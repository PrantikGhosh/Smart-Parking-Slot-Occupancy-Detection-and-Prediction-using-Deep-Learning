"""
Interactive Parking Layout Component
Clickable slots with instant status and predictions
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta


def create_interactive_parking_layout(db, selected_lot_id, annotations):
    """
    Create interactive clickable parking lot layout
    
    Args:
        db: Database instance
        selected_lot_id: Parking lot ID
        annotations: List of slot annotations
    
    Returns:
        selected_slot_id: ID of clicked slot or None
    """
    
    # Get lot info for sizing
    lot_info = db.get_parking_lot_by_id(selected_lot_id)
    width = lot_info.get('video_width', 1280) or 1280
    height = lot_info.get('video_height', 720) or 720
    
    # Get current status of each slot
    slot_statuses = {}
    for ann in annotations:
        events = db.get_occupancy_events(selected_lot_id, ann['slot_id'], limit=1)
        if events:
            slot_statuses[ann['slot_id']] = events[0]['status']
        else:
            slot_statuses[ann['slot_id']] = 'unknown'
    
    # Initialize selected slot in session state
    if 'selected_parking_slot' not in st.session_state:
        st.session_state.selected_parking_slot = None
    
    # Create figure
    fig = go.Figure()
    
    # Add dark background
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=width, y1=height,
        fillcolor="rgba(20,20,30,0.95)",
        line=dict(width=0)
    )
    
    # Draw each slot
    for ann in annotations:
        status = slot_statuses.get(ann['slot_id'], 'unknown')
        is_selected = (st.session_state.selected_parking_slot == ann['slot_id'])
        
        # Color based on status
        if status == 'occupied':
            color = 'rgba(239, 68, 68, 0.7)'  # Red
            border_color = 'rgb(220, 38, 38)'
            status_symbol = '🔴'
        elif status == 'empty':
            color = 'rgba(34, 197, 94, 0.7)'  # Green
            border_color = 'rgb(21, 128, 61)'
            status_symbol = '🟢'
        else:
            color = 'rgba(156, 163, 175, 0.6)'  # Gray
            border_color = 'rgb(107, 114, 128)'
            status_symbol = '⚪'
        
        # Highlight if selected
        if is_selected:
            border_color = 'rgb(251, 191, 36)'  # Yellow
            border_width = 5
        else:
            border_width = 3
        
        # Draw slot rectangle
        fig.add_shape(
            type="rect",
            x0=ann['x1'], y0=ann['y1'],
            x1=ann['x2'], y1=ann['y2'],
            fillcolor=color,
            line=dict(color=border_color, width=border_width)
        )
        
        # Add slot label
        center_x = (ann['x1'] + ann['x2']) / 2
        center_y = (ann['y1'] + ann['y2']) / 2
        
        # Add invisible scatter point for click detection
        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[center_y],
            mode='markers+text',
            marker=dict(size=1, color='rgba(0,0,0,0)'),  # Invisible
            text=f"<b>{ann['slot_id']}</b> {status_symbol}",
            textfont=dict(size=14, color="white", family="Arial Black"),
            textposition="middle center",
            name=ann['slot_id'],
            hovertemplate=f"<b>{ann['slot_id']}</b><br>Status: {status.upper()}<br>Click to view predictions<extra></extra>",
            customdata=[[ann['slot_id']]]
        ))
    
    # Configure layout
    fig.update_xaxes(
        range=[0, width],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )
    
    fig.update_yaxes(
        range=[height, 0],  # Inverted
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1
    )
    
    fig.update_layout(
        height=min(600, int(600 * height / width)),
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text="<b>Click on any slot to view predictions</b><br><sub>🟢 Empty | 🔴 Occupied | ⚪ Unknown</sub>",
            font=dict(size=18, color="white")
        ),
        showlegend=False,
        hovermode='closest'
    )
    
    # Display with click event
    selected_points = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="parking_layout_click"
    )
    
    # Handle click
    if selected_points and 'selection' in selected_points:
        points = selected_points['selection'].get('points', [])
        if points and len(points) > 0:
            # Get the slot_id from customdata
            clicked_slot = points[0].get('customdata', [[None]])[0][0]
            if clicked_slot:
                st.session_state.selected_parking_slot = clicked_slot
                st.rerun()
    
    return st.session_state.selected_parking_slot


def render_slot_predictions(db, parking_lot_id, slot_id):
    """
    Render prediction controls and results for selected slot
    
    Args:
        db: Database instance
        parking_lot_id: Parking lot ID
        slot_id: Selected slot ID
    """
    
    st.markdown(f"### 🎯 Predictions for Slot **{slot_id}**")
    
    # Get current status
    latest_events = db.get_occupancy_events(parking_lot_id, slot_id, limit=1)
    if latest_events:
        current_status = latest_events[0]['status']
        status_time = latest_events[0]['timestamp']
        
        if current_status == 'empty':
            st.success(f"✅ **Currently: EMPTY** (as of {status_time})")
        elif current_status == 'occupied':
            st.error(f"🚗 **Currently: OCCUPIED** (as of {status_time})")
        else:
            st.warning(f"⚪ **Currently: UNKNOWN** (as of {status_time})")
    else:
        st.info("No status data available")
        return
    
    st.markdown("---")
    
    # Prediction controls
    st.markdown("#### ⏰ Predict Future Availability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_interval = st.selectbox(
            "Select Time Ahead",
            options=[
                "15 minutes",
                "30 minutes",
                "45 minutes",
                "1 hour",
                "1.5 hours",
                "2 hours"
            ],
            help="How far into the future to predict"
        )
    
    with col2:
        # Extract minutes from selection
        interval_map = {
            "15 minutes": 15,
            "30 minutes": 30,
            "45 minutes": 45,
            "1 hour": 60,
            "1.5 hours": 90,
            "2 hours": 120
        }
        minutes_ahead = interval_map[time_interval]
        
        # Calculate target time
        if latest_events:
            base_time = datetime.fromisoformat(latest_events[0]['timestamp'])
            target_time = base_time + timedelta(minutes=minutes_ahead)
            
            st.info(f"**Predicting for:**\n\n{target_time.strftime('%I:%M %p')}")
    
    # Get historical data for pattern-based prediction
    all_events = db.get_occupancy_events(parking_lot_id, slot_id)
    
    if len(all_events) > 10:  # Need some historical data
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(all_events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_occupied'] = (df['status'] == 'occupied').astype(int)
        
        # Pattern-based prediction
        target_hour = target_time.hour
        target_dow = target_time.weekday()
        
        # Get similar historical patterns
        similar_data = df[
            (df['hour'] == target_hour) |
            ((df['hour'] == target_hour - 1) & (df['minute'] >= 30)) |
            ((df['hour'] == target_hour + 1) & (df['minute'] <= 30))
        ]
        
        if len(similar_data) > 0:
            prob_occupied = similar_data['is_occupied'].mean()
            prob_empty = 1 - prob_occupied
            
            # Display prediction
            st.markdown("---")
            st.markdown("#### 📊 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability meters
                st.metric(
                    "Probability Empty",
                    f"{prob_empty*100:.1f}%",
                    delta=None
                )
                st.progress(prob_empty)
            
            with col2:
                st.metric(
                    "Probability Occupied",
                    f"{prob_occupied*100:.1f}%",
                    delta=None
                )
                st.progress(prob_occupied)
            
            # Recommendation
            st.markdown("---")
            st.markdown("####  💡 Recommendation")
            
            if prob_empty > 0.7:
                st.success(f"**HIGH chance** slot will be available at {target_time.strftime('%I:%M %p')}! ✅")
            elif prob_empty > 0.4:
                st.warning(f"**MODERATE chance** slot will be available at {target_time.strftime('%I:%M %p')}. ⚠️")
            else:
                st.error(f"**LOW chance** slot will be available at {target_time.strftime('%I:%M %p')}. Try another slot. ❌")
            
            # Show based on sample size
            st.caption(f"_Based on {len(similar_data)} historical observations at similar times_")
        else:
            st.warning("Not enough historical data for this time period to make accurate predictions.")
    else:
        st.warning("⚠️ Insufficient historical data. Need at least 10 observations to predict.")
    
    # Clear selection button
    if st.button("← Back to Layout", use_container_width=True):
        st.session_state.selected_parking_slot = None
        st.rerun()
