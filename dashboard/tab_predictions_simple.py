"""
Simplified Tab 3 - Parking Slot Predictions
Minimal UI with LSTM/Prophet model integration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render_tab_predictions(db):
    """Simplified prediction tab with working models"""
    
    # Initialize session state for predictions
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'prediction_params' not in st.session_state:
        st.session_state.prediction_params = {}
    
    st.markdown("## 🔮 Parking Slot Predictions")
    
    # Get all parking lots
    parking_lots = db.get_all_parking_lots()
    
    if not parking_lots:
        st.info("""
        👋 **No parking lots found!**
        
        1. Go to **Tab 2: Video Annotation**
        2. Upload and annotate a video
        3. Process with YOLO
        4. Return here for predictions
        """)
        return
    
    # Parking lot selector
    lot_options = {lot['name']: lot['id'] for lot in parking_lots}
    selected_lot_name = st.selectbox("📍 Select Parking Lot", list(lot_options.keys()))
    selected_lot_id = lot_options[selected_lot_name]
    
    # Clear predictions if lot changed
    if st.session_state.prediction_params.get('lot_id') != selected_lot_id:
        st.session_state.prediction_result = None
        st.session_state.prediction_params = {}
    
    # Get statistics
    stats = db.get_statistics(selected_lot_id)
    
    if stats['total_events'] == 0:
        st.warning("⚠️ No occupancy data. Please process the video in Tab 2 first.")
        return
    
    # Get slot annotations
    annotations = db.get_slot_annotations(selected_lot_id)
    
    if not annotations:
        st.warning("⚠️ No slots annotated. Please annotate in Tab 2 first.")
        return
    
    st.markdown("---")
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Slots", stats['total_slots'])
    
    with col2:
        st.metric("Events Recorded", f"{stats['total_events']:,}")
    
    with col3:
        # Current occupancy
        latest_events = db.get_occupancy_events(selected_lot_id, limit=stats['total_slots'])
        occupied_count = sum(1 for e in latest_events if e['status'] == 'occupied')
        st.metric("Currently Occupied", f"{occupied_count}/{stats['total_slots']}")
    
    st.markdown("---")
    
    # Collapsible Parking Layout Reference
    with st.expander("🅿️ **View Parking Layout Reference**", expanded=False):
        st.markdown("This diagram shows the position of each parking slot to help you identify which slot is which.")
        
        # Only render when expander is opened (Streamlit handles this automatically)
        from components.parking_layout_diagram import create_parking_layout_diagram
        
        layout_fig = create_parking_layout_diagram(db, selected_lot_id, annotations)
        st.plotly_chart(layout_fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("---")
    
    # Slot selector
    st.markdown("### 🎯 Select Slot for Prediction")
    
    # Natural sort function for slot IDs (e.g., slot_1, slot_2, ... slot_10, slot_11)
    def natural_sort_key(slot_id):
        import re
        parts = re.split(r'(\d+)', slot_id)
        return [int(part) if part.isdigit() else part for part in parts]
    
    slot_ids = [ann['slot_id'] for ann in annotations]
    slot_ids_sorted = sorted(slot_ids, key=natural_sort_key)
    
    selected_slot = st.selectbox("Choose a parking slot", slot_ids_sorted)
    
    # Clear predictions if slot changed
    if st.session_state.prediction_params.get('slot_id') != selected_slot:
        st.session_state.prediction_result = None
        st.session_state.prediction_params = {}
    
    # Get slot data
    slot_events = db.get_occupancy_events(selected_lot_id, slot_id=selected_slot)
    
    if len(slot_events) < 10:
        st.warning(f"⚠️ Insufficient data for slot {selected_slot}. Need at least 10 events (have {len(slot_events)}).")
        return
    
    # Current status
    latest_event = slot_events[0]  # Already sorted DESC
    status = latest_event['status']
    status_time = latest_event['timestamp']
    
    if status == 'empty':
        st.success(f"✅ **Slot {selected_slot} is EMPTY** (as of {status_time})")
    else:
        st.error(f"🚗 **Slot {selected_slot} is OCCUPIED** (as of {status_time})")
    
    st.markdown("---")
    
    # Compact Prediction Controls
    st.markdown("### ⏰ Predict Future Availability")
    
    # Single row with time selector and predict button
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        predict_minutes = st.selectbox(
            "Time ahead",
            [15, 30, 45, 60, 90, 120],
            format_func=lambda x: f"{x} min" if x < 60 else (f"{x/60:.1f}h ~ {x} min" if x % 60 != 0 else f"{x//60}h ~ {x} min"),
            label_visibility="collapsed"
        )
    
    # Clear predictions if time changed
    if st.session_state.prediction_params.get('predict_minutes') != predict_minutes:
        st.session_state.prediction_result = None
        st.session_state.prediction_params = {}
    
    with col2:
        target_time = datetime.fromisoformat(status_time) + timedelta(minutes=predict_minutes)
        st.markdown(f"<div style='text-align: center; padding-top: 8px;'><b>Predicting for:</b> {target_time.strftime('%I:%M %p')}</div>", unsafe_allow_html=True)
    
    with col3:
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        
        # Use new PredictionService
        from models import get_prediction_service
        
        with st.spinner("🔮 Generating predictions..."):
            service = get_prediction_service(db)
            result = service.predict_future_availability(
                lot_id=selected_lot_id,
                slot_id=selected_slot,
                minutes_ahead=predict_minutes
            )
        
        if not result['success']:
            st.error(f"❌ {result.get('error', 'Prediction failed')}")
            st.info(f"📊 Available data points: {result.get('data_points', 0)}")
            return
        
        # Store result in session state
        st.session_state.prediction_result = result
        st.session_state.prediction_params = {
            'lot_id': selected_lot_id,
            'slot_id': selected_slot,
            'predict_minutes': predict_minutes,
            'target_time': target_time
        }
    
    # Retrieve result from session state (if available)
    result = st.session_state.prediction_result
    
    # Display results if we have them (either from button click or session state)
    if result and result.get('success'):
        
        # Get target time from session state params
        target_time = st.session_state.prediction_params.get('target_time', target_time)
        
        # Display data volume info
        st.markdown("---")
        st.markdown("### 📊 Data & Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Points", result['data_points'])
        
        with col2:
            recommended_models = result['data_info'].get('recommended_models', [])
            st.metric("Models Available", len(recommended_models))
        
        with col3:
            st.metric("Confidence", f"{result.get('confidence', 0)*100:.0f}%")
        
        st.caption(f"🤖 Models that can run with your data: {', '.join(recommended_models) if recommended_models else 'None'}")
        
        # Show model errors if any
        if result.get('model_errors'):
            with st.expander("⚠️ Model Issues", expanded=False):
                for model_name, error_msg in result['model_errors'].items():
                    st.warning(f"**{model_name.title()}**: {error_msg}")
        
        # Display recommended prediction (main result)
        if result['recommended_prediction'] is not None:
            st.markdown("---")
            st.markdown(f"### 🎯 Prediction ({result['recommended_model']})")
            
            prob_occupied = result['recommended_prediction']
            prob_empty = 1 - prob_occupied
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Probability EMPTY", f"{prob_empty*100:.1f}%")
                st.progress(float(prob_empty))
            
            with col2:
                st.metric("Probability OCCUPIED", f"{prob_occupied*100:.1f}%")
                st.progress(float(prob_occupied))
            
            # Recommendation
            st.markdown("---")
            st.markdown("#### 💡 Recommendation")
            
            if prob_empty > 0.7:
                st.success(f"✅ **HIGH chance** of availability at {target_time.strftime('%I:%M %p')}")
            elif prob_empty > 0.4:
                st.warning(f"⚠️ **MODERATE chance** of availability at {target_time.strftime('%I:%M %p')}")
            else:
                st.error(f"❌ **LOW chance** of availability at {target_time.strftime('%I:%M %p')}. Try another slot.")
        
        # Model Selector - Allow user to choose specific models
        if len(result['predictions']) > 0:
            st.markdown("---")
            st.markdown("### 🔬 Explore Other Models")
            
            # Create list of available models
            available_models = list(result['predictions'].keys())
            model_display_names = {
                'statistical': '📊 Statistical Ensemble',
                'prophet': '📈 Prophet (Time Series)',
                'lstm': '🧠 LSTM (Deep Learning)'
            }
            
            # Create options for dropdown
            model_options = [model_display_names.get(m, m.title()) for m in available_models]
            
            # Default to recommended model if available, otherwise first model
            default_index = 0
            if result.get('recommended_model'):
                recommended_key = None
                for key in available_models:
                    if result['recommended_model'].lower() in key or key in result['recommended_model'].lower():
                        recommended_key = key
                        break
                if recommended_key and recommended_key in available_models:
                    default_index = available_models.index(recommended_key)
            
            selected_model_display = st.selectbox(
                "Choose a model to view detailed predictions:",
                options=model_options,
                index=default_index,
                help="Select any model to see its prediction, even if it's not the recommended one"
            )
            
            # Get the actual model key from display name
            selected_model = available_models[model_options.index(selected_model_display)]
            selected_pred = result['predictions'][selected_model]
            
            # Display selected model's prediction
            st.markdown(f"#### {selected_model_display} Prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Probability EMPTY", f"{selected_pred['probability_empty']*100:.1f}%")
                st.progress(float(selected_pred['probability_empty']))
            
            with col2:
                st.metric("Probability OCCUPIED", f"{selected_pred['probability_occupied']*100:.1f}%")
                st.progress(float(selected_pred['probability_occupied']))
            
            # Show model-specific details
            if selected_model == 'statistical':
                if 'confidence' in selected_pred:
                    st.metric("Model Confidence", f"{selected_pred['confidence']*100:.1f}%")
                
                if 'models_used' in selected_pred:
                    st.caption(f"🔧 Sub-models: {', '.join(selected_pred['models_used'])}")
                
                if 'individual_predictions' in selected_pred:
                    with st.expander("View Individual Statistical Models", expanded=False):
                        for model_name, pred_value in selected_pred['individual_predictions'].items():
                            st.write(f"**{model_name.replace('_', ' ').title()}:** {pred_value*100:.1f}% occupied")
            
            elif selected_model == 'prophet':
                if 'confidence' in selected_pred:
                    st.metric("Model Confidence", f"{selected_pred['confidence']*100:.1f}%")
                
                if 'prediction_interval' in selected_pred:
                    interval = selected_pred['prediction_interval']
                    st.write(f"**95% Confidence Interval:** [{interval['lower']*100:.1f}%, {interval['upper']*100:.1f}%]")
                    st.caption("Prophet provides uncertainty bounds based on historical patterns")
            
            elif selected_model == 'lstm':
                if 'confidence' in selected_pred:
                    st.metric("Model Confidence", f"{selected_pred.get('confidence', 0.75)*100:.1f}%")
                
                if 'training_samples' in selected_pred:
                    st.caption(f"🎓 Trained on {selected_pred['training_samples']} sequence samples")
                    st.caption("LSTM learns complex temporal patterns from sequential data")
        
        # Original detailed expanders (now collapsed by default)
        if len(result['predictions']) > 1:
            st.markdown("---")
            st.markdown("### 📋 All Model Comparisons")
            st.caption("Expand to compare all available model predictions side-by-side")
            
            # Statistical Models
            if 'statistical' in result['predictions']:
                with st.expander("📊 Statistical Models (Ensemble)", expanded=False):
                    stat = result['predictions']['statistical']
                    
                    st.metric("Ensemble Prediction", f"{stat['probability_occupied']*100:.1f}% occupied")
                    st.metric("Confidence", f"{stat['confidence']*100:.1f}%")
                    
                    if 'models_used' in stat:
                        st.caption(f"Models used: {', '.join(stat['models_used'])}")
                    
                    if 'individual_predictions' in stat:
                        st.markdown("**Individual Model Predictions:**")
                        for model_name, pred_value in stat['individual_predictions'].items():
                            st.write(f"- {model_name.replace('_', ' ').title()}: {pred_value*100:.1f}%")
            
            # Prophet Model
            if 'prophet' in result['predictions']:
                with st.expander("📈 Prophet Model", expanded=False):
                    prophet = result['predictions']['prophet']
                    
                    st.metric("Prophet Prediction", f"{prophet['probability_occupied']*100:.1f}% occupied")
                    st.metric("Confidence", f"{prophet['confidence']*100:.1f}%")
                    
                    if 'prediction_interval' in prophet:
                        interval = prophet['prediction_interval']
                        st.write(f"**95% Confidence Interval:** [{interval['lower']*100:.1f}%, {interval['upper']*100:.1f}%]")
            
            # LSTM Model
            if 'lstm' in result['predictions']:
                with st.expander("🧠 LSTM Deep Learning Model", expanded=False):
                    lstm = result['predictions']['lstm']
                    
                    st.metric("LSTM Prediction", f"{lstm['probability_occupied']*100:.1f}% occupied")
                    
                    if 'training_samples' in lstm:
                        st.caption(f"Trained on {lstm['training_samples']} sequences")
        
        else:
            st.warning("Only one prediction model available with current data volume.")
    
    # Historical pattern
    st.markdown("---")
    st.markdown("### 📈 Historical Pattern")
    
    if len(slot_events) > 0:
        df = pd.DataFrame(slot_events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['is_occupied'] = (df['status'] == 'occupied').astype(int)
        
        # Simple time series chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['is_occupied'],
            mode='lines',
            name='Status',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.3)'
        ))
        
        fig.update_layout(
            title=f"Occupancy History - Slot {selected_slot}",
            xaxis_title="Time",
            yaxis_title="Status",
            yaxis=dict(tickvals=[0, 1], ticktext=['Empty', 'Occupied']),
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, config={'displayModeBar': False})
