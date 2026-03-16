"""
Parking Layout Diagram Component
Generates a schematic parking layout based on slot annotations
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Dict


def create_parking_layout_diagram(db, parking_lot_id: int, annotations: List[Dict]) -> go.Figure:
    """
    Create a schematic parking layout diagram showing slot positions and statuses
    
    Args:
        db: Database instance
        parking_lot_id: Parking lot ID
        annotations: List of slot annotations with coordinates
    
    Returns:
        Plotly figure with parking layout
    """
    
    if not annotations:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No parking slots annotated yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Get current status for each slot
    slot_data = []
    for ann in annotations:
        # Get latest status
        events = db.get_occupancy_events(parking_lot_id, ann['slot_id'], limit=1)
        status = events[0]['status'] if events else 'unknown'
        
        # Calculate center position
        center_x = (ann['x1'] + ann['x2']) / 2
        center_y = (ann['y1'] + ann['y2']) / 2
        width = ann['x2'] - ann['x1']
        height = ann['y2'] - ann['y1']
        
        slot_data.append({
            'slot_id': ann['slot_id'],
            'x': center_x,
            'y': center_y,
            'width': width,
            'height': height,
            'status': status,
            'x1': ann['x1'],
            'y1': ann['y1'],
            'x2': ann['x2'],
            'y2': ann['y2']
        })
    
    # Sort slots by position to create a logical layout
    # Group by rows (similar Y coordinates) and then by columns (X coordinates)
    slot_data.sort(key=lambda s: (s['y'], s['x']))
    
    # Determine rows based on Y clustering
    rows = []
    current_row = []
    threshold_y = 50  # pixels - adjust based on typical slot spacing
    
    for slot in slot_data:
        if not current_row:
            current_row.append(slot)
        else:
            # Check if this slot is in the same row as the last one
            if abs(slot['y'] - current_row[-1]['y']) < threshold_y:
                current_row.append(slot)
            else:
                # Start new row
                rows.append(sorted(current_row, key=lambda s: s['x']))
                current_row = [slot]
    
    # Add last row
    if current_row:
        rows.append(sorted(current_row, key=lambda s: s['x']))
    
    # Create figure
    fig = go.Figure()
    
    # Define colors
    color_map = {
        'empty': '#22c55e',      # Green
        'occupied': '#ef4444',    # Red
        'unknown': '#9ca3af'      # Gray
    }
    
    # Calculate schematic positions
    slot_width = 80
    slot_height = 40
    row_spacing = 60
    col_spacing = 90
    start_x = 50
    start_y = 50
    
    # Draw slots
    for row_idx, row in enumerate(rows):
        for col_idx, slot in enumerate(row):
            # Schematic position
            x = start_x + col_idx * col_spacing
            y = start_y + row_idx * row_spacing
            
            # Get color
            color = color_map.get(slot['status'], color_map['unknown'])
            
            # Draw slot rectangle
            fig.add_shape(
                type="rect",
                x0=x, y0=y,
                x1=x + slot_width, y1=y + slot_height,
                fillcolor=color,
                line=dict(color="white", width=2),
                opacity=0.8
            )
            
            # Add slot label
            label_x = x + slot_width / 2
            label_y = y + slot_height / 2
            
            # Extract slot number from slot_id (e.g., "slot_1" -> "1")
            slot_number = slot['slot_id'].replace('slot_', '').replace('Slot', '')
            
            # Add text annotation
            fig.add_annotation(
                x=label_x, y=label_y,
                text=f"<b>{slot_number}</b>",
                showarrow=False,
                font=dict(size=14, color="white", family="Arial Black"),
                xref="x", yref="y"
            )
            
            # Add invisible scatter for better hover
            status_emoji = {
                'empty': '🟢',
                'occupied': '🔴',
                'unknown': '⚪'
            }
            
            fig.add_trace(go.Scatter(
                x=[label_x],
                y=[label_y],
                mode='markers',
                marker=dict(size=1, color='rgba(0,0,0,0)'),
                showlegend=False,
                hovertemplate=f"<b>{slot['slot_id']}</b><br>Status: {slot['status'].upper()}<extra></extra>",
                name=slot['slot_id']
            ))
    
    # Calculate figure dimensions
    max_cols = max(len(row) for row in rows)
    max_rows = len(rows)
    
    fig_width = max_cols * col_spacing + 100
    fig_height = max_rows * row_spacing + 100
    
    # Update layout
    fig.update_xaxes(
        range=[0, fig_width],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )
    
    fig.update_yaxes(
        range=[0, fig_height],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Parking Layout Reference</b><br><sub>🟢 Empty  |  🔴 Occupied  |  ⚪ Unknown</sub>",
            font=dict(size=16, color="white"),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(30, 30, 40, 0.9)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=min(400, fig_height + 50),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig
