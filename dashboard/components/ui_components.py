"""
Enhanced UI Components for Parking Prediction Dashboard
Professional, minimal, and informative design system
"""

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==================== Color System ====================

class Colors:
    """Dark theme color palette - High contrast, easy on eyes"""
    
    # Primary Gradients (Vibrant for dark theme)
    PRIMARY_GRADIENT = "linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)"
    SUCCESS_GRADIENT = "linear-gradient(90deg, #10b981 0%, #34d399 100%)"
    WARNING_GRADIENT = "linear-gradient(90deg, #f59e0b 0%, #ef4444 100%)"
    INFO_GRADIENT = "linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)"
    
    # Semantic Colors (Brighter for dark backgrounds)
    SUCCESS = "#34d399"
    SUCCESS_DARK = "#10b981"
    WARNING = "#f87171"
    WARNING_DARK = "#fbbf24"
    ERROR = "#ef4444"
    INFO = "#06b6d4"
    
    # Status Colors (High visibility on dark)
    EMPTY = "#34d399"  # Bright green - available
    OCCUPIED = "#f87171"  # Bright red - occupied
    UNKNOWN = "#fbbf24"  # Bright yellow - unknown
    
    # Dark Theme Colors
    BACKGROUND = "#0a0a0a"  # Near black
    CARD_BG = "#1a1a1a"  # Dark gray
    CARD_BG_HOVER = "#222222"  # Slightly lighter on hover
    TEXT_PRIMARY = "#f0f0f0"  # Almost white
    TEXT_SECONDARY = "#a0a0a0"  # Light gray
    BORDER = "#333333"  # Dark border
    
    # Chart Colors (Vivid for dark theme)
    HISTORICAL = "#8b5cf6"  # Bright purple
    PREDICTED = "#f87171"  # Bright red
    CONFIDENCE_BAND = "rgba(248, 113, 113, 0.2)"


def load_custom_css():
    """Load enhanced dark theme CSS"""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Dark Theme - Main Layout */
        .main {
            padding: 2rem;
            background: #0a0a0a;
        }
        
        .stApp {
            background: #0a0a0a;
            color: #f0f0f0;
        }
        
        /* Content Cards - Dark */
        .content-card {
            background: #1a1a1a;
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            margin-bottom: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            border: 1px solid #333333;
        }
        
        .content-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 48px rgba(0,0,0,0.6);
            background: #222222;
        }
        
        /* Metric Cards - Vibrant on Dark */
        .metric-card {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            padding: 24px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(124, 58, 237, 0.3);
            transition: all 0.3s ease;
            border: 2px solid rgba(168, 85, 247, 0.3);
        }
        
        .metric-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 12px 28px rgba(124, 58, 237, 0.5);
            border-color: rgba(168, 85, 247, 0.6);
        }
        
        .metric-card h3 {
            margin: 0;
            font-size: 2.8rem;
            font-weight: 700;
            line-height: 1;
            text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        
        .metric-card p {
            margin: 8px 0 0 0;
            font-size: 0.95rem;
            opacity: 0.95;
            font-weight: 500;
        }
        
        .metric-icon {
            font-size: 2.2rem;
            margin-bottom: 8px;
            opacity: 0.9;
        }
        
        /* Status Banners - High Contrast */
        .status-banner {
            padding: 16px 24px;
            border-radius: 12px;
            margin: 16px 0;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .success-banner {
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            color: white;
        }
        
        .warning-banner {
            background: linear-gradient(90deg, #f59e0b 0%, #ef4444 100%);
            color: white;
        }
        
        .info-banner {
            background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
            color: white;
        }
        
        /* Headers - Light on Dark */
        .header-title {
            color: #f0f0f0;
            font-size: 3.2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 12px;
            text-shadow: 0 4px 12px rgba(0,0,0,0.7);
            letter-spacing: -0.5px;
        }
        
        .header-subtitle {
            color: #a0a0a0;
            font-size: 1.25rem;
            text-align: center;
            margin-bottom: 36px;
            font-weight: 400;
        }
        
        .section-header {
            color: #f0f0f0;
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        /* Buttons - Bright on Dark */
        .stButton>button {
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            color: white;
            border: none;
            padding: 14px 32px;
            border-radius: 28px;
            font-weight: 600;
            font-size: 1.05rem;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
            transition: all 0.3s ease;
            border: 2px solid rgba(52, 211, 153, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 24px rgba(16, 185, 129, 0.6);
            border-color: rgba(52, 211, 153, 0.6);
        }
        
        .stButton>button:active {
            transform: translateY(0px);
        }
        
        /* Tabs - Dark Theme */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: #1a1a1a;
            padding: 8px;
            border-radius: 16px;
            border: 1px solid #333333;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 56px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 12px;
            color: #a0a0a0;
            font-weight: 600;
            font-size: 1.05rem;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #222222;
            color: #f0f0f0;
        }
        
        /* Progress Bar - Bright */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        }
        
        /* File Uploader - Dark */
        [data-testid="stFileUploader"] {
            background: #1a1a1a;
            padding: 24px;
            border-radius: 12px;
            border: 2px dashed #333333;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #7c3aed;
            background: #222222;
        }
        
        /* Selectbox / Dropdown - Dark */
        .stSelectbox > div > div {
            background: #1a1a1a;
            border-radius: 10px;
            border: 2px solid #333333;
            transition: all 0.3s ease;
            color: #f0f0f0;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #7c3aed;
            background: #222222;
        }
        
        /* Text Input - Dark */
        .stTextInput > div > div > input {
            background: #1a1a1a;
            color: #f0f0f0;
            border: 2px solid #333333;
            border-radius: 10px;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #7c3aed;
            background: #222222;
        }
        
        /* Number Input - Dark */
        .stNumberInput > div > div > input {
            background: #1a1a1a;
            color: #f0f0f0;
            border: 2px solid #333333;
        }
        
        /* Slider - Bright */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%);
        }
        
        /* Infographic Elements */
        .info-stat {
            display: inline-block;
            background: #222222;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            color: #7c3aed;
            margin: 4px;
            border: 2px solid #333333;
        }
        
        .status-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 16px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .status-empty {
            background: #065f46;
            color: #34d399;
            border: 1px solid #10b981;
        }
        
        .status-occupied {
            background: #7f1d1d;
            color: #f87171;
            border: 1px solid #ef4444;
        }
        
        /* Sidebar - Dark */
        section[data-testid="stSidebar"] {
            background: #0f0f0f;
            border-right: 1px solid #333333;
        }
        
        section[data-testid="stSidebar"] > div {
            padding-top: 2rem;
            background: #0f0f0f;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #f0f0f0;
        }
        
        /* Divider */
        hr {
            margin: 24px 0;
            border: none;
            border-top: 2px solid #333333;
        }
        
        /* Metric Widget - Dark Theme Override */
        [data-testid="stMetricValue"] {
            color: #f0f0f0;
        }
        
        [data-testid="stMetricDelta"] {
            color: #34d399;
        }
        
        /* Expander - Dark */
        .streamlit-expanderHeader {
            background: #1a1a1a;
            color: #f0f0f0;
            border: 1px solid #333333;
        }
        
        .streamlit-expanderHeader:hover {
            background: #222222;
        }
        
        /* Info/Warning/Error boxes */
        .stAlert {
            background: #1a1a1a;
            border: 1px solid #333333;
            color: #f0f0f0;
        }
        
        /* Smooth Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .content-card {
            animation: fadeIn 0.4s ease;
        }
        
        /* Tooltip */
        [data-testid="stTooltipIcon"] {
            color: #7c3aed;
        }
        
        /* Make all text readable on dark background */
        p, span, div, label {
            color: #f0f0f0;
        }
        
        /* Markdown text */
        .stMarkdown {
            color: #f0f0f0;
        }
        
        /* Code blocks */
        code {
            background: #222222;
            color: #34d399;
            padding: 2px 6px;
            border-radius: 4px;
        }
        
        pre {
            background: #1a1a1a;
            border: 1px solid #333333;
        }
        
        </style>
    """, unsafe_allow_html=True)


# ==================== UI Components ====================


# ==================== UI Components ====================

def render_header(title: str, subtitle: str, icon: str = ""):
    """Render enhanced header with icon"""
    st.markdown(f"""
        <h1 class="header-title">{title}</h1>
        <p class="header-subtitle">{subtitle}</p>
    """, unsafe_allow_html=True)


def render_metric_card(value: str, label: str, icon: str = "", 
                       gradient: str = Colors.PRIMARY_GRADIENT,
                       trend: str = None, trend_value: str = None,
                       subtitle: str = None) -> str:
    """
    Render clean minimal metric card
    
    Args:
        value: Main metric value
        label: Metric label  
        icon: Emoji icon
        gradient: Background gradient
        trend: 'up' or 'down'
        trend_value: Change percentage
        subtitle: Additional text
    """
    
    # Trend indicator
    trend_html = ""
    if trend and trend_value:
        trend_color = "#34d399" if trend == "up" else "#f87171"
        trend_arrow = "▲" if trend == "up" else "▼"
        trend_html = f'<div style="color: {trend_color}; font-size: 13px; font-weight: 600; margin-top: 8px;"><span>{trend_arrow}</span> <span>{trend_value}</span></div>'
    
    # Subtitle
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="color: rgba(255,255,255,0.7); font-size: 12px; margin-top: 6px;">{subtitle}</div>'
    
    return f'''
    <div style="
        background: {gradient};
        border-radius: 16px;
        padding: 24px;
        height: 100%;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    ">
        <div style="font-size: 28px; margin-bottom: 12px;">{icon}</div>
        <div style="font-size: 36px; font-weight: 700; color: white; margin-bottom: 8px;">{value}</div>
        <div style="font-size: 13px; color: rgba(255,255,255,0.85); font-weight: 500;">{label}</div>
        {trend_html}
        {subtitle_html}
    </div>
    '''


def render_stat_card(title: str, value: str, change: str = None, 
                     icon: str = "", color: str = "#7c3aed") -> str:
    """
    Render minimal stat card
    
    Args:
        title: Statistic title
        value: Main value
        change: Change indicator
        icon: Icon/emoji
        color: Accent color
    """
    
    change_html = ""
    if change:
        is_positive = change.startswith("+")
        change_color = "#34d399" if is_positive else "#f87171"
        change_html = f'<span style="color: {change_color}; font-size: 14px; font-weight: 600; margin-left: 8px;">{change}</span>'
    
    return f'''
    <div style="
        background: #1a1a1a;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #2a2a2a;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
            <span style="color: #a0a0a0; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">{title}</span>
            <span style="font-size: 18px;">{icon}</span>
        </div>
        <div style="display: flex; align-items: baseline;">
            <span style="font-size: 32px; font-weight: 700; color: white;">{value}</span>
            {change_html}
        </div>
    </div>
    '''


def render_status_banner(message: str, type: str = "info"):
    """
    Render status banner
    
    Args:
        message: Banner message
        type: 'success', 'warning', or 'info'
    """
    # Emoticons removed
    icons = {
        'success': '',
        'warning': '',
        'info': ''
    }
    st.markdown(f"""
        <div class="status-banner {type}-banner">
            <span>{icons.get(type, '')}</span>
            <span>{message}</span>
        </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = ""):
    """Render section header with icon"""
    st.markdown(f"""
        <h2 class="section-header">
            {f'<span>{icon}</span>' if icon else ''}
            <span>{title}</span>
        </h2>
    """, unsafe_allow_html=True)


def render_progress_indicator(current: int, total: int, label: str = ""):
    """Render enhanced progress indicator"""
    percentage = (current / total * 100) if total > 0 else 0
    st.markdown(f"**{label}** {current}/{total}")
    st.progress(percentage / 100)


def create_occupancy_chart(historical_data: List[Dict], 
                          prediction_data: Optional[List[Dict]] = None):
    """
    Create professional occupancy chart with predictions
    
    Args:
        historical_data: List of dicts with 'timestamp' and 'occupancy'
        prediction_data: Optional predictions with 'timestamp', 'value', 'upper', 'lower'
    """
    fig = go.Figure()
    
    # Historical data
    if historical_data:
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in historical_data],
            y=[d['occupancy'] for d in historical_data],
            mode='lines',
            name='Historical',
            line=dict(color=Colors.HISTORICAL, width=3),
            hovertemplate='<b>Time:</b> %{x}<br><b>Occupancy:</b> %{y:.1%}<extra></extra>'
        ))
    
    # Predictions with confidence band
    if prediction_data:
        # Confidence band upper
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in prediction_data],
            y=[d.get('upper', d['value']) for d in prediction_data],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Confidence band lower
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in prediction_data],
            y=[d.get('lower', d['value']) for d in prediction_data],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            fillcolor=Colors.CONFIDENCE_BAND,
            name='Confidence Interval',
            hoverinfo='skip'
        ))
        
        # Prediction line
        fig.add_trace(go.Scatter(
            x=[d['timestamp'] for d in prediction_data],
            y=[d['value'] for d in prediction_data],
            mode='lines',
            name='Predicted',
            line=dict(color=Colors.PREDICTED, width=3, dash='dash'),
            hovertemplate='<b>Time:</b> %{x}<br><b>Predicted:</b> %{y:.1%}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Occupancy Timeline',
            font=dict(size=20, family='Inter', weight=600, color='#f0f0f0')
        ),
        xaxis_title='Time',
        yaxis_title='Occupancy Probability',
        hovermode='x unified',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(family='Inter', color='#f0f0f0'),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#f0f0f0')
        ),
        yaxis=dict(
            tickformat='.0%',
            gridcolor='#333333',
            color='#f0f0f0'
        ),
        xaxis=dict(
            gridcolor='#333333',
            color='#f0f0f0'
        )
    )
    
    return fig


def create_heatmap(data: Dict[str, List[float]], title: str = "Weekly Occupancy Pattern"):
    """
    Create occupancy heatmap
    
    Args:
        data: Dict with 'days' (list), 'hours' (list), 'values' (2D array)
    """
    fig = go.Figure(data=go.Heatmap(
        z=data['values'],
        x=data['hours'],
        y=data['days'],
        colorscale=[
            [0, Colors.SUCCESS],      # Low occupancy - green
            [0.5, Colors.WARNING_DARK],  # Medium - yellow/orange
            [1, Colors.WARNING]       # High occupancy - red
        ],
        hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Occupancy: %{z:.0%}<extra></extra>',
        colorbar=dict(
            title="Occupancy",
            tickformat='.0%'
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family='Inter', weight=600, color='#f0f0f0')
        ),
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(family='Inter', color='#f0f0f0'),
        margin=dict(l=100, r=40, t=60, b=60),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            color='#f0f0f0'
        ),
        yaxis=dict(
            color='#f0f0f0'
        )
    )
    
    return fig


def create_donut_chart(empty: int, occupied: int):
    """Create donut chart for current occupancy"""
    total = empty + occupied
    if total == 0:
        return None
    
    fig = go.Figure(data=[go.Pie(
        labels=['Empty', 'Occupied'],
        values=[empty, occupied],
        hole=.65,
        marker_colors=[Colors.SUCCESS, Colors.WARNING],
        textinfo='label+percent',
        textfont_size=16,
        textfont_family='Inter',
        textfont_weight=600,
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(
            text=f'<b>{total}</b><br>Total<br>Slots',
            x=0.5, y=0.5,
            font_size=22,
            font_family='Inter',
            showarrow=False
        )]
    )
    
    return fig
