"""
Helper functions for Streamlit app
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# Define data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

@st.cache_data
def load_segments():
    """Load product segmentation data"""
    path = DATA_DIR / "processed" / "products_segmented.csv"
    return pd.read_csv(path)

@st.cache_data
def load_results():
    """Load model performance results"""
    path = OUTPUTS_DIR / "results" / "performance_by_segment_1000products.csv"
    df = pd.read_csv(path)
    # Use original column names (lowercase)
    return df

@st.cache_data
def load_features():
    """Load feature-engineered sales data"""
    path = DATA_DIR / "processed" / "sales_1000_with_features.csv"
    df = pd.read_csv(path, parse_dates=['date'])
    return df

@st.cache_resource
def load_model():
    """Load trained LightGBM model"""
    path = MODELS_DIR / "lgb_model_1000products.txt"
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(path))
    return model

def create_abc_xyz_heatmap(segments_df):
    """Create ABC-XYZ heatmap"""
    # Create cross-tabulation
    matrix = pd.crosstab(
        segments_df['ABC'],
        segments_df['XYZ']
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale='YlOrRd',
        text=matrix.values,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Product Count")
    ))
    
    fig.update_layout(
        title="ABC-XYZ Product Distribution",
        xaxis_title="XYZ (Demand Variability)",
        yaxis_title="ABC (Revenue Importance)",
        height=500
    )
    
    return fig

def create_segment_bar_chart(results_df):
    """Create bar chart of MAPE by segment"""
    # Sort by MAPE (use lowercase column names!)
    df_sorted = results_df.sort_values('mape')
    
    fig = px.bar(
        df_sorted,
        x='segment',  # lowercase!
        y='mape',     # lowercase!
        color='mape', # lowercase!
        color_continuous_scale='RdYlGn_r',
        labels={'segment': 'Segment', 'mape': 'MAPE (%)'},
        title='Forecast Accuracy (MAPE) by Segment',
        hover_data={
            'segment': True,
            'mape': ':.1f',
            'n_products': True
        }
    )
    
    fig.update_layout(
        xaxis_title="Segment",
        yaxis_title="MAPE (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def format_number(num, prefix="", suffix="", decimals=0):
    """Format large numbers with K/M/B suffixes"""
    if num >= 1e9:
        return f"{prefix}{num/1e9:.{decimals}f}B{suffix}"
    elif num >= 1e6:
        return f"{prefix}{num/1e6:.{decimals}f}M{suffix}"
    elif num >= 1e3:
        return f"{prefix}{num/1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{num:.{decimals}f}{suffix}"

def calculate_safety_stock(mean_demand, std_demand, lead_time_days, service_level=0.95):
    """Calculate safety stock using normal distribution"""
    from scipy import stats
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * std_demand * (lead_time_days ** 0.5)
    return safety_stock

def calculate_reorder_point(mean_demand, lead_time_days, safety_stock):
    """Calculate reorder point"""
    lead_time_demand = mean_demand * lead_time_days
    reorder_point = lead_time_demand + safety_stock
    return reorder_point