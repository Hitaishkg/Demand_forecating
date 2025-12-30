"""
Demand Forecasting Dashboard - Homepage
Multi-Model Demand Forecasting for E-Commerce
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.helpers import load_segments, load_results, format_number

# Page config
st.set_page_config(
    page_title="Demand Forecasting Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛒 Demand Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Model Architecture for E-Commerce Inventory Optimization</div>', unsafe_allow_html=True)

st.markdown("---")

# Load data
try:
    segments_df = load_segments()
    results_df = load_results()
    
    # ========================================
    # KEY METRICS
    # ========================================
    st.markdown('<div class="section-header">📊 Project Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Products Analyzed",
            value=format_number(len(segments_df)),
            help="Total unique product-store combinations"
        )
    
    with col2:
        avg_mape = results_df['mape'].mean()
        st.metric(
            label="🎯 Average MAPE",
            value=f"{avg_mape:.1f}%",
            delta=f"{48.8 - avg_mape:.1f}% vs baseline",
            delta_color="inverse",
            help="Mean Absolute Percentage Error"
        )
    
    with col3:
        best_segment = results_df.loc[results_df['mape'].idxmin(), 'segment']
        best_mape = results_df['mape'].min()
        st.metric(
            label="⭐ Best Segment",
            value=best_segment,
            delta=f"{best_mape:.1f}% MAPE",
            help="Segment with lowest forecast error"
        )
    
    with col4:
        total_revenue = segments_df['total_revenue'].sum()
        st.metric(
            label="💰 Total Revenue",
            value=format_number(total_revenue, prefix="$", decimals=1),
            help="5-year cumulative revenue across all products"
        )
    
    st.markdown("---")
    
    # ========================================
    # SYSTEM ARCHITECTURE
    # ========================================
    st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This project implements a **production-grade multi-model forecasting system** mimicking architectures 
    used by Amazon, Walmart, and Target. Key features:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Core Components
        
        1. **ABC-XYZ Segmentation**
           - Revenue-based classification (ABC)
           - Variability-based classification (XYZ)
           - 9 distinct product segments
        
        2. **Multi-Model Architecture**
           - LightGBM for high-volume products
           - Rolling averages for erratic demand
           - Model routing by segment
        
        3. **Feature Engineering**
           - Lag features (1, 7, 14, 28 days)
           - Rolling windows (7, 14, 28, 90 days)
           - Calendar effects & price features
        """)
    
    with col2:
        st.markdown("""
        #### 💼 Business Logic
        
        1. **Safety Stock Calculation**
           - Service level optimization (85-98%)
           - Lead time consideration
           - CV-based buffer sizing
        
        2. **ROI Analysis**
           - Holding cost savings (15-20%)
           - Stockout reduction (30-40%)
           - $2.9M potential annual savings
        
        3. **Strategic Recommendations**
           - Model selection by segment
           - Update frequency guidelines
           - Investment prioritization
        """)
    
    # Architecture Diagram
    st.markdown("#### 📐 Data Flow Architecture")
    
    st.code("""
┌─────────────────────────────────────────────┐
│         DATA LAYER (M5 Walmart)             │
│  30,490 products × 1,913 days               │
│  Sales + Prices + Calendar + Promotions     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│       SEGMENTATION LAYER                     │
│  ABC Analysis (Revenue) × XYZ (Variability) │
│  → Routes products to optimal models        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         MODELING LAYER                       │
│  LightGBM | Rolling Avg | Prophet (future)  │
│  88.7% feature importance: rolling_mean_7   │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│      BUSINESS LOGIC LAYER                    │
│  Safety Stock | Reorder Points | ROI        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          DEPLOYMENT (Streamlit)              │
│  Interactive Dashboard + Forecasting Tool    │
└─────────────────────────────────────────────┘
    """, language="text")
    
    st.markdown("---")
    
    # ========================================
    # TECHNOLOGY STACK
    # ========================================
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Processing**
        - Pandas & NumPy
        - Scipy (statistics)
        - 58.5M rows processed
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning**
        - LightGBM (primary model)
        - Scikit-learn (preprocessing)
        - 5-fold cross-validation
        """)
    
    with col3:
        st.markdown("""
        **Visualization**
        - Streamlit (dashboard)
        - Plotly (interactive charts)
        - Matplotlib & Seaborn
        """)
    
    st.markdown("---")
    
    # ========================================
    # QUICK STATISTICS
    # ========================================
    st.markdown('<div class="section-header">📈 Quick Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Segmentation Distribution**")
        segment_counts = segments_df['ABC_XYZ'].value_counts().head(5)
        for seg, count in segment_counts.items():
            pct = (count / len(segments_df)) * 100
            st.write(f"**{seg}**: {count} ({pct:.1f}%)")
    
    with col2:
        st.markdown("**Performance by ABC Category**")
        for abc in ['A', 'B', 'C']:
            abc_data = results_df[results_df['segment'].str.startswith(abc)]
            if not abc_data.empty:
                avg_mape = abc_data['mape'].mean()
                st.write(f"**Category {abc}**: {avg_mape:.1f}% MAPE")
    
    with col3:
        st.markdown("**Revenue by Category**")
        cat_revenue = segments_df.groupby('cat_id')['total_revenue'].sum().sort_values(ascending=False)
        for cat, rev in cat_revenue.items():
            pct = (rev / total_revenue) * 100
            st.write(f"**{cat}**: ${format_number(rev, decimals=1)} ({pct:.1f}%)")
    
    st.markdown("---")
    
    # ========================================
    # NAVIGATION GUIDE
    # ========================================
    st.markdown('<div class="section-header">🧭 Navigation Guide</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Exploratory Data Analysis**
        - Dataset overview and statistics
        - ABC-XYZ segmentation visualization
        - Product distribution analysis
        - Interactive product explorer
        """)
        
        st.success("""
        **🎯 Forecaster**
        - Select any product for prediction
        - 7-day demand forecast
        - Model confidence intervals
        - Forecast visualization
        """)
    
    with col2:
        st.warning("""
        **🤖 Model Results**
        - Performance metrics by segment
        - Feature importance analysis
        - Baseline vs LightGBM comparison
        - Error distribution analysis
        """)
        
        st.error("""
        **💼 Business Logic**
        - Safety stock calculator
        - Reorder point optimization
        - ROI analysis tool
        - Strategic recommendations
        """)
    
    st.markdown("---")
    
    # ========================================
    # FOOTER
    # ========================================
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Demand Forecasting Dashboard v1.0</strong></p>
        <p>Built with Streamlit | Data: M5 Walmart Dataset | Model: LightGBM</p>
        <p>📧 Contact: hitaishkg@gmail.com | 🔗 GitHub: github.com/Hitaishkg</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure data files are in the correct location relative to the app.")
    import traceback
    st.code(traceback.format_exc())