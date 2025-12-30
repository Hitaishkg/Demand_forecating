"""
EDA Page - Exploratory Data Analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import (
    load_segments, load_results, 
    create_abc_xyz_heatmap, format_number
)

# Page config
st.set_page_config(
    page_title="EDA - Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Exploratory Data Analysis")
st.markdown("Comprehensive analysis of M5 Walmart dataset and ABC-XYZ segmentation")
st.markdown("---")

# Load data
try:
    segments_df = load_segments()
    results_df = load_results()
    
    # ========================================
    # SECTION 1: Dataset Overview
    # ========================================
    st.header("1️⃣ Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", format_number(len(segments_df)))
    
    with col2:
        n_categories = segments_df['cat_id'].nunique()
        st.metric("Categories", n_categories)
    
    with col3:
        n_stores = segments_df['store_id'].nunique()
        st.metric("Stores", n_stores)
    
    with col4:
        n_states = segments_df['state_id'].nunique()
        st.metric("States", n_states)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 2: ABC-XYZ Segmentation
    # ========================================
    st.header("2️⃣ ABC-XYZ Segmentation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap
        fig = create_abc_xyz_heatmap(segments_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Segment distribution
        st.subheader("Segment Distribution")
        
        segment_counts = segments_df['ABC_XYZ'].value_counts().sort_index()
        
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            labels={'x': 'Segment', 'y': 'Product Count'},
            title='Products per Segment',
            color=segment_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_traces(text=segment_counts.values, textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment details table
    st.subheader("Segment Statistics")
    
    segment_stats = segments_df.groupby('ABC_XYZ').agg({
        'id': 'count',
        'mean_sales': 'mean',
        'cv': 'mean',
        'total_revenue': 'sum',
        'zero_pct': 'mean'
    }).reset_index()
    
    segment_stats.columns = ['Segment', 'Product_Count', 'Avg_Daily_Sales', 
                             'Avg_CV', 'Total_Revenue', 'Avg_Zero_Pct']
    
    # Format the dataframe
    segment_stats['Avg_Daily_Sales'] = segment_stats['Avg_Daily_Sales'].round(2)
    segment_stats['Avg_CV'] = segment_stats['Avg_CV'].round(2)
    segment_stats['Total_Revenue'] = segment_stats['Total_Revenue'].apply(lambda x: f"${x:,.0f}")
    segment_stats['Avg_Zero_Pct'] = segment_stats['Avg_Zero_Pct'].round(1)
    
    st.dataframe(segment_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 3: Sales Distribution Analysis
    # ========================================
    st.header("3️⃣ Sales Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by segment - box plot
        st.subheader("Sales Distribution by Segment")
        
        # Prepare data for box plot
        plot_data = segments_df[['ABC_XYZ', 'mean_sales']].copy()
        plot_data = plot_data[plot_data['mean_sales'] < 50]  # Remove outliers for better viz
        
        fig = px.box(
            plot_data,
            x='ABC_XYZ',
            y='mean_sales',
            title='Average Daily Sales by Segment (outliers removed)',
            labels={'mean_sales': 'Avg Daily Sales', 'ABC_XYZ': 'Segment'},
            color='ABC_XYZ'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CV distribution by ABC category
        st.subheader("Demand Variability (CV) by ABC")
        
        fig = px.violin(
            segments_df,
            x='ABC',
            y='cv',
            title='Coefficient of Variation by ABC Category',
            labels={'cv': 'CV', 'ABC': 'ABC Category'},
            color='ABC',
            box=True
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales by ABC category - bar chart
    st.subheader("Average Sales by ABC-XYZ Combination")
    
    abc_xyz_sales = segments_df.groupby(['ABC', 'XYZ'])['mean_sales'].mean().reset_index()
    abc_xyz_sales['ABC_XYZ'] = abc_xyz_sales['ABC'] + abc_xyz_sales['XYZ']
    
    fig = px.bar(
        abc_xyz_sales,
        x='ABC_XYZ',
        y='mean_sales',
        color='ABC',
        title='Average Daily Sales by Segment',
        labels={'mean_sales': 'Avg Daily Sales', 'ABC_XYZ': 'Segment'},
        text='mean_sales',
        color_discrete_map={'A': '#FF4B4B', 'B': '#FFA500', 'C': '#4B4BFF'}
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 4: Revenue Analysis
    # ========================================
    st.header("4️⃣ Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by category - sunburst
        st.subheader("Revenue Distribution")
        
        # Prepare hierarchical data
        cat_dept_revenue = segments_df.groupby(['cat_id', 'dept_id'])['total_revenue'].sum().reset_index()
        
        fig = px.sunburst(
            cat_dept_revenue,
            path=['cat_id', 'dept_id'],
            values='total_revenue',
            title='Revenue by Category & Department',
            color='total_revenue',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by state
        st.subheader("Revenue by State")
        
        state_revenue = segments_df.groupby('state_id')['total_revenue'].sum().reset_index()
        
        fig = px.pie(
            state_revenue,
            values='total_revenue',
            names='state_id',
            title='Revenue Distribution by State',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue concentration (Pareto)
    st.subheader("Revenue Concentration (Pareto Analysis)")
    
    # Sort by revenue
    sorted_segments = segments_df.sort_values('total_revenue', ascending=False).reset_index(drop=True)
    sorted_segments['cumulative_revenue_pct'] = (sorted_segments['total_revenue'].cumsum() / 
                                                   sorted_segments['total_revenue'].sum() * 100)
    
    fig = go.Figure()
    
    # Bar chart of revenue
    fig.add_trace(go.Bar(
        x=list(range(len(sorted_segments))),
        y=sorted_segments['total_revenue'],
        name='Revenue',
        yaxis='y1',
        marker_color='lightblue'
    ))
    
    # Line chart of cumulative percentage
    fig.add_trace(go.Scatter(
        x=list(range(len(sorted_segments))),
        y=sorted_segments['cumulative_revenue_pct'],
        name='Cumulative %',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    # Add 80% line
    fig.add_hline(y=80, line_dash="dash", line_color="green", 
                  annotation_text="80% Revenue", yref='y2')
    
    fig.update_layout(
        title='Pareto Chart - Revenue Concentration',
        xaxis_title='Product Rank',
        yaxis=dict(title='Revenue ($)', side='left'),
        yaxis2=dict(title='Cumulative Revenue (%)', overlaying='y', side='right', range=[0, 100]),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate ABC cutoffs
    abc_80_cutoff = sorted_segments[sorted_segments['cumulative_revenue_pct'] <= 80].shape[0]
    abc_95_cutoff = sorted_segments[sorted_segments['cumulative_revenue_pct'] <= 95].shape[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Category A**: Top {abc_80_cutoff} products ({abc_80_cutoff/len(segments_df)*100:.1f}%) drive 80% revenue")
    with col2:
        st.warning(f"**Category B**: Next {abc_95_cutoff - abc_80_cutoff} products ({(abc_95_cutoff - abc_80_cutoff)/len(segments_df)*100:.1f}%) drive 15% revenue")
    with col3:
        st.success(f"**Category C**: Remaining {len(segments_df) - abc_95_cutoff} products ({(len(segments_df) - abc_95_cutoff)/len(segments_df)*100:.1f}%) drive 5% revenue")
    
    st.markdown("---")
    
    # ========================================
    # SECTION 5: Interactive Product Explorer
    # ========================================
    st.header("5️⃣ Interactive Product Explorer")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_category = st.selectbox(
            "Select Category",
            options=['All'] + sorted(segments_df['cat_id'].unique().tolist())
        )
    
    with col2:
        selected_store = st.selectbox(
            "Select Store",
            options=['All'] + sorted(segments_df['store_id'].unique().tolist())
        )
    
    with col3:
        selected_segment = st.selectbox(
            "Select Segment",
            options=['All'] + sorted(segments_df['ABC_XYZ'].unique().tolist())
        )
    
    # Filter data
    filtered_df = segments_df.copy()
    
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['cat_id'] == selected_category]
    
    if selected_store != 'All':
        filtered_df = filtered_df[filtered_df['store_id'] == selected_store]
    
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['ABC_XYZ'] == selected_segment]
    
    # Display filtered results
    st.subheader(f"Filtered Results: {len(filtered_df)} products")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Daily Sales", f"{filtered_df['mean_sales'].mean():.2f}")
    
    with col2:
        st.metric("Avg CV", f"{filtered_df['cv'].mean():.2f}")
    
    with col3:
        st.metric("Total Revenue", f"${filtered_df['total_revenue'].sum():,.0f}")
    
    with col4:
        st.metric("Avg Zero Days", f"{filtered_df['zero_pct'].mean():.1f}%")
    
    # Display table
    st.dataframe(
        filtered_df[['id', 'item_id', 'cat_id', 'store_id', 'ABC_XYZ', 
                     'mean_sales', 'cv', 'total_revenue']].head(100),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f'filtered_products_{selected_category}_{selected_store}_{selected_segment}.csv',
        mime='text/csv'
    )

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    import traceback
    st.code(traceback.format_exc())