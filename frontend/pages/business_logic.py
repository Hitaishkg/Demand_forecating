"""
Business Logic Page - Safety Stock, ROI, and Strategic Recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import (
    load_segments, 
    load_results, 
    calculate_safety_stock, 
    calculate_reorder_point,
    format_number
)

# Page config
st.set_page_config(
    page_title="Business Logic - Demand Forecasting",
    page_icon="💼",
    layout="wide"
)

# Header
st.title("💼 Business Logic & Strategic Planning")
st.markdown("Convert forecasts into actionable business decisions: safety stock, reorder points, and ROI analysis")
st.markdown("---")

# Load data
try:
    segments_df = load_segments()
    results_df = load_results()
    
    # ========================================
    # SECTION 1: Safety Stock Calculator
    # ========================================
    st.header("1️⃣ Safety Stock Calculator")
    
    st.markdown("""
    **Safety stock** protects against stockouts due to demand variability and supply uncertainty.
    Higher service levels require more safety stock but reduce lost sales.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Input Parameters")
        
        # Product selector for safety stock
        product_for_calc = st.selectbox(
            "Select Product (or use generic parameters)",
            ['Generic Product'] + segments_df['id'].head(100).tolist(),
            help="Choose a specific product or use generic parameters"
        )
        
        if product_for_calc == 'Generic Product':
            # Manual inputs
            avg_daily_demand = st.number_input(
                "Average Daily Demand (units)",
                min_value=0.1,
                value=10.0,
                step=0.5,
                help="Average units sold per day"
            )
            
            std_daily_demand = st.number_input(
                "Std Dev of Daily Demand (units)",
                min_value=0.1,
                value=3.0,
                step=0.5,
                help="Standard deviation of daily sales"
            )
            
            cv = std_daily_demand / avg_daily_demand
            st.metric("Calculated CV", f"{cv:.2f}", help="Coefficient of Variation = Std Dev / Mean")
            
        else:
            # Load from selected product
            product_data = segments_df[segments_df['id'] == product_for_calc].iloc[0]
            avg_daily_demand = product_data['mean_sales']
            std_daily_demand = product_data['std_sales']
            cv = product_data['cv']
            
            st.info(f"""
            **Product Info:**
            - Avg Daily Demand: {avg_daily_demand:.2f} units
            - Std Dev: {std_daily_demand:.2f} units
            - CV: {cv:.2f}
            - Segment: {product_data['ABC_XYZ']}
            """)
        
        # Lead time
        lead_time_days = st.slider(
            "Lead Time (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Time between ordering and receiving inventory"
        )
        
        # Service level
        service_level = st.slider(
            "Target Service Level (%)",
            min_value=85,
            max_value=99,
            value=95,
            step=1,
            help="Probability of NOT stocking out"
        )
        
        service_level_decimal = service_level / 100
    
    with col2:
        st.subheader("📈 Safety Stock Results")
        
        # Calculate safety stock
        z_score = stats.norm.ppf(service_level_decimal)
        safety_stock = z_score * std_daily_demand * np.sqrt(lead_time_days)
        
        # Calculate reorder point
        lead_time_demand = avg_daily_demand * lead_time_days
        reorder_point = lead_time_demand + safety_stock
        
        # Display metrics
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(
                "Safety Stock",
                f"{safety_stock:.0f} units",
                help="Buffer inventory to prevent stockouts"
            )
            
            st.metric(
                "Lead Time Demand",
                f"{lead_time_demand:.0f} units",
                help="Expected sales during lead time"
            )
        
        with col_b:
            st.metric(
                "Reorder Point",
                f"{reorder_point:.0f} units",
                help="When to place new order"
            )
            
            st.metric(
                "Z-Score",
                f"{z_score:.2f}",
                help="Standard deviations for this service level"
            )
        
        # Service level explanation
        st.success(f"""
        **What this means:**
        - Order new stock when inventory drops to **{reorder_point:.0f} units**
        - Keep **{safety_stock:.0f} units** as safety buffer
        - Expected stockout rate: **{100-service_level:.1f}%** of time
        - Annual stockouts (365 days): ~**{(100-service_level)/100 * 365:.0f} days**
        """)
        
        # Cost trade-off
        st.info(f"""
        **Cost Trade-offs:**
        - Lower service level = Lower inventory cost, but MORE stockouts
        - Higher service level = Higher inventory cost, but FEWER stockouts
        
        For this product:
        - 90% service → {stats.norm.ppf(0.90) * std_daily_demand * np.sqrt(lead_time_days):.0f} units safety stock
        - 95% service → {safety_stock:.0f} units safety stock
        - 99% service → {stats.norm.ppf(0.99) * std_daily_demand * np.sqrt(lead_time_days):.0f} units safety stock
        """)
    
    # Visualization: Service Level vs Safety Stock
    st.subheader("📊 Service Level Impact")
    
    service_levels = np.arange(85, 100, 1)
    safety_stocks = [stats.norm.ppf(sl/100) * std_daily_demand * np.sqrt(lead_time_days) for sl in service_levels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=service_levels,
        y=safety_stocks,
        mode='lines',
        name='Safety Stock Required',
        line=dict(color='#FF4B4B', width=3)
    ))
    
    # Highlight current selection
    fig.add_trace(go.Scatter(
        x=[service_level],
        y=[safety_stock],
        mode='markers',
        name='Your Selection',
        marker=dict(size=15, color='green', symbol='star')
    ))
    
    fig.update_layout(
        title="Safety Stock Required vs Service Level",
        xaxis_title="Service Level (%)",
        yaxis_title="Safety Stock (units)",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 2: ROI Calculator
    # ========================================
    st.header("2️⃣ ROI Analysis - Forecast Accuracy Impact")
    
    st.markdown("""
    **Better forecasts = Better inventory decisions = Cost savings**
    
    This calculator estimates annual cost savings from improved forecast accuracy.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💰 Input Business Parameters")
        
        # Number of products
        num_products = st.number_input(
            "Number of Products",
            min_value=100,
            max_value=50000,
            value=1000,
            step=100,
            help="Total SKUs in your inventory"
        )
        
        # Average selling price
        avg_price = st.number_input(
            "Average Selling Price ($)",
            min_value=1.0,
            max_value=1000.0,
            value=25.0,
            step=5.0
        )
        
        # Average daily sales per product
        avg_daily_sales_per_product = st.number_input(
            "Avg Daily Sales per Product (units)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5
        )
        
        # Current forecast MAPE
        current_mape = st.slider(
            "Current Forecast MAPE (%)",
            min_value=30,
            max_value=80,
            value=60,
            help="Your current forecasting accuracy"
        )
        
        # Improved MAPE (with this system)
        improved_mape = st.slider(
            "Improved MAPE (%) - Our System",
            min_value=20,
            max_value=70,
            value=48,
            help="Expected accuracy with our forecasting system"
        )
        
        # Cost parameters
        holding_cost_pct = st.slider(
            "Annual Holding Cost (%)",
            min_value=10,
            max_value=40,
            value=25,
            help="% of product value spent on storage/insurance/depreciation per year"
        )
        
        stockout_cost_pct = st.slider(
            "Stockout Cost (% of sale price)",
            min_value=10,
            max_value=100,
            value=30,
            help="Lost profit + customer dissatisfaction cost"
        )
    
    with col2:
        st.subheader("💵 Estimated Annual Savings")
        
        # Calculate total annual revenue
        annual_revenue = num_products * avg_daily_sales_per_product * 365 * avg_price
        
        # Holding cost savings (better forecasts = less excess inventory)
        mape_improvement_pct = (current_mape - improved_mape) / current_mape
        
        # Assume: 10% MAPE improvement → 15% reduction in safety stock
        inventory_reduction_pct = mape_improvement_pct * 1.5  # Multiplier effect
        
        avg_inventory_value = annual_revenue / 12  # Approximate monthly inventory
        holding_cost_savings = avg_inventory_value * (holding_cost_pct/100) * inventory_reduction_pct
        
        # Stockout cost savings (better forecasts = fewer stockouts)
        # Assume: 10% MAPE improvement → 30% reduction in stockouts
        stockout_reduction_pct = mape_improvement_pct * 3
        
        current_stockout_rate = 0.05  # Assume 5% of demand currently lost to stockouts
        stockout_cost_savings = annual_revenue * current_stockout_rate * (stockout_cost_pct/100) * stockout_reduction_pct
        
        # Total savings
        total_annual_savings = holding_cost_savings + stockout_cost_savings
        
        # Display results
        st.metric(
            "Total Annual Savings",
            f"${format_number(total_annual_savings, decimals=1)}",
            help="Combined holding cost + stockout reduction"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric(
                "Holding Cost Savings",
                f"${format_number(holding_cost_savings, decimals=1)}",
                help="From reduced excess inventory"
            )
        
        with col_b:
            st.metric(
                "Stockout Cost Savings",
                f"${format_number(stockout_cost_savings, decimals=1)}",
                help="From fewer lost sales"
            )
        
        # ROI metrics
        st.metric(
            "Annual Revenue",
            f"${format_number(annual_revenue, decimals=1)}",
            help="Total sales across all products"
        )
        
        roi_percentage = (total_annual_savings / annual_revenue) * 100
        
        st.metric(
            "Savings as % of Revenue",
            f"{roi_percentage:.2f}%",
            help="Cost savings relative to total revenue"
        )
        
        # Payback visualization
        st.success(f"""
        **Key Insights:**
        
        - **MAPE Improvement**: {current_mape}% → {improved_mape}% ({current_mape - improved_mape} percentage points)
        - **Inventory Reduction**: ~{inventory_reduction_pct*100:.1f}%
        - **Stockout Reduction**: ~{stockout_reduction_pct*100:.1f}%
        - **Annual Benefit**: ${format_number(total_annual_savings, decimals=2)}
        
        **For a ${format_number(annual_revenue/1e9, decimals=1)}B revenue company:**
        - This represents **{roi_percentage:.1f}%** cost savings
        - Payback period: < 3 months (assuming implementation cost)
        """)
    
    # Sensitivity analysis
    st.subheader("📊 Sensitivity Analysis: MAPE vs Savings")
    
    mape_range = np.arange(30, 71, 2)
    savings_range = []
    
    for mape in mape_range:
        mape_imp = (current_mape - mape) / current_mape
        inv_red = mape_imp * 1.5
        stock_red = mape_imp * 3
        
        hold_save = avg_inventory_value * (holding_cost_pct/100) * inv_red
        stock_save = annual_revenue * 0.05 * (stockout_cost_pct/100) * stock_red
        savings_range.append(hold_save + stock_save)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=mape_range,
        y=savings_range,
        mode='lines',
        name='Annual Savings',
        line=dict(color='green', width=3),
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.1)'
    ))
    
    # Highlight current selection
    fig.add_trace(go.Scatter(
        x=[improved_mape],
        y=[total_annual_savings],
        mode='markers',
        name='Our System',
        marker=dict(size=15, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title="Annual Savings vs Forecast Accuracy (MAPE)",
        xaxis_title="MAPE (%)",
        yaxis_title="Annual Savings ($)",
        height=300,
        xaxis=dict(autorange="reversed"),  # Lower MAPE is better
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 3: Strategic Recommendations
    # ========================================
    st.header("3️⃣ Strategic Recommendations by Segment")
    
    st.markdown("""
    Based on ABC-XYZ analysis and model performance, here are segment-specific strategies:
    """)
    
    # Merge segments with results
    segment_summary = segments_df.groupby('ABC_XYZ').agg({
        'id': 'count',
        'total_revenue': 'sum',
        'mean_sales': 'mean',
        'cv': 'mean',
        'zero_pct': 'mean'
    }).reset_index()
    
    segment_summary.columns = ['segment', 'n_products', 'total_revenue', 'avg_sales', 'avg_cv', 'avg_zero_pct']
    
    # Merge with model results
    if 'mape' in results_df.columns:
        segment_summary = segment_summary.merge(
            results_df[['segment', 'mape']], 
            on='segment', 
            how='left'
        )
    
    # Define strategies
    strategies = {
        'AX': {
            'priority': '🔴 CRITICAL',
            'strategy': 'Invest heavily in accuracy',
            'model': 'Prophet or ARIMA',
            'update_freq': 'Daily',
            'service_level': '98%',
            'safety_stock': 'Low (1-2 weeks)',
            'actions': [
                'Real-time monitoring',
                'Dedicated forecasting resources',
                'Consider vendor-managed inventory',
                'Premium fulfillment (next-day delivery)'
            ]
        },
        'AY': {
            'priority': '🔴 CRITICAL',
            'strategy': 'Balance accuracy and cost',
            'model': 'LightGBM or Prophet',
            'update_freq': 'Daily',
            'service_level': '97%',
            'safety_stock': 'Medium (2-3 weeks)',
            'actions': [
                'Daily forecast updates',
                'Automatic reorder triggers',
                'Price elasticity monitoring',
                'Promotion planning integration'
            ]
        },
        'AZ': {
            'priority': '🔴 CRITICAL',
            'strategy': 'High safety stock, frequent review',
            'model': 'Ensemble (LightGBM + TFT)',
            'update_freq': 'Daily',
            'service_level': '95%',
            'safety_stock': 'High (4-6 weeks)',
            'actions': [
                'High safety stock (prevent stockouts)',
                'Close vendor relationships',
                'Scenario planning for demand spikes',
                'Consider pre-positioning inventory'
            ]
        },
        'BY': {
            'priority': '🟡 IMPORTANT',
            'strategy': 'Efficient standard processes',
            'model': 'LightGBM',
            'update_freq': 'Weekly',
            'service_level': '92%',
            'safety_stock': 'Medium (2-4 weeks)',
            'actions': [
                'Weekly batch forecasting',
                'Standard reorder policies',
                'Optimize for cost efficiency',
                'Periodic accuracy monitoring'
            ]
        },
        'BZ': {
            'priority': '🟡 IMPORTANT',
            'strategy': 'Minimize holding costs',
            'model': 'LightGBM with high buffer',
            'update_freq': 'Weekly',
            'service_level': '90%',
            'safety_stock': 'High (4-6 weeks)',
            'actions': [
                'Accept some stockouts (low margin)',
                'High safety stock for unpredictability',
                'Consider drop-shipping',
                'Evaluate product rationalization'
            ]
        },
        'CY': {
            'priority': '🟢 LOW',
            'strategy': 'Simple, low-touch',
            'model': 'Rolling Average',
            'update_freq': 'Monthly',
            'service_level': '88%',
            'safety_stock': 'Medium (2-4 weeks)',
            'actions': [
                'Simple moving average forecast',
                'Manual review only when flagged',
                'Longer replenishment cycles',
                'Consider consolidation with similar SKUs'
            ]
        },
        'CZ': {
            'priority': '⚠️ REVIEW',
            'strategy': 'Consider discontinuation',
            'model': 'Rolling Average or None',
            'update_freq': 'Monthly or None',
            'service_level': '85%',
            'safety_stock': 'High or Discontinue',
            'actions': [
                'Evaluate profitability vs carrying cost',
                'Consider discontinuing if unprofitable',
                'Made-to-order if must keep',
                'Focus sales effort on alternatives'
            ]
        }
    }
    
    # Display segment cards
    for _, row in segment_summary.iterrows():
        segment = row['segment']
        
        if segment in strategies:
            with st.expander(f"**{segment}** - {strategies[segment]['priority']} ({row['n_products']} products)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📊 Segment Characteristics**")
                    st.write(f"Products: {row['n_products']}")
                    st.write(f"Total Revenue: ${format_number(row['total_revenue'], decimals=1)}")
                    st.write(f"Avg Daily Sales: {row['avg_sales']:.2f} units")
                    st.write(f"Avg CV: {row['avg_cv']:.2f}")
                    if 'mape' in row and pd.notna(row['mape']):
                        st.write(f"Forecast MAPE: {row['mape']:.1f}%")
                
                with col2:
                    st.markdown("**⚙️ Recommended Approach**")
                    st.write(f"**Strategy:** {strategies[segment]['strategy']}")
                    st.write(f"**Model:** {strategies[segment]['model']}")
                    st.write(f"**Update:** {strategies[segment]['update_freq']}")
                    st.write(f"**Service Level:** {strategies[segment]['service_level']}")
                    st.write(f"**Safety Stock:** {strategies[segment]['safety_stock']}")
                
                with col3:
                    st.markdown("**✅ Action Items**")
                    for action in strategies[segment]['actions']:
                        st.write(f"• {action}")
    
    st.markdown("---")
    
    # ========================================
    # SECTION 4: Investment Priorities
    # ========================================
    st.header("4️⃣ Where to Invest Resources")
    
    st.markdown("""
    **Focus your forecasting and inventory efforts on segments with highest potential impact:**
    """)
    
    # Calculate impact score (Revenue × Forecast Improvement Potential)
    segment_summary['impact_score'] = (
        segment_summary['total_revenue'] / segment_summary['total_revenue'].sum() * 100
    )
    
    # Sort by impact
    segment_summary_sorted = segment_summary.sort_values('impact_score', ascending=False)
    
    # Create priority visualization
    fig = px.bar(
        segment_summary_sorted,
        x='segment',
        y='impact_score',
        color='impact_score',
        color_continuous_scale='Reds',
        title='Investment Priority by Segment (Revenue Impact)',
        labels={'segment': 'Segment', 'impact_score': 'Priority Score (%)'},
        text='impact_score'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 3 priorities
    col1, col2, col3 = st.columns(3)
    
    top_3 = segment_summary_sorted.head(3)
    
    for idx, (col, (_, row)) in enumerate(zip([col1, col2, col3], top_3.iterrows())):
        with col:
            st.info(f"""
            **Priority #{idx+1}: {row['segment']}**
            
            - Revenue Share: {row['impact_score']:.1f}%
            - Products: {row['n_products']}
            - Strategy: {strategies.get(row['segment'], {}).get('strategy', 'N/A')}
            
            **Investment:** {'Critical' if idx == 0 else 'High' if idx == 1 else 'Medium'}
            """)
    
    st.markdown("---")
    
    # ========================================
    # SECTION 5: Quick Reference Guide
    # ========================================
    st.header("5️⃣ Quick Reference Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Service Level Guidelines")
        st.markdown("""
        **When to use each service level:**
        
        - **99%**: Critical items, high margin, can't stockout
          - Medical supplies, fast-moving staples
        
        - **95-97%**: Important items, good margin
          - Most A and B category products
        
        - **90-92%**: Standard items, acceptable stockout rate
          - Most B and C category products
        
        - **85-88%**: Low-value, low-margin items
          - C category, easily substitutable
        """)
    
    with col2:
        st.subheader("💡 Rule of Thumb")
        st.markdown("""
        **Quick calculations without calculator:**
        
        **Safety Stock Approximation:**
        - Service Level 95%: Safety Stock ≈ 2 × Std Dev × √Lead Time
        - Service Level 99%: Safety Stock ≈ 2.5 × Std Dev × √Lead Time
        
        **Example:** 
        - Std Dev = 10 units/day
        - Lead Time = 7 days
        - Safety Stock (95%) ≈ 2 × 10 × √7 ≈ **53 units**
        
        **Reorder Point:**
        - ROP = (Avg Daily Demand × Lead Time) + Safety Stock
        
        **When to Reorder:**
        - When inventory drops below ROP, place order!
        """)
    
    # ========================================
    # FOOTER: Download Strategy Template
    # ========================================
    st.markdown("---")
    st.header("📥 Export Strategic Plan")
    
    # Create comprehensive strategy document
    strategy_export = []
    
    for _, row in segment_summary.iterrows():
        segment = row['segment']
        if segment in strategies:
            strategy_export.append({
                'Segment': segment,
                'Priority': strategies[segment]['priority'],
                'Products': row['n_products'],
                'Revenue': row['total_revenue'],
                'Revenue_Share_%': row['impact_score'],
                'Avg_CV': row['avg_cv'],
                'Forecast_MAPE_%': row.get('mape', 'N/A'),
                'Recommended_Model': strategies[segment]['model'],
                'Update_Frequency': strategies[segment]['update_freq'],
                'Target_Service_Level': strategies[segment]['service_level'],
                'Safety_Stock_Strategy': strategies[segment]['safety_stock'],
                'Key_Action_1': strategies[segment]['actions'][0] if len(strategies[segment]['actions']) > 0 else '',
                'Key_Action_2': strategies[segment]['actions'][1] if len(strategies[segment]['actions']) > 1 else '',
                'Key_Action_3': strategies[segment]['actions'][2] if len(strategies[segment]['actions']) > 2 else ''
            })
    
    strategy_df = pd.DataFrame(strategy_export)
    csv = strategy_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Complete Strategic Plan (CSV)",
        data=csv,
        file_name='demand_forecasting_strategic_plan.csv',
        mime='text/csv',
        help="Download segment-by-segment strategies and recommendations"
    )

except Exception as e:
    st.error(f"Error: {str(e)}")
    import traceback
    st.code(traceback.format_exc())