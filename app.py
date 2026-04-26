"""
Pilgrim Footfall Prediction Dashboard
=====================================
A Streamlit app for predicting and visualizing pilgrim footfall at pilgrimage sites.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pilgrim Footfall Prediction",
    page_icon="🛕",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .peak-alert {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# Helper Functions
# ============================================

@st.cache_resource
def load_models():
    """Load trained models and scalers."""
    models = {}
    
    # Check if model files exist
    if os.path.exists('prophet_model.pkl'):
        with open('prophet_model.pkl', 'rb') as f:
            models['prophet'] = pickle.load(f)
    else:
        models['prophet'] = None
    
    if os.path.exists('lstm_model.h5'):
        from tensorflow.keras.models import load_model
        models['lstm'] = load_model('lstm_model.h5')
    else:
        models['lstm'] = None
    
    if os.path.exists('scalers.pkl'):
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            models['scaler'] = scalers['scaler']
            models['footfall_scaler'] = scalers['footfall_scaler']
    else:
        models['scaler'] = None
        models['footfall_scaler'] = None
    
    if os.path.exists('features.pkl'):
        with open('features.pkl', 'rb') as f:
            models['features'] = pickle.load(f)
    else:
        models['features'] = ['footfall', 'temperature', 'rainfall', 'holiday', 
                              'festival', 'day_of_week', 'month', 'is_weekend']
    
    return models


@st.cache_data
def load_historical_data():
    """Load historical footfall data."""
    if os.path.exists('historical_data.csv'):
        df = pd.read_csv('historical_data.csv', parse_dates=['date'])
        return df
    else:
        # Generate synthetic data if file doesn't exist
        return generate_synthetic_data()


def generate_synthetic_data(start_date='2024-01-01', num_days=730):
    """Generate synthetic pilgrimage footfall data."""
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=num_days, freq='D')
    
    data = {
        'date': dates,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'day': dates.day,
    }
    df = pd.DataFrame(data)
    
    base_footfall = 5000
    seasonal_pattern = np.sin(2 * np.pi * (df['month'] - 3) / 12) * 2000
    weekly_pattern = np.where(df['day_of_week'] >= 5, 1500, 0)
    
    df['temperature'] = 25 + 10 * np.sin(2 * np.pi * (df['month'] - 4) / 12) + np.random.normal(0, 3, num_days)
    df['temperature'] = df['temperature'].clip(15, 42)
    
    monsoon_months = df['month'].isin([6, 7, 8, 9])
    df['rainfall'] = np.where(monsoon_months, np.random.exponential(15, num_days), np.random.exponential(2, num_days))
    df['rainfall'] = df['rainfall'].clip(0, 100)
    
    df['holiday'] = np.random.choice([0, 1], size=num_days, p=[0.95, 0.05])
    
    festival_dates = []
    for year in df['date'].dt.year.unique():
        festival_dates.extend(pd.date_range(f'{year}-10-20', periods=10))
        festival_dates.extend(pd.date_range(f'{year}-03-10', periods=5))
        festival_dates.extend(pd.date_range(f'{year}-09-25', periods=9))
    
    df['festival'] = df['date'].isin(festival_dates).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['footfall'] = (
        base_footfall + seasonal_pattern + weekly_pattern
        + df['holiday'] * np.random.uniform(3000, 5000, num_days)
        + df['festival'] * np.random.uniform(8000, 15000, num_days)
        - df['rainfall'] * 50
        + np.random.normal(0, 500, num_days)
    )
    df['footfall'] = df['footfall'].clip(500, None).astype(int)
    
    return df


def create_forecast(df, models, num_days, include_festival=False, include_holiday=False):
    """
    Create forecast using available models.
    Falls back to statistical forecast if ML models aren't available.
    """
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=num_days, freq='D')
    
    future_df = pd.DataFrame({'date': future_dates})
    future_df['ds'] = future_df['date']
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
    future_df['day_name'] = future_df['date'].dt.day_name()
    
    # Weather estimates
    future_df['temperature'] = 30 + 5 * np.sin(2 * np.pi * (future_df['month'] - 4) / 12)
    future_df['rainfall'] = np.where(future_df['month'].isin([6, 7, 8, 9]), 10, 2)
    
    # User-specified events
    future_df['holiday'] = int(include_holiday)
    future_df['festival'] = int(include_festival)
    
    # Prophet forecast
    if models.get('prophet') is not None:
        try:
            prophet_future = future_df[['ds', 'temperature', 'rainfall', 'holiday', 'festival']].copy()
            prophet_forecast = models['prophet'].predict(prophet_future)
            future_df['prophet_forecast'] = prophet_forecast['yhat'].values
            future_df['prophet_lower'] = prophet_forecast['yhat_lower'].values
            future_df['prophet_upper'] = prophet_forecast['yhat_upper'].values
        except Exception as e:
            st.warning(f"Prophet prediction failed: {e}")
            future_df['prophet_forecast'] = df['footfall'].tail(30).mean()
            future_df['prophet_lower'] = future_df['prophet_forecast'] * 0.8
            future_df['prophet_upper'] = future_df['prophet_forecast'] * 1.2
    else:
        # Fallback: use historical averages
        monthly_avg = df.groupby('month')['footfall'].mean()
        weekday_effect = df.groupby('day_of_week')['footfall'].mean() / df['footfall'].mean()
        
        base_forecast = future_df['month'].map(monthly_avg)
        day_adjustment = future_df['day_of_week'].map(weekday_effect)
        
        future_df['prophet_forecast'] = base_forecast * day_adjustment
        
        if include_festival:
            future_df['prophet_forecast'] *= 2.0
        if include_holiday:
            future_df['prophet_forecast'] *= 1.5
            
        future_df['prophet_lower'] = future_df['prophet_forecast'] * 0.8
        future_df['prophet_upper'] = future_df['prophet_forecast'] * 1.2
    
    # LSTM forecast
    if models.get('lstm') is not None and models.get('scaler') is not None:
        try:
            features = models['features']
            scaler = models['scaler']
            footfall_scaler = models['footfall_scaler']
            
            # Scale historical data
            scaled_data = scaler.transform(df[features].tail(60))
            last_sequence = scaled_data[-30:].copy()
            
            lstm_predictions = []
            for i in range(num_days):
                pred = models['lstm'].predict(last_sequence.reshape(1, 30, len(features)), verbose=0)
                lstm_predictions.append(pred[0, 0])
                
                next_features = np.zeros(len(features))
                next_features[0] = pred[0, 0]
                future_row = future_df.iloc[i]
                next_features[1] = (future_row['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min() + 1e-6)
                next_features[2] = (future_row['rainfall'] - df['rainfall'].min()) / (df['rainfall'].max() - df['rainfall'].min() + 1e-6)
                next_features[3] = future_row['holiday']
                next_features[4] = future_row['festival']
                next_features[5] = future_row['day_of_week'] / 6
                next_features[6] = (future_row['month'] - 1) / 11
                next_features[7] = future_row['is_weekend']
                
                last_sequence = np.vstack([last_sequence[1:], next_features])
            
            lstm_scaled = np.array(lstm_predictions).reshape(-1, 1)
            future_df['lstm_forecast'] = footfall_scaler.inverse_transform(lstm_scaled).flatten()
        except Exception as e:
            st.warning(f"LSTM prediction failed: {e}")
            future_df['lstm_forecast'] = future_df['prophet_forecast']
    else:
        # Fallback: add noise to prophet forecast
        future_df['lstm_forecast'] = future_df['prophet_forecast'] * np.random.uniform(0.95, 1.05, num_days)
    
    # Calculate average forecast
    future_df['avg_forecast'] = (future_df['prophet_forecast'] + future_df['lstm_forecast']) / 2
    
    # Identify peak days
    peak_threshold = future_df['avg_forecast'].quantile(0.75)
    future_df['is_peak'] = future_df['avg_forecast'] >= peak_threshold
    
    return future_df


# ============================================
# Main App
# ============================================

def main():
    # Header
    st.markdown('<div class="main-header">🛕 Pilgrim Footfall Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered forecasting for pilgrimage site management</div>', unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("Loading models and data..."):
        models = load_models()
        df = load_historical_data()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Forecast Settings")
    
    num_days = st.sidebar.slider(
        "Number of days to forecast",
        min_value=7,
        max_value=90,
        value=30,
        step=7
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎉 Special Events")
    
    include_festival = st.sidebar.checkbox("Include Festival Period", value=False)
    include_holiday = st.sidebar.checkbox("Include Public Holiday", value=False)
    
    st.sidebar.markdown("---")
    
    # Model status
    st.sidebar.subheader("📊 Model Status")
    prophet_status = "✅ Loaded" if models.get('prophet') else "⚠️ Using fallback"
    lstm_status = "✅ Loaded" if models.get('lstm') else "⚠️ Using fallback"
    st.sidebar.text(f"Prophet: {prophet_status}")
    st.sidebar.text(f"LSTM: {lstm_status}")
    
    # Generate forecast
    forecast_df = create_forecast(df, models, num_days, include_festival, include_holiday)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Historical Analysis", "ℹ️ About"])
    
    # ============================================
    # Tab 1: Forecast
    # ============================================
    with tab1:
        st.header(f"📅 {num_days}-Day Footfall Forecast")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_footfall = forecast_df['avg_forecast'].mean()
        max_footfall = forecast_df['avg_forecast'].max()
        min_footfall = forecast_df['avg_forecast'].min()
        peak_count = forecast_df['is_peak'].sum()
        
        col1.metric("📊 Average Daily Footfall", f"{avg_footfall:,.0f}")
        col2.metric("📈 Maximum Expected", f"{max_footfall:,.0f}")
        col3.metric("📉 Minimum Expected", f"{min_footfall:,.0f}")
        col4.metric("⚠️ Peak Days", f"{peak_count}")
        
        st.markdown("---")
        
        # Forecast chart
        st.subheader("📈 Forecast Visualization")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical (last 30 days)
        historical = df.tail(30)
        ax.plot(historical['date'], historical['footfall'], 
                label='Historical', color='#1f4e79', linewidth=2)
        
        # Forecasts
        ax.plot(forecast_df['date'], forecast_df['prophet_forecast'], 
                label='Prophet', color='#ff7f0e', linewidth=2, linestyle='--')
        ax.plot(forecast_df['date'], forecast_df['lstm_forecast'], 
                label='LSTM', color='#2ca02c', linewidth=2, linestyle=':')
        ax.plot(forecast_df['date'], forecast_df['avg_forecast'], 
                label='Average (Recommended)', color='#d62728', linewidth=2.5)
        
        # Confidence interval
        ax.fill_between(forecast_df['date'], 
                        forecast_df['prophet_lower'], 
                        forecast_df['prophet_upper'], 
                        alpha=0.15, color='#ff7f0e', label='Confidence Interval')
        
        # Peak markers
        peaks = forecast_df[forecast_df['is_peak']]
        ax.scatter(peaks['date'], peaks['avg_forecast'], 
                   color='red', s=100, marker='*', zorder=5, label='Peak Days')
        
        # Forecast boundary
        ax.axvline(x=df['date'].max(), color='gray', linestyle='--', alpha=0.7)
        ax.text(df['date'].max(), ax.get_ylim()[1]*0.95, ' Forecast →', 
                fontsize=10, color='gray')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Footfall', fontsize=12)
        ax.set_title('Pilgrim Footfall Forecast', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Peak day alerts
        st.subheader("⚠️ Peak Day Alerts")
        
        peak_days = forecast_df[forecast_df['is_peak']].copy()
        
        if len(peak_days) > 0:
            st.warning(f"**{len(peak_days)} high-footfall days detected!** Consider additional staffing and resources.")
            
            peak_display = peak_days[['date', 'day_name', 'avg_forecast']].copy()
            peak_display.columns = ['Date', 'Day', 'Expected Footfall']
            peak_display['Date'] = peak_display['Date'].dt.strftime('%Y-%m-%d')
            peak_display['Expected Footfall'] = peak_display['Expected Footfall'].astype(int).apply(lambda x: f"{x:,}")
            
            st.dataframe(peak_display, use_container_width=True, hide_index=True)
        else:
            st.success("No extreme peak days detected in the forecast period.")
        
        # Detailed forecast table
        st.subheader("📋 Detailed Forecast Table")
        
        display_df = forecast_df[['date', 'day_name', 'prophet_forecast', 'lstm_forecast', 'avg_forecast', 'is_peak']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['prophet_forecast'] = display_df['prophet_forecast'].astype(int)
        display_df['lstm_forecast'] = display_df['lstm_forecast'].astype(int)
        display_df['avg_forecast'] = display_df['avg_forecast'].astype(int)
        display_df['Alert'] = display_df['is_peak'].map({True: '⚠️ PEAK', False: ''})
        display_df = display_df.drop('is_peak', axis=1)
        display_df.columns = ['Date', 'Day', 'Prophet', 'LSTM', 'Average', 'Alert']
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"footfall_forecast_{num_days}days.csv",
            mime="text/csv"
        )
    
    # ============================================
    # Tab 2: Historical Analysis
    # ============================================
    with tab2:
        st.header("📊 Historical Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Footfall Over Time")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df['date'], df['footfall'], color='steelblue', linewidth=0.8)
            ax.axhline(y=df['footfall'].mean(), color='red', linestyle='--', label=f"Mean: {df['footfall'].mean():,.0f}")
            ax.set_xlabel('Date')
            ax.set_ylabel('Footfall')
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Monthly Average")
            monthly = df.groupby(df['date'].dt.month)['footfall'].mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.bar(months, monthly.values, color='coral', edgecolor='darkred')
            ax.set_ylabel('Average Footfall')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("By Day of Week")
            day_avg = df.groupby('day_of_week')['footfall'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(days, day_avg.values, color='teal', edgecolor='darkslategray')
            ax.set_ylabel('Average Footfall')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col4:
            st.subheader("Festival Impact")
            festival_avg = df.groupby('festival')['footfall'].mean()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(['Non-Festival', 'Festival'], festival_avg.values, 
                   color=['lightgray', 'gold'], edgecolor='black')
            ax.set_ylabel('Average Footfall')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Summary statistics
        st.subheader("📈 Key Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        stats_col1.metric("Total Days Recorded", f"{len(df):,}")
        stats_col2.metric("Average Daily Footfall", f"{df['footfall'].mean():,.0f}")
        stats_col3.metric("Record High", f"{df['footfall'].max():,}")
    
    # ============================================
    # Tab 3: About
    # ============================================
    with tab3:
        st.header("ℹ️ About This Dashboard")
        
        st.markdown("""
        ### 🎯 Purpose
        This dashboard helps pilgrimage site administrators predict daily visitor footfall 
        to optimize resource allocation, staffing, and crowd management.
        
        ### 🤖 Models Used
        
        **1. Prophet (Facebook/Meta)**
        - Time-series forecasting model designed for business metrics
        - Handles seasonality, holidays, and trend changes automatically
        - Provides confidence intervals for predictions
        
        **2. LSTM (Long Short-Term Memory)**
        - Deep learning neural network for sequence prediction
        - Learns complex temporal patterns from historical data
        - Considers multiple features: weather, day of week, festivals, etc.
        
        ### 📊 Features Considered
        - **Date/Time**: Day of week, month, weekend indicator
        - **Weather**: Temperature, rainfall
        - **Events**: Festivals, public holidays
        - **Historical patterns**: Seasonal trends, weekly cycles
        
        ### 📝 How to Use
        1. Adjust the forecast period using the sidebar slider
        2. Toggle festival/holiday modes for special scenarios
        3. View predictions in the Forecast tab
        4. Analyze historical patterns in the Historical Analysis tab
        5. Download forecasts for offline planning
        
        ### ⚠️ Limitations
        - Predictions are based on historical patterns and may not account for 
          unprecedented events
        - Weather forecasts are estimated based on seasonal averages
        - Actual footfall may vary due to unforeseen circumstances
        
        ---
        
        **Built with:** Python, Streamlit, Prophet, TensorFlow/Keras, Pandas, Matplotlib
        """)


if __name__ == "__main__":
    main()
