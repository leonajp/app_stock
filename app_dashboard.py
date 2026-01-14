"""
APP Mean Reversion Trading Dashboard
=====================================
Professional trading dashboard with secure API key handling.

Deployment:
1. Push to GitHub (without API keys)
2. On Streamlit Cloud, add secrets via Settings > Secrets
3. Format: POLYGON_API_KEY = "your_key_here"

Author: Limestone Hill Capital
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="APP Mean Reversion Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    /* Dark theme optimized for trading */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #12121a 100%);
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #2a2a4a;
    }
    
    .main-header h1 {
        color: #00d4aa;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        font-family: 'SF Pro Display', -apple-system, sans-serif;
    }
    
    .main-header p {
        color: #8892b0;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #1f1f35 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #2a2a4a;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.15);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-value.positive { color: #00d4aa; }
    .metric-value.negative { color: #ff6b6b; }
    .metric-value.neutral { color: #ffd93d; }
    
    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Signal box */
    .signal-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .signal-long {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(0, 212, 170, 0.05) 100%);
        border: 2px solid #00d4aa;
    }
    
    .signal-short {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 107, 107, 0.05) 100%);
        border: 2px solid #ff6b6b;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.1) 0%, rgba(255, 217, 61, 0.02) 100%);
        border: 2px solid #ffd93d;
    }
    
    .signal-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Section headers */
    .section-header {
        color: #e6e6e6;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2a2a4a;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(0, 212, 170, 0.08);
        border-left: 4px solid #00d4aa;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 217, 61, 0.08);
        border-left: 4px solid #ffd93d;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #12121a;
    }
    
    /* Table styling */
    .dataframe {
        background: #1a1a2e !important;
        color: #e6e6e6 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API KEY HANDLING (SECURE)
# ============================================================================

def get_api_key() -> Optional[str]:
    """
    Get API key from Streamlit secrets or user input.
    Priority: 1. Streamlit secrets (for deployment), 2. User input
    """
    # Try secrets first (Streamlit Cloud deployment)
    try:
        if "POLYGON_API_KEY" in st.secrets:
            return st.secrets["POLYGON_API_KEY"]
    except Exception:
        pass
    
    # Fall back to session state (user input)
    if "user_api_key" in st.session_state and st.session_state.user_api_key:
        return st.session_state.user_api_key
    
    return None


# ============================================================================
# DATA FETCHER
# ============================================================================

class PolygonFetcher:
    """Fetch data from Polygon.io"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_aggregates(_self, symbol: str, multiplier: int, timespan: str, 
                       from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch aggregate bars"""
        url = f"{_self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": _self.api_key
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            
            if "results" not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data["results"])
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
            df = df.rename(columns={
                "o": "open", "h": "high", "l": "low",
                "c": "close", "v": "volume", "vw": "vwap"
            })
            df = df.set_index("timestamp")
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
            
            return df[["open", "high", "low", "close", "volume", "vwap"]]
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return pd.DataFrame()
    
    def get_snapshot(_self, symbol: str) -> Dict:
        """Get current ticker snapshot"""
        url = f"{_self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": _self.api_key}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            return resp.json().get("ticker", {})
        except:
            return {}


# ============================================================================
# ANALYSIS CLASSES
# ============================================================================

class FeatureCalculator:
    """Calculate technical features"""
    
    @staticmethod
    def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 0.0001)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    @staticmethod
    def calc_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ma = df['close'].rolling(period).mean()
        std_dev = df['close'].rolling(period).std()
        upper = ma + std * std_dev
        lower = ma - std * std_dev
        return ma, upper, lower
    
    @staticmethod
    def calc_relative_volume(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        avg_vol = df['volume'].rolling(lookback).mean()
        return df['volume'] / avg_vol
    
    @staticmethod
    def calc_vwap_deviation(df: pd.DataFrame) -> pd.Series:
        if 'vwap' in df.columns:
            return (df['close'] - df['vwap']) / df['vwap'] * 100
        return pd.Series(0, index=df.index)


class SessionAnalyzer:
    """Analyze trading sessions"""
    
    @staticmethod
    def calculate_session_returns(minute_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns for each session"""
        df = minute_df.copy()
        df['date'] = df.index.date
        df['time'] = df.index.time
        
        results = []
        dates = df['date'].unique()
        
        for i, date in enumerate(dates):
            day_data = df[df['date'] == date]
            prev_date = dates[i-1] if i > 0 else None
            prev_day_data = df[df['date'] == prev_date] if prev_date else None
            
            record = {'date': date}
            
            # Previous day close
            if prev_day_data is not None:
                prev_regular = prev_day_data[
                    (prev_day_data['time'] >= dt_time(9, 30)) &
                    (prev_day_data['time'] < dt_time(16, 0))
                ]
                if len(prev_regular) > 0:
                    record['prev_close'] = prev_regular['close'].iloc[-1]
                    record['prev_high'] = prev_regular['high'].max()
                    record['prev_low'] = prev_regular['low'].min()
            
            # Pre-market
            premarket = day_data[
                (day_data['time'] >= dt_time(4, 0)) &
                (day_data['time'] < dt_time(9, 30))
            ]
            if len(premarket) > 0 and 'prev_close' in record:
                record['premarket_return'] = (premarket['close'].iloc[-1] / record['prev_close'] - 1) * 100
                record['premarket_high'] = premarket['high'].max()
                record['premarket_low'] = premarket['low'].min()
            
            # Regular session
            regular = day_data[
                (day_data['time'] >= dt_time(9, 30)) &
                (day_data['time'] < dt_time(16, 0))
            ]
            
            if len(regular) > 0:
                record['open'] = regular['open'].iloc[0]
                record['close'] = regular['close'].iloc[-1]
                record['high'] = regular['high'].max()
                record['low'] = regular['low'].min()
                record['volume'] = regular['volume'].sum()
                
                if 'prev_close' in record:
                    record['gap_return'] = (record['open'] / record['prev_close'] - 1) * 100
                
                # First 5 minutes
                first_5min = regular[regular['time'] <= dt_time(9, 35)]
                if len(first_5min) > 0:
                    record['first_5min_return'] = (first_5min['close'].iloc[-1] / record['open'] - 1) * 100
                
                record['day_return'] = (record['close'] / record['open'] - 1) * 100
            
            results.append(record)
        
        return pd.DataFrame(results)


class RegimeDetector:
    """Detect market regime"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        atr = FeatureCalculator.calc_atr(df)
        atr_pct = atr.iloc[-1] / df['close'].iloc[-1] * 100 if len(df) > 14 else 0
        
        # Volatility regime
        high_vol = atr_pct > df['close'].pct_change().abs().rolling(20).mean().iloc[-1] * 100 * 1.5 if len(df) > 20 else False
        
        # Trend vs mean reversion (autocorrelation)
        returns = df['close'].pct_change()
        autocorr = returns.iloc[-20:].autocorr() if len(returns) > 20 else 0
        trending = autocorr > 0.1
        
        if high_vol and not trending:
            regime = "HIGH_VOL_MR"
            description = "High volatility, mean reverting - BEST for strategy"
            color = "#00d4aa"
        elif high_vol and trending:
            regime = "HIGH_VOL_TREND"
            description = "High volatility, trending - REDUCE position size"
            color = "#ff6b6b"
        elif not high_vol and not trending:
            regime = "LOW_VOL_MR"
            description = "Low volatility, mean reverting - Good, smaller targets"
            color = "#ffd93d"
        else:
            regime = "LOW_VOL_TREND"
            description = "Low volatility, trending - AVOID mean reversion"
            color = "#ff6b6b"
        
        return {
            "regime": regime,
            "description": description,
            "color": color,
            "atr_pct": atr_pct,
            "autocorr": autocorr
        }


class SignalGenerator:
    """Generate trading signals"""
    
    @staticmethod
    def generate_ensemble_signal(df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {"signal": "NEUTRAL", "confidence": 0, "score": 0, "components": {}}
        
        latest = df.iloc[-1]
        
        # Component signals
        rsi = FeatureCalculator.calc_rsi(df['close']).iloc[-1]
        rsi_signal = (30 - rsi) / 30 if rsi < 30 else ((70 - rsi) / 30 if rsi > 70 else 0)
        
        # Bollinger position
        ma, upper, lower = FeatureCalculator.calc_bollinger(df)
        bb_range = upper.iloc[-1] - lower.iloc[-1]
        bb_signal = -((latest['close'] - ma.iloc[-1]) / (bb_range / 2)) if bb_range > 0 else 0
        bb_signal = np.clip(bb_signal, -1, 1)
        
        # VWAP deviation
        vwap_dev = FeatureCalculator.calc_vwap_deviation(df).iloc[-1]
        vwap_signal = -np.clip(vwap_dev / 2, -1, 1)
        
        # Volume confirmation
        rel_vol = FeatureCalculator.calc_relative_volume(df).iloc[-1]
        vol_boost = min(rel_vol, 2.0) / 2.0
        
        # Composite score
        weights = {"rsi": 0.30, "bb": 0.25, "vwap": 0.20}
        score = (
            rsi_signal * weights["rsi"] +
            bb_signal * weights["bb"] +
            vwap_signal * weights["vwap"]
        ) * (1 + vol_boost * 0.25)
        
        # Confidence
        signals = [rsi_signal, bb_signal, vwap_signal]
        n_bullish = sum(1 for s in signals if s > 0.1)
        n_bearish = sum(1 for s in signals if s < -0.1)
        confidence = max(n_bullish, n_bearish) / len(signals) * 100
        
        # Signal determination
        if score > 0.2 and confidence > 40:
            signal = "LONG"
        elif score < -0.2 and confidence > 40:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"
        
        return {
            "signal": signal,
            "confidence": confidence,
            "score": score,
            "components": {
                "RSI": rsi_signal,
                "Bollinger": bb_signal,
                "VWAP": vwap_signal,
                "Volume": vol_boost
            },
            "raw": {
                "rsi": rsi,
                "rel_volume": rel_vol,
                "vwap_dev": vwap_dev
            }
        }


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_price_chart(df: pd.DataFrame, title: str = "Price Action") -> go.Figure:
    """Create candlestick chart with indicators"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, "RSI", "Volume")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#00d4aa',
            decreasing_line_color='#ff6b6b'
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    if len(df) > 20:
        ma, upper, lower = FeatureCalculator.calc_bollinger(df)
        fig.add_trace(go.Scatter(x=df.index, y=ma, name="MA20", line=dict(color='#ffd93d', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="Upper BB", line=dict(color='#4a4a6a', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="Lower BB", line=dict(color='#4a4a6a', width=1, dash='dash')), row=1, col=1)
    
    # VWAP
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name="VWAP", line=dict(color='#9d4edd', width=1.5)), row=1, col=1)
    
    # RSI
    if len(df) > 14:
        rsi = FeatureCalculator.calc_rsi(df['close'])
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#00b4d8', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00d4aa", line_width=1, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.05)", line_width=0, row=2, col=1)
    
    # Volume
    colors = ['#00d4aa' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff6b6b' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color=colors, opacity=0.7), row=3, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(gridcolor='#2a2a4a', showgrid=True)
    fig.update_yaxes(gridcolor='#2a2a4a', showgrid=True)
    
    return fig


def create_session_chart(session_df: pd.DataFrame) -> go.Figure:
    """Create session returns chart"""
    
    fig = go.Figure()
    
    if 'gap_return' in session_df.columns:
        colors = ['#00d4aa' if x >= 0 else '#ff6b6b' for x in session_df['gap_return'].fillna(0)]
        fig.add_trace(go.Bar(
            x=session_df['date'],
            y=session_df['gap_return'],
            name="Gap Return",
            marker_color=colors,
            opacity=0.8
        ))
    
    if 'first_5min_return' in session_df.columns:
        fig.add_trace(go.Scatter(
            x=session_df['date'],
            y=session_df['first_5min_return'],
            name="First 5min Return",
            mode='lines+markers',
            line=dict(color='#ffd93d', width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        height=350,
        title="Session Returns Analysis",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(gridcolor='#2a2a4a')
    fig.update_yaxes(gridcolor='#2a2a4a')
    
    return fig


def create_signal_gauge(signal_data: Dict) -> go.Figure:
    """Create signal strength gauge"""
    
    score = signal_data.get('score', 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Signal Strength", 'font': {'size': 16, 'color': '#e6e6e6'}},
        number={'suffix': "%", 'font': {'color': '#e6e6e6'}},
        gauge={
            'axis': {'range': [-100, 100], 'tickcolor': '#e6e6e6'},
            'bar': {'color': '#00d4aa' if score > 0 else '#ff6b6b'},
            'bgcolor': '#1a1a2e',
            'borderwidth': 2,
            'bordercolor': '#2a2a4a',
            'steps': [
                {'range': [-100, -30], 'color': 'rgba(255, 107, 107, 0.3)'},
                {'range': [-30, 30], 'color': 'rgba(255, 217, 61, 0.2)'},
                {'range': [30, 100], 'color': 'rgba(0, 212, 170, 0.3)'}
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e6e6e6'},
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )
    
    return fig


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def generate_sample_data(n_days: int = 60) -> pd.DataFrame:
    """Generate sample data for demo mode with realistic APP price (~$716)"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days * 390, freq='1min')
    # Filter to market hours only
    dates = dates[(dates.time >= dt_time(9, 30)) & (dates.time < dt_time(16, 0))]
    
    # APP current price ~$716 with higher volatility
    price = 716.0
    prices = []
    
    for i in range(len(dates)):
        # Mean revert to 716 with APP-like volatility
        deviation = (price - 716) / 716
        mean_reversion = -deviation * 0.002
        noise = np.random.normal(0, 0.0015)  # ~0.15% per minute vol
        price = price * (1 + noise + mean_reversion)
        prices.append(price)
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, len(prices))),
        'high': prices * (1 + np.random.uniform(0.001, 0.003, len(prices))),
        'low': prices * (1 - np.random.uniform(0.001, 0.003, len(prices))),
        'close': prices,
        'volume': np.random.randint(50000, 200000, len(prices)),
        'vwap': prices * (1 + np.random.uniform(-0.0005, 0.0005, len(prices)))
    }, index=dates)
    
    df.index = df.index.tz_localize("America/New_York")
    
    return df


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 APP Mean Reversion Dashboard</h1>
        <p>Real-time session analysis, RSI signals, and options strategy recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key handling
        st.markdown("#### 🔑 API Key")
        
        api_key = get_api_key()
        
        if not api_key:
            st.markdown("""
            <div class="warning-box">
            <strong>No API Key Found</strong><br>
            Enter your Polygon API key below or deploy with Streamlit secrets.
            </div>
            """, unsafe_allow_html=True)
            
            user_key = st.text_input(
                "Polygon API Key",
                type="password",
                help="Your API key is stored in session only, never saved to disk or transmitted."
            )
            
            if user_key:
                st.session_state.user_api_key = user_key
                st.rerun()
            
            use_demo = st.checkbox("Use Demo Data", value=True)
        else:
            st.success("✓ API Key configured")
            use_demo = st.checkbox("Use Demo Data Instead", value=False)
        
        st.markdown("---")
        
        # Symbol selection
        symbol = st.text_input("Symbol", value="APP")
        
        # Date range
        lookback_days = st.slider("Lookback Days", 5, 90, 30)
        
        # RSI settings
        st.markdown("#### 📊 RSI Settings")
        rsi_oversold = st.slider("RSI Oversold", 15, 35, 30)
        rsi_overbought = st.slider("RSI Overbought", 65, 85, 70)
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    if use_demo or not api_key:
        st.info("📌 Running in Demo Mode with simulated data")
        minute_df = generate_sample_data(lookback_days)
    else:
        # Fetch real data
        fetcher = PolygonFetcher(api_key)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        with st.spinner("Fetching data from Polygon..."):
            minute_df = fetcher.get_aggregates(symbol, 1, "minute", start_date, end_date)
        
        if len(minute_df) == 0:
            st.error("No data received. Check your API key and symbol.")
            return
    
    # Calculate daily data
    daily_df = minute_df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': 'last'
    }).dropna()
    
    # Generate signals
    signal_data = SignalGenerator.generate_ensemble_signal(daily_df)
    regime_data = RegimeDetector.detect(daily_df)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = daily_df['close'].iloc[-1] if len(daily_df) > 0 else 0
        price_change = (daily_df['close'].iloc[-1] / daily_df['close'].iloc[-2] - 1) * 100 if len(daily_df) > 1 else 0
        color_class = "positive" if price_change >= 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {color_class}">${current_price:.2f}</p>
            <p class="metric-label">Current Price ({price_change:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rsi_val = signal_data['raw'].get('rsi', 50)
        rsi_class = "negative" if rsi_val > 70 else ("positive" if rsi_val < 30 else "neutral")
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {rsi_class}">{rsi_val:.1f}</p>
            <p class="metric-label">RSI (14)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rel_vol = signal_data['raw'].get('rel_volume', 1)
        vol_class = "positive" if rel_vol > 1.5 else ("negative" if rel_vol < 0.5 else "neutral")
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {vol_class}">{rel_vol:.2f}x</p>
            <p class="metric-label">Relative Volume</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        vwap_dev = signal_data['raw'].get('vwap_dev', 0)
        vwap_class = "positive" if vwap_dev > 0.5 else ("negative" if vwap_dev < -0.5 else "neutral")
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {vwap_class}">{vwap_dev:+.2f}%</p>
            <p class="metric-label">VWAP Deviation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        confidence = signal_data.get('confidence', 0)
        conf_class = "positive" if confidence > 60 else ("neutral" if confidence > 40 else "negative")
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value {conf_class}">{confidence:.0f}%</p>
            <p class="metric-label">Signal Confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal and Regime row
    col_signal, col_regime = st.columns(2)
    
    with col_signal:
        signal = signal_data.get('signal', 'NEUTRAL')
        signal_class = "signal-long" if signal == "LONG" else ("signal-short" if signal == "SHORT" else "signal-neutral")
        signal_icon = "🟢" if signal == "LONG" else ("🔴" if signal == "SHORT" else "🟡")
        
        st.markdown(f"""
        <div class="signal-box {signal_class}">
            <p class="signal-text">{signal_icon} {signal}</p>
            <p style="color: #8892b0; margin: 0.5rem 0 0 0;">
                Score: {signal_data.get('score', 0):.3f} | Confidence: {confidence:.0f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_regime:
        st.markdown(f"""
        <div class="signal-box" style="background: rgba(128, 128, 128, 0.1); border: 2px solid {regime_data['color']};">
            <p class="signal-text" style="color: {regime_data['color']};">📈 {regime_data['regime']}</p>
            <p style="color: #8892b0; margin: 0.5rem 0 0 0;">{regime_data['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    st.markdown('<p class="section-header">📈 Price Action & Indicators</p>', unsafe_allow_html=True)
    
    # Use recent minute data for chart
    recent_minutes = minute_df.iloc[-2000:] if len(minute_df) > 2000 else minute_df
    fig_price = create_price_chart(recent_minutes, f"{symbol} - Intraday")
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Session analysis
    col_session, col_gauge = st.columns([2, 1])
    
    with col_session:
        st.markdown('<p class="section-header">📊 Session Returns Analysis</p>', unsafe_allow_html=True)
        session_df = SessionAnalyzer.calculate_session_returns(minute_df)
        if len(session_df) > 0:
            fig_session = create_session_chart(session_df)
            st.plotly_chart(fig_session, use_container_width=True)
    
    with col_gauge:
        st.markdown('<p class="section-header">🎯 Signal Gauge</p>', unsafe_allow_html=True)
        fig_gauge = create_signal_gauge(signal_data)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Component breakdown
        st.markdown("**Component Signals:**")
        for comp, value in signal_data.get('components', {}).items():
            color = "#00d4aa" if value > 0 else ("#ff6b6b" if value < 0 else "#ffd93d")
            bar_width = abs(value) * 100
            direction = "+" if value > 0 else ""
            st.markdown(f"""
            <div style="margin: 0.5rem 0;">
                <span style="color: #8892b0; width: 80px; display: inline-block;">{comp}</span>
                <span style="color: {color}; font-weight: 600;">{direction}{value:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Options Strategy Recommendations
    st.markdown('<p class="section-header">📋 Options Strategy Recommendations</p>', unsafe_allow_html=True)
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        if signal == "LONG":
            st.markdown("""
            <div class="info-box">
                <strong>🟢 BULLISH SETUP</strong><br><br>
                <strong>Primary:</strong> Buy ATM Call (0DTE)<br>
                <strong>Alternative:</strong> Bull Call Spread<br>
                <strong>Entry:</strong> On RSI bounce above 30<br>
                <strong>Target:</strong> 50% premium gain or RSI > 50<br>
                <strong>Stop:</strong> 50% premium loss
            </div>
            """, unsafe_allow_html=True)
        elif signal == "SHORT":
            st.markdown("""
            <div class="info-box" style="border-color: #ff6b6b; background: rgba(255, 107, 107, 0.08);">
                <strong>🔴 BEARISH SETUP</strong><br><br>
                <strong>Primary:</strong> Buy ATM Put (0DTE)<br>
                <strong>Alternative:</strong> Bear Put Spread<br>
                <strong>Entry:</strong> On RSI rejection below 70<br>
                <strong>Target:</strong> 50% premium gain or RSI < 50<br>
                <strong>Stop:</strong> 50% premium loss
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⏸️ NO ACTIONABLE SIGNAL</strong><br><br>
                Waiting for:<br>
                • RSI < 30 (oversold) or > 70 (overbought)<br>
                • Volume confirmation (> 1.5x average)<br>
                • Multiple indicator alignment
            </div>
            """, unsafe_allow_html=True)
    
    with col_opt2:
        st.markdown("""
        <div style="background: #1a1a2e; padding: 1rem; border-radius: 8px; border: 1px solid #2a2a4a;">
            <strong style="color: #e6e6e6;">⚠️ Risk Management</strong><br><br>
            <span style="color: #8892b0;">
            • Position size: 1-2% of account per trade<br>
            • Max daily loss: 5% of account<br>
            • Use 0DTE only with strict discipline<br>
            • Never hold 0DTE into close<br>
            • Consider spreads to reduce premium cost
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Session statistics
    if len(session_df) > 5:
        st.markdown('<p class="section-header">📊 Session Statistics</p>', unsafe_allow_html=True)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            if 'gap_return' in session_df.columns:
                gap_data = session_df['gap_return'].dropna()
                avg_gap = gap_data.mean()
                gap_up_pct = (gap_data > 0).mean() * 100
                
                st.metric("Avg Gap", f"{avg_gap:.2f}%")
                st.metric("Gap Up %", f"{gap_up_pct:.0f}%")
        
        with col_stat2:
            if 'first_5min_return' in session_df.columns:
                first5_data = session_df['first_5min_return'].dropna()
                avg_first5 = first5_data.mean()
                first5_pos = (first5_data > 0).mean() * 100
                
                st.metric("Avg First 5min", f"{avg_first5:.2f}%")
                st.metric("First 5min Up %", f"{first5_pos:.0f}%")
        
        with col_stat3:
            if 'gap_return' in session_df.columns and 'first_5min_return' in session_df.columns:
                # Correlation between gap and first 5 min
                valid = session_df[['gap_return', 'first_5min_return']].dropna()
                if len(valid) > 5:
                    corr = valid['gap_return'].corr(valid['first_5min_return'])
                    st.metric("Gap vs First5min Corr", f"{corr:.3f}")
                    
                    # Mean reversion rate
                    mr_rate = ((valid['gap_return'] > 0) & (valid['first_5min_return'] < 0)).mean() + \
                              ((valid['gap_return'] < 0) & (valid['first_5min_return'] > 0)).mean()
                    st.metric("Mean Reversion Rate", f"{mr_rate*100:.1f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #8892b0; font-size: 0.85rem;">
        <p>Built by Limestone Hill Capital | Data from Polygon.io</p>
        <p>⚠️ This is not financial advice. Trade at your own risk.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()