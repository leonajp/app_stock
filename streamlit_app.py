"""
APP Mean Reversion Trading System
==================================
Multi-page Streamlit app with:
1. Live Dashboard - Real-time signals and monitoring
2. Backtest Analysis - Session predictors and RSI analysis
3. Walk-Forward Validation - Anti-overfitting testing

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
from scipy.stats import percentileofscore, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="APP Mean Reversion System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    /* Clean light-dark theme - no heavy shadows */
    .stApp {
        background: #0f1117;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #3d7ab5;
    }
    
    .main-header h1 {
        color: #00e5ff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #b0c4de;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric cards - cleaner, brighter */
    .metric-card {
        background: #1a2332;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #2a4a6a;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-value.positive { color: #00e676; }
    .metric-value.negative { color: #ff5252; }
    .metric-value.neutral { color: #ffca28; }
    
    .metric-label {
        color: #90a4ae;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Signal boxes - more vibrant */
    .signal-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .signal-long {
        background: rgba(0, 230, 118, 0.15);
        border: 2px solid #00e676;
    }
    
    .signal-short {
        background: rgba(255, 82, 82, 0.15);
        border: 2px solid #ff5252;
    }
    
    .signal-neutral {
        background: rgba(255, 202, 40, 0.12);
        border: 2px solid #ffca28;
    }
    
    .signal-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Section headers - cleaner */
    .section-header {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3d5a80;
    }
    
    /* Info boxes - brighter accents */
    .info-box {
        background: rgba(0, 229, 255, 0.08);
        border-left: 4px solid #00e5ff;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    .warning-box {
        background: rgba(255, 202, 40, 0.08);
        border-left: 4px solid #ffca28;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        color: #e0e0e0;
    }
    
    /* Make text more readable */
    .stMarkdown, .stText, p, span, div {
        color: #e0e0e0;
    }
    
    /* Table styling */
    .stDataFrame {
        background: #1a2332;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a2332;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API KEY HANDLING
# ============================================================================

def get_api_key() -> Optional[str]:
    """Get API key from Streamlit secrets or user input"""
    try:
        if "POLYGON_API_KEY" in st.secrets:
            return st.secrets["POLYGON_API_KEY"]
    except:
        pass
    
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
    
    @st.cache_data(ttl=300)
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


# ============================================================================
# FEATURE CALCULATOR
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
    def calc_bollinger(df: pd.DataFrame, period: int = 20, std: float = 2.0):
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


# ============================================================================
# SESSION ANALYZER (COMPREHENSIVE)
# ============================================================================

class SessionAnalyzer:
    """Comprehensive session analysis for mean reversion"""
    
    @staticmethod
    def classify_session(timestamp: pd.Timestamp) -> str:
        """Classify timestamp into trading session"""
        time = timestamp.time()
        
        if dt_time(9, 30) <= time < dt_time(16, 0):
            return "regular"
        elif dt_time(16, 0) <= time < dt_time(20, 0):
            return "postmarket"
        elif time >= dt_time(20, 0) or time < dt_time(4, 0):
            return "overnight"
        else:
            return "premarket"
    
    @staticmethod
    def calculate_session_returns(minute_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive session returns"""
        df = minute_df.copy()
        df['session'] = df.index.map(SessionAnalyzer.classify_session)
        df['date'] = df.index.date
        df['time'] = df.index.time
        
        results = []
        dates = sorted(df['date'].unique())
        
        for i, date in enumerate(dates):
            day_data = df[df['date'] == date]
            prev_date = dates[i-1] if i > 0 else None
            prev_day_data = df[df['date'] == prev_date] if prev_date else None
            
            record = {'date': date}
            
            # Previous day regular session
            if prev_day_data is not None:
                prev_regular = prev_day_data[prev_day_data['session'] == 'regular']
                if len(prev_regular) > 0:
                    record['prev_day_return'] = (
                        prev_regular['close'].iloc[-1] / prev_regular['open'].iloc[0] - 1
                    ) * 100
                    record['prev_day_close'] = prev_regular['close'].iloc[-1]
                    record['prev_day_high'] = prev_regular['high'].max()
                    record['prev_day_low'] = prev_regular['low'].min()
            
            # Post-market
            if prev_day_data is not None:
                postmarket = prev_day_data[prev_day_data['session'] == 'postmarket']
                if len(postmarket) > 0 and 'prev_day_close' in record:
                    record['postmarket_return'] = (
                        postmarket['close'].iloc[-1] / record['prev_day_close'] - 1
                    ) * 100
                    record['postmarket_close'] = postmarket['close'].iloc[-1]
            
            # Pre-market
            premarket = day_data[day_data['session'] == 'premarket']
            if len(premarket) > 0:
                ref_price = record.get('postmarket_close', record.get('prev_day_close'))
                if ref_price:
                    record['premarket_return'] = (
                        premarket['close'].iloc[-1] / ref_price - 1
                    ) * 100
                    record['premarket_close'] = premarket['close'].iloc[-1]
                    record['premarket_high'] = premarket['high'].max()
                    record['premarket_low'] = premarket['low'].min()
            
            # Regular session
            regular = day_data[day_data['session'] == 'regular']
            if len(regular) > 0:
                record['open'] = regular['open'].iloc[0]
                record['close'] = regular['close'].iloc[-1]
                record['high'] = regular['high'].max()
                record['low'] = regular['low'].min()
                record['volume'] = regular['volume'].sum()
                
                if 'prev_day_close' in record:
                    record['gap_return'] = (
                        record['open'] / record['prev_day_close'] - 1
                    ) * 100
                
                # First 5 minutes
                first_5min = regular[regular['time'] <= dt_time(9, 35)]
                if len(first_5min) > 0:
                    record['first_5min_return'] = (
                        first_5min['close'].iloc[-1] / first_5min['open'].iloc[0] - 1
                    ) * 100
                    record['first_5min_close'] = first_5min['close'].iloc[-1]
                    record['first_5min_high'] = first_5min['high'].max()
                    record['first_5min_low'] = first_5min['low'].min()
                
                # First 30 minutes
                first_30min = regular[regular['time'] <= dt_time(10, 0)]
                if len(first_30min) > 0:
                    record['first_30min_return'] = (
                        first_30min['close'].iloc[-1] / first_30min['open'].iloc[0] - 1
                    ) * 100
                
                record['day_return'] = (record['close'] / record['open'] - 1) * 100
            
            # Total extended hours move
            record['total_extended_return'] = (
                record.get('postmarket_return', 0) +
                record.get('premarket_return', 0)
            )
            
            results.append(record)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def analyze_session_predictors(session_df: pd.DataFrame) -> Dict:
        """Analyze which session returns predict first 5-min movement"""
        df = session_df.dropna(subset=['first_5min_return'])
        
        predictors = ['prev_day_return', 'postmarket_return', 'premarket_return', 'gap_return', 'total_extended_return']
        results = {}
        
        for predictor in predictors:
            if predictor not in df.columns:
                continue
            
            valid = df.dropna(subset=[predictor])
            if len(valid) < 10:
                continue
            
            x = valid[predictor].values
            y = valid['first_5min_return'].values
            
            # Correlation
            corr, p_value = stats.pearsonr(x, y)
            
            # Directional accuracy
            contrarian_correct = np.sum((x < 0) & (y > 0) | (x > 0) & (y < 0)) / len(x)
            momentum_correct = 1 - contrarian_correct
            
            # Mean reversion after extreme moves
            extreme_negative = valid[valid[predictor] < -1]['first_5min_return']
            extreme_positive = valid[valid[predictor] > 1]['first_5min_return']
            
            results[predictor] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'contrarian_accuracy': contrarian_correct * 100,
                'momentum_accuracy': momentum_correct * 100,
                'signal_type': 'CONTRARIAN' if contrarian_correct > 0.5 else 'MOMENTUM',
                'avg_first5min_after_negative': extreme_negative.mean() if len(extreme_negative) > 0 else None,
                'avg_first5min_after_positive': extreme_positive.mean() if len(extreme_positive) > 0 else None,
                'n_samples': len(valid)
            }
        
        return results


# ============================================================================
# RSI MEAN REVERSION BACKTESTER
# ============================================================================

class RSIBacktester:
    """Backtest RSI mean reversion strategy"""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def find_rsi_reversals(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Find RSI reversal points and calculate P&L"""
        df = minute_df.copy()
        df['rsi'] = FeatureCalculator.calc_rsi(df['close'], self.rsi_period)
        
        # Filter to regular hours
        df = df[(df.index.time >= dt_time(9, 30)) & (df.index.time < dt_time(16, 0))]
        
        signals = []
        in_position = False
        entry = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['rsi']):
                continue
            
            # Entry signals
            if not in_position:
                # Oversold bounce
                if prev_row['rsi'] < self.oversold and row['rsi'] >= self.oversold:
                    entry = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'entry_rsi': row['rsi'],
                        'direction': 'long',
                        'signal': 'oversold_bounce'
                    }
                    in_position = True
                
                # Overbought reversal
                elif prev_row['rsi'] > self.overbought and row['rsi'] <= self.overbought:
                    entry = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'entry_rsi': row['rsi'],
                        'direction': 'short',
                        'signal': 'overbought_reversal'
                    }
                    in_position = True
            
            # Exit signals
            else:
                should_exit = False
                exit_reason = None
                
                # RSI neutralized
                if entry['direction'] == 'long' and row['rsi'] >= 50:
                    should_exit = True
                    exit_reason = 'rsi_neutral'
                elif entry['direction'] == 'short' and row['rsi'] <= 50:
                    should_exit = True
                    exit_reason = 'rsi_neutral'
                
                # End of day
                if row.name.time() >= dt_time(15, 55):
                    should_exit = True
                    exit_reason = 'eod'
                
                # Stop loss
                if entry['direction'] == 'long' and row['rsi'] < 20:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif entry['direction'] == 'short' and row['rsi'] > 80:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                if should_exit:
                    pnl = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100
                    if entry['direction'] == 'short':
                        pnl = -pnl
                    
                    signals.append({
                        **entry,
                        'exit_time': row.name,
                        'exit_price': row['close'],
                        'exit_rsi': row['rsi'],
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl,
                        'duration_mins': (row.name - entry['entry_time']).total_seconds() / 60
                    })
                    
                    in_position = False
                    entry = None
        
        return pd.DataFrame(signals)
    
    def calculate_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate backtest statistics"""
        if len(trades_df) == 0:
            return {'error': 'No trades'}
        
        df = trades_df
        
        stats = {
            'total_trades': len(df),
            'win_rate': (df['pnl_pct'] > 0).mean() * 100,
            'avg_pnl': df['pnl_pct'].mean(),
            'total_pnl': df['pnl_pct'].sum(),
            'max_win': df['pnl_pct'].max(),
            'max_loss': df['pnl_pct'].min(),
            'avg_duration_mins': df['duration_mins'].mean(),
            'sharpe': df['pnl_pct'].mean() / df['pnl_pct'].std() * np.sqrt(252) if df['pnl_pct'].std() > 0 else 0,
            'profit_factor': abs(df[df['pnl_pct'] > 0]['pnl_pct'].sum() / df[df['pnl_pct'] < 0]['pnl_pct'].sum()) if df[df['pnl_pct'] < 0]['pnl_pct'].sum() != 0 else float('inf')
        }
        
        # By signal type
        for signal_type in ['oversold_bounce', 'overbought_reversal']:
            subset = df[df['signal'] == signal_type]
            if len(subset) > 0:
                stats[f'{signal_type}_count'] = len(subset)
                stats[f'{signal_type}_win_rate'] = (subset['pnl_pct'] > 0).mean() * 100
                stats[f'{signal_type}_avg_pnl'] = subset['pnl_pct'].mean()
        
        return stats


class ImprovedBacktester:
    """
    Improved RSI strategy with better risk management.
    
    Key improvements over basic RSI:
    1. Profit target exit (don't let winners become losers)
    2. Price-based stop loss (faster than RSI-based)
    3. Volume confirmation (only trade with conviction)
    4. Time filter (avoid choppy open and close)
    
    These are NOT curve-fit parameters - they're standard risk management.
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
        # Risk management parameters (standard, not optimized)
        self.profit_target_pct = 0.4   # Take profit at 0.4% gain
        self.stop_loss_pct = 0.3       # Stop loss at 0.3% loss
        self.max_hold_mins = 20        # Max hold time 20 minutes
        self.min_rel_volume = 1.0      # Minimum relative volume
    
    def find_improved_trades(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Find trades with improved risk management"""
        df = minute_df.copy()
        df['rsi'] = FeatureCalculator.calc_rsi(df['close'], self.rsi_period)
        df['rel_volume'] = FeatureCalculator.calc_relative_volume(df, lookback=50)
        
        # Filter to regular hours, avoid first 5 min and last 15 min
        df = df[
            (df.index.time >= dt_time(9, 35)) & 
            (df.index.time < dt_time(15, 45))
        ]
        
        signals = []
        in_position = False
        entry = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['rsi']) or pd.isna(row['rel_volume']):
                continue
            
            # Entry signals with volume confirmation
            if not in_position:
                # Volume filter - need above average volume for conviction
                has_volume = row['rel_volume'] >= self.min_rel_volume
                
                # Oversold bounce (LONG)
                if prev_row['rsi'] < self.oversold and row['rsi'] >= self.oversold and has_volume:
                    entry = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'entry_rsi': row['rsi'],
                        'entry_volume': row['rel_volume'],
                        'direction': 'long',
                        'signal': 'oversold_bounce'
                    }
                    in_position = True
                
                # Overbought reversal (SHORT)
                elif prev_row['rsi'] > self.overbought and row['rsi'] <= self.overbought and has_volume:
                    entry = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'entry_rsi': row['rsi'],
                        'entry_volume': row['rel_volume'],
                        'direction': 'short',
                        'signal': 'overbought_reversal'
                    }
                    in_position = True
            
            # Exit signals - improved risk management
            else:
                should_exit = False
                exit_reason = None
                
                # Calculate current P&L
                if entry['direction'] == 'long':
                    current_pnl = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100
                else:
                    current_pnl = (entry['entry_price'] - row['close']) / entry['entry_price'] * 100
                
                # 1. PROFIT TARGET - most important, lock in gains
                if current_pnl >= self.profit_target_pct:
                    should_exit = True
                    exit_reason = 'profit_target'
                
                # 2. STOP LOSS - price-based, faster than RSI
                elif current_pnl <= -self.stop_loss_pct:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                # 3. TIME EXIT - don't hold too long
                mins_held = (row.name - entry['entry_time']).total_seconds() / 60
                if mins_held >= self.max_hold_mins and not should_exit:
                    should_exit = True
                    exit_reason = 'time_exit'
                
                # 4. RSI NEUTRAL - original exit (backup)
                if not should_exit:
                    if entry['direction'] == 'long' and row['rsi'] >= 55:
                        should_exit = True
                        exit_reason = 'rsi_neutral'
                    elif entry['direction'] == 'short' and row['rsi'] <= 45:
                        should_exit = True
                        exit_reason = 'rsi_neutral'
                
                # 5. END OF DAY
                if row.name.time() >= dt_time(15, 50):
                    should_exit = True
                    exit_reason = 'eod'
                
                if should_exit:
                    pnl = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100
                    if entry['direction'] == 'short':
                        pnl = -pnl
                    
                    signals.append({
                        **entry,
                        'exit_time': row.name,
                        'exit_price': row['close'],
                        'exit_rsi': row['rsi'],
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl,
                        'duration_mins': (row.name - entry['entry_time']).total_seconds() / 60
                    })
                    
                    in_position = False
                    entry = None
        
        return pd.DataFrame(signals)
    
    def calculate_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate backtest statistics"""
        if len(trades_df) == 0:
            return {'error': 'No trades'}
        
        df = trades_df
        
        stats = {
            'total_trades': len(df),
            'win_rate': (df['pnl_pct'] > 0).mean() * 100,
            'avg_pnl': df['pnl_pct'].mean(),
            'total_pnl': df['pnl_pct'].sum(),
            'max_win': df['pnl_pct'].max(),
            'max_loss': df['pnl_pct'].min(),
            'avg_duration_mins': df['duration_mins'].mean(),
            'sharpe': df['pnl_pct'].mean() / df['pnl_pct'].std() * np.sqrt(252) if df['pnl_pct'].std() > 0 else 0,
            'profit_factor': abs(df[df['pnl_pct'] > 0]['pnl_pct'].sum() / df[df['pnl_pct'] < 0]['pnl_pct'].sum()) if df[df['pnl_pct'] < 0]['pnl_pct'].sum() != 0 else float('inf')
        }
        
        # By exit reason
        for reason in df['exit_reason'].unique():
            subset = df[df['exit_reason'] == reason]
            stats[f'{reason}_count'] = len(subset)
            stats[f'{reason}_win_rate'] = (subset['pnl_pct'] > 0).mean() * 100
            stats[f'{reason}_avg_pnl'] = subset['pnl_pct'].mean()
        
        # By signal type
        for signal_type in ['oversold_bounce', 'overbought_reversal']:
            subset = df[df['signal'] == signal_type]
            if len(subset) > 0:
                stats[f'{signal_type}_count'] = len(subset)
                stats[f'{signal_type}_win_rate'] = (subset['pnl_pct'] > 0).mean() * 100
                stats[f'{signal_type}_avg_pnl'] = subset['pnl_pct'].mean()
        
        return stats


class OptionsBacktester:
    """
    Options-aware backtester with:
    1. Strike selection based on confidence (ATM vs OTM)
    2. Leverage-based P&L calculation (simpler, more realistic)
    3. VIX filter for long trades
    
    Key insight: Options provide leverage.
    - ATM option (~5% of stock price) with delta 0.50 → ~3-4x leverage
    - OTM option (~2% of stock price) with delta 0.30 → ~5-8x leverage
    
    We model this as: option_return = stock_return × leverage_factor
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
        # Leverage factors by strike type (conservative, realistic for 0-1 DTE options)
        self.leverage = {
            'ATM': 3.0,       # ATM: moderate leverage (reduced from 3.5)
            'OTM': 5.0,       # OTM: higher leverage (reduced from 6.0)
            'DEEP_OTM': 8.0   # Deep OTM: max leverage (reduced from 10.0)
        }
        
        # Position size inversely proportional to leverage (KEY RISK MANAGEMENT)
        self.position_size_map = {
            'ATM': 1.0,       # Full size for ATM
            'OTM': 0.60,      # 60% size for OTM 
            'DEEP_OTM': 0.35  # 35% size for Deep OTM
        }
        
        # Max loss is 100% of premium (can't lose more than you paid)
        self.max_loss_pct = -100.0
        
        # VIX thresholds
        self.vix_caution = 20     # Reduce long size above this
        self.vix_fear = 25        # Skip longs above this
        self.vix_rising_threshold = 2  # VIX up 2+ points = rising
    
    def select_strike(self, spot_price: float, direction: str, confidence: str) -> tuple:
        """
        Select strike price based on confidence level.
        Returns: (strike, strike_type, leverage)
        """
        def round_strike(price):
            return round(price / 5) * 5
        
        if direction == 'long':  # Buying calls
            if confidence == 'VERY_HIGH':
                # Deep OTM - max leverage for high conviction
                strike = round_strike(spot_price * 1.03)  # 3% OTM
                return strike, 'DEEP_OTM', self.leverage['DEEP_OTM']
            elif confidence == 'HIGH':
                # OTM - more leverage
                strike = round_strike(spot_price * 1.015)  # 1.5% OTM
                return strike, 'OTM', self.leverage['OTM']
            else:
                # ATM - balanced
                strike = round_strike(spot_price)
                return strike, 'ATM', self.leverage['ATM']
        else:  # Buying puts
            if confidence == 'VERY_HIGH':
                strike = round_strike(spot_price * 0.97)  # 3% OTM
                return strike, 'DEEP_OTM', self.leverage['DEEP_OTM']
            elif confidence == 'HIGH':
                strike = round_strike(spot_price * 0.985)  # 1.5% OTM
                return strike, 'OTM', self.leverage['OTM']
            else:
                strike = round_strike(spot_price)
                return strike, 'ATM', self.leverage['ATM']
    
    def assess_confidence(self, rsi: float, rel_volume: float, vix: float, 
                          vix_change: float, direction: str) -> str:
        """
        Assess trade confidence based on multiple factors.
        More lenient scoring to allow OTM trades when conditions are good.
        """
        confidence_score = 0
        
        # RSI extremity (most important)
        if direction == 'long':
            if rsi < 20:
                confidence_score += 3  # Very oversold
            elif rsi < 25:
                confidence_score += 2
            elif rsi < 28:
                confidence_score += 1
        else:
            if rsi > 80:
                confidence_score += 3  # Very overbought
            elif rsi > 75:
                confidence_score += 2
            elif rsi > 72:
                confidence_score += 1
        
        # Volume confirmation (lowered thresholds)
        if rel_volume > 2.0:
            confidence_score += 2
        elif rel_volume > 1.3:
            confidence_score += 1
        
        # VIX context
        if direction == 'long':
            # High VIX + very oversold = capitulation, good for bounce
            if vix > 20 and rsi < 27:
                confidence_score += 1
            # VIX falling = fear subsiding
            if vix_change < -0.5:
                confidence_score += 1
        else:
            # Low VIX + overbought = complacency, good for puts
            if vix < 16 and rsi > 73:
                confidence_score += 1
            # VIX rising = fear building, good for puts
            if vix_change > 0.5:
                confidence_score += 1
        
        if confidence_score >= 4:
            return 'VERY_HIGH'
        elif confidence_score >= 2:
            return 'HIGH'
        else:
            return 'NORMAL'
    
    def calculate_option_pnl(self, stock_pnl_pct: float, leverage: float, 
                              direction: str, strike_type: str, vix: float = 18.0) -> float:
        """
        Calculate option P&L using leverage model with realistic adjustments.
        
        Key improvements:
        1. VIX adjustment - high VIX = expensive options = lower effective leverage
        2. Asymmetric losses - options lose value faster due to delta + theta
        3. Small win penalty - time decay eats into small profits on OTM
        """
        # VIX adjustment: High VIX means options are expensive, less bang for buck
        # VIX 18 = normal, VIX 30 = ~85% effectiveness, VIX 12 = ~105% effectiveness
        vix_factor = 1.0 - (vix - 18) * 0.008
        vix_factor = np.clip(vix_factor, 0.75, 1.10)
        
        # Adjusted leverage
        adjusted_leverage = leverage * vix_factor
        
        # Base option P&L
        option_pnl = stock_pnl_pct * adjusted_leverage
        
        # ASYMMETRIC ADJUSTMENT (key to realistic options)
        if stock_pnl_pct < 0:
            # Losses hurt more - delta shrinks as you lose, theta still ticking
            if strike_type == 'ATM':
                option_pnl *= 1.1   # 10% worse
            elif strike_type == 'OTM':
                option_pnl *= 1.2   # 20% worse (OTM more sensitive)
            else:  # DEEP_OTM
                option_pnl *= 1.35  # 35% worse
        elif stock_pnl_pct > 0 and stock_pnl_pct < 0.3:
            # Small wins: Theta eats into profit on OTM
            if strike_type in ['OTM', 'DEEP_OTM']:
                option_pnl *= 0.6  # 40% haircut on small OTM wins
        
        # Cap losses at -100% (can't lose more than premium)
        option_pnl = max(self.max_loss_pct, option_pnl)
        
        # Cap gains realistically
        max_gain = {'ATM': 70, 'OTM': 100, 'DEEP_OTM': 150}
        option_pnl = min(max_gain.get(strike_type, 70), option_pnl)
        
        return option_pnl
    
    def find_options_trades(self, minute_df: pd.DataFrame, vix_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Find options trades with strike selection and VIX filter.
        """
        df = minute_df.copy()
        df['rsi'] = FeatureCalculator.calc_rsi(df['close'], self.rsi_period)
        df['rel_volume'] = FeatureCalculator.calc_relative_volume(df, lookback=50)
        
        # Prepare VIX data if provided
        vix_daily = None
        if vix_df is not None and len(vix_df) > 0:
            vix_daily = vix_df.resample('D').agg({'close': 'last'}).dropna()
            vix_daily['vix_change'] = vix_daily['close'].diff()
            vix_daily['vix_rising'] = vix_daily['vix_change'] > self.vix_rising_threshold
        
        # Filter to regular hours
        df = df[
            (df.index.time >= dt_time(9, 35)) & 
            (df.index.time < dt_time(15, 45))
        ]
        
        signals = []
        in_position = False
        entry = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            current_date = row.name.date()
            
            if pd.isna(row['rsi']) or pd.isna(row['rel_volume']):
                continue
            
            # Get VIX for today
            vix = 18.0
            vix_change = 0
            vix_rising = False
            
            if vix_daily is not None:
                try:
                    vix_row = vix_daily.loc[vix_daily.index.date == current_date].iloc[-1]
                    vix = vix_row['close']
                    vix_change = vix_row['vix_change'] if not pd.isna(vix_row['vix_change']) else 0
                    vix_rising = vix_row['vix_rising']
                except:
                    pass
            
            # Entry signals
            if not in_position:
                has_volume = row['rel_volume'] >= 1.0
                
                # Oversold bounce (LONG / CALL)
                if prev_row['rsi'] < self.oversold and row['rsi'] >= self.oversold and has_volume:
                    
                    # VIX FILTER: Skip longs when VIX high AND rising
                    if vix > self.vix_fear and vix_rising:
                        continue
                    
                    # Reduce size in elevated/rising VIX
                    position_size = 1.0
                    if vix > self.vix_caution:
                        position_size = 0.6
                    if vix_rising and vix > 18:
                        position_size *= 0.8
                    
                    confidence = self.assess_confidence(
                        row['rsi'], row['rel_volume'], vix, vix_change, 'long'
                    )
                    
                    strike, strike_type, leverage = self.select_strike(
                        row['close'], 'long', confidence
                    )
                    
                    entry = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'entry_rsi': row['rsi'],
                        'entry_volume': row['rel_volume'],
                        'direction': 'long',
                        'signal': 'oversold_bounce',
                        'option_type': 'CALL',
                        'strike': strike,
                        'strike_type': strike_type,
                        'leverage': leverage,
                        'confidence': confidence,
                        'vix': vix,
                        'vix_change': vix_change,
                        'position_size': position_size
                    }
                    in_position = True
                
                # Overbought reversal (SHORT / PUT)
                elif prev_row['rsi'] > self.overbought and row['rsi'] <= self.overbought and has_volume:
                    
                    # Puts benefit from high VIX (bigger moves), but expensive
                    position_size = 1.0
                    if vix > 28:
                        position_size = 0.7  # Reduce when puts very expensive
                    
                    confidence = self.assess_confidence(
                        row['rsi'], row['rel_volume'], vix, vix_change, 'short'
                    )
                    
                    strike, strike_type, leverage = self.select_strike(
                        row['close'], 'short', confidence
                    )
                    
                    entry = {
                        'entry_time': row.name,
                        'entry_price': row['close'],
                        'entry_rsi': row['rsi'],
                        'entry_volume': row['rel_volume'],
                        'direction': 'short',
                        'signal': 'overbought_reversal',
                        'option_type': 'PUT',
                        'strike': strike,
                        'strike_type': strike_type,
                        'leverage': leverage,
                        'confidence': confidence,
                        'vix': vix,
                        'vix_change': vix_change,
                        'position_size': position_size
                    }
                    in_position = True
            
            # Exit signals
            else:
                should_exit = False
                exit_reason = None
                
                # Calculate stock P&L
                if entry['direction'] == 'long':
                    stock_pnl = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100
                else:
                    stock_pnl = (entry['entry_price'] - row['close']) / entry['entry_price'] * 100
                
                # Profit/loss targets (adjust for strike type)
                if entry['strike_type'] == 'ATM':
                    profit_target = 0.30
                    stop_loss = -0.25
                elif entry['strike_type'] == 'OTM':
                    profit_target = 0.45  # Need bigger move
                    stop_loss = -0.30
                else:  # DEEP_OTM
                    profit_target = 0.60
                    stop_loss = -0.35
                
                if stock_pnl >= profit_target:
                    should_exit = True
                    exit_reason = 'profit_target'
                elif stock_pnl <= stop_loss:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                # Time exit
                mins_held = (row.name - entry['entry_time']).total_seconds() / 60
                if mins_held >= 20 and not should_exit:
                    should_exit = True
                    exit_reason = 'time_exit'
                
                # RSI neutral
                if not should_exit:
                    if entry['direction'] == 'long' and row['rsi'] >= 50:
                        should_exit = True
                        exit_reason = 'rsi_neutral'
                    elif entry['direction'] == 'short' and row['rsi'] <= 50:
                        should_exit = True
                        exit_reason = 'rsi_neutral'
                
                # EOD
                if row.name.time() >= dt_time(15, 50):
                    should_exit = True
                    exit_reason = 'eod'
                
                if should_exit:
                    # Final stock P&L
                    stock_pnl = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100
                    if entry['direction'] == 'short':
                        stock_pnl = -stock_pnl
                    
                    # Option P&L using leverage model with VIX adjustment
                    option_pnl = self.calculate_option_pnl(
                        stock_pnl, entry['leverage'], 
                        entry['direction'], entry['strike_type'],
                        vix=entry.get('vix', 18.0)
                    )
                    
                    # Adjust for position size (inverse to leverage)
                    base_size = self.position_size_map.get(entry['strike_type'], 1.0)
                    final_size = entry['position_size'] * base_size
                    adj_option_pnl = option_pnl * final_size
                    
                    signals.append({
                        **entry,
                        'exit_time': row.name,
                        'exit_price': row['close'],
                        'exit_rsi': row['rsi'],
                        'exit_reason': exit_reason,
                        'stock_pnl_pct': stock_pnl,
                        'option_pnl_pct': option_pnl,
                        'adj_option_pnl_pct': adj_option_pnl,
                        'duration_mins': (row.name - entry['entry_time']).total_seconds() / 60
                    })
                    
                    in_position = False
                    entry = None
        
        return pd.DataFrame(signals)
    
    def calculate_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate backtest statistics for options trades"""
        if len(trades_df) == 0:
            return {'error': 'No trades'}
        
        df = trades_df
        pnl_col = 'adj_option_pnl_pct'
        
        stats = {
            'total_trades': len(df),
            'win_rate': (df[pnl_col] > 0).mean() * 100,
            'avg_pnl': df[pnl_col].mean(),
            'total_pnl': df[pnl_col].sum(),
            'max_win': df[pnl_col].max(),
            'max_loss': df[pnl_col].min(),
            'avg_duration_mins': df['duration_mins'].mean(),
            'sharpe': df[pnl_col].mean() / df[pnl_col].std() * np.sqrt(252) if df[pnl_col].std() > 0 else 0,
            'profit_factor': abs(df[df[pnl_col] > 0][pnl_col].sum() / df[df[pnl_col] < 0][pnl_col].sum()) if df[df[pnl_col] < 0][pnl_col].sum() != 0 else float('inf'),
            
            # Compare to stock
            'stock_total_pnl': df['stock_pnl_pct'].sum(),
            'options_vs_stock': df[pnl_col].sum() - df['stock_pnl_pct'].sum()
        }
        
        # By strike type
        for strike_type in ['ATM', 'OTM', 'DEEP_OTM']:
            subset = df[df['strike_type'] == strike_type]
            if len(subset) > 0:
                stats[f'{strike_type}_trades'] = len(subset)
                stats[f'{strike_type}_win_rate'] = (subset[pnl_col] > 0).mean() * 100
                stats[f'{strike_type}_avg_pnl'] = subset[pnl_col].mean()
                stats[f'{strike_type}_total_pnl'] = subset[pnl_col].sum()
        
        # By confidence
        for conf in ['NORMAL', 'HIGH', 'VERY_HIGH']:
            subset = df[df['confidence'] == conf]
            if len(subset) > 0:
                stats[f'{conf}_trades'] = len(subset)
                stats[f'{conf}_win_rate'] = (subset[pnl_col] > 0).mean() * 100
                stats[f'{conf}_total_pnl'] = subset[pnl_col].sum()
        
        # By direction
        for direction in ['long', 'short']:
            subset = df[df['direction'] == direction]
            if len(subset) > 0:
                stats[f'{direction}_trades'] = len(subset)
                stats[f'{direction}_win_rate'] = (subset[pnl_col] > 0).mean() * 100
                stats[f'{direction}_total_pnl'] = subset[pnl_col].sum()
        
        return stats


def generate_vix_sample_data(n_days: int = 60, base_vix: float = 18.0) -> pd.DataFrame:
    """
    Generate sample VIX data with realistic behavior.
    VIX tends to spike during market drops and mean-revert.
    """
    np.random.seed(456)
    
    all_data = []
    vix = base_vix
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day_offset in range(n_days, -1, -1):
        date = today - timedelta(days=day_offset)
        
        if date.weekday() >= 5:
            continue
        
        # VIX mean reversion (stronger than stocks)
        mean_rev = (base_vix - vix) / base_vix * 0.1
        
        # Random shock (occasional spikes)
        if np.random.random() < 0.05:  # 5% chance of spike
            shock = np.random.uniform(2, 8)
        else:
            shock = np.random.normal(0, 1)
        
        vix = vix * (1 + mean_rev) + shock
        vix = max(10, min(50, vix))  # VIX typically 10-50
        
        # Generate minute data (VIX doesn't trade but we simulate)
        for hour in range(9, 16):
            start_min = 30 if hour == 9 else 0
            for minute in range(start_min, 60):
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if day_offset == 0 and ts > datetime.now():
                    continue
                
                intraday_noise = np.random.normal(0, 0.005)
                intraday_vix = vix * (1 + intraday_noise)
                
                all_data.append({
                    'timestamp': ts,
                    'open': intraday_vix,
                    'high': intraday_vix * 1.01,
                    'low': intraday_vix * 0.99,
                    'close': intraday_vix,
                    'volume': 0,
                    'vwap': intraday_vix
                })
    
    if len(all_data) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
    
    df = pd.DataFrame(all_data)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index).tz_localize('America/New_York')
    
    return df


class MarketAwareBacktester:
    """
    Enhanced backtester that uses QQQ as a market regime filter.
    
    Key insight: Mean reversion works best when market is NOT trending strongly.
    We filter trades based on:
    1. QQQ regime (trending vs range-bound)
    2. QQQ momentum alignment
    3. Market volatility level
    
    Anti-overfitting measures:
    - Simple, interpretable rules (only 3 filter conditions)
    - Walk-forward validation
    - Economic intuition for each rule
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
        # Market filter parameters (kept simple to avoid overfitting)
        self.qqq_trend_threshold = 2.0  # % 5-day move that indicates trending
        self.high_vol_threshold = 2.0   # Daily vol % threshold
    
    def calculate_qqq_regime(self, qqq_minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate QQQ market regime indicators at daily level.
        
        Regimes:
        - STRONG_UP: QQQ 5-day return > +2% (avoid shorts)
        - STRONG_DOWN: QQQ 5-day return < -2% (avoid longs)  
        - HIGH_VOL: Daily vol > 2% (reduce position size)
        - NORMAL: Good for mean reversion
        """
        # Resample to daily
        qqq_daily = qqq_minute_df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # Daily returns
        qqq_daily['daily_return'] = qqq_daily['close'].pct_change() * 100
        
        # 5-day momentum
        qqq_daily['momentum_5d'] = qqq_daily['close'].pct_change(5) * 100
        
        # 10-day volatility
        qqq_daily['volatility'] = qqq_daily['daily_return'].rolling(10).std()
        
        # QQQ RSI
        qqq_daily['qqq_rsi'] = FeatureCalculator.calc_rsi(qqq_daily['close'], 14)
        
        # Classify regime
        def get_regime(row):
            if pd.isna(row['momentum_5d']):
                return 'NORMAL'
            if row['momentum_5d'] > self.qqq_trend_threshold:
                return 'STRONG_UP'
            elif row['momentum_5d'] < -self.qqq_trend_threshold:
                return 'STRONG_DOWN'
            elif row['volatility'] > self.high_vol_threshold:
                return 'HIGH_VOL'
            return 'NORMAL'
        
        qqq_daily['regime'] = qqq_daily.apply(get_regime, axis=1)
        
        return qqq_daily
    
    def find_market_aware_trades(self, app_minute_df: pd.DataFrame, 
                                  qqq_minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find RSI reversal trades with market regime filter.
        
        Filter Rules (simple, interpretable):
        1. LONG trades: Skip if QQQ regime is STRONG_DOWN
        2. SHORT trades: Skip if QQQ regime is STRONG_UP
        3. Position size: 50% in HIGH_VOL regime
        4. Confidence boost: If QQQ RSI confirms direction
        """
        df = app_minute_df.copy()
        df['rsi'] = FeatureCalculator.calc_rsi(df['close'], self.rsi_period)
        
        # Get QQQ regime
        qqq_regime = self.calculate_qqq_regime(qqq_minute_df)
        
        # Filter to regular hours
        df = df[(df.index.time >= dt_time(9, 30)) & (df.index.time < dt_time(16, 0))]
        
        signals = []
        in_position = False
        entry = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            current_date = row.name.date()
            
            if pd.isna(row['rsi']):
                continue
            
            # Get market regime for today
            try:
                regime_row = qqq_regime.loc[qqq_regime.index.date == current_date].iloc[-1]
                regime = regime_row['regime']
                qqq_rsi = regime_row['qqq_rsi']
                qqq_momentum = regime_row['momentum_5d']
            except:
                regime = 'NORMAL'
                qqq_rsi = 50
                qqq_momentum = 0
            
            # Entry signals with market filter
            if not in_position:
                # Oversold bounce (LONG)
                if prev_row['rsi'] < self.oversold and row['rsi'] >= self.oversold:
                    # FILTER: Skip longs in strong downtrend
                    if regime != 'STRONG_DOWN':
                        # Confidence: Higher if QQQ also oversold
                        confidence = 'HIGH' if qqq_rsi < 35 else 'NORMAL'
                        position_size = 0.5 if regime == 'HIGH_VOL' else 1.0
                        
                        entry = {
                            'entry_time': row.name,
                            'entry_price': row['close'],
                            'entry_rsi': row['rsi'],
                            'direction': 'long',
                            'signal': 'oversold_bounce',
                            'regime': regime,
                            'qqq_rsi': qqq_rsi,
                            'qqq_momentum': qqq_momentum,
                            'confidence': confidence,
                            'position_size': position_size
                        }
                        in_position = True
                
                # Overbought reversal (SHORT)
                elif prev_row['rsi'] > self.overbought and row['rsi'] <= self.overbought:
                    # FILTER: Skip shorts in strong uptrend
                    if regime != 'STRONG_UP':
                        confidence = 'HIGH' if qqq_rsi > 65 else 'NORMAL'
                        position_size = 0.5 if regime == 'HIGH_VOL' else 1.0
                        
                        entry = {
                            'entry_time': row.name,
                            'entry_price': row['close'],
                            'entry_rsi': row['rsi'],
                            'direction': 'short',
                            'signal': 'overbought_reversal',
                            'regime': regime,
                            'qqq_rsi': qqq_rsi,
                            'qqq_momentum': qqq_momentum,
                            'confidence': confidence,
                            'position_size': position_size
                        }
                        in_position = True
            
            # Exit signals (same as basic)
            else:
                should_exit = False
                exit_reason = None
                
                if entry['direction'] == 'long' and row['rsi'] >= 50:
                    should_exit = True
                    exit_reason = 'rsi_neutral'
                elif entry['direction'] == 'short' and row['rsi'] <= 50:
                    should_exit = True
                    exit_reason = 'rsi_neutral'
                
                if row.name.time() >= dt_time(15, 55):
                    should_exit = True
                    exit_reason = 'eod'
                
                if entry['direction'] == 'long' and row['rsi'] < 20:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif entry['direction'] == 'short' and row['rsi'] > 80:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                if should_exit:
                    raw_pnl = (row['close'] - entry['entry_price']) / entry['entry_price'] * 100
                    if entry['direction'] == 'short':
                        raw_pnl = -raw_pnl
                    
                    # Adjusted PnL accounts for position sizing
                    adj_pnl = raw_pnl * entry['position_size']
                    
                    signals.append({
                        **entry,
                        'exit_time': row.name,
                        'exit_price': row['close'],
                        'exit_rsi': row['rsi'],
                        'exit_reason': exit_reason,
                        'pnl_pct': raw_pnl,
                        'adj_pnl_pct': adj_pnl,
                        'duration_mins': (row.name - entry['entry_time']).total_seconds() / 60
                    })
                    
                    in_position = False
                    entry = None
        
        return pd.DataFrame(signals)
    
    def calculate_stats(self, trades_df: pd.DataFrame, use_adjusted: bool = True) -> Dict:
        """Calculate backtest statistics"""
        if len(trades_df) == 0:
            return {'error': 'No trades'}
        
        df = trades_df
        pnl_col = 'adj_pnl_pct' if use_adjusted and 'adj_pnl_pct' in df.columns else 'pnl_pct'
        
        stats = {
            'total_trades': len(df),
            'win_rate': (df[pnl_col] > 0).mean() * 100,
            'avg_pnl': df[pnl_col].mean(),
            'total_pnl': df[pnl_col].sum(),
            'max_win': df[pnl_col].max(),
            'max_loss': df[pnl_col].min(),
            'avg_duration_mins': df['duration_mins'].mean(),
            'sharpe': df[pnl_col].mean() / df[pnl_col].std() * np.sqrt(252) if df[pnl_col].std() > 0 else 0,
            'profit_factor': abs(df[df[pnl_col] > 0][pnl_col].sum() / df[df[pnl_col] < 0][pnl_col].sum()) if df[df[pnl_col] < 0][pnl_col].sum() != 0 else float('inf')
        }
        
        # By regime
        if 'regime' in df.columns:
            for regime in df['regime'].unique():
                subset = df[df['regime'] == regime]
                if len(subset) > 0:
                    stats[f'{regime}_trades'] = len(subset)
                    stats[f'{regime}_win_rate'] = (subset[pnl_col] > 0).mean() * 100
                    stats[f'{regime}_total_pnl'] = subset[pnl_col].sum()
        
        # By signal type
        for signal_type in ['oversold_bounce', 'overbought_reversal']:
            subset = df[df['signal'] == signal_type]
            if len(subset) > 0:
                stats[f'{signal_type}_count'] = len(subset)
                stats[f'{signal_type}_win_rate'] = (subset[pnl_col] > 0).mean() * 100
                stats[f'{signal_type}_avg_pnl'] = subset[pnl_col].mean()
        
        return stats
    
    @staticmethod
    def walk_forward_validation(trades_df: pd.DataFrame, train_pct: float = 0.7) -> Dict:
        """
        Walk-forward validation to detect overfitting.
        
        Compares train vs test performance. If test is much worse, we're overfitting.
        """
        if len(trades_df) < 20:
            return {'valid': False, 'reason': 'Not enough trades'}
        
        df = trades_df.sort_values('entry_time').copy()
        pnl_col = 'adj_pnl_pct' if 'adj_pnl_pct' in df.columns else 'pnl_pct'
        
        split_idx = int(len(df) * train_pct)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        
        def calc_metrics(subset):
            if len(subset) == 0:
                return {'n': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'sharpe': 0}
            pnl = subset[pnl_col]
            return {
                'n': len(subset),
                'win_rate': (pnl > 0).mean() * 100,
                'total_pnl': pnl.sum(),
                'avg_pnl': pnl.mean(),
                'sharpe': pnl.mean() / pnl.std() if pnl.std() > 0 else 0
            }
        
        train_metrics = calc_metrics(train)
        test_metrics = calc_metrics(test)
        
        # Overfit detection: test sharpe < 50% of train sharpe
        overfit = False
        if train_metrics['sharpe'] > 0.1:
            if test_metrics['sharpe'] < train_metrics['sharpe'] * 0.5:
                overfit = True
        
        return {
            'valid': True,
            'train': train_metrics,
            'test': test_metrics,
            'train_period': f"{train['entry_time'].iloc[0].strftime('%m/%d')} - {train['entry_time'].iloc[-1].strftime('%m/%d')}",
            'test_period': f"{test['entry_time'].iloc[0].strftime('%m/%d')} - {test['entry_time'].iloc[-1].strftime('%m/%d')}",
            'overfit_warning': overfit,
            'performance_decay': (1 - test_metrics['sharpe'] / train_metrics['sharpe']) * 100 if train_metrics['sharpe'] > 0 else 0
        }


# ============================================================================
# OUT-OF-SAMPLE VALIDATION SYSTEM
# ============================================================================

class RollingWalkForwardValidator:
    """
    Multiple rolling train/test windows through time.

    Tracks performance consistency across periods to detect:
    - Parameter decay over time
    - Regime changes affecting strategy
    - Overfitting to specific market conditions
    """

    def __init__(self, train_window_days: int = 30, test_window_days: int = 10, step_days: int = None):
        self.train_window = train_window_days
        self.test_window = test_window_days
        self.step = step_days or test_window_days

    def run_rolling_validation(self, trades_df: pd.DataFrame, min_trades_per_period: int = 5) -> Dict:
        """
        Run rolling walk-forward validation.

        Returns:
        - periods: List of train/test period results
        - consistency_score: 0-100 score of performance stability
        - performance_trend: Slope of test Sharpes over time
        - aggregate_oos_stats: Combined out-of-sample statistics
        """
        if len(trades_df) < 20:
            return {'valid': False, 'reason': 'Not enough trades for rolling validation'}

        df = trades_df.sort_values('entry_time').copy()
        pnl_col = 'adj_pnl_pct' if 'adj_pnl_pct' in df.columns else 'pnl_pct'

        # Get date range
        start_date = df['entry_time'].min().date()
        end_date = df['entry_time'].max().date()
        total_days = (end_date - start_date).days

        if total_days < self.train_window + self.test_window:
            return {'valid': False, 'reason': f'Not enough days ({total_days}) for train({self.train_window})+test({self.test_window}) windows'}

        periods = []
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=self.train_window)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_window)

            if test_end > end_date:
                break

            # Get trades for each period
            train_mask = (df['entry_time'].dt.date >= current_start) & (df['entry_time'].dt.date < train_end)
            test_mask = (df['entry_time'].dt.date >= test_start) & (df['entry_time'].dt.date < test_end)

            train_trades = df[train_mask]
            test_trades = df[test_mask]

            if len(train_trades) >= min_trades_per_period and len(test_trades) >= min_trades_per_period:
                train_stats = self._calc_period_stats(train_trades, pnl_col)
                test_stats = self._calc_period_stats(test_trades, pnl_col)

                periods.append({
                    'period_num': len(periods) + 1,
                    'train_start': current_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_stats': train_stats,
                    'test_stats': test_stats
                })

            current_start = current_start + timedelta(days=self.step)

        if len(periods) < 2:
            return {'valid': False, 'reason': 'Not enough valid periods (need at least 2)'}

        # Calculate aggregate metrics
        consistency = self.calculate_consistency_metrics(periods)
        decay = self.detect_parameter_decay(periods)

        # Aggregate OOS stats
        all_oos_pnls = []
        for p in periods:
            all_oos_pnls.extend([p['test_stats']['avg_pnl']] * p['test_stats']['n_trades'])

        aggregate_oos = {
            'total_periods': len(periods),
            'total_oos_trades': sum(p['test_stats']['n_trades'] for p in periods),
            'avg_oos_pnl': np.mean(all_oos_pnls) if all_oos_pnls else 0,
            'avg_oos_sharpe': np.mean([p['test_stats']['sharpe'] for p in periods]),
            'avg_oos_win_rate': np.mean([p['test_stats']['win_rate'] for p in periods])
        }

        return {
            'valid': True,
            'periods': periods,
            'consistency': consistency,
            'decay': decay,
            'aggregate_oos': aggregate_oos
        }

    def _calc_period_stats(self, trades: pd.DataFrame, pnl_col: str) -> Dict:
        """Calculate statistics for a single period"""
        pnl = trades[pnl_col]
        return {
            'n_trades': len(trades),
            'win_rate': (pnl > 0).mean() * 100,
            'total_pnl': pnl.sum(),
            'avg_pnl': pnl.mean(),
            'sharpe': pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0
        }

    def calculate_consistency_metrics(self, periods: List[Dict]) -> Dict:
        """Calculate consistency metrics across periods"""
        test_sharpes = [p['test_stats']['sharpe'] for p in periods]
        test_win_rates = [p['test_stats']['win_rate'] for p in periods]
        test_pnls = [p['test_stats']['avg_pnl'] for p in periods]

        # Coefficient of variation (lower = more consistent)
        sharpe_cv = np.std(test_sharpes) / np.mean(test_sharpes) if np.mean(test_sharpes) != 0 else float('inf')

        # Sign consistency (% of periods with positive Sharpe)
        sign_consistency = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) * 100

        # Consistency score: 100 - (CV * 50), clamped to 0-100
        consistency_score = max(0, min(100, 100 - sharpe_cv * 50))

        return {
            'consistency_score': consistency_score,
            'sharpe_cv': sharpe_cv,
            'sign_consistency': sign_consistency,
            'sharpe_mean': np.mean(test_sharpes),
            'sharpe_std': np.std(test_sharpes),
            'sharpe_min': min(test_sharpes),
            'sharpe_max': max(test_sharpes),
            'win_rate_mean': np.mean(test_win_rates),
            'pnl_mean': np.mean(test_pnls)
        }

    def detect_parameter_decay(self, periods: List[Dict]) -> Dict:
        """Detect if performance is decaying over time"""
        test_sharpes = [p['test_stats']['sharpe'] for p in periods]
        period_nums = list(range(len(test_sharpes)))

        # Linear regression for trend
        if len(period_nums) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(period_nums, test_sharpes)
        else:
            slope, r_value, p_value = 0, 0, 1

        # Compare first half vs second half
        mid = len(test_sharpes) // 2
        first_half_avg = np.mean(test_sharpes[:mid]) if mid > 0 else 0
        second_half_avg = np.mean(test_sharpes[mid:]) if mid < len(test_sharpes) else 0

        decay_pct = ((first_half_avg - second_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0

        return {
            'slope': slope,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'first_half_sharpe': first_half_avg,
            'second_half_sharpe': second_half_avg,
            'decay_pct': decay_pct,
            'is_decaying': slope < -0.1 and p_value < 0.1  # Significant negative trend
        }


class HoldoutValidator:
    """
    Reserved holdout period management.

    Locks away final N days completely until user explicitly reveals.
    Prevents lookahead bias contamination.
    """

    def __init__(self, holdout_days: int = 60):
        self.holdout_days = holdout_days

    def split_data(self, trades_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split trades into development and holdout sets"""
        if len(trades_df) == 0:
            return pd.DataFrame(), pd.DataFrame()

        df = trades_df.sort_values('entry_time').copy()
        end_date = df['entry_time'].max().date()
        holdout_start = end_date - timedelta(days=self.holdout_days)

        dev_mask = df['entry_time'].dt.date < holdout_start
        holdout_mask = df['entry_time'].dt.date >= holdout_start

        return df[dev_mask], df[holdout_mask]

    def calculate_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate statistics for a trade set"""
        if len(trades_df) == 0:
            return {'n_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0, 'sharpe': 0, 'profit_factor': 0}

        pnl_col = 'adj_pnl_pct' if 'adj_pnl_pct' in trades_df.columns else 'pnl_pct'
        pnl = trades_df[pnl_col]

        wins = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())

        return {
            'n_trades': len(trades_df),
            'win_rate': (pnl > 0).mean() * 100,
            'total_pnl': pnl.sum(),
            'avg_pnl': pnl.mean(),
            'sharpe': pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else 0,
            'profit_factor': wins / losses if losses > 0 else float('inf'),
            'max_win': pnl.max(),
            'max_loss': pnl.min()
        }

    def reveal_holdout(self, trades_df: pd.DataFrame) -> Dict:
        """
        Reveal holdout results with comparison to development period.
        """
        dev_df, holdout_df = self.split_data(trades_df)

        if len(holdout_df) == 0:
            return {'valid': False, 'reason': 'No trades in holdout period'}

        dev_stats = self.calculate_stats(dev_df)
        holdout_stats = self.calculate_stats(holdout_df)

        # Calculate degradation
        degradation = self.calculate_degradation(dev_stats, holdout_stats)

        # Determine pass/fail
        pass_criteria = {
            'sharpe_ratio': holdout_stats['sharpe'] >= dev_stats['sharpe'] * 0.6 if dev_stats['sharpe'] > 0 else True,
            'win_rate': holdout_stats['win_rate'] >= dev_stats['win_rate'] * 0.8 if dev_stats['win_rate'] > 0 else True,
            'profit_factor': holdout_stats['profit_factor'] >= 1.0,
            'positive_pnl': holdout_stats['total_pnl'] > 0
        }

        overall_pass = sum(pass_criteria.values()) >= 3  # Pass at least 3 of 4 criteria

        return {
            'valid': True,
            'dev_stats': dev_stats,
            'holdout_stats': holdout_stats,
            'degradation': degradation,
            'pass_criteria': pass_criteria,
            'overall_pass': overall_pass,
            'dev_period': f"{dev_df['entry_time'].min().strftime('%m/%d/%y')} - {dev_df['entry_time'].max().strftime('%m/%d/%y')}" if len(dev_df) > 0 else "N/A",
            'holdout_period': f"{holdout_df['entry_time'].min().strftime('%m/%d/%y')} - {holdout_df['entry_time'].max().strftime('%m/%d/%y')}" if len(holdout_df) > 0 else "N/A"
        }

    def calculate_degradation(self, dev_stats: Dict, holdout_stats: Dict) -> Dict:
        """Measure performance decay from development to holdout"""
        def safe_pct_change(dev, holdout):
            if dev == 0:
                return 0
            return ((holdout - dev) / abs(dev)) * 100

        return {
            'sharpe_change_pct': safe_pct_change(dev_stats['sharpe'], holdout_stats['sharpe']),
            'win_rate_change_pct': safe_pct_change(dev_stats['win_rate'], holdout_stats['win_rate']),
            'avg_pnl_change_pct': safe_pct_change(dev_stats['avg_pnl'], holdout_stats['avg_pnl']),
            'holdout_to_dev_ratio': holdout_stats['sharpe'] / dev_stats['sharpe'] if dev_stats['sharpe'] > 0 else 0
        }


class MonteCarloSimulator:
    """
    Bootstrap and permutation testing for statistical significance.

    Tests whether strategy performance is statistically significant
    vs random chance.
    """

    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_sims = n_simulations
        self.confidence = confidence_level

    def bootstrap_trades(self, trades_df: pd.DataFrame) -> Dict:
        """
        Resample trades with replacement to estimate confidence intervals.
        """
        if len(trades_df) < 10:
            return {'valid': False, 'reason': 'Not enough trades for bootstrap'}

        pnl_col = 'adj_pnl_pct' if 'adj_pnl_pct' in trades_df.columns else 'pnl_pct'
        pnls = trades_df[pnl_col].values

        # Bootstrap resampling
        bootstrap_totals = []
        bootstrap_means = []
        bootstrap_sharpes = []

        for _ in range(self.n_sims):
            # Sample with replacement
            sample = np.random.choice(pnls, size=len(pnls), replace=True)
            bootstrap_totals.append(sample.sum())
            bootstrap_means.append(sample.mean())
            if sample.std() > 0:
                bootstrap_sharpes.append(sample.mean() / sample.std() * np.sqrt(252))
            else:
                bootstrap_sharpes.append(0)

        # Calculate confidence intervals
        alpha = 1 - self.confidence
        ci_lower_pct = alpha / 2 * 100
        ci_upper_pct = (1 - alpha / 2) * 100

        return {
            'valid': True,
            'actual_total_pnl': pnls.sum(),
            'actual_mean_pnl': pnls.mean(),
            'actual_sharpe': pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0,
            'bootstrap_mean_total': np.mean(bootstrap_totals),
            'bootstrap_std_total': np.std(bootstrap_totals),
            'ci_total_lower': np.percentile(bootstrap_totals, ci_lower_pct),
            'ci_total_upper': np.percentile(bootstrap_totals, ci_upper_pct),
            'ci_mean_lower': np.percentile(bootstrap_means, ci_lower_pct),
            'ci_mean_upper': np.percentile(bootstrap_means, ci_upper_pct),
            'ci_sharpe_lower': np.percentile(bootstrap_sharpes, ci_lower_pct),
            'ci_sharpe_upper': np.percentile(bootstrap_sharpes, ci_upper_pct),
            'bootstrap_totals': bootstrap_totals,
            'bootstrap_sharpes': bootstrap_sharpes,
            'ci_excludes_zero': np.percentile(bootstrap_totals, ci_lower_pct) > 0,
            'pct_positive': sum(1 for t in bootstrap_totals if t > 0) / len(bootstrap_totals) * 100
        }

    def random_baseline_test(self, trades_df: pd.DataFrame) -> Dict:
        """
        Compare strategy returns vs random entry/exit timing.
        Shuffles trade PnLs and compares distributions.
        """
        if len(trades_df) < 10:
            return {'valid': False, 'reason': 'Not enough trades for random baseline test'}

        pnl_col = 'adj_pnl_pct' if 'adj_pnl_pct' in trades_df.columns else 'pnl_pct'
        pnls = trades_df[pnl_col].values

        actual_total = pnls.sum()
        actual_sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0

        # Generate random baseline by shuffling
        random_totals = []
        random_sharpes = []

        for _ in range(self.n_sims):
            # Random signs on same absolute values (random direction)
            random_pnls = np.abs(pnls) * np.random.choice([-1, 1], size=len(pnls))
            random_totals.append(random_pnls.sum())
            if random_pnls.std() > 0:
                random_sharpes.append(random_pnls.mean() / random_pnls.std() * np.sqrt(252))
            else:
                random_sharpes.append(0)

        # Calculate percentile rank
        pct_rank = percentileofscore(random_totals, actual_total)
        sharpe_pct_rank = percentileofscore(random_sharpes, actual_sharpe)

        # P-value: probability of achieving this by chance
        p_value = 1 - pct_rank / 100

        return {
            'valid': True,
            'actual_total_pnl': actual_total,
            'actual_sharpe': actual_sharpe,
            'random_mean_total': np.mean(random_totals),
            'random_std_total': np.std(random_totals),
            'random_mean_sharpe': np.mean(random_sharpes),
            'percentile_rank': pct_rank,
            'sharpe_percentile_rank': sharpe_pct_rank,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'random_totals': random_totals,
            'random_sharpes': random_sharpes
        }

    def shuffle_analysis(self, trades_df: pd.DataFrame) -> Dict:
        """
        Analyze if trade sequence matters by shuffling order.
        Tests for path-dependency in drawdowns.
        """
        if len(trades_df) < 10:
            return {'valid': False, 'reason': 'Not enough trades for shuffle analysis'}

        pnl_col = 'adj_pnl_pct' if 'adj_pnl_pct' in trades_df.columns else 'pnl_pct'
        pnls = trades_df[pnl_col].values

        # Calculate actual max drawdown
        cumsum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumsum)
        drawdowns = cumsum - running_max
        actual_max_dd = drawdowns.min()

        # Shuffle and calculate drawdowns
        shuffled_max_dds = []
        for _ in range(self.n_sims):
            shuffled = np.random.permutation(pnls)
            cumsum = np.cumsum(shuffled)
            running_max = np.maximum.accumulate(cumsum)
            drawdowns = cumsum - running_max
            shuffled_max_dds.append(drawdowns.min())

        # Compare actual vs shuffled
        dd_percentile = percentileofscore(shuffled_max_dds, actual_max_dd)

        return {
            'valid': True,
            'actual_max_drawdown': actual_max_dd,
            'shuffled_mean_dd': np.mean(shuffled_max_dds),
            'shuffled_median_dd': np.median(shuffled_max_dds),
            'shuffled_worst_dd': min(shuffled_max_dds),
            'shuffled_best_dd': max(shuffled_max_dds),
            'dd_percentile': dd_percentile,
            'sequence_helps': actual_max_dd > np.median(shuffled_max_dds),  # Better than median shuffled
            'shuffled_dds': shuffled_max_dds
        }


class ValidationOrchestrator:
    """
    Coordinates all validation methods and provides unified interface.
    """

    def __init__(self, train_window: int = 30, test_window: int = 10,
                 holdout_days: int = 60, n_simulations: int = 1000):
        self.rolling_validator = RollingWalkForwardValidator(train_window, test_window)
        self.holdout_validator = HoldoutValidator(holdout_days)
        self.monte_carlo = MonteCarloSimulator(n_simulations)

    def run_full_validation(self, trades_df: pd.DataFrame, reveal_holdout: bool = False) -> Dict:
        """
        Run all validation methods and compile comprehensive report.
        """
        results = {
            'rolling': None,
            'holdout': None,
            'monte_carlo': None,
            'summary': None
        }

        # Rolling walk-forward
        results['rolling'] = self.rolling_validator.run_rolling_validation(trades_df)

        # Holdout (only if requested)
        if reveal_holdout:
            results['holdout'] = self.holdout_validator.reveal_holdout(trades_df)
        else:
            dev_df, holdout_df = self.holdout_validator.split_data(trades_df)
            dev_stats = self.holdout_validator.calculate_stats(dev_df)
            results['holdout'] = {
                'valid': True,
                'revealed': False,
                'dev_stats': dev_stats,
                'holdout_trade_count': len(holdout_df),
                'dev_period': f"{dev_df['entry_time'].min().strftime('%m/%d/%y')} - {dev_df['entry_time'].max().strftime('%m/%d/%y')}" if len(dev_df) > 0 else "N/A"
            }

        # Monte Carlo (on development data only if holdout not revealed)
        if reveal_holdout:
            mc_df = trades_df
        else:
            mc_df, _ = self.holdout_validator.split_data(trades_df)

        results['monte_carlo'] = {
            'bootstrap': self.monte_carlo.bootstrap_trades(mc_df),
            'random_baseline': self.monte_carlo.random_baseline_test(mc_df),
            'shuffle': self.monte_carlo.shuffle_analysis(mc_df)
        }

        # Generate summary
        results['summary'] = self.generate_validation_summary(results)

        return results

    def generate_validation_summary(self, results: Dict) -> Dict:
        """Generate summary with pass/fail indicators"""
        tests = []

        # Rolling validation tests
        if results['rolling'] and results['rolling'].get('valid'):
            rolling = results['rolling']

            # Consistency score test
            consistency_score = rolling['consistency']['consistency_score']
            tests.append({
                'name': 'Rolling Consistency',
                'value': f"{consistency_score:.0f}/100",
                'pass': consistency_score >= 60,
                'threshold': '>= 60'
            })

            # Sign consistency test
            sign_consistency = rolling['consistency']['sign_consistency']
            tests.append({
                'name': 'Positive Period Rate',
                'value': f"{sign_consistency:.0f}%",
                'pass': sign_consistency >= 60,
                'threshold': '>= 60%'
            })

            # Parameter decay test
            decay_pct = rolling['decay']['decay_pct']
            tests.append({
                'name': 'Parameter Decay',
                'value': f"{decay_pct:.0f}%",
                'pass': decay_pct < 30,
                'threshold': '< 30%'
            })

        # Holdout test (if revealed)
        if results['holdout'] and results['holdout'].get('valid') and results['holdout'].get('overall_pass') is not None:
            holdout = results['holdout']
            ratio = holdout['degradation']['holdout_to_dev_ratio'] * 100
            tests.append({
                'name': 'Holdout vs Dev Ratio',
                'value': f"{ratio:.0f}%",
                'pass': ratio >= 60,
                'threshold': '>= 60%'
            })

        # Monte Carlo tests
        if results['monte_carlo']:
            mc = results['monte_carlo']

            # Bootstrap CI test
            if mc['bootstrap'].get('valid'):
                ci_excludes_zero = mc['bootstrap']['ci_excludes_zero']
                tests.append({
                    'name': 'Bootstrap CI > 0',
                    'value': 'Yes' if ci_excludes_zero else 'No',
                    'pass': ci_excludes_zero,
                    'threshold': 'CI excludes zero'
                })

            # Statistical significance test
            if mc['random_baseline'].get('valid'):
                p_value = mc['random_baseline']['p_value']
                tests.append({
                    'name': 'Statistical Significance',
                    'value': f"p={p_value:.3f}",
                    'pass': p_value < 0.05,
                    'threshold': 'p < 0.05'
                })

        # Calculate overall score
        passed = sum(1 for t in tests if t['pass'])
        total = len(tests)
        overall_score = (passed / total * 100) if total > 0 else 0

        return {
            'tests': tests,
            'passed': passed,
            'total': total,
            'overall_score': overall_score,
            'overall_pass': passed >= total * 0.7  # Pass 70% of tests
        }


def generate_qqq_sample_data(n_days: int = 60, base_price: float = 525.0) -> pd.DataFrame:
    """Generate sample QQQ minute data for demo mode"""
    np.random.seed(123)  # Different seed than APP
    
    all_data = []
    price = base_price
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day_offset in range(n_days, -1, -1):
        date = today - timedelta(days=day_offset)
        
        if date.weekday() >= 5:
            continue
        
        # QQQ has lower vol than APP
        daily_mean_rev = (base_price - price) / base_price * 0.015
        
        # Pre-market
        gap = np.random.normal(0, 0.006)
        price *= (1 + gap)
        price = max(base_price * 0.90, min(base_price * 1.10, price))
        
        for hour in range(4, 9):
            for minute in range(60):
                if hour == 9 and minute >= 30:
                    break
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if day_offset == 0 and ts > datetime.now():
                    continue
                
                noise = np.random.normal(0, 0.0005)
                price *= (1 + noise)
                price = max(base_price * 0.90, min(base_price * 1.10, price))
                
                all_data.append({
                    'timestamp': ts,
                    'open': price, 'high': price * 1.001, 'low': price * 0.999,
                    'close': price, 'volume': np.random.randint(10000, 50000), 'vwap': price
                })
        
        # Regular hours
        open_price = price
        for hour in range(9, 16):
            start_min = 30 if hour == 9 else 0
            for minute in range(start_min, 60):
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if day_offset == 0 and ts > datetime.now():
                    continue
                
                deviation = (price - open_price) / open_price
                mean_rev = -deviation * 0.002
                base_rev = (base_price - price) / base_price * 0.0008
                noise = np.random.normal(0, 0.0008)
                price *= (1 + noise + mean_rev + base_rev)
                price = max(base_price * 0.90, min(base_price * 1.10, price))
                
                all_data.append({
                    'timestamp': ts,
                    'open': price, 'high': price * 1.001, 'low': price * 0.999,
                    'close': price, 'volume': np.random.randint(50000, 200000), 'vwap': price
                })
        
        # Post-market
        if day_offset > 0:
            for hour in range(16, 20):
                for minute in range(60):
                    ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    noise = np.random.normal(0, 0.0003)
                    price *= (1 + noise)
                    price = max(base_price * 0.90, min(base_price * 1.10, price))
                    all_data.append({
                        'timestamp': ts,
                        'open': price, 'high': price * 1.0005, 'low': price * 0.9995,
                        'close': price, 'volume': np.random.randint(5000, 20000), 'vwap': price
                    })
    
    if len(all_data) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
    
    df = pd.DataFrame(all_data)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index).tz_localize('America/New_York')
    
    return df


# ============================================================================
# INTRADAY REVERSION ANALYZER
# ============================================================================

class IntradayReversionAnalyzer:
    """Analyze how price reverts during the trading day"""
    
    @staticmethod
    def analyze_reversion_from_open(minute_df: pd.DataFrame, session_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze how first 5-min moves revert during the day"""
        results = []
        
        for _, day_session in session_df.iterrows():
            date = day_session['date']
            
            if pd.isna(day_session.get('first_5min_return')):
                continue
            
            # Get regular hours data
            day_data = minute_df[
                (minute_df.index.date == date) &
                (minute_df.index.time >= dt_time(9, 30)) &
                (minute_df.index.time < dt_time(16, 0))
            ]
            
            if len(day_data) < 10:
                continue
            
            first_5min_return = day_session['first_5min_return']
            open_price = day_session['open']
            first_5min_close = day_session.get('first_5min_close', day_data.iloc[5]['close'] if len(day_data) > 5 else None)
            
            if first_5min_close is None:
                continue
            
            direction = 'up' if first_5min_return > 0 else 'down'
            
            # Find max reversion after first 5 min
            after_5min = day_data[day_data.index.time > dt_time(9, 35)]
            
            if len(after_5min) == 0:
                continue
            
            after_5min_returns = (after_5min['close'] - first_5min_close) / first_5min_close * 100
            
            if direction == 'up':
                max_reversion = after_5min_returns.min()
                reverted_to_open = (after_5min['low'].min() < open_price)
            else:
                max_reversion = after_5min_returns.max()
                reverted_to_open = (after_5min['high'].max() > open_price)
            
            results.append({
                'date': date,
                'first_5min_return': first_5min_return,
                'direction': direction,
                'max_reversion': max_reversion,
                'reverted_to_open': reverted_to_open,
                'day_return': day_session['day_return'],
                'gap_return': day_session.get('gap_return', None)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def calculate_reversion_probabilities(reversion_df: pd.DataFrame) -> Dict:
        """Calculate probability of hitting various reversion levels"""
        results = {}
        
        for direction in ['up', 'down']:
            subset = reversion_df[reversion_df['direction'] == direction]
            
            if len(subset) < 5:
                continue
            
            if direction == 'up':
                levels = [-0.5, -1.0, -1.5, -2.0, -2.5]
            else:
                levels = [0.5, 1.0, 1.5, 2.0, 2.5]
            
            level_probs = {}
            for level in levels:
                if direction == 'up':
                    hit_rate = (subset['max_reversion'] <= level).mean() * 100
                else:
                    hit_rate = (subset['max_reversion'] >= level).mean() * 100
                level_probs[f'{level}%'] = hit_rate
            
            results[direction] = {
                'n_samples': len(subset),
                'avg_first_5min_move': subset['first_5min_return'].mean(),
                'avg_reversion': subset['max_reversion'].mean(),
                'reversion_hit_rates': level_probs,
                'reverted_to_open_rate': subset['reverted_to_open'].mean() * 100,
                'same_direction_day': (
                    (subset['day_return'] > 0).mean() if direction == 'up'
                    else (subset['day_return'] < 0).mean()
                ) * 100
            }

        return results


class OvernightPremktDayAnalyzer:
    """
    Statistical analysis of overnight/premarket impact on full day movement.

    Tests hypotheses:
    1. Overnight + Premarket direction predicts full day direction
    2. First N minutes after open is a reversal window
    3. If no reversal by N minutes, continuation is likely
    """

    @staticmethod
    def analyze_overnight_premarket_to_day(session_df: pd.DataFrame) -> Dict:
        """
        Test if overnight+premarket predicts full day direction.

        Returns comprehensive statistics with p-values for significance.
        """
        df = session_df.dropna(subset=['day_return']).copy()

        # Calculate combined overnight+premarket
        if 'total_extended_return' not in df.columns:
            df['total_extended_return'] = df.get('postmarket_return', 0).fillna(0) + df.get('premarket_return', 0).fillna(0)

        # Also calculate gap (overnight only)
        if 'overnight_return' not in df.columns and 'gap_return' in df.columns:
            df['overnight_return'] = df['gap_return']

        results = {}

        # Test different predictor combinations
        predictors = {
            'premarket_only': 'premarket_return',
            'overnight_only': 'gap_return',
            'overnight_premarket': 'total_extended_return'
        }

        for name, col in predictors.items():
            if col not in df.columns:
                continue

            valid = df.dropna(subset=[col])
            if len(valid) < 10:
                continue

            x = valid[col].values * 100  # Convert to %
            y = valid['day_return'].values * 100

            # Correlation test
            corr, corr_p = stats.pearsonr(x, y)

            # Directional accuracy: Does sign of predictor match sign of day?
            same_direction = ((x > 0) & (y > 0)) | ((x < 0) & (y < 0))
            direction_accuracy = same_direction.mean() * 100

            # Chi-square test for independence
            contingency = pd.crosstab(x > 0, y > 0)
            if contingency.shape == (2, 2):
                chi2, chi2_p, dof, expected = stats.chi2_contingency(contingency)
            else:
                chi2, chi2_p = 0, 1

            # Breakdown by predictor direction
            down_premarket = valid[valid[col] < -0.005]  # Down > 0.5%
            up_premarket = valid[valid[col] > 0.005]     # Up > 0.5%

            down_then_day = down_premarket['day_return'].values * 100 if len(down_premarket) > 0 else []
            up_then_day = up_premarket['day_return'].values * 100 if len(up_premarket) > 0 else []

            results[name] = {
                'n_samples': len(valid),
                'correlation': corr,
                'correlation_p': corr_p,
                'correlation_significant': corr_p < 0.05,
                'direction_accuracy': direction_accuracy,
                'chi2_statistic': chi2,
                'chi2_p': chi2_p,
                'chi2_significant': chi2_p < 0.05,
                'signal_type': 'MOMENTUM' if direction_accuracy > 50 else 'CONTRARIAN',
                'down_premarket': {
                    'n': len(down_premarket),
                    'avg_day_return': np.mean(down_then_day) if len(down_then_day) > 0 else 0,
                    'pct_day_down': (np.array(down_then_day) < 0).mean() * 100 if len(down_then_day) > 0 else 0,
                    'pct_day_up': (np.array(down_then_day) > 0).mean() * 100 if len(down_then_day) > 0 else 0,
                },
                'up_premarket': {
                    'n': len(up_premarket),
                    'avg_day_return': np.mean(up_then_day) if len(up_then_day) > 0 else 0,
                    'pct_day_up': (np.array(up_then_day) > 0).mean() * 100 if len(up_then_day) > 0 else 0,
                    'pct_day_down': (np.array(up_then_day) < 0).mean() * 100 if len(up_then_day) > 0 else 0,
                }
            }

        return results

    @staticmethod
    def analyze_reversal_window(minute_df: pd.DataFrame, session_df: pd.DataFrame,
                                reversal_windows: List[int] = [15, 30, 45, 60]) -> Dict:
        """
        Analyze the first N minutes as a reversal window.

        Tests: If market opens down, does it reverse within N minutes?
        """
        results = {}

        for window_mins in reversal_windows:
            window_results = []

            for _, day_session in session_df.iterrows():
                date = day_session['date']

                # Get overnight/premarket direction
                premarket_return = day_session.get('premarket_return', 0) or 0
                gap_return = day_session.get('gap_return', 0) or 0
                extended_return = premarket_return + gap_return

                if abs(extended_return) < 0.003:  # Skip if < 0.3% move
                    continue

                premarket_direction = 'down' if extended_return < 0 else 'up'

                # Get regular hours data
                day_data = minute_df[
                    (minute_df.index.date == date) &
                    (minute_df.index.time >= dt_time(9, 30)) &
                    (minute_df.index.time < dt_time(16, 0))
                ]

                if len(day_data) < window_mins + 5:
                    continue

                open_price = day_data.iloc[0]['open']

                # First N minutes data
                window_end_time = (datetime.combine(date, dt_time(9, 30)) + timedelta(minutes=window_mins)).time()
                first_n_mins = day_data[day_data.index.time <= window_end_time]

                if len(first_n_mins) < 5:
                    continue

                # After window data
                after_window = day_data[day_data.index.time > window_end_time]

                # Check for reversal within window
                if premarket_direction == 'down':
                    # If premarket down, reversal = price goes above open
                    reversed_in_window = first_n_mins['high'].max() > open_price
                    window_high = first_n_mins['high'].max()
                    window_low = first_n_mins['low'].min()
                    reversal_pct = ((window_high - open_price) / open_price * 100) if open_price > 0 else 0
                else:
                    # If premarket up, reversal = price goes below open
                    reversed_in_window = first_n_mins['low'].min() < open_price
                    window_high = first_n_mins['high'].max()
                    window_low = first_n_mins['low'].min()
                    reversal_pct = ((window_low - open_price) / open_price * 100) if open_price > 0 else 0

                # Day close relative to open
                day_close = day_data.iloc[-1]['close']
                day_return = (day_close - open_price) / open_price * 100
                day_direction = 'up' if day_return > 0 else 'down'

                # Continuation after window (if no reversal)
                continued_direction = None
                if len(after_window) > 0:
                    after_close = after_window.iloc[-1]['close']
                    window_close = first_n_mins.iloc[-1]['close']
                    after_return = (after_close - window_close) / window_close * 100
                    continued_direction = 'same' if (extended_return > 0 and after_return > 0) or (extended_return < 0 and after_return < 0) else 'reversed'

                window_results.append({
                    'date': date,
                    'premarket_direction': premarket_direction,
                    'extended_return_pct': extended_return * 100,
                    'reversed_in_window': reversed_in_window,
                    'reversal_pct': reversal_pct,
                    'day_return_pct': day_return,
                    'day_direction': day_direction,
                    'followed_premarket': premarket_direction == day_direction,
                    'continued_after_window': continued_direction
                })

            if len(window_results) < 5:
                continue

            df_results = pd.DataFrame(window_results)

            # Calculate statistics
            down_premarket = df_results[df_results['premarket_direction'] == 'down']
            up_premarket = df_results[df_results['premarket_direction'] == 'up']

            results[f'{window_mins}min'] = {
                'n_samples': len(df_results),
                'overall_reversal_rate': df_results['reversed_in_window'].mean() * 100,
                'overall_followed_premarket': df_results['followed_premarket'].mean() * 100,
                'down_premarket': {
                    'n': len(down_premarket),
                    'reversal_rate': down_premarket['reversed_in_window'].mean() * 100 if len(down_premarket) > 0 else 0,
                    'avg_reversal_pct': down_premarket['reversal_pct'].mean() if len(down_premarket) > 0 else 0,
                    'day_ended_down_pct': (down_premarket['day_direction'] == 'down').mean() * 100 if len(down_premarket) > 0 else 0,
                    'day_ended_up_pct': (down_premarket['day_direction'] == 'up').mean() * 100 if len(down_premarket) > 0 else 0,
                    'no_reversal_continued': down_premarket[~down_premarket['reversed_in_window']]['followed_premarket'].mean() * 100 if len(down_premarket[~down_premarket['reversed_in_window']]) > 0 else 0
                },
                'up_premarket': {
                    'n': len(up_premarket),
                    'reversal_rate': up_premarket['reversed_in_window'].mean() * 100 if len(up_premarket) > 0 else 0,
                    'avg_reversal_pct': up_premarket['reversal_pct'].mean() if len(up_premarket) > 0 else 0,
                    'day_ended_up_pct': (up_premarket['day_direction'] == 'up').mean() * 100 if len(up_premarket) > 0 else 0,
                    'day_ended_down_pct': (up_premarket['day_direction'] == 'down').mean() * 100 if len(up_premarket) > 0 else 0,
                    'no_reversal_continued': up_premarket[~up_premarket['reversed_in_window']]['followed_premarket'].mean() * 100 if len(up_premarket[~up_premarket['reversed_in_window']]) > 0 else 0
                },
                'raw_data': df_results
            }

        return results

    @staticmethod
    def create_hypothesis_summary(overnight_results: Dict, reversal_results: Dict) -> pd.DataFrame:
        """Create a summary table for all hypothesis tests"""
        rows = []

        # Hypothesis 1: Overnight/Premarket predicts day
        for predictor, data in overnight_results.items():
            rows.append({
                'Hypothesis': 'Overnight/Premarket → Day Direction',
                'Test': predictor.replace('_', ' ').title(),
                'N': data['n_samples'],
                'Result': f"{data['direction_accuracy']:.1f}% accuracy",
                'Correlation': f"{data['correlation']:.3f}",
                'P-Value': f"{data['correlation_p']:.4f}",
                'Significant': 'Yes' if data['correlation_significant'] else 'No',
                'Signal': data['signal_type']
            })

        # Hypothesis 2: Reversal window
        for window, data in reversal_results.items():
            if 'raw_data' in data:
                rows.append({
                    'Hypothesis': f'Reversal Window ({window})',
                    'Test': 'After Down Premarket',
                    'N': data['down_premarket']['n'],
                    'Result': f"{data['down_premarket']['reversal_rate']:.1f}% reverse",
                    'Correlation': f"Day down: {data['down_premarket']['day_ended_down_pct']:.1f}%",
                    'P-Value': '-',
                    'Significant': '-',
                    'Signal': 'MOMENTUM' if data['down_premarket']['day_ended_down_pct'] > 50 else 'REVERSAL'
                })
                rows.append({
                    'Hypothesis': f'Reversal Window ({window})',
                    'Test': 'After Up Premarket',
                    'N': data['up_premarket']['n'],
                    'Result': f"{data['up_premarket']['reversal_rate']:.1f}% reverse",
                    'Correlation': f"Day up: {data['up_premarket']['day_ended_up_pct']:.1f}%",
                    'P-Value': '-',
                    'Significant': '-',
                    'Signal': 'MOMENTUM' if data['up_premarket']['day_ended_up_pct'] > 50 else 'REVERSAL'
                })

                # No reversal → continuation
                if data['down_premarket']['no_reversal_continued'] > 0:
                    rows.append({
                        'Hypothesis': f'No Reversal by {window} → Continuation',
                        'Test': 'Down premarket, no reversal',
                        'N': int(data['down_premarket']['n'] * (100 - data['down_premarket']['reversal_rate']) / 100),
                        'Result': f"{data['down_premarket']['no_reversal_continued']:.1f}% continued down",
                        'Correlation': '-',
                        'P-Value': '-',
                        'Significant': 'Yes' if data['down_premarket']['no_reversal_continued'] > 60 else 'No',
                        'Signal': 'CONTINUATION'
                    })

        return pd.DataFrame(rows)


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Generate ensemble trading signals"""
    
    @staticmethod
    def generate_signal(df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'signal': 'NEUTRAL', 'confidence': 0, 'score': 0, 'components': {}}
        
        latest = df.iloc[-1]
        
        # RSI signal
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
        score = (rsi_signal * 0.30 + bb_signal * 0.25 + vwap_signal * 0.20) * (1 + vol_boost * 0.25)
        
        # Confidence
        signals = [rsi_signal, bb_signal, vwap_signal]
        n_bullish = sum(1 for s in signals if s > 0.1)
        n_bearish = sum(1 for s in signals if s < -0.1)
        confidence = max(n_bullish, n_bearish) / len(signals) * 100
        
        # Signal
        if score > 0.2 and confidence > 40:
            signal = 'LONG'
        elif score < -0.2 and confidence > 40:
            signal = 'SHORT'
        else:
            signal = 'NEUTRAL'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'score': score,
            'components': {'RSI': rsi_signal, 'Bollinger': bb_signal, 'VWAP': vwap_signal, 'Volume': vol_boost},
            'raw': {'rsi': rsi, 'rel_volume': rel_vol, 'vwap_dev': vwap_dev}
        }


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """Detect market regime"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        if len(df) < 20:
            return {'regime': 'UNKNOWN', 'description': 'Insufficient data', 'color': '#888888'}
        
        atr = FeatureCalculator.calc_atr(df)
        atr_pct = atr.iloc[-1] / df['close'].iloc[-1] * 100
        
        # Volatility regime
        avg_vol = df['close'].pct_change().abs().rolling(20).mean().iloc[-1] * 100
        high_vol = atr_pct > avg_vol * 1.5
        
        # Trend vs mean reversion
        returns = df['close'].pct_change()
        autocorr = returns.iloc[-20:].autocorr() if len(returns) > 20 else 0
        trending = autocorr > 0.1
        
        if high_vol and not trending:
            return {'regime': 'HIGH_VOL_MR', 'description': 'High volatility, mean reverting - BEST for strategy', 'color': '#00d4aa'}
        elif high_vol and trending:
            return {'regime': 'HIGH_VOL_TREND', 'description': 'High volatility, trending - REDUCE size', 'color': '#ff6b6b'}
        elif not high_vol and not trending:
            return {'regime': 'LOW_VOL_MR', 'description': 'Low volatility, mean reverting - Good', 'color': '#ffd93d'}
        else:
            return {'regime': 'LOW_VOL_TREND', 'description': 'Low volatility, trending - AVOID', 'color': '#ff6b6b'}


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def generate_sample_data(n_days: int = 60, base_price: float = 672.0) -> pd.DataFrame:
    """
    Generate sample minute data with REALISTIC mean reversion patterns.
    
    Key insight: Real stocks like APP have exploitable mean reversion because:
    1. Momentum pushes cause RSI extremes
    2. Market makers and profit-taking cause reversals
    3. These reversals are predictable from RSI signals
    
    This generator creates data where RSI extremes actually predict reversals.
    """
    np.random.seed(42)
    
    all_data = []
    price = base_price
    momentum = 0  # Track recent momentum
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for day_offset in range(n_days, -1, -1):
        date = today - timedelta(days=day_offset)
        
        if date.weekday() >= 5:
            continue
        
        # Overnight gap with mean reversion
        gap = np.random.normal(0, 0.012)
        price *= (1 + gap)
        price = max(base_price * 0.85, min(base_price * 1.15, price))
        
        # Pre-market
        for hour in range(4, 9):
            for minute in range(60):
                if hour == 9 and minute >= 30:
                    break
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if day_offset == 0 and ts > datetime.now():
                    continue
                
                noise = np.random.normal(0, 0.0006)
                price *= (1 + noise)
                price = max(base_price * 0.85, min(base_price * 1.15, price))
                
                all_data.append({
                    'timestamp': ts, 'open': price, 'high': price * 1.001,
                    'low': price * 0.999, 'close': price,
                    'volume': np.random.randint(5000, 30000), 'vwap': price
                })
        
        # Regular hours - THIS IS WHERE MEAN REVERSION HAPPENS
        open_price = price
        momentum = 0
        returns_buffer = []  # Track recent returns for momentum
        
        for hour in range(9, 16):
            start_min = 30 if hour == 9 else 0
            for minute in range(start_min, 60):
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if day_offset == 0 and ts > datetime.now():
                    continue
                
                # Calculate momentum from recent returns
                if len(returns_buffer) >= 14:
                    recent_ups = sum(1 for r in returns_buffer[-14:] if r > 0)
                    recent_downs = sum(1 for r in returns_buffer[-14:] if r < 0)
                    momentum = (recent_ups - recent_downs) / 14  # -1 to +1
                
                # MEAN REVERSION LOGIC:
                # Strong momentum (like RSI extremes) triggers counter-move
                if momentum > 0.4:  # Overbought-like
                    # High probability of down move
                    expected_return = -0.0015 * momentum  # Stronger reversal
                    vol = 0.0012
                elif momentum < -0.4:  # Oversold-like
                    # High probability of up move
                    expected_return = -0.0015 * momentum  # Reversal (positive because momentum is negative)
                    vol = 0.0012
                else:
                    # Normal random walk
                    expected_return = 0
                    vol = 0.0010
                
                # Add some persistence (trends don't reverse instantly)
                persistence = 0.3
                if len(returns_buffer) > 0:
                    expected_return = expected_return * (1 - persistence) + returns_buffer[-1] * persistence * 0.5
                
                # Generate return
                ret = np.random.normal(expected_return, vol)
                
                # Also add base price mean reversion
                base_pull = (base_price - price) / base_price * 0.0003
                ret += base_pull
                
                # Update price
                old_price = price
                price *= (1 + ret)
                price = max(base_price * 0.85, min(base_price * 1.15, price))
                
                # Track actual return
                actual_ret = (price - old_price) / old_price
                returns_buffer.append(actual_ret)
                if len(returns_buffer) > 20:
                    returns_buffer.pop(0)
                
                # Generate realistic OHLC
                high = price * (1 + abs(np.random.normal(0, 0.001)))
                low = price * (1 - abs(np.random.normal(0, 0.001)))
                
                all_data.append({
                    'timestamp': ts,
                    'open': old_price,
                    'high': max(old_price, price, high),
                    'low': min(old_price, price, low),
                    'close': price,
                    'volume': np.random.randint(30000, 150000),
                    'vwap': (old_price + price) / 2
                })
        
        # Post-market
        if day_offset > 0:
            for hour in range(16, 20):
                for minute in range(60):
                    ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    noise = np.random.normal(0, 0.0004)
                    price *= (1 + noise)
                    price = max(base_price * 0.85, min(base_price * 1.15, price))
                    
                    all_data.append({
                        'timestamp': ts, 'open': price, 'high': price * 1.0005,
                        'low': price * 0.9995, 'close': price,
                        'volume': np.random.randint(3000, 15000), 'vwap': price
                    })
    
    if len(all_data) == 0:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'vwap'])
    
    df = pd.DataFrame(all_data)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index).tz_localize('America/New_York')
    
    return df


# ============================================================================
# VALIDATION CHART FUNCTIONS
# ============================================================================

def create_rolling_performance_chart(periods: List[Dict]) -> go.Figure:
    """Create line chart showing test Sharpe ratio over rolling periods"""
    period_nums = [p['period_num'] for p in periods]
    test_sharpes = [p['test_stats']['sharpe'] for p in periods]
    train_sharpes = [p['train_stats']['sharpe'] for p in periods]

    fig = go.Figure()

    # Test period Sharpes (main line)
    fig.add_trace(go.Scatter(
        x=period_nums, y=test_sharpes,
        mode='lines+markers',
        name='Test Sharpe',
        line=dict(color='#00e676', width=3),
        marker=dict(size=10, symbol='circle')
    ))

    # Train period Sharpes (reference)
    fig.add_trace(go.Scatter(
        x=period_nums, y=train_sharpes,
        mode='lines+markers',
        name='Train Sharpe',
        line=dict(color='#4fc3f7', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    # Add trend line for test Sharpes
    if len(period_nums) >= 2:
        z = np.polyfit(period_nums, test_sharpes, 1)
        p = np.poly1d(z)
        trend_color = '#ff5252' if z[0] < 0 else '#00e676'
        fig.add_trace(go.Scatter(
            x=period_nums, y=p(period_nums),
            mode='lines',
            name='Trend',
            line=dict(color=trend_color, width=2, dash='dot')
        ))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#90a4ae", line_width=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a2332',
        title=dict(text='Rolling Walk-Forward Performance', font=dict(color='#ffffff')),
        xaxis_title='Period',
        yaxis_title='Sharpe Ratio',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )

    return fig


def create_holdout_comparison_chart(dev_stats: Dict, holdout_stats: Dict) -> go.Figure:
    """Create side-by-side bar chart comparing development vs holdout metrics"""
    metrics = ['Sharpe', 'Win Rate', 'Profit Factor']
    dev_values = [dev_stats['sharpe'], dev_stats['win_rate'], min(dev_stats['profit_factor'], 5)]
    holdout_values = [holdout_stats['sharpe'], holdout_stats['win_rate'], min(holdout_stats['profit_factor'], 5)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Development',
        x=metrics,
        y=dev_values,
        marker_color='#4fc3f7',
        text=[f"{v:.2f}" for v in dev_values],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='Holdout',
        x=metrics,
        y=holdout_values,
        marker_color='#00e676',
        text=[f"{v:.2f}" for v in holdout_values],
        textposition='outside'
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a2332',
        title=dict(text='Development vs Holdout Performance', font=dict(color='#ffffff')),
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=400
    )

    return fig


def create_bootstrap_histogram(bootstrap_results: Dict) -> go.Figure:
    """Create histogram of bootstrap distribution with confidence interval"""
    totals = bootstrap_results['bootstrap_totals']
    actual = bootstrap_results['actual_total_pnl']
    ci_lower = bootstrap_results['ci_total_lower']
    ci_upper = bootstrap_results['ci_total_upper']

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=totals,
        nbinsx=50,
        name='Bootstrap Distribution',
        marker_color='#4fc3f7',
        opacity=0.7
    ))

    # Confidence interval shading
    fig.add_vrect(
        x0=ci_lower, x1=ci_upper,
        fillcolor='rgba(0, 230, 118, 0.2)',
        line_width=0,
        annotation_text=f"95% CI",
        annotation_position="top"
    )

    # Actual value line
    fig.add_vline(
        x=actual,
        line_dash="solid",
        line_color="#ffca28",
        line_width=3,
        annotation_text=f"Actual: {actual:.2f}%"
    )

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="#ff5252", line_width=2)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a2332',
        title=dict(text='Bootstrap Distribution of Total P&L', font=dict(color='#ffffff')),
        xaxis_title='Total P&L (%)',
        yaxis_title='Frequency',
        showlegend=False,
        height=400
    )

    return fig


def create_significance_chart(random_baseline_results: Dict) -> go.Figure:
    """Create histogram showing strategy vs random baseline"""
    random_totals = random_baseline_results['random_totals']
    actual = random_baseline_results['actual_total_pnl']
    pct_rank = random_baseline_results['percentile_rank']

    fig = go.Figure()

    # Random distribution histogram
    fig.add_trace(go.Histogram(
        x=random_totals,
        nbinsx=50,
        name='Random Trading',
        marker_color='#90a4ae',
        opacity=0.7
    ))

    # Actual strategy line
    color = '#00e676' if pct_rank > 95 else '#ffca28' if pct_rank > 50 else '#ff5252'
    fig.add_vline(
        x=actual,
        line_dash="solid",
        line_color=color,
        line_width=3,
        annotation_text=f"Your Strategy\n{pct_rank:.1f}th percentile"
    )

    # Zero line
    fig.add_vline(x=0, line_dash="dash", line_color="#ffffff", line_width=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a2332',
        title=dict(text='Strategy vs Random Baseline', font=dict(color='#ffffff')),
        xaxis_title='Total P&L (%)',
        yaxis_title='Frequency',
        showlegend=False,
        height=400
    )

    return fig


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_price_chart(df: pd.DataFrame, title: str = "Price") -> go.Figure:
    """Create candlestick chart with brighter colors"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2], subplot_titles=(title, "RSI", "Volume"))
    
    # Brighter candlestick colors
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                  name="Price", increasing_line_color='#00e676', decreasing_line_color='#ff5252',
                                  increasing_fillcolor='#00e676', decreasing_fillcolor='#ff5252'), row=1, col=1)
    
    if len(df) > 20:
        ma, upper, lower = FeatureCalculator.calc_bollinger(df)
        fig.add_trace(go.Scatter(x=df.index, y=ma, name="MA20", line=dict(color='#ffca28', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="Upper BB", line=dict(color='#80cbc4', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="Lower BB", line=dict(color='#80cbc4', width=1, dash='dash')), row=1, col=1)
    
    if 'vwap' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name="VWAP", line=dict(color='#ce93d8', width=1.5)), row=1, col=1)
    
    if len(df) > 14:
        rsi = FeatureCalculator.calc_rsi(df['close'])
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color='#4fc3f7', width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff5252", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00e676", line_width=1, row=2, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.03)", line_width=0, row=2, col=1)
    
    colors = ['#00e676' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff5252' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color=colors, opacity=0.8), row=3, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a2332',
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color='#ffffff')),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(color='#e0e0e0')
    )
    
    fig.update_xaxes(gridcolor='#2a4a6a', showgrid=True)
    fig.update_yaxes(gridcolor='#2a4a6a', showgrid=True)
    
    return fig


def create_session_heatmap(session_df: pd.DataFrame) -> go.Figure:
    """Create session returns heatmap with brighter colors"""
    cols = ['prev_day_return', 'postmarket_return', 'premarket_return', 'gap_return', 'first_5min_return', 'day_return']
    available_cols = [c for c in cols if c in session_df.columns]
    
    if len(available_cols) == 0:
        return go.Figure()
    
    data = session_df[available_cols].tail(20).T
    
    # Brighter colorscale
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=[str(d)[:10] for d in data.columns],
        y=[c.replace('_', ' ').title() for c in data.index],
        colorscale=[[0, '#ff5252'], [0.5, '#1a2332'], [1, '#00e676']],
        zmid=0,
        text=np.round(data.values, 2),
        texttemplate="%{text}%",
        textfont={"size": 10, "color": "#ffffff"},
        hovertemplate="Date: %{x}<br>Session: %{y}<br>Return: %{z:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a2332',
        height=300,
        title=dict(text="Session Returns Heatmap (Last 20 Days)", font=dict(color='#ffffff')),
        margin=dict(l=150, r=50, t=50, b=50),
        font=dict(color='#e0e0e0')
    )
    
    return fig


def create_predictor_chart(predictor_results: Dict) -> go.Figure:
    """Create predictor analysis chart with brighter colors"""
    predictors = list(predictor_results.keys())
    correlations = [predictor_results[p]['correlation'] for p in predictors]
    accuracies = [predictor_results[p]['contrarian_accuracy'] for p in predictors]
    
    # Color by signal type - brighter
    colors = ['#00e676' if predictor_results[p]['signal_type'] == 'CONTRARIAN' else '#4fc3f7' for p in predictors]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlation with First 5min", "Contrarian Accuracy %"))
    
    fig.add_trace(go.Bar(
        x=[p.replace('_', ' ').title() for p in predictors], 
        y=correlations, 
        marker_color=colors, 
        name="Correlation",
        text=[f"{c:.3f}" for c in correlations],
        textposition='outside',
        textfont=dict(color='#ffffff')
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=[p.replace('_', ' ').title() for p in predictors], 
        y=accuracies, 
        marker_color=colors, 
        name="Accuracy",
        text=[f"{a:.1f}%" for a in accuracies],
        textposition='outside',
        textfont=dict(color='#ffffff')
    ), row=1, col=2)
    
    fig.add_hline(y=50, line_dash="dash", line_color="#ffca28", line_width=2, row=1, col=2)
    fig.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=1, row=1, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a2332',
        height=350,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=80),
        font=dict(color='#e0e0e0')
    )
    fig.update_xaxes(tickangle=45, gridcolor='#2a4a6a')
    fig.update_yaxes(gridcolor='#2a4a6a')
    
    return fig


def create_trades_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Create trades P&L chart with datetime x-axis and bright colors"""
    if len(trades_df) == 0:
        return go.Figure()
    
    trades_df = trades_df.copy()
    trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
    
    # Use entry_time for x-axis
    x_values = trades_df['entry_time'] if 'entry_time' in trades_df.columns else list(range(len(trades_df)))
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Cumulative P&L (%)", "Individual Trade P&L (%)"),
                        row_heights=[0.5, 0.5])
    
    # Cumulative P&L line - bright green
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=trades_df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#00e676', width=2),
        marker=dict(size=6, color='#00e676'),
        hovertemplate='<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>'
    ), row=1, col=1)
    
    # Fill area under cumulative curve
    fig.add_trace(go.Scatter(
        x=x_values,
        y=trades_df['cumulative_pnl'],
        fill='tozeroy',
        fillcolor='rgba(0, 230, 118, 0.2)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=1, row=1, col=1)
    
    # Individual trade bars - bright colors
    colors = ['#00e676' if p > 0 else '#ff5252' for p in trades_df['pnl_pct']]
    
    # Create hover text with trade details
    hover_texts = []
    for _, row in trades_df.iterrows():
        hover_text = (
            f"<b>{row.get('signal', 'N/A').replace('_', ' ').title()}</b><br>"
            f"Direction: {row.get('direction', 'N/A').upper()}<br>"
            f"Entry: ${row.get('entry_price', 0):.2f}<br>"
            f"Exit: ${row.get('exit_price', 0):.2f}<br>"
            f"P&L: {row['pnl_pct']:+.2f}%<br>"
            f"Duration: {row.get('duration_mins', 0):.0f} min"
        )
        hover_texts.append(hover_text)
    
    fig.add_trace(go.Bar(
        x=x_values, 
        y=trades_df['pnl_pct'],
        marker_color=colors,
        name='Trade P&L',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    ), row=2, col=1)
    
    # Add zero line for trade bars
    fig.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=1, row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a2332',
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=80),
        font=dict(color='#e0e0e0'),
        xaxis2=dict(
            tickformat='%m/%d %H:%M',
            tickangle=45
        )
    )
    
    fig.update_xaxes(gridcolor='#2a4a6a', title_text="Entry Time", row=2, col=1)
    fig.update_yaxes(gridcolor='#2a4a6a', title_text="Cumulative %", row=1, col=1)
    fig.update_yaxes(gridcolor='#2a4a6a', title_text="Trade P&L %", row=2, col=1)
    
    return fig


# ============================================================================
# PAGES
# ============================================================================

def page_validation():
    """Out-of-Sample Validation Page"""
    st.markdown("""
    <div class="main-header">
        <h1>Out-of-Sample Validation</h1>
        <p>Validate your strategy before risking real capital</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for holdout
    if 'holdout_revealed' not in st.session_state:
        st.session_state.holdout_revealed = False

    # Sidebar configuration
    st.sidebar.markdown("### Validation Settings")

    # Signal type selection
    signal_type = st.sidebar.selectbox(
        "Signal Type",
        ["RSI Mean Reversion", "Market-Aware RSI", "Session Predictors"],
        help="Select which signal type to validate"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Rolling Walk-Forward**")
    train_window = st.sidebar.slider("Train Window (days)", 10, 90, 30)
    test_window = st.sidebar.slider("Test Window (days)", 5, 30, 10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Holdout Period**")
    holdout_days = st.sidebar.slider("Holdout Days", 30, 180, 60)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Monte Carlo**")
    n_simulations = st.sidebar.slider("Simulations", 100, 5000, 1000, step=100)

    # Demo mode toggle
    use_demo = st.sidebar.checkbox("Demo Mode", value=True, key="validation_demo")

    # Generate or fetch data
    if use_demo:
        minute_df = generate_sample_data(90)
        qqq_df = generate_qqq_sample_data(90)
    else:
        api_key = get_api_key()
        if not api_key:
            st.warning("Please enter your Polygon API key in Settings to use live data.")
            return

        fetcher = PolygonFetcher(api_key)
        minute_df = fetcher.get_minute_data("APP", 90)
        qqq_df = fetcher.get_minute_data("QQQ", 90)

        if minute_df.empty:
            st.error("Failed to fetch data. Please check your API key.")
            return

    # Generate trades based on signal type
    if signal_type == "RSI Mean Reversion":
        backtester = RSIBacktester(oversold=30, overbought=70)
        trades_df = backtester.find_rsi_reversals(minute_df)
    elif signal_type == "Market-Aware RSI":
        backtester = MarketAwareBacktester(oversold=30, overbought=70)
        trades_df = backtester.find_market_aware_trades(minute_df, qqq_df)
    else:  # Session Predictors
        backtester = RSIBacktester(oversold=30, overbought=70)
        trades_df = backtester.find_rsi_reversals(minute_df)

    if len(trades_df) < 10:
        st.warning(f"Not enough trades ({len(trades_df)}) for meaningful validation. Need at least 10 trades.")
        return

    st.info(f"Analyzing **{len(trades_df)} trades** from {signal_type} strategy")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Rolling Walk-Forward",
        "Holdout Test",
        "Monte Carlo",
        "Summary Report"
    ])

    # Initialize validators
    orchestrator = ValidationOrchestrator(
        train_window=train_window,
        test_window=test_window,
        holdout_days=holdout_days,
        n_simulations=n_simulations
    )

    # Run validation
    results = orchestrator.run_full_validation(
        trades_df,
        reveal_holdout=st.session_state.holdout_revealed
    )

    # Tab 1: Rolling Walk-Forward
    with tab1:
        st.markdown('<p class="section-header">Rolling Walk-Forward Analysis</p>', unsafe_allow_html=True)

        rolling = results['rolling']
        if rolling and rolling.get('valid'):
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            consistency = rolling['consistency']
            score_color = "#00e676" if consistency['consistency_score'] >= 60 else "#ffca28" if consistency['consistency_score'] >= 40 else "#ff5252"

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <p class="metric-value" style="color: {score_color};">{consistency['consistency_score']:.0f}</p>
                <p class="metric-label">Consistency Score</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                <p class="metric-value neutral">{len(rolling['periods'])}</p>
                <p class="metric-label">Test Periods</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                sign_color = "#00e676" if consistency['sign_consistency'] >= 60 else "#ff5252"
                st.markdown(f"""
                <div class="metric-card">
                <p class="metric-value" style="color: {sign_color};">{consistency['sign_consistency']:.0f}%</p>
                <p class="metric-label">Positive Periods</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                decay = rolling['decay']
                decay_color = "#00e676" if decay['decay_pct'] < 20 else "#ffca28" if decay['decay_pct'] < 40 else "#ff5252"
                st.markdown(f"""
                <div class="metric-card">
                <p class="metric-value" style="color: {decay_color};">{decay['decay_pct']:.0f}%</p>
                <p class="metric-label">Performance Decay</p>
                </div>
                """, unsafe_allow_html=True)

            # Performance chart
            fig = create_rolling_performance_chart(rolling['periods'])
            st.plotly_chart(fig, use_container_width=True)

            # Period details table
            with st.expander("Period Details", expanded=False):
                period_data = []
                for p in rolling['periods']:
                    period_data.append({
                        'Period': p['period_num'],
                        'Train': f"{p['train_start']} - {p['train_end']}",
                        'Test': f"{p['test_start']} - {p['test_end']}",
                        'Train Trades': p['train_stats']['n_trades'],
                        'Test Trades': p['test_stats']['n_trades'],
                        'Train Sharpe': f"{p['train_stats']['sharpe']:.2f}",
                        'Test Sharpe': f"{p['test_stats']['sharpe']:.2f}",
                        'Test Win Rate': f"{p['test_stats']['win_rate']:.1f}%"
                    })
                st.dataframe(pd.DataFrame(period_data), use_container_width=True)

            # Decay warning
            if decay['is_decaying']:
                st.warning(f"Performance is decaying over time (slope: {decay['slope']:.3f}, p-value: {decay['p_value']:.3f})")
            else:
                st.success("No significant performance decay detected")
        else:
            reason = rolling.get('reason', 'Unknown error') if rolling else 'No data'
            st.warning(f"Rolling validation not available: {reason}")

    # Tab 2: Holdout Test
    with tab2:
        st.markdown('<p class="section-header">Time-Period Holdout Validation</p>', unsafe_allow_html=True)

        holdout = results['holdout']

        if not st.session_state.holdout_revealed:
            # Show locked state
            st.markdown("""
            <div class="warning-box">
            <strong>HOLDOUT PERIOD LOCKED</strong><br>
            The last {} days are reserved for final out-of-sample validation.<br>
            Complete all strategy optimization before revealing these results.
            </div>
            """.format(holdout_days), unsafe_allow_html=True)

            if holdout and holdout.get('valid'):
                st.markdown("**Development Period Performance:**")
                dev_stats = holdout.get('dev_stats', {})

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Dev Trades", dev_stats.get('n_trades', 0))
                col2.metric("Dev Win Rate", f"{dev_stats.get('win_rate', 0):.1f}%")
                col3.metric("Dev Sharpe", f"{dev_stats.get('sharpe', 0):.2f}")
                col4.metric("Holdout Trades", holdout.get('holdout_trade_count', 0))

            st.markdown("---")
            st.markdown("""
            **Before revealing holdout results, ensure:**
            - All parameter optimization is complete
            - You are satisfied with development period performance
            - You understand this is your final validation test
            """)

            if st.button("REVEAL HOLDOUT RESULTS", type="primary"):
                st.session_state.holdout_revealed = True
                st.rerun()

        else:
            # Show revealed results
            if holdout and holdout.get('valid') and holdout.get('holdout_stats'):
                dev_stats = holdout['dev_stats']
                holdout_stats = holdout['holdout_stats']
                degradation = holdout['degradation']

                # Pass/Fail indicator
                if holdout.get('overall_pass'):
                    st.success("HOLDOUT VALIDATION PASSED")
                else:
                    st.error("HOLDOUT VALIDATION FAILED")

                # Comparison chart
                fig = create_holdout_comparison_chart(dev_stats, holdout_stats)
                st.plotly_chart(fig, use_container_width=True)

                # Detailed comparison
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="info-box" style="border-color: #4fc3f7;">
                    <strong>Development Period</strong> ({holdout.get('dev_period', 'N/A')})<br>
                    Trades: {dev_stats['n_trades']} | Win Rate: {dev_stats['win_rate']:.1f}%<br>
                    Total P&L: {dev_stats['total_pnl']:.2f}% | Sharpe: {dev_stats['sharpe']:.2f}<br>
                    Profit Factor: {dev_stats['profit_factor']:.2f}
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    holdout_color = "#00e676" if holdout.get('overall_pass') else "#ff5252"
                    st.markdown(f"""
                    <div class="info-box" style="border-color: {holdout_color};">
                    <strong>Holdout Period</strong> ({holdout.get('holdout_period', 'N/A')})<br>
                    Trades: {holdout_stats['n_trades']} | Win Rate: {holdout_stats['win_rate']:.1f}%<br>
                    Total P&L: {holdout_stats['total_pnl']:.2f}% | Sharpe: {holdout_stats['sharpe']:.2f}<br>
                    Profit Factor: {holdout_stats['profit_factor']:.2f}
                    </div>
                    """, unsafe_allow_html=True)

                # Degradation metrics
                st.markdown("**Performance Change:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Sharpe Change", f"{degradation['sharpe_change_pct']:.0f}%")
                col2.metric("Win Rate Change", f"{degradation['win_rate_change_pct']:.0f}%")
                col3.metric("Holdout/Dev Ratio", f"{degradation['holdout_to_dev_ratio']*100:.0f}%")

                # Reset button
                if st.button("Reset Holdout (Start Over)"):
                    st.session_state.holdout_revealed = False
                    st.rerun()
            else:
                st.warning("Holdout validation failed. Check if there are trades in the holdout period.")

    # Tab 3: Monte Carlo
    with tab3:
        st.markdown('<p class="section-header">Monte Carlo Simulation</p>', unsafe_allow_html=True)

        mc = results['monte_carlo']

        # Bootstrap Analysis
        st.markdown("**Bootstrap Confidence Intervals:**")
        bootstrap = mc['bootstrap']

        if bootstrap and bootstrap.get('valid'):
            col1, col2, col3 = st.columns(3)

            ci_color = "#00e676" if bootstrap['ci_excludes_zero'] else "#ff5252"
            col1.markdown(f"""
            <div class="metric-card">
            <p class="metric-value" style="color: {ci_color};">{bootstrap['pct_positive']:.0f}%</p>
            <p class="metric-label">Positive Simulations</p>
            </div>
            """, unsafe_allow_html=True)

            col2.markdown(f"""
            <div class="metric-card">
            <p class="metric-value neutral">[{bootstrap['ci_total_lower']:.1f}%, {bootstrap['ci_total_upper']:.1f}%]</p>
            <p class="metric-label">95% CI for Total P&L</p>
            </div>
            """, unsafe_allow_html=True)

            col3.markdown(f"""
            <div class="metric-card">
            <p class="metric-value positive">{bootstrap['actual_total_pnl']:.2f}%</p>
            <p class="metric-label">Actual Total P&L</p>
            </div>
            """, unsafe_allow_html=True)

            # Bootstrap histogram
            fig = create_bootstrap_histogram(bootstrap)
            st.plotly_chart(fig, use_container_width=True)

            if bootstrap['ci_excludes_zero']:
                st.success("95% confidence interval excludes zero - strategy has positive expected value")
            else:
                st.warning("95% confidence interval includes zero - results may not be statistically significant")
        else:
            st.warning("Bootstrap analysis not available")

        st.markdown("---")

        # Random Baseline Test
        st.markdown("**Statistical Significance (vs Random Trading):**")
        random_baseline = mc['random_baseline']

        if random_baseline and random_baseline.get('valid'):
            col1, col2, col3 = st.columns(3)

            sig_color = "#00e676" if random_baseline['is_significant'] else "#ff5252"
            col1.markdown(f"""
            <div class="metric-card">
            <p class="metric-value" style="color: {sig_color};">p={random_baseline['p_value']:.3f}</p>
            <p class="metric-label">P-Value</p>
            </div>
            """, unsafe_allow_html=True)

            pct_color = "#00e676" if random_baseline['percentile_rank'] > 95 else "#ffca28" if random_baseline['percentile_rank'] > 50 else "#ff5252"
            col2.markdown(f"""
            <div class="metric-card">
            <p class="metric-value" style="color: {pct_color};">{random_baseline['percentile_rank']:.1f}th</p>
            <p class="metric-label">Percentile vs Random</p>
            </div>
            """, unsafe_allow_html=True)

            col3.markdown(f"""
            <div class="metric-card">
            <p class="metric-value neutral">{random_baseline['random_mean_total']:.2f}%</p>
            <p class="metric-label">Random Mean P&L</p>
            </div>
            """, unsafe_allow_html=True)

            # Significance chart
            fig = create_significance_chart(random_baseline)
            st.plotly_chart(fig, use_container_width=True)

            if random_baseline['is_significant']:
                st.success(f"Strategy is statistically significant (p={random_baseline['p_value']:.3f} < 0.05)")
            else:
                st.warning(f"Strategy is NOT statistically significant (p={random_baseline['p_value']:.3f} >= 0.05)")
        else:
            st.warning("Random baseline test not available")

        st.markdown("---")

        # Sequence Analysis
        st.markdown("**Trade Sequence Analysis:**")
        shuffle = mc['shuffle']

        if shuffle and shuffle.get('valid'):
            col1, col2, col3 = st.columns(3)

            col1.metric("Actual Max Drawdown", f"{shuffle['actual_max_drawdown']:.2f}%")
            col2.metric("Shuffled Median DD", f"{shuffle['shuffled_median_dd']:.2f}%")
            col3.metric("DD Percentile", f"{shuffle['dd_percentile']:.0f}%")

            if shuffle['sequence_helps']:
                st.success("Trade sequence is favorable - actual drawdown is better than most random orderings")
            else:
                st.info("Trade sequence is neutral or unfavorable - drawdown similar to random ordering")

    # Tab 4: Summary Report
    with tab4:
        st.markdown('<p class="section-header">Validation Summary Report</p>', unsafe_allow_html=True)

        summary = results['summary']

        # Overall score
        score = summary['overall_score']
        score_color = "#00e676" if score >= 70 else "#ffca28" if score >= 50 else "#ff5252"

        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: #1a2332; border-radius: 12px; margin-bottom: 2rem;">
        <p style="color: #90a4ae; margin: 0;">Overall Validation Score</p>
        <p style="font-size: 4rem; font-weight: 700; color: {score_color}; margin: 0;">{score:.0f}%</p>
        <p style="color: #90a4ae; margin: 0;">{summary['passed']}/{summary['total']} tests passed</p>
        </div>
        """, unsafe_allow_html=True)

        # Test results table
        st.markdown("**Individual Test Results:**")

        for test in summary['tests']:
            icon = "PASS" if test['pass'] else "FAIL"
            color = "#00e676" if test['pass'] else "#ff5252"
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: #1a2332; border-radius: 8px; margin-bottom: 0.5rem; border-left: 4px solid {color};">
            <span style="color: #e0e0e0;">{test['name']}</span>
            <span style="color: {color}; font-weight: bold;">{test['value']} ({icon})</span>
            </div>
            """, unsafe_allow_html=True)

        # Final recommendation
        st.markdown("---")
        if summary['overall_pass']:
            st.success("""
            **STRATEGY VALIDATED**

            Your strategy has passed the out-of-sample validation tests. Key findings:
            - Performance is consistent across multiple time periods
            - Results are statistically significant vs random trading
            - Strategy appears robust and not overfit

            Consider paper trading before committing real capital.
            """)
        else:
            st.error("""
            **VALIDATION CONCERNS**

            Your strategy has failed some validation tests. Recommendations:
            - Review periods where strategy underperformed
            - Consider simplifying the strategy (fewer parameters)
            - Gather more data for testing
            - Re-examine the economic rationale for your signals

            Do not trade this strategy with real capital until validation improves.
            """)


def page_dashboard():
    """Live Dashboard Page"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Live Dashboard</h1>
        <p>Real-time signals and market monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    api_key = get_api_key()
    symbol = st.session_state.get('symbol', 'APP')
    lookback = st.session_state.get('lookback', 30)
    
    # Auto-detect: use real data if API key exists, unless explicitly set to demo
    if 'use_demo' not in st.session_state:
        st.session_state.use_demo = not bool(api_key)
    use_demo = st.session_state.use_demo
    
    # Data source status bar
    col_src1, col_src2, col_src3 = st.columns([2, 1, 1])
    with col_src1:
        if api_key and not use_demo:
            st.success(f"📡 **LIVE DATA** from Polygon.io | Symbol: {symbol}")
        elif api_key and use_demo:
            st.info(f"📊 **DEMO DATA** (API key available - uncheck Demo Mode for live data)")
        else:
            st.warning(f"📊 **DEMO DATA** | Add Polygon API key in Settings for live data")
    with col_src2:
        demo_toggle = st.checkbox("Demo Mode", value=use_demo, key="dash_demo")
        st.session_state.use_demo = demo_toggle
        use_demo = demo_toggle
    with col_src3:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Fetch data
    minute_df = None
    
    if use_demo or not api_key:
        minute_df = generate_sample_data(lookback)
    else:
        fetcher = PolygonFetcher(api_key)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
        with st.spinner(f"Fetching {symbol} data from Polygon..."):
            minute_df = fetcher.get_aggregates(symbol, 1, "minute", start_date, end_date)
        if len(minute_df) == 0:
            st.error("❌ No data received from Polygon. Check your API key or try Demo Mode.")
            st.info("💡 Tip: Free Polygon accounts have limited access to minute data. Try a smaller lookback period or use Demo Mode.")
            return
    
    daily_df = minute_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                             'close': 'last', 'volume': 'sum', 'vwap': 'last'}).dropna()
    
    signal_data = SignalGenerator.generate_signal(daily_df)
    regime_data = RegimeDetector.detect(daily_df)
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        price = daily_df['close'].iloc[-1] if len(daily_df) > 0 else 0
        change = (daily_df['close'].iloc[-1] / daily_df['close'].iloc[-2] - 1) * 100 if len(daily_df) > 1 else 0
        color = "positive" if change >= 0 else "negative"
        st.markdown(f'<div class="metric-card"><p class="metric-value {color}">${price:.2f}</p><p class="metric-label">Price ({change:+.2f}%)</p></div>', unsafe_allow_html=True)
    
    with col2:
        rsi = signal_data['raw'].get('rsi', 50)
        color = "negative" if rsi > 70 else ("positive" if rsi < 30 else "neutral")
        st.markdown(f'<div class="metric-card"><p class="metric-value {color}">{rsi:.1f}</p><p class="metric-label">RSI</p></div>', unsafe_allow_html=True)
    
    with col3:
        vol = signal_data['raw'].get('rel_volume', 1)
        color = "positive" if vol > 1.5 else "neutral"
        st.markdown(f'<div class="metric-card"><p class="metric-value {color}">{vol:.2f}x</p><p class="metric-label">Rel Volume</p></div>', unsafe_allow_html=True)
    
    with col4:
        vwap = signal_data['raw'].get('vwap_dev', 0)
        color = "positive" if vwap > 0.5 else ("negative" if vwap < -0.5 else "neutral")
        st.markdown(f'<div class="metric-card"><p class="metric-value {color}">{vwap:+.2f}%</p><p class="metric-label">VWAP Dev</p></div>', unsafe_allow_html=True)
    
    with col5:
        conf = signal_data.get('confidence', 0)
        color = "positive" if conf > 60 else "neutral"
        st.markdown(f'<div class="metric-card"><p class="metric-value {color}">{conf:.0f}%</p><p class="metric-label">Confidence</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Signal and Regime
    col_s, col_r = st.columns(2)
    
    with col_s:
        signal = signal_data.get('signal', 'NEUTRAL')
        signal_class = "signal-long" if signal == "LONG" else ("signal-short" if signal == "SHORT" else "signal-neutral")
        icon = "🟢" if signal == "LONG" else ("🔴" if signal == "SHORT" else "🟡")
        st.markdown(f'<div class="signal-box {signal_class}"><p class="signal-text">{icon} {signal}</p></div>', unsafe_allow_html=True)
    
    with col_r:
        st.markdown(f'<div class="signal-box" style="border-color: {regime_data["color"]};"><p class="signal-text" style="color: {regime_data["color"]};">📈 {regime_data["regime"]}</p><p style="color: #8892b0;">{regime_data["description"]}</p></div>', unsafe_allow_html=True)
    
    # Chart
    st.markdown('<p class="section-header">📈 Price Chart</p>', unsafe_allow_html=True)
    recent = minute_df.iloc[-2000:] if len(minute_df) > 2000 else minute_df
    st.plotly_chart(create_price_chart(recent, f"{symbol} - Intraday"), use_container_width=True)


def page_backtest():
    """Backtest Analysis Page"""
    st.markdown("""
    <div class="main-header">
        <h1>📈 Backtest Analysis</h1>
        <p>Session predictors, RSI reversals, and intraday reversion patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add explanation expander at the top
    with st.expander("📚 How to Read This Analysis & Trade Options", expanded=False):
        st.markdown("""
        ### Understanding the Correlation Chart
        
        The **"Correlation with First 5 Min"** chart shows how well each session predicts the first 5 minutes of trading:
        
        | Correlation | Signal Type | Meaning | Trade Action |
        |-------------|-------------|---------|--------------|
        | **Negative** (< 0) | CONTRARIAN | Gap UP → First 5 min DOWN | Gap up? **BUY PUTS** |
        | **Positive** (> 0) | MOMENTUM | Gap UP → First 5 min UP | Gap up? **BUY CALLS** |
        
        **Strength Guide:**
        - |corr| > 0.20: Strong signal ✅
        - |corr| 0.10-0.20: Moderate signal ⚠️
        - |corr| < 0.10: Weak/noise ❌
        
        ---
        
        ### Quick Trading Rules
        
        **Gap Trades (at 9:30 AM):**
        ```
        IF Gap > +2% AND correlation is NEGATIVE:
           → BUY ATM PUT at open
           → Target: 50% gain, Stop: 40% loss
           → Exit by 9:45 AM
        
        IF Gap < -2% AND correlation is NEGATIVE:
           → BUY ATM CALL at open
           → Target: 50% gain, Stop: 40% loss
           → Exit by 9:45 AM
        ```
        
        **RSI Trades (during the day):**
        ```
        IF RSI < 30 AND Volume > 1.5x average:
           → BUY ATM CALL
           → Exit when RSI crosses 50
        
        IF RSI > 70 AND Volume > 1.5x average:
           → BUY ATM PUT
           → Exit when RSI crosses 50
        ```
        """)
    
    api_key = get_api_key()
    symbol = st.session_state.get('symbol', 'APP')
    lookback = st.session_state.get('lookback', 60)
    rsi_oversold = st.session_state.get('rsi_oversold', 30)
    rsi_overbought = st.session_state.get('rsi_overbought', 70)
    
    # Auto-detect: use real data if API key exists
    if 'use_demo' not in st.session_state:
        st.session_state.use_demo = not bool(api_key)
    use_demo = st.session_state.use_demo
    
    # Data source status bar
    col_src1, col_src2, col_src3 = st.columns([2, 1, 1])
    with col_src1:
        if api_key and not use_demo:
            st.success(f"📡 **LIVE DATA** from Polygon.io | {symbol} | {lookback} days")
        elif api_key and use_demo:
            st.info(f"📊 **DEMO DATA** | Uncheck Demo Mode for live {symbol} data")
        else:
            st.warning(f"📊 **DEMO DATA** | Add Polygon API key in Settings for live data")
    with col_src2:
        demo_toggle = st.checkbox("Demo Mode", value=use_demo, key="backtest_demo")
        st.session_state.use_demo = demo_toggle
        use_demo = demo_toggle
    with col_src3:
        if st.button("🔄 Refresh", key="backtest_refresh"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Fetch data
    minute_df = None
    
    if use_demo or not api_key:
        minute_df = generate_sample_data(lookback)
        data_source = "Demo"
    else:
        fetcher = PolygonFetcher(api_key)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
        with st.spinner(f"Fetching {lookback} days of {symbol} minute data from Polygon..."):
            minute_df = fetcher.get_aggregates(symbol, 1, "minute", start_date, end_date)
        if len(minute_df) == 0:
            st.error("❌ No data received from Polygon API")
            st.info("""
            **Possible reasons:**
            - Free Polygon accounts have limited minute-bar access (only 2 years of history, delayed data)
            - Try a **Stocks Starter** subscription for real-time minute data
            - Or use **Demo Mode** to test the system
            
            **Alternative:** Try fetching daily bars instead (change timespan in code)
            """)
            if st.button("Switch to Demo Mode"):
                st.session_state.use_demo = True
                st.rerun()
            return
        data_source = "Polygon"
    
    # Show data info
    if len(minute_df) > 0:
        st.caption(f"📊 Loaded {len(minute_df):,} bars | Range: {minute_df.index.min().strftime('%Y-%m-%d')} to {minute_df.index.max().strftime('%Y-%m-%d')} | Source: {data_source}")
    
    # Session Analysis
    st.markdown('<p class="section-header">📊 Session Predictor Analysis</p>', unsafe_allow_html=True)
    
    session_df = SessionAnalyzer.calculate_session_returns(minute_df)
    predictor_results = SessionAnalyzer.analyze_session_predictors(session_df)
    
    if predictor_results:
        st.plotly_chart(create_predictor_chart(predictor_results), use_container_width=True)
        
        st.markdown("**📊 Predictor Details & Trading Implications:**")
        pred_data = []
        for pred, stats in predictor_results.items():
            # Determine trading implication
            if abs(stats['correlation']) < 0.10:
                trade_impl = "⚪ Too weak to trade"
            elif stats['signal_type'] == 'CONTRARIAN':
                if 'gap' in pred.lower():
                    trade_impl = "🔴 Gap UP → Buy Puts | Gap DOWN → Buy Calls"
                else:
                    trade_impl = "🔴 Fade the move"
            else:
                if 'gap' in pred.lower():
                    trade_impl = "🟢 Gap UP → Buy Calls | Gap DOWN → Buy Puts"
                else:
                    trade_impl = "🟢 Follow the momentum"
            
            pred_data.append({
                'Predictor': pred.replace('_', ' ').title(),
                'Correlation': f"{stats['correlation']:.3f}",
                'Signal': stats['signal_type'],
                'Accuracy': f"{stats['contrarian_accuracy']:.1f}%",
                'P-Value': f"{stats['p_value']:.3f}" + (" ✓" if stats['p_value'] < 0.05 else " ✗"),
                'Trade Implication': trade_impl,
                'Samples': stats['n_samples']
            })
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)
        
        # Add today's trade plan if we have the data
        st.markdown('<p class="section-header">🎯 Today\'s Trade Setup</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            gap_stats = predictor_results.get('gap_return', {})
            if gap_stats:
                signal_type = gap_stats.get('signal_type', 'UNKNOWN')
                corr = gap_stats.get('correlation', 0)
                
                st.markdown(f"""
                <div class="info-box">
                <strong>Gap Trade Strategy</strong><br><br>
                Historical Signal: <strong>{signal_type}</strong> (corr: {corr:.3f})<br><br>
                <strong>If gap > +2% today:</strong><br>
                → {'BUY ATM PUT at 9:30' if signal_type == 'CONTRARIAN' else 'BUY ATM CALL at 9:30'}<br>
                → Target: 50% premium gain<br>
                → Stop: 40% premium loss<br>
                → Exit by 9:45 AM<br><br>
                <strong>If gap < -2% today:</strong><br>
                → {'BUY ATM CALL at 9:30' if signal_type == 'CONTRARIAN' else 'BUY ATM PUT at 9:30'}<br>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box" style="border-color: #ffd93d; background: rgba(255, 217, 61, 0.08);">
            <strong>RSI Trade Strategy</strong><br><br>
            <strong>Oversold Bounce (RSI < 30):</strong><br>
            → BUY ATM CALL<br>
            → Entry: RSI crosses above 30<br>
            → Exit: RSI crosses 50<br><br>
            <strong>Overbought Reversal (RSI > 70):</strong><br>
            → BUY ATM PUT<br>
            → Entry: RSI crosses below 70<br>
            → Exit: RSI crosses 50<br><br>
            <em>Confirm with volume > 1.5x average</em>
            </div>
            """, unsafe_allow_html=True)
    
    # Session Heatmap
    st.markdown('<p class="section-header">🗓️ Session Returns Heatmap</p>', unsafe_allow_html=True)
    st.plotly_chart(create_session_heatmap(session_df), use_container_width=True)

    # =========================================================================
    # OVERNIGHT/PREMARKET → DAY HYPOTHESIS TESTING
    # =========================================================================
    st.markdown('<p class="section-header">🔬 Overnight/Premarket → Day Direction Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
    **Your Hypothesis:** Overnight + premarket movement predicts full day direction.
    If premarket is down, the day tends to close down (and vice versa).
    """)

    # Run the analysis
    overnight_results = OvernightPremktDayAnalyzer.analyze_overnight_premarket_to_day(session_df)

    if overnight_results:
        # Summary metrics
        for name, data in overnight_results.items():
            with st.expander(f"**{name.replace('_', ' ').title()}** - {data['n_samples']} samples", expanded=(name == 'overnight_premarket')):
                col1, col2, col3 = st.columns(3)

                acc_color = "#00e676" if data['direction_accuracy'] > 55 else "#ffca28" if data['direction_accuracy'] > 45 else "#ff5252"
                col1.markdown(f"""
                <div class="metric-card">
                <p class="metric-value" style="color: {acc_color};">{data['direction_accuracy']:.1f}%</p>
                <p class="metric-label">Direction Accuracy</p>
                </div>
                """, unsafe_allow_html=True)

                corr_color = "#00e676" if data['correlation_significant'] else "#90a4ae"
                col2.markdown(f"""
                <div class="metric-card">
                <p class="metric-value" style="color: {corr_color};">{data['correlation']:.3f}</p>
                <p class="metric-label">Correlation (p={data['correlation_p']:.3f})</p>
                </div>
                """, unsafe_allow_html=True)

                col3.markdown(f"""
                <div class="metric-card">
                <p class="metric-value neutral">{data['signal_type']}</p>
                <p class="metric-label">Signal Type</p>
                </div>
                """, unsafe_allow_html=True)

                # Breakdown table
                breakdown_data = []
                if data['down_premarket']['n'] > 0:
                    breakdown_data.append({
                        'Condition': f'Down {name.replace("_", " ")} (>{0.5}%)',
                        'N': data['down_premarket']['n'],
                        'Avg Day Return': f"{data['down_premarket']['avg_day_return']:.2f}%",
                        'Day Ended Down': f"{data['down_premarket']['pct_day_down']:.1f}%",
                        'Day Ended Up': f"{data['down_premarket']['pct_day_up']:.1f}%"
                    })
                if data['up_premarket']['n'] > 0:
                    breakdown_data.append({
                        'Condition': f'Up {name.replace("_", " ")} (>{0.5}%)',
                        'N': data['up_premarket']['n'],
                        'Avg Day Return': f"{data['up_premarket']['avg_day_return']:.2f}%",
                        'Day Ended Up': f"{data['up_premarket']['pct_day_up']:.1f}%",
                        'Day Ended Down': f"{data['up_premarket']['pct_day_down']:.1f}%"
                    })
                if breakdown_data:
                    st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

                # Trading implication
                if data['direction_accuracy'] > 55 and data['correlation_significant']:
                    if data['signal_type'] == 'MOMENTUM':
                        st.success(f"**Trade Signal:** Follow premarket direction. Down premarket → Day likely closes down. Up premarket → Day likely closes up.")
                    else:
                        st.warning(f"**Trade Signal:** Fade premarket direction (contrarian).")
                else:
                    st.info("**Trade Signal:** Not statistically significant - use with caution.")

    # =========================================================================
    # REVERSAL WINDOW ANALYSIS
    # =========================================================================
    st.markdown('<p class="section-header">⏱️ Reversal Window Analysis</p>', unsafe_allow_html=True)

    st.markdown("""
    **Your Hypothesis:** The first 30 minutes after open is the key reversal window.
    If no reversal happens, the move continues in the premarket direction.
    """)

    # Configurable reversal window
    reversal_window = st.select_slider(
        "Select Reversal Window (minutes after open)",
        options=[15, 30, 45, 60, 90, 120],
        value=30,
        key="reversal_window_slider"
    )

    reversal_results = OvernightPremktDayAnalyzer.analyze_reversal_window(
        minute_df, session_df,
        reversal_windows=[15, 30, 45, 60]
    )

    if reversal_results:
        # Show selected window details
        selected_key = f"{reversal_window}min"
        if selected_key not in reversal_results:
            selected_key = list(reversal_results.keys())[0] if reversal_results else None

        if selected_key and selected_key in reversal_results:
            data = reversal_results[selected_key]

            col1, col2, col3 = st.columns(3)
            col1.metric("Sample Days", data['n_samples'])
            col2.metric("Overall Reversal Rate", f"{data['overall_reversal_rate']:.1f}%")
            col3.metric("Day Followed Premarket", f"{data['overall_followed_premarket']:.1f}%")

            # Down premarket analysis
            st.markdown(f"**After DOWN Premarket ({data['down_premarket']['n']} days):**")
            down_cols = st.columns(4)
            down_cols[0].metric("Reversed in Window", f"{data['down_premarket']['reversal_rate']:.1f}%")
            down_cols[1].metric("Avg Reversal", f"{data['down_premarket']['avg_reversal_pct']:.2f}%")
            down_cols[2].metric("Day Ended Down", f"{data['down_premarket']['day_ended_down_pct']:.1f}%")
            down_cols[3].metric("No Rev → Continued", f"{data['down_premarket']['no_reversal_continued']:.1f}%")

            # Up premarket analysis
            st.markdown(f"**After UP Premarket ({data['up_premarket']['n']} days):**")
            up_cols = st.columns(4)
            up_cols[0].metric("Reversed in Window", f"{data['up_premarket']['reversal_rate']:.1f}%")
            up_cols[1].metric("Avg Reversal", f"{data['up_premarket']['avg_reversal_pct']:.2f}%")
            up_cols[2].metric("Day Ended Up", f"{data['up_premarket']['day_ended_up_pct']:.1f}%")
            up_cols[3].metric("No Rev → Continued", f"{data['up_premarket']['no_reversal_continued']:.1f}%")

            # Trading implications
            st.markdown("---")
            st.markdown("**Trading Implications:**")

            down_follow = data['down_premarket']['day_ended_down_pct']
            up_follow = data['up_premarket']['day_ended_up_pct']
            no_rev_cont_down = data['down_premarket']['no_reversal_continued']
            no_rev_cont_up = data['up_premarket']['no_reversal_continued']

            if down_follow > 55 and up_follow > 55:
                st.success(f"""
                **MOMENTUM STRATEGY CONFIRMED:**
                - Down premarket → {down_follow:.0f}% chance day closes down
                - Up premarket → {up_follow:.0f}% chance day closes up
                - Trade WITH the premarket direction
                """)
            elif down_follow < 45 and up_follow < 45:
                st.warning(f"""
                **REVERSAL STRATEGY:**
                - Premarket direction does NOT predict day close
                - Consider fading the gap
                """)
            else:
                st.info("Mixed results - no clear edge")

            if no_rev_cont_down > 60 or no_rev_cont_up > 60:
                st.success(f"""
                **NO REVERSAL → CONTINUATION:**
                - If no reversal by {selected_key}, {max(no_rev_cont_down, no_rev_cont_up):.0f}% continuation
                - Wait for {selected_key} window, if no reversal → enter with trend
                """)

        # Comparison table across all windows
        with st.expander("Compare All Reversal Windows", expanded=False):
            window_comparison = []
            for window, wdata in reversal_results.items():
                window_comparison.append({
                    'Window': window,
                    'Samples': wdata['n_samples'],
                    'Reversal Rate': f"{wdata['overall_reversal_rate']:.1f}%",
                    'Day Followed Premarket': f"{wdata['overall_followed_premarket']:.1f}%",
                    'Down PM → Day Down': f"{wdata['down_premarket']['day_ended_down_pct']:.1f}%",
                    'Up PM → Day Up': f"{wdata['up_premarket']['day_ended_up_pct']:.1f}%",
                    'No Rev → Continue (Down)': f"{wdata['down_premarket']['no_reversal_continued']:.1f}%",
                    'No Rev → Continue (Up)': f"{wdata['up_premarket']['no_reversal_continued']:.1f}%"
                })
            st.dataframe(pd.DataFrame(window_comparison), use_container_width=True, hide_index=True)

    # =========================================================================
    # HYPOTHESIS SUMMARY TABLE
    # =========================================================================
    st.markdown('<p class="section-header">📋 Statistical Hypothesis Summary</p>', unsafe_allow_html=True)

    if overnight_results and reversal_results:
        summary_df = OvernightPremktDayAnalyzer.create_hypothesis_summary(overnight_results, reversal_results)
        if len(summary_df) > 0:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Overall conclusion
            significant_tests = summary_df[summary_df['Significant'] == 'Yes']
            if len(significant_tests) > 0:
                st.success(f"**{len(significant_tests)} statistically significant patterns found.** These can be traded with higher confidence.")
            else:
                st.warning("**No statistically significant patterns found.** Consider using more data or different parameters.")

    # =========================================================================
    # RSI MEAN REVERSION - PRIMARY STRATEGY
    # =========================================================================
    st.markdown('<p class="section-header">📈 RSI Mean Reversion Strategy</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **Strategy:** Buy when RSI crosses above oversold (30), sell/short when RSI crosses below overbought (70).
    Exit when RSI returns to 50, stop loss at RSI extremes, or end of day.
    """)
    
    backtester = RSIBacktester(oversold=rsi_oversold, overbought=rsi_overbought)
    trades_df = backtester.find_rsi_reversals(minute_df)
    
    if len(trades_df) > 0:
        stats = backtester.calculate_stats(trades_df)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", stats['total_trades'])
        col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        col3.metric("Total P&L", f"{stats['total_pnl']:+.1f}%")
        col4.metric("Sharpe", f"{stats['sharpe']:.2f}")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Avg P&L/Trade", f"{stats['avg_pnl']:.3f}%")
        col6.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
        col7.metric("Max Win", f"{stats['max_win']:.2f}%")
        col8.metric("Max Loss", f"{stats['max_loss']:.2f}%")
        
        # Calculate cumulative P&L
        trades_df = trades_df.copy()
        trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
        
        # MAIN CUMULATIVE P&L CHART
        fig_main = go.Figure()
        
        # Cumulative line
        fig_main.add_trace(go.Scatter(
            x=trades_df['exit_time'], 
            y=trades_df['cumulative_pnl'],
            mode='lines',
            name='Cumulative P&L',
            line=dict(color='#00e676', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 230, 118, 0.3)'
        ))
        
        # Zero line
        fig_main.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=1)
        
        # Win/loss markers
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]
        
        fig_main.add_trace(go.Scatter(
            x=wins['exit_time'],
            y=wins['cumulative_pnl'],
            mode='markers',
            name=f'Wins ({len(wins)})',
            marker=dict(color='#00e676', size=6, symbol='circle')
        ))
        
        fig_main.add_trace(go.Scatter(
            x=losses['exit_time'],
            y=losses['cumulative_pnl'],
            mode='markers',
            name=f'Losses ({len(losses)})',
            marker=dict(color='#ff5252', size=6, symbol='circle')
        ))
        
        fig_main.update_layout(
            title=f"📈 Cumulative P&L: {stats['total_pnl']:+.1f}% ({stats['total_trades']} trades, {stats['win_rate']:.0f}% win rate)",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor='#2a4a6a', zerolinecolor='#ffffff'),
            xaxis=dict(gridcolor='#2a4a6a')
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Performance by signal type
        st.markdown("**📊 Performance by Signal:**")
        col_long, col_short = st.columns(2)
        
        with col_long:
            long_trades = trades_df[trades_df['direction'] == 'long']
            if len(long_trades) > 0:
                long_wr = (long_trades['pnl_pct'] > 0).mean() * 100
                long_pnl = long_trades['pnl_pct'].sum()
                color = "#00e676" if long_pnl > 0 else "#ff5252"
                st.markdown(f"""
                <div class="metric-card" style="border-color: #00e676;">
                <h4 style="color: #00e676; margin-top: 0;">📈 LONG (Oversold Bounce)</h4>
                <p style="color: #e0e0e0;">Trades: {len(long_trades)} | Win Rate: {long_wr:.0f}%</p>
                <p style="color: {color}; font-size: 1.3rem; font-weight: bold;">P&L: {long_pnl:+.2f}%</p>
                <p style="color: #90a4ae;">→ BUY CALLs when RSI &lt; {rsi_oversold}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_short:
            short_trades = trades_df[trades_df['direction'] == 'short']
            if len(short_trades) > 0:
                short_wr = (short_trades['pnl_pct'] > 0).mean() * 100
                short_pnl = short_trades['pnl_pct'].sum()
                color = "#00e676" if short_pnl > 0 else "#ff5252"
                st.markdown(f"""
                <div class="metric-card" style="border-color: #ff5252;">
                <h4 style="color: #ff5252; margin-top: 0;">📉 SHORT (Overbought Reversal)</h4>
                <p style="color: #e0e0e0;">Trades: {len(short_trades)} | Win Rate: {short_wr:.0f}%</p>
                <p style="color: {color}; font-size: 1.3rem; font-weight: bold;">P&L: {short_pnl:+.2f}%</p>
                <p style="color: #90a4ae;">→ BUY PUTs when RSI &gt; {rsi_overbought}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Trade log
        st.markdown("**📋 Trade Log:**")
        
        display_df = trades_df.copy()
        display_df['Entry'] = display_df['entry_time'].dt.strftime('%m/%d %H:%M')
        display_df['Exit'] = display_df['exit_time'].dt.strftime('%H:%M')
        display_df['Dir'] = display_df['direction'].str.upper().str[:1]
        
        # Options trade column
        display_df['Option'] = display_df.apply(
            lambda r: f"${round(r['entry_price']/5)*5:.0f} {'CALL' if r['direction']=='long' else 'PUT'}", 
            axis=1
        )
        display_df['Entry $'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
        display_df['Exit $'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
        display_df['RSI'] = display_df['entry_rsi'].apply(lambda x: f"{x:.0f}")
        display_df['P&L'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
        display_df['Cumul'] = display_df['cumulative_pnl'].apply(lambda x: f"{x:+.1f}%")
        display_df['Reason'] = display_df['exit_reason'].str.replace('_', ' ').str.title()
        
        st.dataframe(
            display_df[['Entry', 'Exit', 'Dir', 'Option', 'Entry $', 'Exit $', 'RSI', 'P&L', 'Cumul', 'Reason']],
            use_container_width=True,
            height=400
        )
        
        # Download
        csv_data = display_df[['Entry', 'Exit', 'Dir', 'Option', 'Entry $', 'Exit $', 'RSI', 'P&L', 'Cumul', 'Reason']].to_csv(index=False)
        st.download_button(
            label="📥 Download Trades CSV",
            data=csv_data,
            file_name=f"app_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No trades found. Try adjusting RSI parameters or using more data.")
    
    # =========================================================================
    # ROBUST OPTIONS STRATEGY
    # =========================================================================
    st.markdown('<p class="section-header">📈 Options Strategy (Leverage + VIX Filter)</p>', unsafe_allow_html=True)
    
    with st.expander("📚 How the Options Strategy Works", expanded=False):
        st.markdown("""
        ### Options Provide Leverage on Stock Moves
        
        Instead of trading stock directly, we buy calls/puts to amplify returns:
        
        | Strike Type | Leverage | Position Size | Best For |
        |-------------|----------|---------------|----------|
        | **ATM** | 3x | 100% | Normal signals |
        | **OTM** | 5x | 60% | High confidence |
        | **Deep OTM** | 8x | 35% | Very high confidence |
        
        ### Key Risk Management
        1. **Position size inversely proportional to leverage** - Higher leverage = smaller position
        2. **VIX-adjusted leverage** - High VIX = expensive options = lower effective returns
        3. **Asymmetric loss model** - Options lose value faster than they gain
        4. **VIX filter** - Skip longs when VIX > 25 and rising (panic mode)
        """)
    
    # Generate VIX data
    vix_df = generate_vix_sample_data(lookback)
    
    # Run options backtest
    opt_backtester = OptionsBacktester(oversold=rsi_oversold, overbought=rsi_overbought)
    opt_trades_df = opt_backtester.find_options_trades(minute_df, vix_df)
    
    if len(opt_trades_df) > 0:
        opt_stats = opt_backtester.calculate_stats(opt_trades_df)
        
        # Comparison metrics
        st.markdown("**💰 Stock vs Options Comparison:**")
        col_stock, col_options = st.columns(2)
        
        with col_stock:
            stock_pnl = opt_stats['stock_total_pnl']
            st.markdown(f"""
            <div class="metric-card" style="border-color: #90a4ae;">
            <h4 style="color: #90a4ae; margin-top: 0;">📊 Stock Only</h4>
            <p style="color: #e0e0e0; font-size: 1.5rem; margin: 0.5rem 0;">{stock_pnl:+.1f}%</p>
            <p style="color: #90a4ae;">No leverage, direct exposure</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_options:
            opt_pnl = opt_stats['total_pnl']
            boost = opt_stats['options_vs_stock']
            boost_color = "#00e676" if boost > 0 else "#ff5252"
            st.markdown(f"""
            <div class="metric-card" style="border-color: #00e676;">
            <h4 style="color: #00e676; margin-top: 0;">📈 With Options</h4>
            <p style="color: #e0e0e0; font-size: 1.5rem; margin: 0.5rem 0;">{opt_pnl:+.1f}%</p>
            <p style="color: {boost_color};">{boost:+.1f}% leverage boost</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Trades", opt_stats['total_trades'])
        col2.metric("Win Rate", f"{opt_stats['win_rate']:.1f}%")
        col3.metric("Sharpe", f"{opt_stats['sharpe']:.2f}")
        col4.metric("Profit Factor", f"{opt_stats['profit_factor']:.2f}")
        
        # Cumulative P&L chart
        opt_trades_df = opt_trades_df.copy()
        opt_trades_df['cumulative_pnl'] = opt_trades_df['adj_option_pnl_pct'].cumsum()
        opt_trades_df['stock_cumulative'] = opt_trades_df['stock_pnl_pct'].cumsum()
        
        fig_opt = go.Figure()
        
        # Options P&L line
        fig_opt.add_trace(go.Scatter(
            x=opt_trades_df['exit_time'], 
            y=opt_trades_df['cumulative_pnl'],
            mode='lines',
            name='Options P&L',
            line=dict(color='#00e676', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 230, 118, 0.3)'
        ))
        
        # Stock P&L line (comparison)
        fig_opt.add_trace(go.Scatter(
            x=opt_trades_df['exit_time'], 
            y=opt_trades_df['stock_cumulative'],
            mode='lines',
            name='Stock P&L',
            line=dict(color='#90a4ae', width=2, dash='dash')
        ))
        
        fig_opt.add_hline(y=0, line_dash="solid", line_color="#ffffff", line_width=1)
        
        fig_opt.update_layout(
            title=f"Options P&L: {opt_stats['total_pnl']:+.1f}% (vs Stock: {opt_stats['stock_total_pnl']:+.1f}%)",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor='#2a4a6a', zerolinecolor='#ffffff'),
            xaxis=dict(gridcolor='#2a4a6a')
        )
        
        st.plotly_chart(fig_opt, use_container_width=True)
        
        # Performance by strike type
        st.markdown("**📊 Performance by Strike Type:**")
        strike_cols = st.columns(3)
        
        for i, st_type in enumerate(['ATM', 'OTM', 'DEEP_OTM']):
            with strike_cols[i]:
                key = f'{st_type}_trades'
                if key in opt_stats and opt_stats[key] > 0:
                    pnl = opt_stats[f'{st_type}_total_pnl']
                    wr = opt_stats[f'{st_type}_win_rate']
                    color = "#00e676" if pnl > 0 else "#ff5252"
                    st.markdown(f"""
                    <div class="metric-card">
                    <p style="color: #4fc3f7; font-weight: bold; margin: 0;">{st_type}</p>
                    <p style="color: #e0e0e0; margin: 0.3rem 0;">Trades: {opt_stats[key]}</p>
                    <p style="color: #e0e0e0; margin: 0.3rem 0;">Win Rate: {wr:.0f}%</p>
                    <p style="color: {color}; font-weight: bold; margin: 0;">P&L: {pnl:+.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #4a4a6a;">
                    <p style="color: #90a4ae; margin: 0;">{st_type}</p>
                    <p style="color: #90a4ae; margin: 0;">No trades</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Trade log (collapsed)
        with st.expander("📋 Options Trade Log", expanded=False):
            opt_display = opt_trades_df.copy()
            opt_display['Entry'] = opt_display['entry_time'].dt.strftime('%m/%d %H:%M')
            opt_display['Option'] = opt_display.apply(
                lambda r: f"${r['strike']:.0f} {r['option_type']}", axis=1
            )
            opt_display['Type'] = opt_display['strike_type']
            opt_display['Conf'] = opt_display['confidence']
            opt_display['VIX'] = opt_display['vix'].apply(lambda x: f"{x:.1f}")
            opt_display['Stock %'] = opt_display['stock_pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            opt_display['Option %'] = opt_display['adj_option_pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            opt_display['Cumul'] = opt_display['cumulative_pnl'].apply(lambda x: f"{x:+.1f}%")
            
            st.dataframe(
                opt_display[['Entry', 'Option', 'Type', 'Conf', 'VIX', 'Stock %', 'Option %', 'Cumul']],
                use_container_width=True,
                height=300
            )
    else:
        st.info("No options trades found. Try adjusting parameters.")
    
    # =========================================================================
    # THETADATA REAL OPTIONS (if enabled)
    # =========================================================================
    if st.session_state.get('theta_enabled', False):
        st.markdown('<p class="section-header">🎯 Real Options Data (ThetaData)</p>', unsafe_allow_html=True)
        
        theta_url = st.session_state.get('theta_url', 'http://127.0.0.1:25510')

        st.markdown(f"""
        <div class="info-box">
        <strong>📡 ThetaData Connection:</strong> {theta_url}<br>
        Using real historical options prices and Greeks instead of simulated leverage.
        </div>
        """, unsafe_allow_html=True)

        # Try to connect and fetch real data
        try:
            import requests
            from io import StringIO

            # Test connection with v2 API
            response = requests.get(f"{theta_url}/v2/snapshot/stock/quote",
                                   params={"root": symbol}, timeout=5)

            if response.status_code == 200:
                st.success(f"✅ Connected to Theta Terminal - Fetching {symbol} options...")

                # Get option chain with v2 API
                chain_response = requests.get(
                    f"{theta_url}/v2/snapshot/option/quote",
                    params={"root": symbol},
                    timeout=30
                )
                
                if chain_response.status_code == 200 and chain_response.text:
                    chain_df = pd.read_csv(StringIO(chain_response.text))
                    
                    if not chain_df.empty:
                        st.markdown(f"**📊 {symbol} Option Chain:**")
                        
                        # Get unique expirations
                        if 'expiration' in chain_df.columns:
                            expirations = sorted(chain_df['expiration'].unique())
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Available Expirations", len(expirations))
                            with col2:
                                if 'strike' in chain_df.columns:
                                    st.metric("Total Contracts", len(chain_df))
                            
                            # Show next 5 expirations
                            st.markdown(f"**Next expirations:** {', '.join([str(e) for e in expirations[:5]])}")
                            
                            # Sample option data
                            with st.expander("📋 Sample Option Chain Data", expanded=False):
                                st.dataframe(chain_df.head(20), use_container_width=True)
                        
                        # Get Greeks for ATM option
                        if 'strike' in chain_df.columns and len(expirations) > 0:
                            nearest_exp = expirations[0]
                            
                            # Find ATM strike (closest to current price)
                            current_price = minute_df['close'].iloc[-1] if len(minute_df) > 0 else 670
                            exp_options = chain_df[chain_df['expiration'] == nearest_exp]
                            
                            if 'strike' in exp_options.columns and len(exp_options) > 0:
                                strikes = exp_options['strike'].unique()
                                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                                
                                # Fetch Greeks
                                greeks_response = requests.get(
                                    f"{theta_url}/v2/snapshot/option/greeks",
                                    params={
                                        "symbol": symbol,
                                        "expiration": str(nearest_exp),
                                        "strike": str(atm_strike),
                                        "right": "call"
                                    },
                                    timeout=10
                                )
                                
                                if greeks_response.status_code == 200 and greeks_response.text:
                                    greeks_df = pd.read_csv(StringIO(greeks_response.text))
                                    
                                    if not greeks_df.empty:
                                        st.markdown(f"**📐 ATM Call Greeks (${atm_strike:.0f}, Exp: {nearest_exp}):**")
                                        
                                        g_cols = st.columns(5)
                                        row = greeks_df.iloc[0]
                                        
                                        with g_cols[0]:
                                            delta = row.get('delta', 0)
                                            st.metric("Delta", f"{delta:.3f}")
                                        with g_cols[1]:
                                            gamma = row.get('gamma', 0)
                                            st.metric("Gamma", f"{gamma:.4f}")
                                        with g_cols[2]:
                                            theta = row.get('theta', 0)
                                            st.metric("Theta", f"{theta:.2f}")
                                        with g_cols[3]:
                                            vega = row.get('vega', 0)
                                            st.metric("Vega", f"{vega:.2f}")
                                        with g_cols[4]:
                                            iv = row.get('implied_volatility', 0)
                                            st.metric("IV", f"{iv:.1%}" if iv else "N/A")
                    else:
                        st.warning("No option chain data returned")
            else:
                st.error(f"❌ Cannot connect to Theta Terminal (Status: {response.status_code})")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to Theta Terminal. Make sure it's running!")
        except Exception as e:
            st.error(f"❌ Error fetching ThetaData: {str(e)}")
    
    # Market-Aware Strategy Section
    st.markdown('<p class="section-header">🎯 Market-Aware Strategy (QQQ Filter)</p>', unsafe_allow_html=True)
    
    with st.expander("📚 How the Market Filter Works", expanded=False):
        st.markdown("""
        ### Why Add a Market Filter?
        
        Mean reversion strategies suffer in **trending markets**. By filtering trades based on QQQ's regime, we can:
        - **Avoid longs** when QQQ is in a strong downtrend (market dragging everything down)
        - **Avoid shorts** when QQQ is in a strong uptrend (momentum overrides mean reversion)
        - **Reduce size** in high volatility (wider swings = more risk)
        
        ### Filter Rules (Simple to Avoid Overfitting)
        
        | QQQ Regime | 5-Day Return | Long Trades | Short Trades | Position Size |
        |------------|--------------|-------------|--------------|---------------|
        | **STRONG_UP** | > +2% | ✅ Allowed | ❌ Skip | 100% |
        | **STRONG_DOWN** | < -2% | ❌ Skip | ✅ Allowed | 100% |
        | **HIGH_VOL** | Any, vol > 2% | ✅ Allowed | ✅ Allowed | 50% |
        | **NORMAL** | -2% to +2% | ✅ Allowed | ✅ Allowed | 100% |
        
        ### Anti-Overfitting Measures
        - Only 3 filter parameters (trend threshold, vol threshold, position adjustment)
        - Walk-forward validation to check out-of-sample performance
        - Rules based on economic intuition, not curve-fitting
        """)
    
    # Fetch or generate QQQ data
    qqq_df = None
    
    if not use_demo and api_key:
        try:
            fetcher = PolygonFetcher(api_key)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
            with st.spinner("Fetching QQQ data..."):
                qqq_df = fetcher.get_aggregates("QQQ", 1, "minute", start_date, end_date)
        except:
            pass
    
    if qqq_df is None or len(qqq_df) == 0:
        qqq_df = generate_qqq_sample_data(lookback)
        st.caption("📊 Using demo QQQ data")
    else:
        st.caption("📡 Using live QQQ data from Polygon")
    
    # Run market-aware backtest
    ma_backtester = MarketAwareBacktester(oversold=rsi_oversold, overbought=rsi_overbought)
    ma_trades_df = ma_backtester.find_market_aware_trades(minute_df, qqq_df)
    
    if len(ma_trades_df) > 0:
        ma_stats = ma_backtester.calculate_stats(ma_trades_df, use_adjusted=True)
        
        # Comparison with basic strategy
        st.markdown("**📊 Strategy Comparison:**")
        
        col_basic, col_market = st.columns(2)
        
        with col_basic:
            basic_pnl = trades_df['pnl_pct'].sum() if len(trades_df) > 0 else 0
            basic_wr = (trades_df['pnl_pct'] > 0).mean() * 100 if len(trades_df) > 0 else 0
            basic_sharpe = trades_df['pnl_pct'].mean() / trades_df['pnl_pct'].std() * np.sqrt(252) if len(trades_df) > 0 and trades_df['pnl_pct'].std() > 0 else 0
            
            st.markdown(f"""
            <div class="metric-card" style="border-color: #4a4a6a;">
            <h4 style="color: #90a4ae; margin-top: 0;">📉 Basic RSI Strategy</h4>
            <table style="width: 100%; color: #e0e0e0;">
            <tr><td>Trades:</td><td><strong>{len(trades_df)}</strong></td></tr>
            <tr><td>Win Rate:</td><td><strong>{basic_wr:.1f}%</strong></td></tr>
            <tr><td>Total P&L:</td><td><strong>{basic_pnl:.2f}%</strong></td></tr>
            <tr><td>Sharpe:</td><td><strong>{basic_sharpe:.2f}</strong></td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col_market:
            improvement = ma_stats['total_pnl'] - basic_pnl
            imp_color = "#00e676" if improvement > 0 else "#ff5252"
            
            st.markdown(f"""
            <div class="metric-card" style="border-color: #00e676;">
            <h4 style="color: #00e676; margin-top: 0;">🎯 Market-Aware Strategy</h4>
            <table style="width: 100%; color: #e0e0e0;">
            <tr><td>Trades:</td><td><strong>{ma_stats['total_trades']}</strong></td></tr>
            <tr><td>Win Rate:</td><td><strong>{ma_stats['win_rate']:.1f}%</strong></td></tr>
            <tr><td>Total P&L:</td><td><strong>{ma_stats['total_pnl']:.2f}%</strong></td></tr>
            <tr><td>Sharpe:</td><td><strong>{ma_stats['sharpe']:.2f}</strong></td></tr>
            </table>
            <p style="color: {imp_color}; margin-top: 0.5rem; font-weight: bold;">
            {'+' if improvement > 0 else ''}{improvement:.2f}% vs Basic
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance by regime
        st.markdown("**📊 Performance by Market Regime:**")
        
        regime_cols = st.columns(4)
        regime_names = ['NORMAL', 'STRONG_UP', 'STRONG_DOWN', 'HIGH_VOL']
        regime_colors = {'NORMAL': '#4fc3f7', 'STRONG_UP': '#00e676', 'STRONG_DOWN': '#ff5252', 'HIGH_VOL': '#ffca28'}
        
        for i, regime in enumerate(regime_names):
            with regime_cols[i]:
                trades_key = f'{regime}_trades'
                wr_key = f'{regime}_win_rate'
                pnl_key = f'{regime}_total_pnl'
                
                if trades_key in ma_stats:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {regime_colors[regime]};">
                    <p style="color: {regime_colors[regime]}; font-weight: bold; margin: 0;">{regime}</p>
                    <p style="color: #e0e0e0; margin: 0.3rem 0;">Trades: {ma_stats[trades_key]}</p>
                    <p style="color: #e0e0e0; margin: 0.3rem 0;">Win: {ma_stats[wr_key]:.1f}%</p>
                    <p style="color: #e0e0e0; margin: 0;">P&L: {ma_stats[pnl_key]:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #4a4a6a;">
                    <p style="color: #90a4ae; font-weight: bold; margin: 0;">{regime}</p>
                    <p style="color: #90a4ae; margin: 0.3rem 0;">No trades</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Walk-forward validation
        st.markdown("**🔬 Walk-Forward Validation (Overfit Check):**")
        
        validation = MarketAwareBacktester.walk_forward_validation(ma_trades_df)
        
        if validation.get('valid'):
            col_train, col_test = st.columns(2)
            
            with col_train:
                train = validation['train']
                st.markdown(f"""
                <div class="info-box" style="border-color: #4fc3f7;">
                <strong>Training Period</strong> ({validation['train_period']})<br>
                Trades: {train['n']} | Win Rate: {train['win_rate']:.1f}%<br>
                Total P&L: {train['total_pnl']:.2f}% | Sharpe: {train['sharpe']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            with col_test:
                test = validation['test']
                test_color = "#00e676" if not validation['overfit_warning'] else "#ffca28"
                st.markdown(f"""
                <div class="info-box" style="border-color: {test_color};">
                <strong>Test Period</strong> ({validation['test_period']})<br>
                Trades: {test['n']} | Win Rate: {test['win_rate']:.1f}%<br>
                Total P&L: {test['total_pnl']:.2f}% | Sharpe: {test['sharpe']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            if validation['overfit_warning']:
                st.warning(f"⚠️ **Potential Overfitting Detected**: Test performance decayed {validation['performance_decay']:.0f}% vs training. Consider simplifying the strategy or using more data.")
            else:
                st.success(f"✅ **Strategy appears robust**: Test performance within acceptable range of training ({validation['performance_decay']:.0f}% decay)")
        
        # Show market-aware trade log
        with st.expander("📋 Market-Aware Trade Log", expanded=False):
            ma_display = ma_trades_df.copy()
            ma_display['cumulative_pnl'] = ma_display['adj_pnl_pct'].cumsum()
            ma_display['Entry Time'] = ma_display['entry_time'].dt.strftime('%m/%d %H:%M')
            ma_display['Signal'] = ma_display['signal'].str.replace('_', ' ').str.title()
            ma_display['Direction'] = ma_display['direction'].str.upper()
            ma_display['Regime'] = ma_display['regime']
            ma_display['QQQ RSI'] = ma_display['qqq_rsi'].apply(lambda x: f"{x:.1f}")
            ma_display['Size'] = ma_display['position_size'].apply(lambda x: f"{x*100:.0f}%")
            ma_display['Raw P&L'] = ma_display['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            ma_display['Adj P&L'] = ma_display['adj_pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            ma_display['Cumulative'] = ma_display['cumulative_pnl'].apply(lambda x: f"{x:+.2f}%")
            
            st.dataframe(
                ma_display[['Entry Time', 'Signal', 'Direction', 'Regime', 'QQQ RSI', 'Size', 'Raw P&L', 'Adj P&L', 'Cumulative']],
                use_container_width=True,
                height=400
            )
    else:
        st.info("No market-aware trades found. Try adjusting parameters or using more data.")
    
    st.markdown('<p class="section-header">🔄 Intraday Reversion Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **What this shows:** After the first 5 minutes move, how often and how much does price revert?
    This helps time your exit for gap trades.
    """)
    
    reversion_df = IntradayReversionAnalyzer.analyze_reversion_from_open(minute_df, session_df)
    reversion_stats = IntradayReversionAnalyzer.calculate_reversion_probabilities(reversion_df)
    
    for direction, stats in reversion_stats.items():
        icon = "📈" if direction == "up" else "📉"
        st.markdown(f"**{icon} After {direction.upper()} first 5 minutes ({stats['n_samples']} samples):**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg First 5min", f"{stats['avg_first_5min_move']:.2f}%")
        col2.metric("Avg Reversion", f"{stats['avg_reversion']:.2f}%")
        col3.metric("Reverted to Open", f"{stats['reverted_to_open_rate']:.1f}%")
        
        # Options implication
        if direction == "up":
            st.markdown(f"""
            <div class="info-box" style="border-color: #ff6b6b; background: rgba(255, 107, 107, 0.08);">
            <strong>Options Implication:</strong> After UP first 5 min, price reverts {stats['reverted_to_open_rate']:.0f}% of time.<br>
            → If you bought PUTS at open on a gap up, expect avg {abs(stats['avg_reversion']):.1f}% pullback<br>
            → Exit when price returns to open or hits your target
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
            <strong>Options Implication:</strong> After DOWN first 5 min, price bounces {stats['reverted_to_open_rate']:.0f}% of time.<br>
            → If you bought CALLS at open on a gap down, expect avg {abs(stats['avg_reversion']):.1f}% bounce<br>
            → Exit when price returns to open or hits your target
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Reversion Target Hit Rates:**")
        hit_data = [{"Level": k, "Hit Rate": f"{v:.1f}%"} for k, v in stats['reversion_hit_rates'].items()]
        st.dataframe(pd.DataFrame(hit_data).T, use_container_width=True, hide_index=True)


def page_trade_plan():
    """Today's Trade Plan Page"""
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Today's Trade Plan</h1>
        <p>Real-time signal analysis and options trade recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date display
    st.markdown(f"**📅 Date: {datetime.now().strftime('%B %d, %Y')}**")
    
    api_key = get_api_key()
    symbol = st.session_state.get('symbol', 'APP')
    
    # Try to auto-fetch live data
    live_data = None
    market_data = {'spy_change': 0.0, 'qqq_change': 0.0, 'vix': 15.0}
    
    if api_key:
        with st.spinner(f"Fetching latest {symbol}, SPY, QQQ data..."):
            try:
                fetcher = PolygonFetcher(api_key)
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                
                # Get APP data
                minute_df = fetcher.get_aggregates(symbol, 1, "minute", start_date, end_date)
                
                if len(minute_df) > 0:
                    # Get daily data for yesterday's OHLC
                    daily_df = minute_df.resample('D').agg({
                        'open': 'first', 'high': 'max', 'low': 'min',
                        'close': 'last', 'volume': 'sum'
                    }).dropna()
                    
                    if len(daily_df) >= 2:
                        yesterday = daily_df.iloc[-2]
                        today_data = minute_df[minute_df.index.date == datetime.now().date()]
                        
                        # Calculate RSI
                        rsi = FeatureCalculator.calc_rsi(minute_df['close']).iloc[-1]
                        
                        # Calculate relative volume
                        avg_vol = minute_df['volume'].rolling(100).mean().iloc[-1]
                        current_vol = today_data['volume'].sum() if len(today_data) > 0 else minute_df['volume'].iloc[-1]
                        rel_vol = current_vol / avg_vol if avg_vol > 0 else 1.0
                        
                        live_data = {
                            'prev_close': yesterday['close'],
                            'prev_high': yesterday['high'],
                            'prev_low': yesterday['low'],
                            'current_price': minute_df['close'].iloc[-1],
                            'current_rsi': rsi,
                            'relative_volume': min(rel_vol, 5.0),  # Cap at 5x
                            'last_updated': minute_df.index[-1]
                        }
                
                # Fetch SPY data
                spy_df = fetcher.get_aggregates("SPY", 1, "minute", start_date, end_date)
                if len(spy_df) > 0:
                    spy_daily = spy_df.resample('D').agg({'open': 'first', 'close': 'last'}).dropna()
                    if len(spy_daily) >= 2:
                        spy_prev_close = spy_daily.iloc[-2]['close']
                        spy_current = spy_df['close'].iloc[-1]
                        market_data['spy_change'] = (spy_current / spy_prev_close - 1) * 100
                
                # Fetch QQQ data
                qqq_df = fetcher.get_aggregates("QQQ", 1, "minute", start_date, end_date)
                if len(qqq_df) > 0:
                    qqq_daily = qqq_df.resample('D').agg({'open': 'first', 'close': 'last'}).dropna()
                    if len(qqq_daily) >= 2:
                        qqq_prev_close = qqq_daily.iloc[-2]['close']
                        qqq_current = qqq_df['close'].iloc[-1]
                        market_data['qqq_change'] = (qqq_current / qqq_prev_close - 1) * 100
                
                if live_data:
                    st.success(f"📡 **LIVE DATA** loaded | {symbol}: ${live_data['current_price']:.2f} | SPY: {market_data['spy_change']:+.2f}% | QQQ: {market_data['qqq_change']:+.2f}%")
                    
            except Exception as e:
                st.warning(f"Could not fetch live data: {str(e)}")
    
    if not live_data:
        st.info("📊 **MANUAL ENTRY** | Add Polygon API key in Settings for auto-fetch")
    
    # Input section
    st.markdown('<p class="section-header">📊 Market Data</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prev_close = st.number_input("Yesterday's Close ($)", 
                                     value=live_data['prev_close'] if live_data else 672.00, 
                                     step=0.01, format="%.2f")
        prev_high = st.number_input("Yesterday's High ($)", 
                                    value=live_data['prev_high'] if live_data else 680.00, 
                                    step=0.01, format="%.2f")
    
    with col2:
        current_price = st.number_input("Current Price ($)", 
                                        value=live_data['current_price'] if live_data else 665.00, 
                                        step=0.01, format="%.2f")
        prev_low = st.number_input("Yesterday's Low ($)", 
                                   value=live_data['prev_low'] if live_data else 660.00, 
                                   step=0.01, format="%.2f")
    
    with col3:
        current_rsi = st.number_input("Current RSI", 
                                      value=live_data['current_rsi'] if live_data else 35.0, 
                                      min_value=0.0, max_value=100.0, step=0.1)
        relative_volume = st.number_input("Relative Volume", 
                                          value=live_data['relative_volume'] if live_data else 1.2, 
                                          min_value=0.0, step=0.1)
    
    # Calculate signals
    gap_pct = (current_price / prev_close - 1) * 100
    
    # Market context input
    st.markdown('<p class="section-header">📈 Market Context</p>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        spy_change = st.number_input("SPY Change (%)", 
                                     value=market_data['spy_change'], 
                                     step=0.1, format="%.2f")
    with col_m2:
        qqq_change = st.number_input("QQQ/Nasdaq Change (%)", 
                                     value=market_data['qqq_change'], 
                                     step=0.1, format="%.2f")
    with col_m3:
        vix_level = st.number_input("VIX Level", value=market_data['vix'], step=0.1)
    
    # Generate Signal Button
    if st.button("🎯 Generate Trade Plan", type="primary", use_container_width=True):
        
        st.markdown("---")
        
        # Signal Analysis
        st.markdown('<p class="section-header">🎯 Signal Analysis</p>', unsafe_allow_html=True)
        
        # Determine signal
        if gap_pct < -3.0:
            signal = "STRONG BUY"
            signal_color = "#00e676"
            confidence = "HIGH"
            signal_class = "signal-long"
        elif gap_pct < -2.0:
            signal = "BUY"
            signal_color = "#00e676"
            confidence = "MODERATE"
            signal_class = "signal-long"
        elif gap_pct < -1.0:
            signal = "LEAN BULLISH"
            signal_color = "#ffca28"
            confidence = "LOW"
            signal_class = "signal-neutral"
        elif gap_pct > 3.0:
            signal = "STRONG SELL"
            signal_color = "#ff5252"
            confidence = "HIGH"
            signal_class = "signal-short"
        elif gap_pct > 2.0:
            signal = "SELL"
            signal_color = "#ff5252"
            confidence = "MODERATE"
            signal_class = "signal-short"
        elif gap_pct > 1.0:
            signal = "LEAN BEARISH"
            signal_color = "#ffca28"
            confidence = "LOW"
            signal_class = "signal-neutral"
        else:
            signal = "NEUTRAL"
            signal_color = "#ffca28"
            confidence = "N/A"
            signal_class = "signal-neutral"
        
        # RSI confirmation
        rsi_signal = ""
        if current_rsi < 30:
            rsi_signal = "OVERSOLD ✅"
            if "BUY" not in signal and signal != "NEUTRAL":
                confidence = "CONFLICTING"
        elif current_rsi > 70:
            rsi_signal = "OVERBOUGHT ✅"
            if "SELL" not in signal and signal != "NEUTRAL":
                confidence = "CONFLICTING"
        else:
            rsi_signal = "NEUTRAL"
        
        # Display signal
        col_sig1, col_sig2 = st.columns(2)
        
        with col_sig1:
            st.markdown(f"""
            <div class="signal-box {signal_class}">
                <p class="signal-text" style="color: {signal_color};">{signal}</p>
                <p style="color: #b0c4de; margin: 0.5rem 0 0 0;">
                    Today's Move: {gap_pct:+.2f}% | Confidence: {confidence}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_sig2:
            rsi_color = "#00e676" if current_rsi < 30 else ("#ff5252" if current_rsi > 70 else "#ffca28")
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color: {rsi_color};">RSI: {current_rsi:.1f}</p>
                <p class="metric-label">{rsi_signal}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Trade Recommendation
        st.markdown('<p class="section-header">📋 Trade Recommendation</p>', unsafe_allow_html=True)
        
        if "BUY" in signal or (signal == "LEAN BULLISH" and current_rsi < 35):
            # Bullish trade plan
            atm_strike = round(current_price / 5) * 5  # Round to nearest 5
            target_1 = round(current_price * 1.02 / 5) * 5
            target_2 = round(prev_close / 5) * 5
            stop_price = round(current_price * 0.97 / 5) * 5
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown(f"""
                <div class="info-box">
                <h3 style="color: #00e676; margin-top: 0;">🟢 PRIMARY: BUY CALL</h3>
                
                <table style="width: 100%; color: #e0e0e0;">
                <tr><td><strong>Strike:</strong></td><td>${atm_strike} (ATM)</td></tr>
                <tr><td><strong>Expiry:</strong></td><td>0DTE or Next Day</td></tr>
                <tr><td><strong>Size:</strong></td><td>1-2% of account</td></tr>
                </table>
                
                <hr style="border-color: #2a4a6a;">
                
                <strong>Entry Triggers:</strong>
                <ul style="margin: 0.5rem 0;">
                <li>RSI crosses above 30</li>
                <li>Price holds ${stop_price} support</li>
                <li>First green 5-min candle</li>
                </ul>
                
                <hr style="border-color: #2a4a6a;">
                
                <strong>Targets:</strong>
                <ul style="margin: 0.5rem 0;">
                <li>T1: ${target_1} → Take 50%</li>
                <li>T2: ${target_2} → Take rest</li>
                </ul>
                
                <strong>Stop Loss:</strong> ${stop_price} or -40% premium
                </div>
                """, unsafe_allow_html=True)
            
            with col_t2:
                st.markdown(f"""
                <div class="info-box" style="border-color: #4fc3f7; background: rgba(79, 195, 247, 0.08);">
                <h3 style="color: #4fc3f7; margin-top: 0;">🔵 ALTERNATIVE: Bull Call Spread</h3>
                
                <table style="width: 100%; color: #e0e0e0;">
                <tr><td><strong>Buy:</strong></td><td>${atm_strike} Call</td></tr>
                <tr><td><strong>Sell:</strong></td><td>${target_1} Call</td></tr>
                <tr><td><strong>Max Profit:</strong></td><td>${target_1 - atm_strike} per share</td></tr>
                <tr><td><strong>Max Loss:</strong></td><td>Premium paid</td></tr>
                </table>
                
                <hr style="border-color: #2a4a6a;">
                
                <strong>Why Spread:</strong>
                <ul style="margin: 0.5rem 0;">
                <li>Lower cost than naked call</li>
                <li>Defined risk</li>
                <li>Works if APP reaches ${target_1}+</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        elif "SELL" in signal or (signal == "LEAN BEARISH" and current_rsi > 65):
            # Bearish trade plan
            atm_strike = round(current_price / 5) * 5
            target_1 = round(current_price * 0.98 / 5) * 5
            target_2 = round(prev_close / 5) * 5
            stop_price = round(current_price * 1.03 / 5) * 5
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown(f"""
                <div class="info-box" style="border-color: #ff5252; background: rgba(255, 82, 82, 0.08);">
                <h3 style="color: #ff5252; margin-top: 0;">🔴 PRIMARY: BUY PUT</h3>
                
                <table style="width: 100%; color: #e0e0e0;">
                <tr><td><strong>Strike:</strong></td><td>${atm_strike} (ATM)</td></tr>
                <tr><td><strong>Expiry:</strong></td><td>0DTE or Next Day</td></tr>
                <tr><td><strong>Size:</strong></td><td>1-2% of account</td></tr>
                </table>
                
                <hr style="border-color: #2a4a6a;">
                
                <strong>Entry Triggers:</strong>
                <ul style="margin: 0.5rem 0;">
                <li>RSI crosses below 70</li>
                <li>Price rejected at ${stop_price}</li>
                <li>First red 5-min candle</li>
                </ul>
                
                <hr style="border-color: #2a4a6a;">
                
                <strong>Targets:</strong>
                <ul style="margin: 0.5rem 0;">
                <li>T1: ${target_1} → Take 50%</li>
                <li>T2: ${target_2} → Take rest</li>
                </ul>
                
                <strong>Stop Loss:</strong> ${stop_price} or -40% premium
                </div>
                """, unsafe_allow_html=True)
            
            with col_t2:
                st.markdown(f"""
                <div class="info-box" style="border-color: #ce93d8; background: rgba(206, 147, 216, 0.08);">
                <h3 style="color: #ce93d8; margin-top: 0;">🟣 ALTERNATIVE: Bear Put Spread</h3>
                
                <table style="width: 100%; color: #e0e0e0;">
                <tr><td><strong>Buy:</strong></td><td>${atm_strike} Put</td></tr>
                <tr><td><strong>Sell:</strong></td><td>${target_1} Put</td></tr>
                <tr><td><strong>Max Profit:</strong></td><td>${atm_strike - target_1} per share</td></tr>
                <tr><td><strong>Max Loss:</strong></td><td>Premium paid</td></tr>
                </table>
                
                <hr style="border-color: #2a4a6a;">
                
                <strong>Why Spread:</strong>
                <ul style="margin: 0.5rem 0;">
                <li>Lower cost than naked put</li>
                <li>Defined risk</li>
                <li>Works if APP drops to ${target_1}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Neutral - no clear trade
            st.markdown("""
            <div class="warning-box">
            <h3 style="color: #ffca28; margin-top: 0;">⏸️ NO CLEAR SIGNAL</h3>
            
            <p>The current setup doesn't meet our criteria for a high-probability trade.</p>
            
            <strong>Wait for:</strong>
            <ul>
            <li>Gap > 2% (up or down) for mean reversion</li>
            <li>RSI < 30 (oversold) or > 70 (overbought)</li>
            <li>Volume confirmation > 1.5x average</li>
            </ul>
            
            <strong>Consider:</strong>
            <ul>
            <li>Iron Condor if expecting range-bound action</li>
            <li>Wait for better setup tomorrow</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Levels
        st.markdown('<p class="section-header">📍 Key Levels</p>', unsafe_allow_html=True)
        
        # Calculate key levels
        support_1 = round(current_price * 0.99 / 5) * 5
        support_2 = round(current_price * 0.97 / 5) * 5
        support_3 = round(current_price * 0.95 / 5) * 5
        resistance_1 = round(current_price * 1.01 / 5) * 5
        resistance_2 = round(current_price * 1.03 / 5) * 5
        resistance_3 = round(prev_high / 5) * 5
        
        col_l1, col_l2 = st.columns(2)
        
        with col_l1:
            st.markdown(f"""
            <div class="metric-card">
            <h4 style="color: #00e676;">Support Levels</h4>
            <table style="width: 100%; color: #e0e0e0;">
            <tr><td>S1:</td><td><strong>${support_1}</strong></td><td style="color: #90a4ae;">First support</td></tr>
            <tr><td>S2:</td><td><strong>${support_2}</strong></td><td style="color: #90a4ae;">Key support</td></tr>
            <tr><td>S3:</td><td><strong>${support_3}</strong></td><td style="color: #90a4ae;">Strong support</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        with col_l2:
            st.markdown(f"""
            <div class="metric-card">
            <h4 style="color: #ff5252;">Resistance Levels</h4>
            <table style="width: 100%; color: #e0e0e0;">
            <tr><td>R1:</td><td><strong>${resistance_1}</strong></td><td style="color: #90a4ae;">First resistance</td></tr>
            <tr><td>R2:</td><td><strong>${resistance_2}</strong></td><td style="color: #90a4ae;">Key resistance</td></tr>
            <tr><td>R3:</td><td><strong>${resistance_3}</strong></td><td style="color: #90a4ae;">Prev high</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)
        
        # Timing
        st.markdown('<p class="section-header">⏰ Entry Timing</p>', unsafe_allow_html=True)
        
        col_time1, col_time2 = st.columns(2)
        
        with col_time1:
            st.markdown("""
            <div class="info-box" style="border-color: #00e676;">
            <h4 style="color: #00e676; margin-top: 0;">✅ Best Entry Windows</h4>
            <ul style="color: #e0e0e0;">
            <li><strong>9:35-9:45 AM</strong> - After initial volatility</li>
            <li><strong>10:00-10:30 AM</strong> - If testing support</li>
            <li><strong>2:00-2:30 PM</strong> - Afternoon reversal</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_time2:
            st.markdown("""
            <div class="info-box" style="border-color: #ff5252; background: rgba(255, 82, 82, 0.08);">
            <h4 style="color: #ff5252; margin-top: 0;">❌ Avoid Trading</h4>
            <ul style="color: #e0e0e0;">
            <li><strong>9:30-9:35 AM</strong> - Too volatile</li>
            <li><strong>12:00-1:00 PM</strong> - Low volume lunch</li>
            <li><strong>After 3:30 PM</strong> - Gamma risk on 0DTE</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Management
        st.markdown('<p class="section-header">⚠️ Risk Management</p>', unsafe_allow_html=True)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown("""
            <div class="metric-card">
            <p class="metric-value neutral">1-2%</p>
            <p class="metric-label">Max Position Size</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown("""
            <div class="metric-card">
            <p class="metric-value negative">5%</p>
            <p class="metric-label">Max Daily Loss</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            st.markdown("""
            <div class="metric-card">
            <p class="metric-value positive">50%+</p>
            <p class="metric-label">Take Profit Target</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Do Not Trade Conditions
        st.markdown('<p class="section-header">🚫 Do NOT Trade If</p>', unsafe_allow_html=True)
        
        warnings = []
        if abs(qqq_change) > 2.5:
            warnings.append(f"❌ Nasdaq move too extreme ({qqq_change:+.1f}%) - wait for stabilization")
        if vix_level > 20:
            warnings.append(f"❌ VIX too high ({vix_level:.1f}) - elevated risk")
        if relative_volume < 0.5:
            warnings.append("❌ Volume too low - lack of conviction")
        if confidence == "CONFLICTING":
            warnings.append("❌ RSI and price signals conflicting - wait for alignment")
        
        if warnings:
            for w in warnings:
                st.warning(w)
        else:
            st.success("✅ No major warning flags - setup looks tradeable")
        
        # Execution Checklist
        st.markdown('<p class="section-header">✅ Execution Checklist</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
        <table style="width: 100%; color: #e0e0e0; text-align: left;">
        <tr><td>☐</td><td>Confirmed current price and RSI</td></tr>
        <tr><td>☐</td><td>Checked SPY/QQQ direction</td></tr>
        <tr><td>☐</td><td>Set price alerts at key levels</td></tr>
        <tr><td>☐</td><td>Calculated position size (1-2% max)</td></tr>
        <tr><td>☐</td><td>Stop loss order ready BEFORE entry</td></tr>
        <tr><td>☐</td><td>Know exit plan: target OR stop OR time</td></tr>
        <tr><td>☐</td><td>Not already at max daily risk</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)


def page_settings():
    """Settings Page"""
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ Settings</h1>
        <p>Configure API keys and parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Section
    st.markdown('<p class="section-header">🔑 Polygon API Configuration</p>', unsafe_allow_html=True)
    
    api_key = get_api_key()
    
    if api_key:
        st.success("✅ API Key configured - Live data available!")
        st.markdown(f"Key: `{api_key[:8]}...{api_key[-4:]}`")
        if st.button("🗑️ Remove API Key"):
            if 'user_api_key' in st.session_state:
                del st.session_state.user_api_key
            st.rerun()
    else:
        st.warning("⚠️ No API key - Using demo data only")
        st.markdown("""
        **To use live market data:**
        1. Get a free API key from [polygon.io](https://polygon.io/)
        2. Enter it below or add to Streamlit Secrets for production
        """)
        
        user_key = st.text_input("Enter Polygon API Key", type="password", placeholder="Your API key here...")
        if user_key:
            st.session_state.user_api_key = user_key
            st.session_state.use_demo = False  # Auto-switch to live data
            st.success("✅ API key saved! Switching to live data...")
            st.rerun()
    
    st.markdown("""
    <div class="info-box">
    <strong>🔒 Security:</strong> Your API key is stored only in your browser session. 
    For Streamlit Cloud deployment, use Settings → Secrets to add:<br>
    <code>POLYGON_API_KEY = "your_key_here"</code>
    </div>
    """, unsafe_allow_html=True)
    
    # ThetaData Configuration (for real options data)
    st.markdown('<p class="section-header">📈 ThetaData Options Configuration</p>', unsafe_allow_html=True)
    
    st.markdown("""
    **ThetaData** provides real historical options data (prices, Greeks, IV).
    This replaces the simulated leverage model with actual option premiums.
    """)
    
    theta_enabled = st.checkbox(
        "Enable ThetaData (Real Options)", 
        value=st.session_state.get('theta_enabled', False),
        help="Enable to use real options data. Requires Theta Terminal running locally."
    )
    st.session_state.theta_enabled = theta_enabled
    
    if theta_enabled:
        col_theta1, col_theta2 = st.columns(2)
        
        with col_theta1:
            theta_url = st.text_input(
                "Theta Terminal URL",
                value=st.session_state.get('theta_url', 'http://127.0.0.1:25510'),
                help="Default: http://127.0.0.1:25510 (ThetaData REST API v2)"
            )
            st.session_state.theta_url = theta_url
            
            # Test connection
            if st.button("🔌 Test Connection"):
                try:
                    import requests
                    # V3 uses different endpoints - test with stock snapshot
                    response = requests.get(f"{theta_url}/v2/snapshot/stock/quote", 
                                          params={"symbol": "AAPL"}, timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Connected to Theta Terminal V3!")
                    else:
                        st.error(f"❌ Connection failed: Status {response.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect. Is Theta Terminal running?")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col_theta2:
            st.markdown("""
            **Setup Instructions:**
            1. Get subscription at [thetadata.net](https://www.thetadata.net/subscribe)
               - Free tier: 30 days EOD data
               - Value ($25/mo): Real-time snapshots
               - Standard ($75/mo): Tick data + streaming
            2. Download & run [Theta Terminal](https://www.thetadata.net/choose-your-api)
            3. Terminal runs on localhost:25510
            """)
        
        st.markdown("""
        <div class="info-box">
        <strong>📊 What ThetaData provides:</strong>
        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
        <li>Real option chain data (all strikes, expirations)</li>
        <li>Historical OHLC prices for options</li>
        <li>Greeks (delta, gamma, theta, vega, rho)</li>
        <li>Implied volatility</li>
        <li>Open interest & volume</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📊 Using simulated leverage model for options (no real options data)")
    
    # Data Source
    st.markdown('<p class="section-header">📡 Data Source</p>', unsafe_allow_html=True)
    
    col_ds1, col_ds2 = st.columns(2)
    
    with col_ds1:
        # Set default based on API key availability
        if 'use_demo' not in st.session_state:
            st.session_state.use_demo = not bool(api_key)
        
        use_demo = st.checkbox(
            "Use Demo Data", 
            value=st.session_state.get('use_demo', not bool(api_key)),
            help="Check to use simulated data. Uncheck to use live Polygon data (requires API key)"
        )
        st.session_state.use_demo = use_demo
        
        if use_demo:
            st.info("📊 Using simulated demo data")
        elif api_key:
            st.success("📡 Using LIVE data from Polygon.io")
        else:
            st.warning("⚠️ Need API key for live data")
    
    with col_ds2:
        if api_key and not use_demo:
            st.markdown("""
            **Polygon Plan Notes:**
            - **Free**: Delayed data, limited history
            - **Starter** ($29/mo): Real-time, full history
            - Minute bars work best with paid plans
            """)
    
    # Parameters
    st.markdown('<p class="section-header">📊 Analysis Parameters</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Symbol", value=st.session_state.get('symbol', 'APP'))
        st.session_state.symbol = symbol
        
        lookback = st.slider("Lookback Days", 5, 120, st.session_state.get('lookback', 60))
        st.session_state.lookback = lookback
    
    with col2:
        rsi_oversold = st.slider("RSI Oversold", 15, 35, st.session_state.get('rsi_oversold', 30))
        st.session_state.rsi_oversold = rsi_oversold
        
        rsi_overbought = st.slider("RSI Overbought", 65, 85, st.session_state.get('rsi_overbought', 70))
        st.session_state.rsi_overbought = rsi_overbought
    
    # Actions
    st.markdown('<p class="section-header">🔧 Actions</p>', unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        if st.button("🔄 Clear Cache & Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col_a2:
        if st.button("🔁 Reset to Defaults", use_container_width=True):
            st.session_state.symbol = 'APP'
            st.session_state.lookback = 60
            st.session_state.rsi_oversold = 30
            st.session_state.rsi_overbought = 70
            st.session_state.use_demo = not bool(api_key)
            st.rerun()


# ============================================================================
# MORNING DIP STRATEGY PAGE
# ============================================================================

class MorningDipBacktester:
    """Backtest the APP morning dip strategy with real option prices."""
    
    def __init__(self, polygon_api_key: str = None,
                 min_gap_pct: float = 0.8,
                 min_dip_pct: float = 1.0,
                 confirm_candles: int = 2,
                 exit_time: str = "11:20",
                 use_volume_filter: bool = True,
                 require_pm_bullish: bool = True):
        self.polygon_api_key = polygon_api_key
        self.min_gap_pct = min_gap_pct
        self.min_dip_pct = min_dip_pct
        self.confirm_candles = confirm_candles
        self.exit_time = exit_time
        self.use_volume_filter = use_volume_filter
        self.require_pm_bullish = require_pm_bullish
        
        # Try to connect to ThetaData
        self.theta_available = False
        self.theta_url = "http://127.0.0.1:25510"
        self.theta_error = None
        
        try:
            response = requests.get(
                f"{self.theta_url}/v2/snapshot/stock/quote", 
                params={"symbol": "AAPL"}, 
                timeout=5  # Increased timeout
            )
            if response.status_code == 200:
                self.theta_available = True
            else:
                self.theta_error = f"Status {response.status_code}"
        except requests.exceptions.ConnectionError:
            self.theta_error = "Connection refused - ThetaData Terminal not running"
        except requests.exceptions.Timeout:
            self.theta_error = "Connection timeout"
        except Exception as e:
            self.theta_error = str(e)
    
    def get_polygon_bars(self, symbol: str, date: str, timespan: str = "minute") -> pd.DataFrame:
        """Get intraday data from Polygon including extended hours."""
        if not self.polygon_api_key:
            return pd.DataFrame()
        
        if len(date) == 8:
            date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            date_formatted = date
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{date_formatted}/{date_formatted}"
        
        try:
            response = requests.get(url, params={
                "apiKey": self.polygon_api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except:
            pass
        
        return pd.DataFrame()
    
    def get_daily_data(self, symbol: str, date: str) -> Dict:
        """Get daily data with extended hours."""
        result = {
            'date': date,
            'premarket_close': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'postmarket_close': None,
            'df': None
        }
        
        df = self.get_polygon_bars(symbol, date)
        
        if df.empty:
            return result
        
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        result['df'] = df
        
        premarket = df[(df['hour'] >= 4) & (df['hour'] < 9) | 
                       ((df['hour'] == 9) & (df['minute'] < 30))]
        regular = df[((df['hour'] == 9) & (df['minute'] >= 30)) | 
                     ((df['hour'] >= 10) & (df['hour'] < 16))]
        postmarket = df[(df['hour'] >= 16) & (df['hour'] < 20)]
        
        if not premarket.empty:
            result['premarket_close'] = premarket['close'].iloc[-1]
        
        if not regular.empty:
            result['open'] = regular['open'].iloc[0]
            result['high'] = regular['high'].max()
            result['low'] = regular['low'].min()
            result['close'] = regular['close'].iloc[-1]
            result['avg_volume'] = regular['volume'].mean()
        
        if not postmarket.empty:
            pm_valid = postmarket[postmarket['close'] > 0]
            if not pm_valid.empty:
                result['postmarket_close'] = pm_valid['close'].iloc[-1]
        
        return result
    
    def check_entry_conditions(self, prev_day: Dict, today: Dict) -> Tuple[bool, str, float]:
        """Check if entry conditions are met."""
        prev_eod = prev_day.get('close')
        prev_postmarket = prev_day.get('postmarket_close')
        today_premarket = today.get('premarket_close')
        today_open = today.get('open')
        
        if not prev_eod:
            return False, "no_prev_eod", 0
        
        pm_gap_pct = 0
        premarket_gap_pct = 0
        
        if self.require_pm_bullish and prev_postmarket and prev_postmarket > 0:
            pm_gap_pct = (prev_postmarket - prev_eod) / prev_eod * 100
            if pm_gap_pct < -0.3:
                return False, f"postmarket_weak_{pm_gap_pct:+.1f}%", pm_gap_pct
        elif prev_postmarket and prev_postmarket > 0:
            pm_gap_pct = (prev_postmarket - prev_eod) / prev_eod * 100
        
        if today_premarket and today_premarket > 0:
            premarket_gap_pct = (today_premarket - prev_eod) / prev_eod * 100
        elif today_open:
            premarket_gap_pct = (today_open - prev_eod) / prev_eod * 100
        else:
            return False, "no_premarket_or_open", 0
        
        if premarket_gap_pct < self.min_gap_pct:
            return False, f"premarket_gap_{premarket_gap_pct:+.1f}%", premarket_gap_pct
        
        if self.require_pm_bullish and prev_postmarket and prev_postmarket > 0:
            if pm_gap_pct < 0:
                if premarket_gap_pct < self.min_gap_pct + abs(pm_gap_pct):
                    return False, f"pm_down_{pm_gap_pct:+.1f}%", premarket_gap_pct
        
        return True, f"pm_{pm_gap_pct:+.1f}%_pre_{premarket_gap_pct:+.1f}%", premarket_gap_pct
    
    def find_morning_dip_with_confirmation(self, df: pd.DataFrame, avg_volume: float = None) -> Optional[Dict]:
        """Find morning dip with momentum confirmation."""
        morning = df[
            ((df['hour'] == 9) & (df['minute'] >= 30)) |
            ((df['hour'] == 10) & (df['minute'] <= 30))
        ].copy()
        
        if len(morning) < 10:
            return None
        
        morning = morning.reset_index(drop=True)
        
        dip_window = morning[
            ((morning['hour'] == 9) & (morning['minute'] >= 30)) |
            ((morning['hour'] == 10) & (morning['minute'] <= 10))
        ]
        
        if dip_window.empty:
            return None
        
        open_price = morning['open'].iloc[0]
        low_idx_original = dip_window['low'].idxmin()
        low_price = dip_window.loc[low_idx_original, 'low']
        low_time = dip_window.loc[low_idx_original, 'timestamp']
        
        dip_pct = (open_price - low_price) / open_price * 100
        if dip_pct < self.min_dip_pct:
            return {'rejected': True, 'reason': f'dip_too_small_{dip_pct:.1f}%'}
        
        if self.use_volume_filter and avg_volume:
            dip_volume = dip_window['volume'].mean()
            if dip_volume < avg_volume * 0.8:
                return {'rejected': True, 'reason': 'low_volume_on_dip'}
        
        low_idx_in_morning = morning[morning['timestamp'] == low_time].index[0]
        candles_after_low = morning.iloc[low_idx_in_morning + 1:low_idx_in_morning + 1 + self.confirm_candles + 2]
        
        if len(candles_after_low) < self.confirm_candles:
            return {'rejected': True, 'reason': 'not_enough_candles_after_low'}
        
        candles_after_low = candles_after_low.copy()
        candles_after_low['green'] = candles_after_low['close'] > candles_after_low['open']
        
        green_count = 0
        entry_idx = None
        for idx, row in candles_after_low.iterrows():
            if row['green']:
                green_count += 1
                if green_count >= self.confirm_candles:
                    entry_idx = idx
                    break
            else:
                green_count = 0
        
        if green_count < self.confirm_candles:
            return {'rejected': True, 'reason': f'no_momentum_{green_count}_green'}
        
        entry_row = morning.loc[entry_idx]
        entry_price = entry_row['close']
        entry_time = entry_row['timestamp']
        
        return {
            'rejected': False,
            'low_price': low_price,
            'low_time': low_time,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'open_price': open_price,
            'dip_pct': dip_pct,
            'green_candles': green_count
        }
    
    def get_exit_price_and_time(self, df: pd.DataFrame, entry_time) -> Tuple[Optional[float], Optional[datetime]]:
        """Get exit price based on exit_time setting."""
        if df is None or df.empty:
            return None, None
        
        if df['timestamp'].dt.tz is not None:
            df = df.copy()
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        if self.exit_time.upper() == "EOD":
            regular = df[((df['hour'] == 9) & (df['minute'] >= 30)) | 
                        ((df['hour'] >= 10) & (df['hour'] < 16))]
            if not regular.empty:
                return regular['close'].iloc[-1], regular['timestamp'].iloc[-1]
            return None, None
        
        try:
            exit_hour, exit_minute = map(int, self.exit_time.split(':'))
        except:
            exit_hour, exit_minute = 11, 20
        
        exit_candles = df[(df['hour'] == exit_hour) & (df['minute'] >= exit_minute)]
        
        if not exit_candles.empty:
            return exit_candles['close'].iloc[0], exit_candles['timestamp'].iloc[0]
        
        after_entry = df[df['timestamp'] > entry_time]
        if not after_entry.empty:
            return after_entry['close'].iloc[-1], after_entry['timestamp'].iloc[-1]
        
        return None, None
    
    def get_option_price_at_time(self, symbol: str, expiration: str, strike: float,
                                  right: str, date: str, target_time, get_last: bool = False) -> Optional[float]:
        """Get option price from ThetaData."""
        if not self.theta_available:
            return None
        
        try:
            from io import StringIO
            response = requests.get(
                f"{self.theta_url}/v2/hist/option/ohlc",
                params={
                    "symbol": symbol,
                    "expiration": expiration,
                    "strike": str(strike),
                    "right": right,
                    "date": date
                },
                timeout=60
            )
            
            if response.status_code != 200 or not response.text:
                return None
            
            df = pd.read_csv(StringIO(response.text))
            if df.empty:
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to rows with actual trades (close > 0 OR volume > 0)
            df_valid = df[(df['close'] > 0) | (df['volume'] > 0)].copy()
            
            if df_valid.empty:
                # No trades - try using high/low if available
                df_valid = df[(df['high'] > 0) | (df['low'] > 0)].copy()
                if df_valid.empty:
                    return None
            
            if df_valid['timestamp'].dt.tz is not None:
                df_valid['timestamp'] = df_valid['timestamp'].dt.tz_localize(None)
            
            if hasattr(target_time, 'tzinfo') and target_time.tzinfo is not None:
                target_time = target_time.replace(tzinfo=None)
            
            if get_last:
                row = df_valid.iloc[-1]
            else:
                df_valid['time_diff'] = abs((df_valid['timestamp'] - target_time).dt.total_seconds())
                closest_idx = df_valid['time_diff'].idxmin()
                row = df_valid.loc[closest_idx]
            
            if row['close'] > 0:
                return float(row['close'])
            elif row['high'] > 0 and row['low'] > 0:
                return float((row['high'] + row['low']) / 2)
            elif row['high'] > 0:
                return float(row['high'])
            elif row['low'] > 0:
                return float(row['low'])
            return None
        except Exception as e:
            # Store error for debugging
            self._last_option_error = str(e)
            return None
    
    def get_expirations(self, symbol: str) -> List[str]:
        """Get option expirations from ThetaData."""
        if not self.theta_available:
            return []
        try:
            from io import StringIO
            response = requests.get(
                f"{self.theta_url}/v2/snapshot/option/quote",
                params={"symbol": symbol, "expiration": "*"},
                timeout=30
            )
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty and 'expiration' in df.columns:
                    return sorted(df['expiration'].unique().tolist())
        except:
            pass
        return []
    
    def get_strikes(self, symbol: str, expiration: str) -> List[float]:
        """Get option strikes from ThetaData."""
        if not self.theta_available:
            return []
        try:
            from io import StringIO
            
            # First try snapshot (for current expirations)
            response = requests.get(
                f"{self.theta_url}/v2/snapshot/option/quote",
                params={"symbol": symbol, "expiration": expiration},
                timeout=30
            )
            if response.status_code == 200 and response.text:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty and 'strike' in df.columns:
                    # Filter for CALLs only
                    if 'right' in df.columns:
                        df = df[df['right'] == 'CALL']
                    return sorted(df['strike'].unique().tolist())
            
            # For historical expirations, try getting data directly
            # by requesting option history - if it returns data, the expiration exists
            test_strike = 650  # Test with a likely strike
            response = requests.get(
                f"{self.theta_url}/v2/hist/option/ohlc",
                params={
                    "symbol": symbol,
                    "expiration": expiration,
                    "strike": str(test_strike),
                    "right": "CALL",
                    "date": datetime.now().strftime("%Y%m%d")
                },
                timeout=10
            )
            if response.status_code == 200 and response.text:
                # Expiration exists, return common strikes
                entry_price = 665  # Approximate APP price
                return [float(x) for x in range(int(entry_price - 50), int(entry_price + 50), 5)]
        except:
            pass
        return []
    
    def backtest_day(self, symbol: str, date: str, prev_day: Dict) -> Dict:
        """Run backtest for a single day."""
        result = {'date': date, 'signal': False}
        
        today = self.get_daily_data(symbol, date)
        
        if not today.get('open'):
            result['reason'] = 'no_data'
            return result
        
        conditions_met, condition_reason, gap_pct = self.check_entry_conditions(prev_day, today)
        result['gap_pct'] = gap_pct
        
        if not conditions_met:
            result['reason'] = condition_reason
            return result
        
        df = today.get('df')
        if df is None or df.empty:
            result['reason'] = 'no_intraday_data'
            return result
        
        dip_result = self.find_morning_dip_with_confirmation(df, today.get('avg_volume'))
        
        if dip_result is None:
            result['reason'] = 'no_dip_pattern'
            return result
        
        if dip_result.get('rejected'):
            result['reason'] = dip_result.get('reason', 'rejected')
            return result
        
        result['signal'] = True
        result['condition'] = condition_reason
        result['entry_time'] = str(dip_result['entry_time'])
        result['entry_stock_price'] = dip_result['entry_price']
        result['low_price'] = dip_result['low_price']
        result['morning_open'] = dip_result['open_price']
        result['dip_pct'] = dip_result['dip_pct']
        
        exit_price, exit_time = self.get_exit_price_and_time(df, dip_result['entry_time'])
        result['exit_stock_price'] = exit_price
        result['exit_time'] = str(exit_time) if exit_time else None
        
        if exit_price and dip_result['entry_price']:
            result['stock_pnl_pct'] = (exit_price - dip_result['entry_price']) / dip_result['entry_price'] * 100
        
        # Initialize option columns (will be filled if ThetaData available)
        result['strike'] = None
        result['expiration'] = None
        result['entry_option_price'] = None
        result['exit_option_price'] = None
        result['option_pnl_pct'] = None
        result['option_pnl_dollar'] = None
        result['_debug_theta_available'] = self.theta_available
        
        # Get option prices if ThetaData available
        if self.theta_available:
            date_dt = datetime.strptime(date, "%Y%m%d")
            entry_price = dip_result['entry_price']
            
            # Generate potential expiration dates to try
            exp_candidates = []
            
            # Add next 4 Fridays (weeklies)
            days_until_friday = (4 - date_dt.weekday()) % 7
            if days_until_friday == 0:
                days_until_friday = 7
            for weeks_ahead in range(4):
                days_to_add = days_until_friday + (weeks_ahead * 7)
                exp_date = date_dt + timedelta(days=days_to_add)
                exp_candidates.append(exp_date.strftime("%Y-%m-%d"))
            
            # Add 3rd Friday of current month (monthly)
            first_of_month = date_dt.replace(day=1)
            days_to_first_friday = (4 - first_of_month.weekday()) % 7
            third_friday = first_of_month + timedelta(days=days_to_first_friday + 14)
            if third_friday > date_dt:
                exp_candidates.append(third_friday.strftime("%Y-%m-%d"))
            
            # Add 3rd Friday of next month
            if date_dt.month == 12:
                next_month = date_dt.replace(year=date_dt.year + 1, month=1, day=1)
            else:
                next_month = date_dt.replace(month=date_dt.month + 1, day=1)
            days_to_first_friday = (4 - next_month.weekday()) % 7
            third_friday_next = next_month + timedelta(days=days_to_first_friday + 14)
            exp_candidates.append(third_friday_next.strftime("%Y-%m-%d"))
            
            # Remove duplicates and sort
            exp_candidates = sorted(list(set(exp_candidates)))
            result['_debug_exp_candidates'] = str(exp_candidates[:5])
            
            # Calculate strikes to try based on entry price
            # Try wider range since stock price may have been very different historically
            base_strike = round(entry_price / 5) * 5
            strikes_near_entry = [base_strike + (i * 5) for i in range(-10, 11)]  # ±$50 range
            
            # Also try common round strikes in case entry price was very different
            common_strikes = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
            all_strikes_to_try = sorted(list(set(strikes_near_entry + common_strikes)))
            
            # Find working expiration and strike
            valid_exp = None
            entry_opt_price = None
            exit_opt_price = None
            working_strike = None
            
            entry_time = dip_result['entry_time']
            if isinstance(entry_time, str):
                try:
                    entry_time = pd.to_datetime(entry_time)
                except:
                    pass
            
            from io import StringIO
            
            for exp in exp_candidates:
                if valid_exp:
                    break
                
                # Sort strikes by distance from entry price (prefer ATM)
                sorted_strikes = sorted(all_strikes_to_try, key=lambda x: abs(x - entry_price))
                
                for strike in sorted_strikes[:15]:  # Try up to 15 strikes per expiration
                    try:
                        response = requests.get(
                            f"{self.theta_url}/v2/hist/option/ohlc",
                            params={
                                "symbol": symbol,
                                "expiration": exp,
                                "strike": str(strike),
                                "right": "CALL",
                                "date": date
                            },
                            timeout=15
                        )
                        
                        if response.status_code == 200 and response.text:
                            df = pd.read_csv(StringIO(response.text))
                            df_valid = df[(df['close'] > 0) | (df['volume'] > 0)].copy()
                            
                            if len(df_valid) > 0:
                                # Found data! Get entry and exit prices
                                df_valid['timestamp'] = pd.to_datetime(df_valid['timestamp'])
                                
                                if df_valid['timestamp'].dt.tz is not None:
                                    df_valid['timestamp'] = df_valid['timestamp'].dt.tz_localize(None)
                                
                                target_entry = entry_time
                                if hasattr(target_entry, 'tzinfo') and target_entry.tzinfo is not None:
                                    target_entry = target_entry.replace(tzinfo=None)
                                
                                # Get entry price (closest to entry time)
                                df_valid['time_diff'] = abs((df_valid['timestamp'] - target_entry).dt.total_seconds())
                                entry_idx = df_valid['time_diff'].idxmin()
                                entry_row = df_valid.loc[entry_idx]
                                entry_opt = float(entry_row['close']) if entry_row['close'] > 0 else None
                                
                                # Get exit price (last available or at exit time)
                                exit_opt = float(df_valid['close'].iloc[-1])
                                
                                if entry_opt and entry_opt > 0 and exit_opt and exit_opt > 0:
                                    valid_exp = exp
                                    working_strike = strike
                                    entry_opt_price = entry_opt
                                    exit_opt_price = exit_opt
                                    break
                    except:
                        continue
            
            # Store results
            if valid_exp and working_strike and entry_opt_price and exit_opt_price:
                result['expiration'] = valid_exp
                result['strike'] = working_strike
                result['entry_option_price'] = entry_opt_price
                result['exit_option_price'] = exit_opt_price
                result['option_pnl_pct'] = (exit_opt_price - entry_opt_price) / entry_opt_price * 100
                result['option_pnl_dollar'] = (exit_opt_price - entry_opt_price) * 100
                result['_debug_valid_exp'] = valid_exp
                result['_debug_found_strike'] = working_strike
            else:
                result['_debug_valid_exp'] = 'none_found'
                result['_debug_tried_exps'] = str(exp_candidates[:4])
        
        return result
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                     progress_callback=None) -> pd.DataFrame:
        """Run full backtest."""
        results = []
        
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        total_days = (end_dt - start_dt).days + 1
        prev_day = None
        current = start_dt
        day_count = 0
        
        while current <= end_dt:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue
            
            date_str = current.strftime("%Y%m%d")
            day_count += 1
            
            if progress_callback:
                progress_callback(day_count / total_days, f"Processing {date_str}...")
            
            today_data = self.get_daily_data(symbol, date_str)
            
            if prev_day and today_data.get('open'):
                result = self.backtest_day(symbol, date_str, prev_day)
                results.append(result)
            
            prev_day = today_data
            current += timedelta(days=1)
        
        return pd.DataFrame(results)


def page_morning_dip():
    """Morning Dip Strategy Page"""
    st.markdown("""
    <div class="main-header">
        <h1>🌅 Morning Dip Strategy</h1>
        <p>Buy calls on morning dips after gap-up with momentum confirmation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar parameters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Strategy Parameters")
    
    min_gap_pct = st.sidebar.slider(
        "Min Premarket Gap %",
        min_value=0.0, max_value=5.0, value=0.8, step=0.1,
        help="Premarket must be this % above previous EOD"
    )
    
    require_pm_bullish = st.sidebar.checkbox(
        "Require Post-Market Bullish",
        value=True,
        help="Previous day's post-market must not be down"
    )
    
    min_dip_pct = st.sidebar.slider(
        "Min Dip %",
        min_value=0.5, max_value=5.0, value=1.0, step=0.1,
        help="Price must drop this % from open"
    )
    
    confirm_candles = st.sidebar.slider(
        "Confirm Candles",
        min_value=1, max_value=5, value=2,
        help="Green candles after low to confirm entry"
    )
    
    use_volume_filter = st.sidebar.checkbox("Volume Filter", value=True)
    
    st.sidebar.markdown("### 📤 Exit Strategy")
    exit_type = st.sidebar.radio("Exit Type", ["Time-Based", "EOD"], index=0)
    
    if exit_type == "Time-Based":
        exit_hour = st.sidebar.slider("Exit Hour", 10, 15, 11)
        exit_minute = st.sidebar.slider("Exit Minute", 0, 55, 20, step=5)
        exit_time = f"{exit_hour:02d}:{exit_minute:02d}"
    else:
        exit_time = "EOD"
    
    st.sidebar.markdown(f"**Exit:** {exit_time}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input("Symbol", value="APP")
    
    with col2:
        lookback = st.number_input("Lookback Days", min_value=5, max_value=400, value=30)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback)
    
    st.markdown(f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Check ThetaData status before running
    theta_available = False
    theta_error = None
    try:
        response = requests.get(
            "http://127.0.0.1:25510/v2/snapshot/stock/quote", 
            params={"symbol": "AAPL"}, 
            timeout=5  # Longer timeout
        )
        if response.status_code == 200:
            theta_available = True
        else:
            theta_error = f"Status {response.status_code}"
    except requests.exceptions.ConnectionError:
        theta_error = "Connection refused"
    except requests.exceptions.Timeout:
        theta_error = "Timeout"
    except Exception as e:
        theta_error = str(e)
    
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if get_api_key():
            st.success("✅ Polygon API: Connected")
        else:
            st.error("❌ Polygon API: Not configured")
    with col_status2:
        if theta_available:
            st.success("✅ ThetaData: Connected (Option prices available)")
        else:
            st.warning(f"⚠️ ThetaData: {theta_error or 'Not available'}")
            if st.button("🔄 Retry ThetaData Connection"):
                st.rerun()
    
    # Explanation of columns
    with st.expander("📊 What are Gap % and Dip %?"):
        st.markdown("""
        **Gap %** = How much stock gapped up from previous day's close
        ```
        Gap % = (Today's Premarket - Previous EOD) / Previous EOD × 100
        ```
        
        **Dip %** = How much stock dropped in morning window (9:30-10:10 AM)
        ```
        Dip % = (Morning Open - Morning Low) / Morning Open × 100
        ```
        
        **Option Prices**: Requires ThetaData Terminal running on port 25510
        """)
    
    # Run backtest
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        api_key = get_api_key()
        
        if not api_key:
            st.error("❌ Polygon API key required. Configure in Settings page.")
            return
        
        backtester = MorningDipBacktester(
            polygon_api_key=api_key,
            min_gap_pct=min_gap_pct,
            min_dip_pct=min_dip_pct,
            confirm_candles=confirm_candles,
            exit_time=exit_time,
            use_volume_filter=use_volume_filter,
            require_pm_bullish=require_pm_bullish
        )
        
        # Check ThetaData
        if backtester.theta_available:
            st.success("✅ ThetaData connected - Real option prices available!")
        else:
            error_msg = backtester.theta_error if hasattr(backtester, 'theta_error') and backtester.theta_error else "Unknown error"
            st.error(f"❌ ThetaData connection failed: {error_msg}")
            st.info("💡 Make sure ThetaData Terminal is running on http://127.0.0.1:25510")
        
        progress = st.progress(0)
        status = st.empty()
        
        def update_progress(pct, msg):
            progress.progress(pct)
            status.text(msg)
        
        results = backtester.run_backtest(
            symbol=symbol,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            progress_callback=update_progress
        )
        
        progress.empty()
        status.empty()
        
        st.session_state.morning_dip_results = results
    
    # Display results
    if 'morning_dip_results' in st.session_state and not st.session_state.morning_dip_results.empty:
        results = st.session_state.morning_dip_results
        signals = results[results['signal'] == True]
        
        # Summary metrics
        st.markdown("### 📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", len(results))
        with col2:
            st.metric("Signals", len(signals))
        with col3:
            if len(signals) > 0:
                win_rate = (signals['stock_pnl_pct'] > 0).mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.metric("Win Rate", "N/A")
        with col4:
            if len(signals) > 0:
                total = signals['stock_pnl_pct'].sum()
                st.metric("Total Stock P&L", f"{total:+.2f}%")
            else:
                st.metric("Total Stock P&L", "N/A")
        
        if len(signals) > 0:
            # Stock performance
            st.markdown("### 📈 Stock Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg P&L", f"{signals['stock_pnl_pct'].mean():+.2f}%")
            with col2:
                st.metric("Best", f"{signals['stock_pnl_pct'].max():+.2f}%")
            with col3:
                st.metric("Worst", f"{signals['stock_pnl_pct'].min():+.2f}%")
            
            # Option performance
            if 'option_pnl_pct' in signals.columns:
                opt_signals = signals[signals['option_pnl_pct'].notna()]
                if len(opt_signals) > 0:
                    st.markdown("### 📊 Option Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Trades", len(opt_signals))
                    with col2:
                        opt_win = (opt_signals['option_pnl_pct'] > 0).mean() * 100
                        st.metric("Win Rate", f"{opt_win:.1f}%")
                    with col3:
                        st.metric("Avg P&L", f"{opt_signals['option_pnl_pct'].mean():+.2f}%")
                    with col4:
                        if 'option_pnl_dollar' in opt_signals.columns:
                            st.metric("Avg $/Contract", f"${opt_signals['option_pnl_dollar'].mean():+.0f}")
            
            # Trade log
            st.markdown("### 📋 Trade Log")
            
            # Show debug info if available (collapsed by default)
            debug_cols = [c for c in signals.columns if c.startswith('_debug')]
            if debug_cols:
                with st.expander("🔧 Debug Info (Option Lookup)", expanded=False):
                    for _, row in signals.iterrows():
                        st.write(f"**Date: {row['date']}**")
                        for col in debug_cols:
                            if col in row and pd.notna(row[col]) and row[col] is not None:
                                st.write(f"  {col.replace('_debug_', '')}: {row[col]}")
                        st.write("---")
            
            display_cols = ['date', 'entry_stock_price', 'exit_stock_price', 'stock_pnl_pct',
                           'expiration', 'strike', 'entry_option_price', 'exit_option_price', 'option_pnl_pct',
                           'option_pnl_dollar', 'gap_pct', 'dip_pct', 'exit_time']
            
            display_df = signals[[c for c in display_cols if c in signals.columns]].copy()
            
            # Rename columns for clarity
            column_names = {
                'date': 'Date',
                'entry_stock_price': 'Stock Entry',
                'exit_stock_price': 'Stock Exit', 
                'stock_pnl_pct': 'Stock P&L',
                'expiration': 'Exp',
                'strike': 'Strike',
                'entry_option_price': 'Opt Buy',
                'exit_option_price': 'Opt Sell',
                'option_pnl_pct': 'Opt P&L %',
                'option_pnl_dollar': 'Opt $/Contract',
                'gap_pct': 'Gap %',
                'dip_pct': 'Dip %',
                'exit_time': 'Exit Time'
            }
            
            # Format values
            for col in display_df.columns:
                if col in ['entry_stock_price', 'exit_stock_price', 'entry_option_price', 'exit_option_price']:
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                elif col == 'option_pnl_dollar':
                    display_df[col] = display_df[col].apply(lambda x: f"${x:+,.0f}" if pd.notna(x) else "N/A")
                elif 'pct' in col:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
                elif col == 'strike':
                    display_df[col] = display_df[col].apply(lambda x: f"${x:.0f}C" if pd.notna(x) else "N/A")
                elif col == 'exit_time':
                    display_df[col] = display_df[col].apply(lambda x: str(x)[-8:-3] if pd.notna(x) and x else "N/A")
            
            # Rename columns
            display_df = display_df.rename(columns={k: v for k, v in column_names.items() if k in display_df.columns})
            
            st.dataframe(display_df, use_container_width=True)
            
            # Equity curves
            st.markdown("### 📈 Equity Curves")
            
            # Stock P&L curve
            cumulative_stock = signals['stock_pnl_pct'].cumsum()
            
            # Option P&L curve (only if we have option data)
            has_option_data = signals['option_pnl_pct'].notna().any()
            
            if has_option_data:
                # Fill NaN with 0 for cumulative calculation
                option_pnl = signals['option_pnl_pct'].fillna(0)
                cumulative_option = option_pnl.cumsum()
                
                # Also calculate dollar P&L
                option_dollar = signals['option_pnl_dollar'].fillna(0)
                cumulative_dollar = option_dollar.cumsum()
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Percentage P&L", "💵 Dollar P&L (Options)"])
            
            with tab1:
                fig = go.Figure()
                
                # Stock P&L trace
                fig.add_trace(go.Scatter(
                    x=signals['date'],
                    y=cumulative_stock,
                    mode='lines+markers',
                    name='Stock P&L',
                    line=dict(color='#2196F3', width=2),
                    marker=dict(size=8)
                ))
                
                # Option P&L trace (if available)
                if has_option_data:
                    fig.add_trace(go.Scatter(
                        x=signals['date'],
                        y=cumulative_option,
                        mode='lines+markers',
                        name='Option P&L',
                        line=dict(color='#FF9800', width=2),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title="Cumulative P&L (%)" + (" - Stock vs Options" if has_option_data else " - Stock Only"),
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L (%)",
                    template="plotly_dark",
                    hovermode='x unified',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Stock Total P&L",
                        f"{cumulative_stock.iloc[-1]:+.1f}%",
                        delta=f"{signals['stock_pnl_pct'].mean():+.1f}% avg/trade"
                    )
                with col2:
                    if has_option_data:
                        st.metric(
                            "Option Total P&L",
                            f"{cumulative_option.iloc[-1]:+.1f}%",
                            delta=f"{option_pnl[option_pnl != 0].mean():+.1f}% avg/trade" if (option_pnl != 0).any() else "N/A"
                        )
                    else:
                        st.info("No option data available")
            
            with tab2:
                if has_option_data:
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=signals['date'],
                        y=cumulative_dollar,
                        mode='lines+markers',
                        name='Option $ P&L',
                        line=dict(color='#4CAF50' if cumulative_dollar.iloc[-1] > 0 else '#f44336', width=3),
                        marker=dict(size=10),
                        fill='tozeroy',
                        fillcolor='rgba(76, 175, 80, 0.2)' if cumulative_dollar.iloc[-1] > 0 else 'rgba(244, 67, 54, 0.2)'
                    ))
                    
                    fig2.update_layout(
                        title="Cumulative Option P&L ($ per Contract)",
                        xaxis_title="Date",
                        yaxis_title="Cumulative $ P&L",
                        template="plotly_dark",
                        hovermode='x unified'
                    )
                    
                    # Add zero line
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Dollar summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total $ P&L",
                            f"${cumulative_dollar.iloc[-1]:+,.0f}",
                            delta=f"per contract"
                        )
                    with col2:
                        trades_with_options = (option_dollar != 0).sum()
                        st.metric(
                            "Trades with Options",
                            f"{trades_with_options}/{len(signals)}",
                            delta=f"{trades_with_options/len(signals)*100:.0f}% coverage"
                        )
                    with col3:
                        if trades_with_options > 0:
                            avg_dollar = option_dollar[option_dollar != 0].mean()
                            st.metric(
                                "Avg $/Trade",
                                f"${avg_dollar:+,.0f}",
                                delta="per contract"
                            )
                else:
                    st.warning("No option data available. Make sure ThetaData is connected.")
                    st.info("Option P&L shows the dollar profit per contract (100 shares).")
            
            # Download
            csv = results.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                data=csv,
                file_name=f"morning_dip_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No signals generated. Try adjusting parameters.")
            
            # Show rejection reasons
            st.markdown("### 📋 Rejection Reasons")
            rejections = results[results['signal'] == False]['reason'].value_counts()
            st.dataframe(rejections.reset_index().rename(columns={'index': 'Reason', 'reason': 'Count'}))
    
    # Strategy explanation
    with st.expander("📖 Strategy Explanation"):
        st.markdown("""
        ## Morning Dip Strategy
        
        ### Entry Conditions:
        1. **Post-Market Bullish**: Previous day's post-market ≥ EOD (if enabled)
        2. **Gap Up**: Today's premarket/open > previous EOD by min gap %
        3. **Morning Dip**: Price drops > min dip % between 9:30-10:10 AM
        4. **Volume**: Above-average volume during dip (if enabled)
        5. **Momentum Confirmation**: N consecutive green candles after the low
        
        ### Exit:
        - **Time-Based**: Exit at specified time (default 11:20 AM)
        - **EOD**: Hold until market close
        
        ### Rationale:
        APP tends to have morning selling pressure after gap-ups. If the dip reverses 
        with momentum, it often continues higher. Early exit captures the bounce.
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📊 APP Trading System")
        st.markdown("---")
        
        page = st.radio("Navigation", ["📊 Dashboard", "📈 Backtest", "🔬 Validation", "🎯 Trade Plan", "🌅 Morning Dip", "⚙️ Settings"], label_visibility="collapsed")

        st.markdown("---")

        # Quick status
        api_key = get_api_key()
        if api_key:
            st.markdown("✅ API: Connected")
        else:
            st.markdown("⚠️ API: Demo Mode")

        st.markdown(f"📅 Lookback: {st.session_state.get('lookback', 60)} days")
        st.markdown(f"📊 RSI: {st.session_state.get('rsi_oversold', 30)}/{st.session_state.get('rsi_overbought', 70)}")

    # Page routing
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "📈 Backtest":
        page_backtest()
    elif page == "🔬 Validation":
        page_validation()
    elif page == "🎯 Trade Plan":
        page_trade_plan()
    elif page == "🌅 Morning Dip":
        page_morning_dip()
    else:
        page_settings()
    
    # Footer
    st.markdown("---")
    st.markdown('<p style="text-align: center; color: #8892b0;">Built by Limestone Hill Capital | ⚠️ Not financial advice</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()