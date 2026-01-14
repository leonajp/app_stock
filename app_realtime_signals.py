"""
APP Real-Time Mean Reversion Signal Generator
==============================================
Live trading signal generator based on session returns and RSI patterns.

Usage:
    python app_realtime_signals.py --api-key YOUR_KEY --live
    python app_realtime_signals.py --api-key YOUR_KEY --backtest

Author: Limestone Hill Capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import requests
import time
import argparse
from enum import Enum
import json

# ============================================================================
# ENUMS & CONFIG
# ============================================================================

class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

class SignalStrength(Enum):
    STRONG = 3
    MODERATE = 2
    WEAK = 1

@dataclass
class TradingConfig:
    # API
    POLYGON_API_KEY: str = ""
    SYMBOL: str = "APP"
    
    # RSI Settings
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_EXTREME_OVERSOLD: float = 20.0
    RSI_EXTREME_OVERBOUGHT: float = 80.0
    
    # Signal Thresholds
    LARGE_GAP_THRESHOLD: float = 2.0  # %
    MEDIUM_GAP_THRESHOLD: float = 1.0  # %
    EXTENDED_MOVE_THRESHOLD: float = 3.0  # % total overnight move
    
    # Options Parameters
    PREFERRED_EXPIRY_DTE: int = 0  # 0DTE
    MAX_OPTION_PREMIUM_PCT: float = 2.0  # Max % of underlying to pay
    
    # Risk Management
    MAX_POSITION_SIZE_USD: float = 10000
    STOP_LOSS_PCT: float = 50.0  # % of premium
    PROFIT_TARGET_PCT: float = 50.0  # % gain on premium


@dataclass
class MarketState:
    """Current market state for signal generation"""
    timestamp: datetime = None
    current_price: float = 0.0
    
    # Session data
    prev_day_close: float = 0.0
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    postmarket_close: float = 0.0
    premarket_close: float = 0.0
    premarket_high: float = 0.0
    premarket_low: float = 0.0
    today_open: float = 0.0
    today_high: float = 0.0
    today_low: float = 0.0
    first_5min_close: float = 0.0
    
    # Calculated returns
    prev_day_return: float = 0.0
    postmarket_return: float = 0.0
    overnight_return: float = 0.0
    premarket_return: float = 0.0
    gap_return: float = 0.0
    total_extended_return: float = 0.0
    first_5min_return: float = 0.0
    current_day_return: float = 0.0
    
    # Indicators
    rsi: float = 50.0
    vwap: float = 0.0
    volume_ratio: float = 1.0  # vs avg
    
    # Levels
    key_resistance: List[float] = field(default_factory=list)
    key_support: List[float] = field(default_factory=list)


@dataclass 
class Signal:
    """Trading signal with options recommendation"""
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    reason: str
    
    # Entry details
    entry_price: float
    stop_loss: float
    target_price: float
    
    # Options recommendation
    options_strategy: str
    strike_recommendation: str
    expiry_recommendation: str
    position_size_suggestion: str
    
    # Risk metrics
    risk_reward_ratio: float
    confidence_pct: float
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal": self.signal_type.value,
            "strength": self.strength.name,
            "reason": self.reason,
            "entry": self.entry_price,
            "stop": self.stop_loss,
            "target": self.target_price,
            "options_strategy": self.options_strategy,
            "strike": self.strike_recommendation,
            "expiry": self.expiry_recommendation,
            "size": self.position_size_suggestion,
            "rr_ratio": self.risk_reward_ratio,
            "confidence": self.confidence_pct
        }


# ============================================================================
# DATA FETCHER
# ============================================================================

class PolygonRealtime:
    """Real-time data from Polygon"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    def get_last_quote(self, symbol: str) -> Dict:
        """Get last quote"""
        url = f"{self.base_url}/v2/last/nbbo/{symbol}"
        params = {"apiKey": self.api_key}
        resp = requests.get(url, params=params)
        return resp.json().get("results", {})
    
    def get_last_trade(self, symbol: str) -> Dict:
        """Get last trade"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {"apiKey": self.api_key}
        resp = requests.get(url, params=params)
        return resp.json().get("results", {})
    
    def get_snapshot(self, symbol: str) -> Dict:
        """Get ticker snapshot with current day data"""
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {"apiKey": self.api_key}
        resp = requests.get(url, params=params)
        return resp.json().get("ticker", {})
    
    def get_aggregates(
        self, 
        symbol: str, 
        multiplier: int, 
        timespan: str, 
        from_date: str, 
        to_date: str
    ) -> pd.DataFrame:
        """Get aggregate bars"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key
        }
        
        resp = requests.get(url, params=params)
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
    
    def get_options_chain(self, symbol: str, expiry: str = None) -> pd.DataFrame:
        """Get options chain"""
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": symbol,
            "expired": "false",
            "limit": 250,
            "apiKey": self.api_key
        }
        if expiry:
            params["expiration_date"] = expiry
        
        resp = requests.get(url, params=params)
        data = resp.json()
        
        if "results" not in data:
            return pd.DataFrame()
        
        return pd.DataFrame(data["results"])


# ============================================================================
# RSI CALCULATOR
# ============================================================================

class RSICalculator:
    """Efficient RSI calculation with rolling updates"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.prices = []
        self.gains = []
        self.losses = []
        self.avg_gain = None
        self.avg_loss = None
    
    def update(self, price: float) -> float:
        """Update with new price, return current RSI"""
        self.prices.append(price)
        
        if len(self.prices) < 2:
            return 50.0
        
        change = self.prices[-1] - self.prices[-2]
        gain = max(change, 0)
        loss = max(-change, 0)
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) < self.period:
            return 50.0
        
        if self.avg_gain is None:
            # Initial SMA
            self.avg_gain = np.mean(self.gains[-self.period:])
            self.avg_loss = np.mean(self.losses[-self.period:])
        else:
            # Wilder's smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) + gain) / self.period
            self.avg_loss = (self.avg_loss * (self.period - 1) + loss) / self.period
        
        if self.avg_loss == 0:
            return 100.0
        
        rs = self.avg_gain / self.avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def reset(self):
        """Reset calculator for new session"""
        self.prices = []
        self.gains = []
        self.losses = []
        self.avg_gain = None
        self.avg_loss = None


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class MeanReversionSignalGenerator:
    """Generate trading signals based on mean reversion patterns"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.fetcher = PolygonRealtime(config.POLYGON_API_KEY)
        self.rsi_calc = RSICalculator(config.RSI_PERIOD)
        self.state = MarketState()
        self.signals: List[Signal] = []
        self.in_position = False
        self.current_signal: Optional[Signal] = None
    
    def initialize_state(self) -> MarketState:
        """Initialize market state from historical data"""
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        
        # Get minute data
        minute_df = self.fetcher.get_aggregates(
            self.config.SYMBOL, 1, "minute", start_date, end_date
        )
        
        if len(minute_df) == 0:
            print("Warning: No historical data available")
            return self.state
        
        # Find sessions
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        
        # Yesterday's regular session
        yesterday_data = minute_df[minute_df.index.date == yesterday]
        yesterday_regular = yesterday_data[
            (yesterday_data.index.time >= dt_time(9, 30)) &
            (yesterday_data.index.time < dt_time(16, 0))
        ]
        
        if len(yesterday_regular) > 0:
            self.state.prev_day_close = yesterday_regular["close"].iloc[-1]
            self.state.prev_day_high = yesterday_regular["high"].max()
            self.state.prev_day_low = yesterday_regular["low"].min()
            prev_open = yesterday_regular["open"].iloc[0]
            self.state.prev_day_return = (self.state.prev_day_close / prev_open - 1) * 100
        
        # Post-market (yesterday 16:00-20:00)
        postmarket = yesterday_data[
            (yesterday_data.index.time >= dt_time(16, 0)) &
            (yesterday_data.index.time < dt_time(20, 0))
        ]
        if len(postmarket) > 0 and self.state.prev_day_close > 0:
            self.state.postmarket_close = postmarket["close"].iloc[-1]
            self.state.postmarket_return = (
                self.state.postmarket_close / self.state.prev_day_close - 1
            ) * 100
        
        # Today's data
        today_data = minute_df[minute_df.index.date == today]
        
        # Pre-market (04:00-09:30)
        premarket = today_data[
            (today_data.index.time >= dt_time(4, 0)) &
            (today_data.index.time < dt_time(9, 30))
        ]
        if len(premarket) > 0:
            ref_price = self.state.postmarket_close or self.state.prev_day_close
            if ref_price > 0:
                self.state.premarket_close = premarket["close"].iloc[-1]
                self.state.premarket_high = premarket["high"].max()
                self.state.premarket_low = premarket["low"].min()
                self.state.premarket_return = (
                    self.state.premarket_close / ref_price - 1
                ) * 100
        
        # Regular hours today
        today_regular = today_data[
            (today_data.index.time >= dt_time(9, 30)) &
            (today_data.index.time < dt_time(16, 0))
        ]
        
        if len(today_regular) > 0:
            self.state.today_open = today_regular["open"].iloc[0]
            self.state.today_high = today_regular["high"].max()
            self.state.today_low = today_regular["low"].min()
            self.state.current_price = today_regular["close"].iloc[-1]
            
            if self.state.prev_day_close > 0:
                self.state.gap_return = (
                    self.state.today_open / self.state.prev_day_close - 1
                ) * 100
            
            # First 5 minutes
            first_5min = today_regular[today_regular.index.time <= dt_time(9, 35)]
            if len(first_5min) > 0:
                self.state.first_5min_close = first_5min["close"].iloc[-1]
                self.state.first_5min_return = (
                    self.state.first_5min_close / self.state.today_open - 1
                ) * 100
            
            # Current day return
            self.state.current_day_return = (
                self.state.current_price / self.state.today_open - 1
            ) * 100
            
            # Initialize RSI
            for price in today_regular["close"].values:
                self.state.rsi = self.rsi_calc.update(price)
            
            # VWAP
            self.state.vwap = today_regular["vwap"].iloc[-1]
        
        # Total extended hours return
        self.state.total_extended_return = (
            (self.state.postmarket_return or 0) +
            (self.state.overnight_return or 0) +
            (self.state.premarket_return or 0)
        )
        
        # Key levels
        self.state.key_resistance = [
            self.state.prev_day_high,
            self.state.premarket_high
        ]
        self.state.key_support = [
            self.state.prev_day_low,
            self.state.premarket_low
        ]
        
        self.state.timestamp = datetime.now()
        
        return self.state
    
    def update_price(self, price: float, timestamp: datetime = None):
        """Update with new price tick"""
        self.state.current_price = price
        self.state.timestamp = timestamp or datetime.now()
        
        # Update RSI
        self.state.rsi = self.rsi_calc.update(price)
        
        # Update intraday high/low
        if price > self.state.today_high:
            self.state.today_high = price
        if price < self.state.today_low:
            self.state.today_low = price
        
        # Update day return
        if self.state.today_open > 0:
            self.state.current_day_return = (
                price / self.state.today_open - 1
            ) * 100
    
    def evaluate_signals(self) -> Optional[Signal]:
        """Evaluate current state and generate signal if warranted"""
        
        current_time = self.state.timestamp.time() if self.state.timestamp else datetime.now().time()
        
        # Only trade during regular hours
        if current_time < dt_time(9, 35) or current_time >= dt_time(15, 50):
            return None
        
        signal = None
        
        # ===== STRATEGY 1: Gap Fade (First 30 min) =====
        if current_time <= dt_time(10, 0):
            signal = self._evaluate_gap_fade()
        
        # ===== STRATEGY 2: RSI Mean Reversion (All day) =====
        if signal is None:
            signal = self._evaluate_rsi_reversal()
        
        # ===== STRATEGY 3: Extended Hours Reversal =====
        if signal is None and current_time <= dt_time(10, 30):
            signal = self._evaluate_extended_reversal()
        
        if signal:
            self.signals.append(signal)
        
        return signal
    
    def _evaluate_gap_fade(self) -> Optional[Signal]:
        """Gap fade strategy for large gaps"""
        
        gap = self.state.gap_return
        
        if abs(gap) < self.config.LARGE_GAP_THRESHOLD:
            return None
        
        # Large gap up - look for fade
        if gap > self.config.LARGE_GAP_THRESHOLD:
            # Confirm weakness (first 5 min negative or flat)
            if self.state.first_5min_return > 0.5:
                return None  # Momentum continuing, don't fade
            
            entry = self.state.current_price
            stop = self.state.today_high * 1.005  # Just above HOD
            target = self.state.prev_day_close + (self.state.today_open - self.state.prev_day_close) * 0.5
            
            return Signal(
                timestamp=self.state.timestamp,
                signal_type=SignalType.SHORT,
                strength=SignalStrength.STRONG if gap > 3 else SignalStrength.MODERATE,
                reason=f"Gap fade: +{gap:.1f}% gap, first 5min weak ({self.state.first_5min_return:.1f}%)",
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                options_strategy="Buy ATM Put or Bear Put Spread",
                strike_recommendation=f"ATM put at ${round(entry, 0)}",
                expiry_recommendation="0DTE",
                position_size_suggestion=self._calculate_position_size(entry, stop),
                risk_reward_ratio=abs(entry - target) / abs(stop - entry),
                confidence_pct=70 if gap > 3 else 60
            )
        
        # Large gap down - look for bounce
        elif gap < -self.config.LARGE_GAP_THRESHOLD:
            # Confirm support holding
            if self.state.first_5min_return < -0.5:
                return None  # Selling continuing
            
            entry = self.state.current_price
            stop = self.state.today_low * 0.995
            target = self.state.prev_day_close - (self.state.prev_day_close - self.state.today_open) * 0.5
            
            return Signal(
                timestamp=self.state.timestamp,
                signal_type=SignalType.LONG,
                strength=SignalStrength.STRONG if gap < -3 else SignalStrength.MODERATE,
                reason=f"Gap fade: {gap:.1f}% gap, first 5min stabilizing ({self.state.first_5min_return:.1f}%)",
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                options_strategy="Buy ATM Call or Bull Call Spread",
                strike_recommendation=f"ATM call at ${round(entry, 0)}",
                expiry_recommendation="0DTE",
                position_size_suggestion=self._calculate_position_size(entry, stop),
                risk_reward_ratio=abs(target - entry) / abs(entry - stop),
                confidence_pct=70 if gap < -3 else 60
            )
        
        return None
    
    def _evaluate_rsi_reversal(self) -> Optional[Signal]:
        """RSI-based mean reversion"""
        
        rsi = self.state.rsi
        
        # Oversold bounce
        if rsi < self.config.RSI_OVERSOLD:
            # Wait for RSI to start turning up
            # In real-time, we'd track previous RSI
            entry = self.state.current_price
            stop = self.state.today_low * 0.995
            target = entry * 1.015  # 1.5% target
            
            strength = SignalStrength.STRONG if rsi < self.config.RSI_EXTREME_OVERSOLD else SignalStrength.MODERATE
            
            return Signal(
                timestamp=self.state.timestamp,
                signal_type=SignalType.LONG,
                strength=strength,
                reason=f"RSI oversold bounce: RSI={rsi:.1f}",
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                options_strategy="Buy ATM Call" if strength == SignalStrength.STRONG else "Bull Call Spread",
                strike_recommendation=f"ATM call at ${round(entry, 0)}",
                expiry_recommendation="0DTE",
                position_size_suggestion=self._calculate_position_size(entry, stop),
                risk_reward_ratio=abs(target - entry) / abs(entry - stop),
                confidence_pct=65 if rsi < 20 else 55
            )
        
        # Overbought reversal
        elif rsi > self.config.RSI_OVERBOUGHT:
            entry = self.state.current_price
            stop = self.state.today_high * 1.005
            target = entry * 0.985
            
            strength = SignalStrength.STRONG if rsi > self.config.RSI_EXTREME_OVERBOUGHT else SignalStrength.MODERATE
            
            return Signal(
                timestamp=self.state.timestamp,
                signal_type=SignalType.SHORT,
                strength=strength,
                reason=f"RSI overbought reversal: RSI={rsi:.1f}",
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                options_strategy="Buy ATM Put" if strength == SignalStrength.STRONG else "Bear Put Spread",
                strike_recommendation=f"ATM put at ${round(entry, 0)}",
                expiry_recommendation="0DTE",
                position_size_suggestion=self._calculate_position_size(entry, stop),
                risk_reward_ratio=abs(entry - target) / abs(stop - entry),
                confidence_pct=65 if rsi > 80 else 55
            )
        
        return None
    
    def _evaluate_extended_reversal(self) -> Optional[Signal]:
        """Trade against large extended hours moves"""
        
        extended_move = self.state.total_extended_return
        
        if abs(extended_move) < self.config.EXTENDED_MOVE_THRESHOLD:
            return None
        
        # Large overnight up move - expect reversion
        if extended_move > self.config.EXTENDED_MOVE_THRESHOLD:
            # Confirm not breaking out further
            if self.state.current_price > self.state.premarket_high:
                return None
            
            entry = self.state.current_price
            stop = self.state.premarket_high * 1.005
            target = self.state.prev_day_close + extended_move * 0.5 / 100 * self.state.prev_day_close
            
            return Signal(
                timestamp=self.state.timestamp,
                signal_type=SignalType.SHORT,
                strength=SignalStrength.MODERATE,
                reason=f"Extended hours reversal: +{extended_move:.1f}% overnight, failing premarket high",
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                options_strategy="Bear Put Spread",
                strike_recommendation=f"Buy ${round(entry, 0)} put, sell ${round(entry * 0.97, 0)} put",
                expiry_recommendation="0DTE or 1DTE",
                position_size_suggestion=self._calculate_position_size(entry, stop),
                risk_reward_ratio=abs(entry - target) / abs(stop - entry),
                confidence_pct=55
            )
        
        # Large overnight down move - expect bounce
        elif extended_move < -self.config.EXTENDED_MOVE_THRESHOLD:
            if self.state.current_price < self.state.premarket_low:
                return None
            
            entry = self.state.current_price
            stop = self.state.premarket_low * 0.995
            target = self.state.prev_day_close - abs(extended_move) * 0.5 / 100 * self.state.prev_day_close
            
            return Signal(
                timestamp=self.state.timestamp,
                signal_type=SignalType.LONG,
                strength=SignalStrength.MODERATE,
                reason=f"Extended hours reversal: {extended_move:.1f}% overnight, holding premarket low",
                entry_price=entry,
                stop_loss=stop,
                target_price=target,
                options_strategy="Bull Call Spread",
                strike_recommendation=f"Buy ${round(entry, 0)} call, sell ${round(entry * 1.03, 0)} call",
                expiry_recommendation="0DTE or 1DTE",
                position_size_suggestion=self._calculate_position_size(entry, stop),
                risk_reward_ratio=abs(target - entry) / abs(entry - stop),
                confidence_pct=55
            )
        
        return None
    
    def _calculate_position_size(self, entry: float, stop: float) -> str:
        """Calculate suggested position size based on risk"""
        risk_per_share = abs(entry - stop)
        max_risk = self.config.MAX_POSITION_SIZE_USD * 0.02  # 2% risk
        shares = int(max_risk / risk_per_share) if risk_per_share > 0 else 0
        contracts = max(1, shares // 100)  # Options contracts
        
        return f"{contracts} contracts (≈{shares} share equivalent)"
    
    def get_state_summary(self) -> Dict:
        """Get current market state summary"""
        return {
            "timestamp": self.state.timestamp.isoformat() if self.state.timestamp else None,
            "price": self.state.current_price,
            "rsi": round(self.state.rsi, 1),
            "session_returns": {
                "prev_day": round(self.state.prev_day_return, 2),
                "postmarket": round(self.state.postmarket_return, 2) if self.state.postmarket_return else None,
                "premarket": round(self.state.premarket_return, 2) if self.state.premarket_return else None,
                "gap": round(self.state.gap_return, 2) if self.state.gap_return else None,
                "total_extended": round(self.state.total_extended_return, 2),
                "first_5min": round(self.state.first_5min_return, 2) if self.state.first_5min_return else None,
                "current_day": round(self.state.current_day_return, 2)
            },
            "levels": {
                "prev_close": self.state.prev_day_close,
                "prev_high": self.state.prev_day_high,
                "prev_low": self.state.prev_day_low,
                "today_open": self.state.today_open,
                "today_high": self.state.today_high,
                "today_low": self.state.today_low,
                "vwap": round(self.state.vwap, 2) if self.state.vwap else None
            },
            "active_signals": len([s for s in self.signals if s.timestamp.date() == datetime.now().date()])
        }


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Backtest mean reversion signals"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.fetcher = PolygonRealtime(config.POLYGON_API_KEY)
    
    def run_backtest(
        self, 
        start_date: str, 
        end_date: str,
        verbose: bool = True
    ) -> Dict:
        """Run backtest over date range"""
        
        if verbose:
            print(f"Running backtest from {start_date} to {end_date}")
        
        # Get data
        minute_df = self.fetcher.get_aggregates(
            self.config.SYMBOL, 1, "minute", start_date, end_date
        )
        
        if len(minute_df) == 0:
            return {"error": "No data"}
        
        trades = []
        dates = minute_df.index.date
        unique_dates = pd.unique(dates)
        
        for date in unique_dates:
            day_data = minute_df[minute_df.index.date == date]
            regular = day_data[
                (day_data.index.time >= dt_time(9, 30)) &
                (day_data.index.time < dt_time(16, 0))
            ]
            
            if len(regular) < 30:
                continue
            
            # Calculate RSI for the day
            rsi_calc = RSICalculator(14)
            for _, row in regular.iterrows():
                rsi = rsi_calc.update(row["close"])
            
            # Simulate signals
            day_trades = self._simulate_day(regular, date)
            trades.extend(day_trades)
        
        # Calculate stats
        if len(trades) == 0:
            return {"error": "No trades generated"}
        
        trades_df = pd.DataFrame(trades)
        
        stats = {
            "total_trades": len(trades_df),
            "win_rate": (trades_df["pnl_pct"] > 0).mean() * 100,
            "avg_pnl": trades_df["pnl_pct"].mean(),
            "total_pnl": trades_df["pnl_pct"].sum(),
            "max_win": trades_df["pnl_pct"].max(),
            "max_loss": trades_df["pnl_pct"].min(),
            "avg_duration_mins": trades_df["duration_mins"].mean(),
            "sharpe": (
                trades_df["pnl_pct"].mean() / trades_df["pnl_pct"].std() 
                if trades_df["pnl_pct"].std() > 0 else 0
            )
        }
        
        # By signal type
        for signal_type in trades_df["signal_type"].unique():
            subset = trades_df[trades_df["signal_type"] == signal_type]
            stats[f"{signal_type}_trades"] = len(subset)
            stats[f"{signal_type}_win_rate"] = (subset["pnl_pct"] > 0).mean() * 100
            stats[f"{signal_type}_avg_pnl"] = subset["pnl_pct"].mean()
        
        if verbose:
            print(f"\n{'='*50}")
            print("BACKTEST RESULTS")
            print(f"{'='*50}")
            print(f"Total trades: {stats['total_trades']}")
            print(f"Win rate: {stats['win_rate']:.1f}%")
            print(f"Average P&L: {stats['avg_pnl']:.2f}%")
            print(f"Total P&L: {stats['total_pnl']:.2f}%")
            print(f"Sharpe: {stats['sharpe']:.2f}")
            print(f"\nBy signal type:")
            for signal_type in trades_df["signal_type"].unique():
                print(f"  {signal_type}: {stats[f'{signal_type}_trades']} trades, "
                      f"{stats[f'{signal_type}_win_rate']:.1f}% win rate, "
                      f"{stats[f'{signal_type}_avg_pnl']:.2f}% avg")
        
        return {
            "stats": stats,
            "trades": trades_df.to_dict("records")
        }
    
    def _simulate_day(self, day_data: pd.DataFrame, date) -> List[Dict]:
        """Simulate trading for a single day"""
        
        trades = []
        rsi_calc = RSICalculator(14)
        in_position = False
        entry = None
        
        # Pre-calculate day stats
        open_price = day_data["open"].iloc[0]
        
        for i, (ts, row) in enumerate(day_data.iterrows()):
            rsi = rsi_calc.update(row["close"])
            
            if i < 14:  # Wait for RSI warmup
                continue
            
            current_time = ts.time()
            
            # Exit at end of day
            if current_time >= dt_time(15, 55) and in_position:
                pnl = (row["close"] - entry["price"]) / entry["price"] * 100
                if entry["direction"] == "short":
                    pnl = -pnl
                
                trades.append({
                    "date": date,
                    "signal_type": entry["signal_type"],
                    "direction": entry["direction"],
                    "entry_time": entry["time"],
                    "entry_price": entry["price"],
                    "exit_time": current_time,
                    "exit_price": row["close"],
                    "exit_reason": "eod",
                    "pnl_pct": pnl,
                    "duration_mins": (ts - entry["timestamp"]).total_seconds() / 60
                })
                in_position = False
                entry = None
                continue
            
            # Entry signals
            if not in_position:
                # RSI oversold
                if rsi < self.config.RSI_OVERSOLD:
                    entry = {
                        "signal_type": "rsi_oversold",
                        "direction": "long",
                        "time": current_time,
                        "price": row["close"],
                        "rsi": rsi,
                        "timestamp": ts
                    }
                    in_position = True
                
                # RSI overbought
                elif rsi > self.config.RSI_OVERBOUGHT:
                    entry = {
                        "signal_type": "rsi_overbought",
                        "direction": "short",
                        "time": current_time,
                        "price": row["close"],
                        "rsi": rsi,
                        "timestamp": ts
                    }
                    in_position = True
            
            # Exit signals
            elif in_position:
                should_exit = False
                exit_reason = None
                
                # RSI normalized
                if entry["direction"] == "long" and rsi >= 50:
                    should_exit = True
                    exit_reason = "rsi_target"
                elif entry["direction"] == "short" and rsi <= 50:
                    should_exit = True
                    exit_reason = "rsi_target"
                
                # Stop loss (RSI goes more extreme)
                if entry["direction"] == "long" and rsi < self.config.RSI_EXTREME_OVERSOLD:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif entry["direction"] == "short" and rsi > self.config.RSI_EXTREME_OVERBOUGHT:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                if should_exit:
                    pnl = (row["close"] - entry["price"]) / entry["price"] * 100
                    if entry["direction"] == "short":
                        pnl = -pnl
                    
                    trades.append({
                        "date": date,
                        "signal_type": entry["signal_type"],
                        "direction": entry["direction"],
                        "entry_time": entry["time"],
                        "entry_price": entry["price"],
                        "exit_time": current_time,
                        "exit_price": row["close"],
                        "exit_reason": exit_reason,
                        "pnl_pct": pnl,
                        "duration_mins": (ts - entry["timestamp"]).total_seconds() / 60
                    })
                    in_position = False
                    entry = None
        
        return trades


# ============================================================================
# MAIN RUNNER
# ============================================================================

def print_signal_alert(signal: Signal):
    """Print formatted signal alert"""
    
    color = "\033[92m" if signal.signal_type == SignalType.LONG else "\033[91m"
    reset = "\033[0m"
    
    print(f"\n{'='*60}")
    print(f"{color}🚨 SIGNAL: {signal.signal_type.value} {signal.strength.name}{reset}")
    print(f"{'='*60}")
    print(f"Time: {signal.timestamp}")
    print(f"Reason: {signal.reason}")
    print(f"\n📊 Trade Setup:")
    print(f"  Entry: ${signal.entry_price:.2f}")
    print(f"  Stop: ${signal.stop_loss:.2f}")
    print(f"  Target: ${signal.target_price:.2f}")
    print(f"  R:R Ratio: {signal.risk_reward_ratio:.2f}")
    print(f"\n📋 Options Play:")
    print(f"  Strategy: {signal.options_strategy}")
    print(f"  Strike: {signal.strike_recommendation}")
    print(f"  Expiry: {signal.expiry_recommendation}")
    print(f"  Size: {signal.position_size_suggestion}")
    print(f"\n  Confidence: {signal.confidence_pct}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="APP Mean Reversion Signal Generator")
    parser.add_argument("--api-key", type=str, help="Polygon API key")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2024-03-01", help="Backtest end date")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")
    
    args = parser.parse_args()
    
    config = TradingConfig()
    if args.api_key:
        config.POLYGON_API_KEY = args.api_key
    
    if args.backtest:
        print("Running backtest mode...")
        engine = BacktestEngine(config)
        results = engine.run_backtest(args.start, args.end)
        
    elif args.live:
        print("Initializing live signal generator...")
        generator = MeanReversionSignalGenerator(config)
        
        print("Loading market state...")
        state = generator.initialize_state()
        
        print("\n📊 Current Market State:")
        summary = generator.get_state_summary()
        print(json.dumps(summary, indent=2))
        
        print("\nStarting signal monitoring (Ctrl+C to stop)...")
        
        try:
            while True:
                # In production, you'd update from real-time feed
                signal = generator.evaluate_signals()
                
                if signal:
                    print_signal_alert(signal)
                
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\nStopping signal generator...")
            
    else:  # Demo mode
        print("Running demo mode with sample signal generation...\n")
        
        generator = MeanReversionSignalGenerator(config)
        
        # Simulate market state
        generator.state = MarketState(
            timestamp=datetime.now(),
            current_price=285.50,
            prev_day_close=280.00,
            prev_day_high=282.00,
            prev_day_low=277.50,
            postmarket_close=281.00,
            premarket_close=284.00,
            premarket_high=285.00,
            premarket_low=280.50,
            today_open=285.00,
            today_high=287.50,
            today_low=283.00,
            first_5min_close=284.50,
            prev_day_return=1.2,
            postmarket_return=0.36,
            overnight_return=0.5,
            premarket_return=1.07,
            gap_return=1.79,
            total_extended_return=1.93,
            first_5min_return=-0.18,
            current_day_return=0.18,
            rsi=72.5,  # Overbought
            vwap=285.20
        )
        
        print("📊 Demo Market State:")
        print(json.dumps(generator.get_state_summary(), indent=2))
        
        # Evaluate signals
        signal = generator.evaluate_signals()
        
        if signal:
            print_signal_alert(signal)
        else:
            print("\n✅ No signal at current levels")
            print("\nWaiting for:")
            print(f"  - RSI > {config.RSI_OVERBOUGHT} or < {config.RSI_OVERSOLD}")
            print(f"  - Gap > {config.LARGE_GAP_THRESHOLD}%")
            print(f"  - Extended move > {config.EXTENDED_MOVE_THRESHOLD}%")


if __name__ == "__main__":
    main()
