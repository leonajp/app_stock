"""
APP Stock Mean Reversion Analysis
=================================
Analyzes session-based predictors and RSI mean reversion patterns for intraday trading.

Sessions:
- Previous Day: 9:30-16:00 prior day
- Post-Market: 16:00-20:00 prior day  
- Overnight: 20:00-04:00
- Pre-Market: 04:00-09:30
- First 5 Min: 09:30-09:35
- Regular Hours: 09:30-16:00

Author: Limestone Hill Capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    POLYGON_API_KEY: str = "YOUR_POLYGON_API_KEY"  # Replace with your key
    SYMBOL: str = "APP"
    LOOKBACK_DAYS: int = 90  # Days of history to analyze
    
    # Session times (ET)
    REGULAR_OPEN: str = "09:30"
    REGULAR_CLOSE: str = "16:00"
    POSTMARKET_CLOSE: str = "20:00"
    PREMARKET_OPEN: str = "04:00"
    
    # RSI parameters
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    RSI_EXTREME_OVERSOLD: float = 20.0
    RSI_EXTREME_OVERBOUGHT: float = 80.0
    
    # Mean reversion thresholds
    REVERSION_TARGET_PCT: float = 0.5  # 50% reversion of the move


# ============================================================================
# DATA FETCHING - POLYGON
# ============================================================================

class PolygonDataFetcher:
    """Fetch intraday and daily data from Polygon.io"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    def get_aggregates(
        self, 
        symbol: str, 
        multiplier: int, 
        timespan: str, 
        from_date: str, 
        to_date: str,
        adjusted: bool = True,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Fetch aggregate bars from Polygon.
        
        timespan: 'minute', 'hour', 'day'
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": limit,
            "apiKey": self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "results" not in data:
            print(f"Warning: No data returned. Response: {data}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data["results"])
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low", 
            "c": "close", "v": "volume", "vw": "vwap"
        })
        df = df.set_index("timestamp")
        
        # Convert to ET
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        
        return df[["open", "high", "low", "close", "volume", "vwap"]]
    
    def get_minute_data(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch 1-minute bars including extended hours"""
        return self.get_aggregates(symbol, 1, "minute", from_date, to_date)
    
    def get_daily_data(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch daily bars"""
        return self.get_aggregates(symbol, 1, "day", from_date, to_date)


# ============================================================================
# SESSION DECOMPOSITION
# ============================================================================

class SessionAnalyzer:
    """Decompose price action into trading sessions"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def classify_session(self, timestamp: pd.Timestamp) -> str:
        """Classify a timestamp into a trading session"""
        time = timestamp.time()
        
        from datetime import time as dt_time
        
        regular_open = dt_time(9, 30)
        regular_close = dt_time(16, 0)
        postmarket_close = dt_time(20, 0)
        premarket_open = dt_time(4, 0)
        
        if regular_open <= time < regular_close:
            return "regular"
        elif regular_close <= time < postmarket_close:
            return "postmarket"
        elif time >= postmarket_close or time < premarket_open:
            return "overnight"
        else:  # premarket_open <= time < regular_open
            return "premarket"
    
    def calculate_session_returns(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns for each session.
        
        Returns DataFrame with columns:
        - date
        - prev_day_return: Previous day's regular session return
        - postmarket_return: Post-market return (16:00-20:00)
        - overnight_return: Overnight return (20:00-04:00)
        - premarket_return: Pre-market return (04:00-09:30)
        - first_5min_return: First 5 minutes return (09:30-09:35)
        - gap_return: Open vs previous close
        - day_return: Full day return
        """
        
        # Add session classification
        df = minute_df.copy()
        df["session"] = df.index.map(self.classify_session)
        df["date"] = df.index.date
        df["time"] = df.index.time
        
        results = []
        dates = df["date"].unique()
        
        from datetime import time as dt_time
        
        for i, date in enumerate(dates):
            day_data = df[df["date"] == date]
            
            # Get previous trading day
            prev_date = dates[i-1] if i > 0 else None
            prev_day_data = df[df["date"] == prev_date] if prev_date else None
            
            record = {"date": date}
            
            # Previous day regular session return
            if prev_day_data is not None:
                prev_regular = prev_day_data[prev_day_data["session"] == "regular"]
                if len(prev_regular) > 0:
                    record["prev_day_return"] = (
                        prev_regular["close"].iloc[-1] / prev_regular["open"].iloc[0] - 1
                    ) * 100
                    record["prev_day_close"] = prev_regular["close"].iloc[-1]
                    record["prev_day_high"] = prev_regular["high"].max()
                    record["prev_day_low"] = prev_regular["low"].min()
            
            # Post-market return (from prev day)
            if prev_day_data is not None:
                postmarket = prev_day_data[prev_day_data["session"] == "postmarket"]
                if len(postmarket) > 0 and "prev_day_close" in record:
                    record["postmarket_return"] = (
                        postmarket["close"].iloc[-1] / record["prev_day_close"] - 1
                    ) * 100
                    record["postmarket_close"] = postmarket["close"].iloc[-1]
            
            # Overnight return
            overnight = day_data[day_data["session"] == "overnight"]
            if len(overnight) > 0:
                ref_price = record.get("postmarket_close", record.get("prev_day_close"))
                if ref_price:
                    record["overnight_return"] = (
                        overnight["close"].iloc[-1] / ref_price - 1
                    ) * 100
                    record["overnight_close"] = overnight["close"].iloc[-1]
            
            # Pre-market return
            premarket = day_data[day_data["session"] == "premarket"]
            if len(premarket) > 0:
                ref_price = record.get("overnight_close", 
                           record.get("postmarket_close", 
                           record.get("prev_day_close")))
                if ref_price:
                    record["premarket_return"] = (
                        premarket["close"].iloc[-1] / ref_price - 1
                    ) * 100
                    record["premarket_close"] = premarket["close"].iloc[-1]
                    record["premarket_high"] = premarket["high"].max()
                    record["premarket_low"] = premarket["low"].min()
            
            # Regular session
            regular = day_data[day_data["session"] == "regular"]
            if len(regular) > 0:
                record["open"] = regular["open"].iloc[0]
                record["close"] = regular["close"].iloc[-1]
                record["high"] = regular["high"].max()
                record["low"] = regular["low"].min()
                record["volume"] = regular["volume"].sum()
                
                # Gap return (open vs prev close)
                if "prev_day_close" in record:
                    record["gap_return"] = (
                        record["open"] / record["prev_day_close"] - 1
                    ) * 100
                
                # First 5 minutes return
                first_5min = regular[regular["time"] <= dt_time(9, 35)]
                if len(first_5min) > 0:
                    record["first_5min_return"] = (
                        first_5min["close"].iloc[-1] / first_5min["open"].iloc[0] - 1
                    ) * 100
                    record["first_5min_close"] = first_5min["close"].iloc[-1]
                    record["first_5min_high"] = first_5min["high"].max()
                    record["first_5min_low"] = first_5min["low"].min()
                
                # First 30 minutes
                first_30min = regular[regular["time"] <= dt_time(10, 0)]
                if len(first_30min) > 0:
                    record["first_30min_return"] = (
                        first_30min["close"].iloc[-1] / first_30min["open"].iloc[0] - 1
                    ) * 100
                
                # Full day return
                record["day_return"] = (
                    record["close"] / record["open"] - 1
                ) * 100
            
            results.append(record)
        
        return pd.DataFrame(results)


# ============================================================================
# SIGNAL ANALYSIS
# ============================================================================

class SignalAnalyzer:
    """Analyze predictive signals for mean reversion"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_session_predictors(self, session_df: pd.DataFrame) -> Dict:
        """
        Analyze which session returns predict first 5-min movement.
        
        Tests:
        1. Correlation analysis
        2. Directional accuracy (does negative premarket predict positive first 5min?)
        3. Magnitude analysis
        """
        
        df = session_df.dropna(subset=["first_5min_return"])
        
        predictors = [
            "prev_day_return", "postmarket_return", "overnight_return", 
            "premarket_return", "gap_return"
        ]
        
        results = {}
        
        for predictor in predictors:
            if predictor not in df.columns:
                continue
                
            valid = df.dropna(subset=[predictor])
            if len(valid) < 10:
                continue
            
            x = valid[predictor].values
            y = valid["first_5min_return"].values
            
            # Correlation
            corr, p_value = stats.pearsonr(x, y)
            
            # Directional accuracy (contrarian signal)
            # If predictor is negative, do we get positive first 5min?
            contrarian_correct = np.sum(
                (x < 0) & (y > 0) | (x > 0) & (y < 0)
            ) / len(x)
            
            # Same direction (momentum signal)
            momentum_correct = np.sum(
                (x < 0) & (y < 0) | (x > 0) & (y > 0)
            ) / len(x)
            
            # Mean reversion: average first_5min return when predictor is extreme
            extreme_negative = valid[valid[predictor] < -1]["first_5min_return"]
            extreme_positive = valid[valid[predictor] > 1]["first_5min_return"]
            
            results[predictor] = {
                "correlation": corr,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "contrarian_accuracy": contrarian_correct,
                "momentum_accuracy": momentum_correct,
                "signal_type": "contrarian" if contrarian_correct > 0.5 else "momentum",
                "avg_first5min_after_negative": extreme_negative.mean() if len(extreme_negative) > 0 else None,
                "avg_first5min_after_positive": extreme_positive.mean() if len(extreme_positive) > 0 else None,
                "n_samples": len(valid)
            }
        
        return results
    
    def analyze_combined_signal(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create combined signal from multiple session returns.
        
        Signal = weighted sum of session returns, flipped if contrarian.
        """
        
        df = session_df.copy()
        
        # Composite gap signal (all extended hours moves)
        df["total_extended_move"] = (
            df["postmarket_return"].fillna(0) + 
            df["overnight_return"].fillna(0) + 
            df["premarket_return"].fillna(0)
        )
        
        # Signal strength (absolute value indicates conviction)
        df["signal_strength"] = df["total_extended_move"].abs()
        
        # Contrarian signal: expect reversion when extended move is large
        df["mean_reversion_signal"] = -df["total_extended_move"]
        
        return df


# ============================================================================
# RSI MEAN REVERSION
# ============================================================================

class RSIMeanReversion:
    """RSI-based intraday mean reversion detection"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def find_rsi_reversals(
        self, 
        minute_df: pd.DataFrame,
        rsi_period: int = 14
    ) -> pd.DataFrame:
        """
        Find RSI reversal points during regular trading hours.
        
        Returns DataFrame with:
        - Entry point (RSI extreme)
        - Exit point (RSI normalized or target hit)
        - P&L
        """
        
        df = minute_df.copy()
        df["rsi"] = self.calculate_rsi(df["close"], rsi_period)
        
        # Filter to regular hours only
        from datetime import time as dt_time
        df = df[(df.index.time >= dt_time(9, 30)) & (df.index.time < dt_time(16, 0))]
        
        signals = []
        in_position = False
        entry = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Skip if RSI not calculated yet
            if pd.isna(row["rsi"]):
                continue
            
            # Entry signals
            if not in_position:
                # Oversold bounce
                if prev_row["rsi"] < self.config.RSI_OVERSOLD and row["rsi"] >= self.config.RSI_OVERSOLD:
                    entry = {
                        "entry_time": row.name,
                        "entry_price": row["close"],
                        "entry_rsi": row["rsi"],
                        "direction": "long",
                        "signal": "oversold_bounce"
                    }
                    in_position = True
                
                # Overbought reversal
                elif prev_row["rsi"] > self.config.RSI_OVERBOUGHT and row["rsi"] <= self.config.RSI_OVERBOUGHT:
                    entry = {
                        "entry_time": row.name,
                        "entry_price": row["close"],
                        "entry_rsi": row["rsi"],
                        "direction": "short",
                        "signal": "overbought_reversal"
                    }
                    in_position = True
            
            # Exit signals
            else:
                should_exit = False
                exit_reason = None
                
                # RSI neutralized (reached 50)
                if entry["direction"] == "long" and row["rsi"] >= 50:
                    should_exit = True
                    exit_reason = "rsi_neutral"
                elif entry["direction"] == "short" and row["rsi"] <= 50:
                    should_exit = True
                    exit_reason = "rsi_neutral"
                
                # End of day exit
                if row.name.time() >= dt_time(15, 55):
                    should_exit = True
                    exit_reason = "eod"
                
                # Stop loss (RSI goes more extreme)
                if entry["direction"] == "long" and row["rsi"] < self.config.RSI_EXTREME_OVERSOLD:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif entry["direction"] == "short" and row["rsi"] > self.config.RSI_EXTREME_OVERBOUGHT:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                if should_exit:
                    pnl = (row["close"] - entry["entry_price"]) / entry["entry_price"] * 100
                    if entry["direction"] == "short":
                        pnl = -pnl
                    
                    signals.append({
                        **entry,
                        "exit_time": row.name,
                        "exit_price": row["close"],
                        "exit_rsi": row["rsi"],
                        "exit_reason": exit_reason,
                        "pnl_pct": pnl,
                        "duration_mins": (row.name - entry["entry_time"]).total_seconds() / 60
                    })
                    
                    in_position = False
                    entry = None
        
        return pd.DataFrame(signals)
    
    def analyze_rsi_patterns(self, minute_df: pd.DataFrame) -> Dict:
        """
        Comprehensive RSI pattern analysis.
        """
        
        reversals = self.find_rsi_reversals(minute_df)
        
        if len(reversals) == 0:
            return {"error": "No RSI reversal signals found"}
        
        # Overall stats
        stats_dict = {
            "total_signals": len(reversals),
            "win_rate": (reversals["pnl_pct"] > 0).mean() * 100,
            "avg_pnl": reversals["pnl_pct"].mean(),
            "total_pnl": reversals["pnl_pct"].sum(),
            "avg_duration_mins": reversals["duration_mins"].mean(),
            "sharpe": reversals["pnl_pct"].mean() / reversals["pnl_pct"].std() if reversals["pnl_pct"].std() > 0 else 0
        }
        
        # By signal type
        for signal_type in ["oversold_bounce", "overbought_reversal"]:
            subset = reversals[reversals["signal"] == signal_type]
            if len(subset) > 0:
                stats_dict[f"{signal_type}_count"] = len(subset)
                stats_dict[f"{signal_type}_win_rate"] = (subset["pnl_pct"] > 0).mean() * 100
                stats_dict[f"{signal_type}_avg_pnl"] = subset["pnl_pct"].mean()
        
        # By exit reason
        for reason in reversals["exit_reason"].unique():
            subset = reversals[reversals["exit_reason"] == reason]
            stats_dict[f"exit_{reason}_count"] = len(subset)
            stats_dict[f"exit_{reason}_avg_pnl"] = subset["pnl_pct"].mean()
        
        return {
            "stats": stats_dict,
            "trades": reversals
        }


# ============================================================================
# INTRADAY REVERSION ANALYSIS
# ============================================================================

class IntradayReversionAnalyzer:
    """Analyze how price reverts during the trading day"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def analyze_reversion_from_open(
        self, 
        minute_df: pd.DataFrame,
        session_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze how first 5-min moves revert during the day.
        
        For each day:
        - If first 5 min was up, when does it pull back and by how much?
        - If first 5 min was down, when does it bounce and by how much?
        """
        
        from datetime import time as dt_time
        
        results = []
        
        for _, day_session in session_df.iterrows():
            date = day_session["date"]
            
            if pd.isna(day_session.get("first_5min_return")):
                continue
            
            # Get regular hours data for this day
            day_data = minute_df[
                (minute_df.index.date == date) &
                (minute_df.index.time >= dt_time(9, 30)) &
                (minute_df.index.time < dt_time(16, 0))
            ]
            
            if len(day_data) < 10:
                continue
            
            first_5min_return = day_session["first_5min_return"]
            open_price = day_session["open"]
            first_5min_close = day_session.get("first_5min_close", day_data.iloc[5]["close"])
            
            # Track reversion
            direction = "up" if first_5min_return > 0 else "down"
            
            # Find max adverse excursion and max favorable excursion after first 5 min
            after_5min = day_data[day_data.index.time > dt_time(9, 35)]
            
            if len(after_5min) == 0:
                continue
            
            # Relative to first 5 min close
            after_5min_returns = (after_5min["close"] - first_5min_close) / first_5min_close * 100
            
            if direction == "up":
                # Looking for pullback (negative move from first 5 min high)
                max_pullback = after_5min_returns.min()
                pullback_time = after_5min_returns.idxmin()
                
                # Did it revert past open?
                reverted_to_open = (after_5min["low"].min() < open_price)
                
            else:
                # Looking for bounce (positive move from first 5 min low)
                max_bounce = after_5min_returns.max()
                pullback_time = after_5min_returns.idxmax()
                
                # Did it revert past open?
                reverted_to_open = (after_5min["high"].max() > open_price)
            
            # Calculate full day stats
            day_high_time = day_data["high"].idxmax()
            day_low_time = day_data["low"].idxmin()
            
            results.append({
                "date": date,
                "first_5min_return": first_5min_return,
                "direction": direction,
                "max_reversion": max_bounce if direction == "down" else max_pullback,
                "reversion_time": pullback_time,
                "reverted_to_open": reverted_to_open,
                "day_return": day_session["day_return"],
                "day_high_time": day_high_time.time() if pd.notna(day_high_time) else None,
                "day_low_time": day_low_time.time() if pd.notna(day_low_time) else None,
                "gap_return": day_session.get("gap_return", None),
                "premarket_return": day_session.get("premarket_return", None)
            })
        
        return pd.DataFrame(results)
    
    def calculate_optimal_reversion_levels(self, reversion_df: pd.DataFrame) -> Dict:
        """
        Calculate probability of hitting various reversion levels.
        """
        
        results = {}
        
        for direction in ["up", "down"]:
            subset = reversion_df[reversion_df["direction"] == direction]
            
            if len(subset) < 5:
                continue
            
            # Reversion levels to test
            if direction == "up":
                # After up move, probability of X% pullback
                levels = [-0.5, -1.0, -1.5, -2.0, -2.5]
            else:
                # After down move, probability of X% bounce
                levels = [0.5, 1.0, 1.5, 2.0, 2.5]
            
            level_probs = {}
            for level in levels:
                if direction == "up":
                    hit_rate = (subset["max_reversion"] <= level).mean() * 100
                else:
                    hit_rate = (subset["max_reversion"] >= level).mean() * 100
                level_probs[f"{level}%"] = hit_rate
            
            results[direction] = {
                "n_samples": len(subset),
                "avg_first_5min_move": subset["first_5min_return"].mean(),
                "avg_reversion": subset["max_reversion"].mean(),
                "reversion_hit_rates": level_probs,
                "reverted_to_open_rate": subset["reverted_to_open"].mean() * 100,
                "same_direction_day": (
                    (subset["day_return"] > 0).mean() if direction == "up" 
                    else (subset["day_return"] < 0).mean()
                ) * 100
            }
        
        return results


# ============================================================================
# OPTIONS STRATEGY RECOMMENDATIONS
# ============================================================================

class OptionsStrategyAdvisor:
    """Generate options strategy recommendations based on signals"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def recommend_strategy(
        self,
        signal_strength: float,
        direction: str,
        expected_reversion: float,
        holding_period: str = "intraday"
    ) -> Dict:
        """
        Recommend options strategy based on signal characteristics.
        
        signal_strength: Absolute magnitude of signal (0-10 scale)
        direction: 'long' (expecting up) or 'short' (expecting down)
        expected_reversion: Expected % move
        holding_period: 'intraday', 'overnight', 'weekly'
        """
        
        strategies = []
        
        # High conviction, directional move expected
        if signal_strength > 5 and abs(expected_reversion) > 1.5:
            if direction == "long":
                strategies.append({
                    "strategy": "Buy ATM Call",
                    "rationale": "High conviction bullish signal, maximize delta exposure",
                    "entry": "At open or after first 5-min confirm",
                    "exit": "Target 50% of expected move or RSI neutral",
                    "risk": "Premium paid",
                    "strike_selection": "ATM or slightly OTM (1-2%)",
                    "expiration": "0DTE or 1DTE for intraday"
                })
                
                # Add spread for risk management
                strategies.append({
                    "strategy": "Bull Call Spread",
                    "rationale": "Lower cost alternative, capped upside",
                    "entry": "Buy ATM call, sell OTM call (2-3% higher)",
                    "exit": "Close at 50% profit or end of day",
                    "risk": "Net debit paid",
                    "max_profit": "Spread width - premium"
                })
            else:
                strategies.append({
                    "strategy": "Buy ATM Put",
                    "rationale": "High conviction bearish signal",
                    "entry": "At open or after first 5-min confirm",
                    "exit": "Target 50% of expected move or RSI neutral",
                    "risk": "Premium paid"
                })
                
                strategies.append({
                    "strategy": "Bear Put Spread",
                    "rationale": "Lower cost bearish play",
                    "entry": "Buy ATM put, sell OTM put (2-3% lower)",
                    "exit": "Close at 50% profit or end of day",
                    "risk": "Net debit paid"
                })
        
        # Medium conviction - use spreads
        elif signal_strength > 2.5:
            if direction == "long":
                strategies.append({
                    "strategy": "Bull Put Spread (Credit)",
                    "rationale": "Collect premium, bullish bias with downside buffer",
                    "entry": "Sell OTM put, buy further OTM put",
                    "exit": "Let expire worthless or buy back at 50% profit",
                    "risk": "Spread width - credit received",
                    "ideal_setup": "After oversold bounce confirmation"
                })
            else:
                strategies.append({
                    "strategy": "Bear Call Spread (Credit)",
                    "rationale": "Collect premium, bearish bias with upside buffer",
                    "entry": "Sell OTM call, buy further OTM call",
                    "exit": "Let expire worthless or buy back at 50% profit",
                    "risk": "Spread width - credit received"
                })
        
        # Low conviction - neutral/theta strategies
        else:
            strategies.append({
                "strategy": "Iron Condor",
                "rationale": "Range-bound expectation, collect theta",
                "entry": "Sell OTM put spread + sell OTM call spread",
                "exit": "Close at 50% profit or manage at 2x loss",
                "risk": "Wider spread width - credit",
                "ideal_setup": "After volatile open settles"
            })
            
            strategies.append({
                "strategy": "Butterfly",
                "rationale": "Pin to expected price, limited risk",
                "entry": "Buy wing, sell 2x body, buy wing",
                "exit": "Close before expiration theta decay",
                "risk": "Net debit paid"
            })
        
        return {
            "signal_strength": signal_strength,
            "direction": direction,
            "expected_move": expected_reversion,
            "recommended_strategies": strategies,
            "general_notes": [
                "0DTE options have high gamma - size positions appropriately",
                "Consider IV percentile before buying premium",
                "Use limit orders, not market orders on options",
                "Monitor bid-ask spread - wide spreads hurt P&L"
            ]
        }
    
    def generate_daily_trade_plan(
        self,
        session_data: Dict,
        rsi_analysis: Dict,
        reversion_stats: Dict
    ) -> Dict:
        """
        Generate a complete daily trade plan based on analysis.
        """
        
        plan = {
            "pre_market_checklist": [],
            "scenarios": [],
            "key_levels": {},
            "options_plays": []
        }
        
        # Pre-market checklist
        plan["pre_market_checklist"] = [
            "Check overnight/premarket move magnitude",
            "Note key support/resistance from prior day",
            "Check IV percentile for options pricing",
            "Review any news/earnings catalyst",
            "Set alerts for RSI extremes"
        ]
        
        # Scenario 1: Large gap up
        plan["scenarios"].append({
            "condition": "Gap up > 2%",
            "expected_action": "Mean reversion likely",
            "strategy": "Watch for RSI overbought, buy puts on reversal",
            "target": "50% gap fill",
            "stop": "New high above gap"
        })
        
        # Scenario 2: Large gap down
        plan["scenarios"].append({
            "condition": "Gap down > 2%",
            "expected_action": "Bounce likely",
            "strategy": "Watch for RSI oversold bounce, buy calls",
            "target": "50% gap fill",
            "stop": "New low below gap"
        })
        
        # Scenario 3: Flat open
        plan["scenarios"].append({
            "condition": "Gap < 0.5%",
            "expected_action": "Wait for direction",
            "strategy": "Trade RSI extremes, use spreads",
            "target": "Prior day high/low",
            "stop": "Based on ATR"
        })
        
        # Add specific RSI-based plays
        if rsi_analysis and "stats" in rsi_analysis:
            stats = rsi_analysis["stats"]
            
            if stats.get("oversold_bounce_win_rate", 0) > 55:
                plan["options_plays"].append({
                    "setup": "RSI Oversold Bounce",
                    "trigger": f"RSI crosses above {self.config.RSI_OVERSOLD}",
                    "action": "Buy ATM call or bull call spread",
                    "historical_win_rate": f"{stats.get('oversold_bounce_win_rate', 0):.1f}%",
                    "avg_gain": f"{stats.get('oversold_bounce_avg_pnl', 0):.2f}%"
                })
            
            if stats.get("overbought_reversal_win_rate", 0) > 55:
                plan["options_plays"].append({
                    "setup": "RSI Overbought Reversal",
                    "trigger": f"RSI crosses below {self.config.RSI_OVERBOUGHT}",
                    "action": "Buy ATM put or bear put spread",
                    "historical_win_rate": f"{stats.get('overbought_reversal_win_rate', 0):.1f}%",
                    "avg_gain": f"{stats.get('overbought_reversal_avg_pnl', 0):.2f}%"
                })
        
        return plan


# ============================================================================
# MAIN ANALYSIS RUNNER
# ============================================================================

class APPMeanReversionAnalysis:
    """Main analysis orchestrator"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.fetcher = PolygonDataFetcher(self.config.POLYGON_API_KEY)
        self.session_analyzer = SessionAnalyzer(self.config)
        self.signal_analyzer = SignalAnalyzer(self.config)
        self.rsi_analyzer = RSIMeanReversion(self.config)
        self.reversion_analyzer = IntradayReversionAnalyzer(self.config)
        self.options_advisor = OptionsStrategyAdvisor(self.config)
    
    def run_full_analysis(self, use_sample_data: bool = False) -> Dict:
        """Run complete analysis pipeline"""
        
        print("=" * 60)
        print(f"APP Mean Reversion Analysis")
        print("=" * 60)
        
        # Get data
        if use_sample_data:
            print("\n[Using sample data for demonstration]")
            minute_df, session_df = self._generate_sample_data()
        else:
            print(f"\nFetching {self.config.LOOKBACK_DAYS} days of data from Polygon...")
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=self.config.LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            
            minute_df = self.fetcher.get_minute_data(
                self.config.SYMBOL, start_date, end_date
            )
            
            if len(minute_df) == 0:
                print("Error: No data received. Check API key.")
                return {"error": "No data"}
            
            print(f"Received {len(minute_df)} minute bars")
            
            # Calculate session returns
            print("\nCalculating session returns...")
            session_df = self.session_analyzer.calculate_session_returns(minute_df)
        
        results = {}
        
        # 1. Session predictor analysis
        print("\n" + "-" * 40)
        print("1. SESSION PREDICTOR ANALYSIS")
        print("-" * 40)
        predictor_results = self.signal_analyzer.analyze_session_predictors(session_df)
        results["session_predictors"] = predictor_results
        
        for predictor, stats in predictor_results.items():
            print(f"\n{predictor}:")
            print(f"  Correlation with first 5-min: {stats['correlation']:.3f} (p={stats['p_value']:.3f})")
            print(f"  Signal type: {stats['signal_type'].upper()}")
            signal_type = stats['signal_type']
            print(f"  Accuracy: {stats[f'{signal_type}_accuracy']*100:.1f}%")
            if stats['avg_first5min_after_negative']:
                print(f"  Avg first 5-min after negative {predictor}: {stats['avg_first5min_after_negative']:.2f}%")
            if stats['avg_first5min_after_positive']:
                print(f"  Avg first 5-min after positive {predictor}: {stats['avg_first5min_after_positive']:.2f}%")
        
        # 2. RSI Mean Reversion Analysis
        print("\n" + "-" * 40)
        print("2. RSI MEAN REVERSION ANALYSIS")
        print("-" * 40)
        rsi_results = self.rsi_analyzer.analyze_rsi_patterns(minute_df)
        results["rsi_analysis"] = rsi_results
        
        if "stats" in rsi_results:
            stats = rsi_results["stats"]
            print(f"\nTotal RSI signals: {stats['total_signals']}")
            print(f"Win rate: {stats['win_rate']:.1f}%")
            print(f"Average P&L: {stats['avg_pnl']:.2f}%")
            print(f"Total P&L: {stats['total_pnl']:.2f}%")
            print(f"Sharpe ratio: {stats['sharpe']:.2f}")
            print(f"Avg holding time: {stats['avg_duration_mins']:.0f} minutes")
            
            if "oversold_bounce_win_rate" in stats:
                print(f"\nOversold Bounce: {stats['oversold_bounce_count']} trades, "
                      f"{stats['oversold_bounce_win_rate']:.1f}% win rate, "
                      f"{stats['oversold_bounce_avg_pnl']:.2f}% avg")
            
            if "overbought_reversal_win_rate" in stats:
                print(f"Overbought Reversal: {stats['overbought_reversal_count']} trades, "
                      f"{stats['overbought_reversal_win_rate']:.1f}% win rate, "
                      f"{stats['overbought_reversal_avg_pnl']:.2f}% avg")
        
        # 3. Intraday Reversion Analysis
        print("\n" + "-" * 40)
        print("3. INTRADAY REVERSION ANALYSIS")
        print("-" * 40)
        reversion_df = self.reversion_analyzer.analyze_reversion_from_open(minute_df, session_df)
        reversion_stats = self.reversion_analyzer.calculate_optimal_reversion_levels(reversion_df)
        results["reversion_analysis"] = reversion_stats
        
        for direction, stats in reversion_stats.items():
            print(f"\nAfter {direction.upper()} first 5 minutes ({stats['n_samples']} samples):")
            print(f"  Average first 5-min move: {stats['avg_first_5min_move']:.2f}%")
            print(f"  Average reversion: {stats['avg_reversion']:.2f}%")
            print(f"  Reverted to open: {stats['reverted_to_open_rate']:.1f}% of time")
            print(f"  Day closed same direction: {stats['same_direction_day']:.1f}%")
            print(f"  Reversion hit rates:")
            for level, rate in stats['reversion_hit_rates'].items():
                print(f"    {level}: {rate:.1f}%")
        
        # 4. Options Strategy Recommendations
        print("\n" + "-" * 40)
        print("4. OPTIONS STRATEGY RECOMMENDATIONS")
        print("-" * 40)
        
        # Generate trade plan
        trade_plan = self.options_advisor.generate_daily_trade_plan(
            predictor_results,
            rsi_results,
            reversion_stats
        )
        results["trade_plan"] = trade_plan
        
        print("\nDaily Trade Plan:")
        print("\nPre-market Checklist:")
        for item in trade_plan["pre_market_checklist"]:
            print(f"  ☐ {item}")
        
        print("\nScenarios:")
        for scenario in trade_plan["scenarios"]:
            print(f"\n  {scenario['condition']}:")
            print(f"    Expected: {scenario['expected_action']}")
            print(f"    Strategy: {scenario['strategy']}")
            print(f"    Target: {scenario['target']}")
        
        print("\nOptions Plays Based on Historical Data:")
        for play in trade_plan["options_plays"]:
            print(f"\n  {play['setup']}:")
            print(f"    Trigger: {play['trigger']}")
            print(f"    Action: {play['action']}")
            print(f"    Historical Win Rate: {play['historical_win_rate']}")
            print(f"    Average Gain: {play['avg_gain']}")
        
        # 5. Generate specific options recommendations
        print("\n" + "-" * 40)
        print("5. SPECIFIC OPTIONS STRATEGIES")
        print("-" * 40)
        
        # High conviction bullish (after oversold)
        bullish_rec = self.options_advisor.recommend_strategy(
            signal_strength=6,
            direction="long",
            expected_reversion=2.0
        )
        results["bullish_strategy"] = bullish_rec
        
        print("\nHigh Conviction Bullish (After Oversold):")
        for strat in bullish_rec["recommended_strategies"]:
            print(f"\n  {strat['strategy']}:")
            print(f"    Rationale: {strat['rationale']}")
            print(f"    Entry: {strat['entry']}")
            print(f"    Exit: {strat['exit']}")
        
        # High conviction bearish (after overbought)
        bearish_rec = self.options_advisor.recommend_strategy(
            signal_strength=6,
            direction="short",
            expected_reversion=-2.0
        )
        results["bearish_strategy"] = bearish_rec
        
        print("\nHigh Conviction Bearish (After Overbought):")
        for strat in bearish_rec["recommended_strategies"]:
            print(f"\n  {strat['strategy']}:")
            print(f"    Rationale: {strat['rationale']}")
            print(f"    Entry: {strat['entry']}")
            print(f"    Exit: {strat['exit']}")
        
        print("\n" + "=" * 60)
        print("Analysis Complete")
        print("=" * 60)
        
        return results
    
    def _generate_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate realistic sample data for demonstration"""
        
        np.random.seed(42)
        
        # Generate 60 days of minute data
        dates = pd.date_range(
            start="2024-01-01", 
            end="2024-03-01", 
            freq="B"
        )
        
        all_data = []
        base_price = 716.0  # APP current price level
        
        for date in dates:
            # Generate intraday pattern
            # Pre-market: 4:00-9:30 (330 minutes)
            premarket_times = pd.date_range(
                start=f"{date} 04:00",
                end=f"{date} 09:29",
                freq="1min",
                tz="America/New_York"
            )
            
            # Regular hours: 9:30-16:00 (390 minutes)
            regular_times = pd.date_range(
                start=f"{date} 09:30",
                end=f"{date} 15:59",
                freq="1min",
                tz="America/New_York"
            )
            
            # Post-market: 16:00-20:00 (240 minutes)
            postmarket_times = pd.date_range(
                start=f"{date} 16:00",
                end=f"{date} 19:59",
                freq="1min",
                tz="America/New_York"
            )
            
            # Generate prices with realistic patterns
            gap = np.random.normal(0, 0.02)  # Gap from prior close
            
            # Pre-market drift
            premarket_drift = np.random.normal(0, 0.005, len(premarket_times))
            premarket_prices = base_price * (1 + gap) * np.cumprod(1 + premarket_drift)
            
            # Regular hours with mean reversion tendency
            open_price = premarket_prices[-1]
            regular_returns = np.random.normal(0, 0.001, len(regular_times))
            
            # Add mean reversion after first 5 minutes
            first_5min_move = np.sum(regular_returns[:5])
            if abs(first_5min_move) > 0.003:  # If first 5 min move is large
                # Add contrarian drift
                regular_returns[5:60] -= first_5min_move * 0.01
            
            regular_prices = open_price * np.cumprod(1 + regular_returns)
            
            # Post-market
            postmarket_drift = np.random.normal(0, 0.002, len(postmarket_times))
            postmarket_prices = regular_prices[-1] * np.cumprod(1 + postmarket_drift)
            
            # Combine
            for times, prices in [
                (premarket_times, premarket_prices),
                (regular_times, regular_prices),
                (postmarket_times, postmarket_prices)
            ]:
                for t, p in zip(times, prices):
                    all_data.append({
                        "timestamp": t,
                        "open": p * (1 - np.random.uniform(0, 0.001)),
                        "high": p * (1 + np.random.uniform(0, 0.002)),
                        "low": p * (1 - np.random.uniform(0, 0.002)),
                        "close": p,
                        "volume": np.random.randint(10000, 100000),
                        "vwap": p
                    })
            
            base_price = postmarket_prices[-1]
        
        minute_df = pd.DataFrame(all_data)
        minute_df = minute_df.set_index("timestamp")
        
        # Calculate session returns
        session_df = self.session_analyzer.calculate_session_returns(minute_df)
        
        return minute_df, session_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize with your API key
    config = Config()
    # config.POLYGON_API_KEY = "your_actual_api_key"  # Uncomment and add your key
    
    analysis = APPMeanReversionAnalysis(config)
    
    # Run with sample data for demonstration
    # Set use_sample_data=False and add your API key for real data
    results = analysis.run_full_analysis(use_sample_data=True)