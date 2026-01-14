"""
APP Enhanced Mean Reversion System
===================================
Robust feature engineering with anti-overfitting measures.

Key Principles:
1. Features must have economic intuition (not just statistical artifacts)
2. Walk-forward validation (no lookahead bias)
3. Feature stability testing across regimes
4. Ensemble signals with confidence weighting
5. Adaptive thresholds based on volatility regime

Author: Limestone Hill Capital
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EnhancedConfig:
    SYMBOL: str = "APP"
    
    # RSI - adaptive based on volatility
    RSI_PERIOD: int = 14
    RSI_OVERSOLD_BASE: float = 30.0
    RSI_OVERBOUGHT_BASE: float = 70.0
    
    # Volume thresholds
    VOLUME_SURGE_THRESHOLD: float = 2.0  # 2x average
    VOLUME_WEAK_THRESHOLD: float = 0.5   # Below 0.5x = low conviction
    
    # Volatility regime
    ATR_PERIOD: int = 14
    VOL_LOOKBACK: int = 20
    
    # VWAP
    VWAP_BAND_MULT: float = 2.0  # Standard deviations
    
    # Mean reversion
    MEAN_REVERSION_LOOKBACK: int = 20  # Days for calculating mean
    
    # Feature selection
    MIN_FEATURE_IMPORTANCE: float = 0.05
    MAX_CORRELATION: float = 0.7  # Remove highly correlated features
    
    # Walk-forward
    TRAIN_WINDOW: int = 60  # Days
    TEST_WINDOW: int = 20   # Days
    MIN_SAMPLES_PER_BUCKET: int = 30


# ============================================================================
# ROBUST FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """
    Generate features with economic intuition.
    
    Categories:
    1. Volume Profile - Confirmation/divergence signals
    2. Volatility Regime - Adapt thresholds to current vol
    3. Price Structure - Support/resistance, trend context
    4. Market Context - Relative strength, correlation
    5. Time Features - Intraday patterns
    6. Mean Reversion Indicators - Oversold/overbought with confirmation
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
    
    # =========== VOLUME FEATURES ===========
    
    def calc_relative_volume(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Current volume vs N-day average.
        Economic intuition: High volume = conviction, validates signal.
        """
        avg_vol = df['volume'].rolling(lookback).mean()
        return df['volume'] / avg_vol
    
    def calc_volume_price_trend(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Volume-weighted price trend.
        Economic intuition: Price moves on high volume are more meaningful.
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vpt = (df['volume'] * ((typical_price - typical_price.shift(1)) / typical_price.shift(1))).cumsum()
        return vpt / vpt.rolling(period).std()  # Normalized
    
    def calc_up_down_volume_ratio(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Ratio of volume on up bars vs down bars.
        Economic intuition: >1 = buying pressure, <1 = selling pressure.
        """
        up_vol = df['volume'].where(df['close'] > df['open'], 0).rolling(period).sum()
        down_vol = df['volume'].where(df['close'] <= df['open'], 0).rolling(period).sum()
        return up_vol / (down_vol + 1)  # +1 to avoid div by zero
    
    def calc_first_5min_volume_ratio(self, minute_df: pd.DataFrame, session_df: pd.DataFrame) -> pd.Series:
        """
        First 5 min volume as % of daily average.
        Economic intuition: High opening volume = institutional activity.
        """
        # This would be calculated during session decomposition
        pass
    
    # =========== VOLATILITY FEATURES ===========
    
    def calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def calc_atr_percentile(self, df: pd.DataFrame, period: int = 14, lookback: int = 252) -> pd.Series:
        """
        ATR percentile vs historical.
        Economic intuition: High vol = wider stops, adjusted RSI thresholds.
        """
        atr = self.calc_atr(df, period)
        return atr.rolling(lookback).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]))
    
    def calc_intraday_range_ratio(self, df: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        Today's range vs average range.
        Economic intuition: Expansion = breakout potential, contraction = mean reversion.
        """
        daily_range = df['high'] - df['low']
        avg_range = daily_range.rolling(lookback).mean()
        return daily_range / avg_range
    
    def calc_close_position_in_range(self, df: pd.DataFrame) -> pd.Series:
        """
        Where close is within day's range (0-1).
        Economic intuition: Close near low = weak, near high = strong.
        """
        return (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
    
    def calc_overnight_gap_vs_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Gap size normalized by ATR.
        Economic intuition: Gaps > 1 ATR more likely to fill partially.
        """
        gap = df['open'] - df['close'].shift(1)
        atr = self.calc_atr(df, period)
        return gap / atr
    
    # =========== PRICE STRUCTURE FEATURES ===========
    
    def calc_distance_from_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Price distance from VWAP in standard deviations.
        Economic intuition: Mean reversion to VWAP is well-documented.
        """
        if 'vwap' not in df.columns:
            # Calculate VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        vwap_std = (df['close'] - df['vwap']).rolling(20).std()
        return (df['close'] - df['vwap']) / (vwap_std + 0.0001)
    
    def calc_distance_from_ma(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Price distance from moving average in %.
        Economic intuition: Extreme deviations tend to revert.
        """
        ma = df['close'].rolling(period).mean()
        return (df['close'] - ma) / ma * 100
    
    def calc_prior_day_close_position(self, df: pd.DataFrame) -> pd.Series:
        """
        Where prior day closed in its range.
        Economic intuition: Closes near low often see follow-through selling.
        """
        return self.calc_close_position_in_range(df).shift(1)
    
    def calc_support_resistance_distance(self, df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Distance to recent support and resistance.
        Economic intuition: Price reacts at key levels.
        """
        rolling_high = df['high'].rolling(lookback).max()
        rolling_low = df['low'].rolling(lookback).min()
        
        dist_to_resistance = (rolling_high - df['close']) / df['close'] * 100
        dist_to_support = (df['close'] - rolling_low) / df['close'] * 100
        
        return dist_to_resistance, dist_to_support
    
    def calc_higher_highs_lower_lows(self, df: pd.DataFrame, lookback: int = 5) -> pd.Series:
        """
        Trend structure: count of higher highs minus lower lows.
        Economic intuition: Trend context for mean reversion timing.
        """
        hh = (df['high'] > df['high'].shift(1)).rolling(lookback).sum()
        ll = (df['low'] < df['low'].shift(1)).rolling(lookback).sum()
        return hh - ll
    
    # =========== MEAN REVERSION INDICATORS ===========
    
    def calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Standard RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 0.0001)
        return 100 - (100 / (1 + rs))
    
    def calc_stochastic_rsi(self, prices: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
        """
        Stochastic RSI - more sensitive to oversold/overbought.
        Economic intuition: RSI of RSI catches extremes better.
        """
        rsi = self.calc_rsi(prices, rsi_period)
        stoch_rsi = (rsi - rsi.rolling(stoch_period).min()) / (
            rsi.rolling(stoch_period).max() - rsi.rolling(stoch_period).min() + 0.0001
        )
        return stoch_rsi * 100
    
    def calc_bollinger_position(self, df: pd.DataFrame, period: int = 20, std_mult: float = 2.0) -> pd.Series:
        """
        Position within Bollinger Bands (-1 to +1, beyond = extreme).
        Economic intuition: Mean reversion from bands is classic strategy.
        """
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        
        # Normalized position: 0 = at MA, +1 = at upper band, -1 = at lower band
        return (df['close'] - ma) / (std_mult * std + 0.0001)
    
    def calc_mean_reversion_zscore(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Z-score of close vs rolling mean.
        Economic intuition: Direct measure of deviation from mean.
        """
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        return (df['close'] - ma) / (std + 0.0001)
    
    def calc_rsi_divergence(self, df: pd.DataFrame, period: int = 14, lookback: int = 10) -> pd.Series:
        """
        RSI divergence from price.
        Economic intuition: Divergence often precedes reversals.
        """
        rsi = self.calc_rsi(df['close'], period)
        
        # Price making lower low but RSI making higher low = bullish divergence
        price_change = df['close'].rolling(lookback).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.0001)
        )
        rsi_change = rsi.rolling(lookback).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.0001)
        )
        
        return rsi_change - price_change  # Positive = bullish divergence
    
    # =========== TIME FEATURES ===========
    
    def calc_day_of_week(self, df: pd.DataFrame) -> pd.Series:
        """Day of week (0=Monday, 4=Friday)"""
        return df.index.dayofweek
    
    def calc_time_of_day_bucket(self, timestamp: datetime) -> str:
        """
        Categorize time of day.
        Economic intuition: Different dynamics in open, midday, close.
        """
        time = timestamp.time()
        if time < dt_time(10, 0):
            return "open_30min"
        elif time < dt_time(11, 30):
            return "morning"
        elif time < dt_time(14, 0):
            return "lunch"
        elif time < dt_time(15, 30):
            return "afternoon"
        else:
            return "close_30min"
    
    # =========== COMPOSITE FEATURES ===========
    
    def calc_volume_confirmed_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        RSI adjusted by volume confirmation.
        Economic intuition: Oversold on high volume = more reliable.
        """
        rsi = self.calc_rsi(df['close'], period)
        rel_vol = self.calc_relative_volume(df)
        
        # Boost signal when volume confirms
        vol_factor = np.clip(rel_vol, 0.5, 2.0)  # Cap influence
        
        # Oversold with high volume = more extreme effective RSI
        adjusted_rsi = 50 + (rsi - 50) * vol_factor
        return adjusted_rsi
    
    def calc_adaptive_rsi_thresholds(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        RSI thresholds that adapt to volatility regime.
        Economic intuition: In high vol, need more extreme RSI for signal.
        """
        atr_pct = self.calc_atr_percentile(df)
        
        # Widen thresholds in high volatility
        vol_adjustment = (atr_pct - 50) / 100 * 10  # ±5 points max
        
        oversold = self.config.RSI_OVERSOLD_BASE - vol_adjustment
        overbought = self.config.RSI_OVERBOUGHT_BASE + vol_adjustment
        
        return oversold.clip(20, 35), overbought.clip(65, 80)


# ============================================================================
# FEATURE SELECTION & VALIDATION
# ============================================================================

class FeatureSelector:
    """
    Select features that are robust and not overfit.
    
    Methods:
    1. Correlation filtering - remove redundant features
    2. Stability testing - feature must work across time periods
    3. Economic significance - not just statistical
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
    
    def remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
        """Remove highly correlated features, keeping most predictive"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return [col for col in df.columns if col not in to_drop]
    
    def test_feature_stability(
        self, 
        df: pd.DataFrame, 
        feature: str, 
        target: str,
        n_splits: int = 5
    ) -> Dict:
        """
        Test if feature is stable across time periods.
        Returns consistency metrics.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        correlations = []
        
        try:
            for train_idx, test_idx in tscv.split(df):
                if len(test_idx) < 10:
                    continue
                
                test = df.iloc[test_idx]
                
                # Get as Series
                feat_series = test[feature] if isinstance(test[feature], pd.Series) else test[feature].iloc[:, 0]
                targ_series = test[target] if isinstance(test[target], pd.Series) else test[target].iloc[:, 0]
                
                # Calculate correlation in each fold
                if feat_series.std() > 0 and targ_series.std() > 0:
                    corr = feat_series.corr(targ_series)
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if len(correlations) < 2:
                return {"mean_corr": 0, "std_corr": 1, "min_corr": 0, "max_corr": 0, 
                        "sign_consistent": False, "stable": False}
            
            return {
                "mean_corr": np.mean(correlations),
                "std_corr": np.std(correlations),
                "min_corr": np.min(correlations),
                "max_corr": np.max(correlations),
                "sign_consistent": all(c > 0 for c in correlations) or all(c < 0 for c in correlations),
                "stable": np.std(correlations) < 0.15  # Low variance across folds
            }
        except Exception as e:
            return {"mean_corr": 0, "std_corr": 1, "min_corr": 0, "max_corr": 0, 
                    "sign_consistent": False, "stable": False}
    
    def calculate_feature_importance(
        self, 
        df: pd.DataFrame,
        features: List[str],
        target: str
    ) -> pd.DataFrame:
        """
        Calculate feature importance using multiple methods.
        """
        results = []
        
        for feature in features:
            if feature not in df.columns or target not in df.columns:
                continue
            
            # Get feature and target as Series
            try:
                feature_series = df[feature].copy()
                target_series = df[target].copy()
                
                # Handle case where column selection returns DataFrame
                if isinstance(feature_series, pd.DataFrame):
                    feature_series = feature_series.iloc[:, 0]
                if isinstance(target_series, pd.DataFrame):
                    target_series = target_series.iloc[:, 0]
                
                valid_mask = feature_series.notna() & target_series.notna()
                feature_valid = feature_series[valid_mask]
                target_valid = target_series[valid_mask]
                
                if len(feature_valid) < 30:
                    continue
                
                # Correlation
                corr, p_value = stats.pearsonr(feature_valid, target_valid)
                
                # Spearman (rank correlation, more robust)
                spearman, sp_p = stats.spearmanr(feature_valid, target_valid)
                
                # Stability test
                valid_df = pd.DataFrame({feature: feature_valid, target: target_valid})
                stability = self.test_feature_stability(valid_df, feature, target)
                
                results.append({
                    "feature": feature,
                    "correlation": corr,
                    "p_value": p_value,
                    "spearman": spearman,
                    "spearman_p": sp_p,
                    "stable": stability["stable"],
                    "sign_consistent": stability["sign_consistent"],
                    "importance_score": abs(corr) * (1 if stability["stable"] else 0.5) * (1 if p_value < 0.05 else 0.5)
                })
            except Exception as e:
                continue
        
        if not results:
            return pd.DataFrame(columns=["feature", "correlation", "p_value", "importance_score"])
        
        return pd.DataFrame(results).sort_values("importance_score", ascending=False)


# ============================================================================
# WALK-FORWARD BACKTESTER
# ============================================================================

class WalkForwardBacktester:
    """
    Walk-forward validation to avoid lookahead bias.
    
    Process:
    1. Train on window N days
    2. Test on next M days
    3. Roll forward, retrain
    4. Aggregate out-of-sample results
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.feature_eng = FeatureEngineering(config)
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        train_window: int = 60,
        test_window: int = 20
    ) -> Dict:
        """
        Run walk-forward backtest.
        """
        results = []
        
        # Ensure we have enough data
        if len(df) < train_window + test_window:
            return {"error": "Insufficient data"}
        
        # Walk forward
        for start in range(0, len(df) - train_window - test_window, test_window):
            train_end = start + train_window
            test_end = train_end + test_window
            
            train_data = df.iloc[start:train_end]
            test_data = df.iloc[train_end:test_end]
            
            # Train: Calculate optimal thresholds on training data
            params = self._optimize_parameters(train_data)
            
            # Test: Apply to out-of-sample
            trades = self._generate_trades(test_data, params)
            
            results.append({
                "train_start": df.index[start],
                "train_end": df.index[train_end-1],
                "test_start": df.index[train_end],
                "test_end": df.index[test_end-1],
                "params": params,
                "trades": trades,
                "n_trades": len(trades),
                "win_rate": (pd.DataFrame(trades)["pnl"].sum() > 0).mean() * 100 if trades else 0,
                "total_pnl": sum(t["pnl"] for t in trades) if trades else 0
            })
        
        # Aggregate
        all_trades = [t for r in results for t in r["trades"]]
        
        return {
            "periods": results,
            "all_trades": all_trades,
            "aggregate_stats": self._calculate_stats(all_trades)
        }
    
    def _optimize_parameters(self, train_data: pd.DataFrame) -> Dict:
        """
        Find optimal parameters on training data.
        Keep it simple to avoid overfitting.
        """
        # Calculate features
        rsi = self.feature_eng.calc_rsi(train_data['close'])
        
        # Find RSI levels with best mean reversion
        # Use wider thresholds to generate more trades
        oversold = rsi.quantile(0.15)  # 15th percentile
        overbought = rsi.quantile(0.85)  # 85th percentile
        
        # Ensure reasonable bounds
        oversold = min(max(oversold, 25), 40)
        overbought = max(min(overbought, 75), 60)
        
        # Volume threshold - more lenient for daily data
        rel_vol = self.feature_eng.calc_relative_volume(train_data)
        vol_threshold = rel_vol.quantile(0.5)  # Above median
        
        return {
            "rsi_oversold": oversold,
            "rsi_overbought": overbought,
            "volume_threshold": vol_threshold,
            "atr_mult_stop": 1.5,  # ATR multiplier for stops
            "atr_mult_target": 2.0  # ATR multiplier for targets
        }
    
    def _generate_trades(self, test_data: pd.DataFrame, params: Dict) -> List[Dict]:
        """Generate trades using optimized parameters"""
        trades = []
        
        rsi = self.feature_eng.calc_rsi(test_data['close'])
        rel_vol = self.feature_eng.calc_relative_volume(test_data)
        atr = self.feature_eng.calc_atr(test_data)
        
        in_position = False
        entry = None
        
        for i in range(14, len(test_data)):  # Skip warmup
            row = test_data.iloc[i]
            current_rsi = rsi.iloc[i]
            current_vol = rel_vol.iloc[i]
            current_atr = atr.iloc[i]
            
            if not in_position:
                # Entry conditions - volume is optional boost, not required
                volume_boost = current_vol >= params["volume_threshold"]
                
                # Long signal
                if current_rsi < params["rsi_oversold"]:
                    entry = {
                        "direction": "long",
                        "entry_idx": i,
                        "entry_price": row["close"],
                        "stop": row["close"] - params["atr_mult_stop"] * current_atr,
                        "target": row["close"] + params["atr_mult_target"] * current_atr,
                        "entry_rsi": current_rsi,
                        "volume_confirmed": volume_boost
                    }
                    in_position = True
                
                # Short signal
                elif current_rsi > params["rsi_overbought"]:
                    entry = {
                        "direction": "short",
                        "entry_idx": i,
                        "entry_price": row["close"],
                        "stop": row["close"] + params["atr_mult_stop"] * current_atr,
                        "target": row["close"] - params["atr_mult_target"] * current_atr,
                        "entry_rsi": current_rsi,
                        "volume_confirmed": volume_boost
                    }
                    in_position = True
            
            else:
                # Exit conditions
                should_exit = False
                exit_reason = None
                
                if entry["direction"] == "long":
                    if row["low"] <= entry["stop"]:
                        should_exit = True
                        exit_reason = "stop"
                        exit_price = entry["stop"]
                    elif row["high"] >= entry["target"]:
                        should_exit = True
                        exit_reason = "target"
                        exit_price = entry["target"]
                    elif current_rsi >= 50:  # RSI normalized
                        should_exit = True
                        exit_reason = "rsi_neutral"
                        exit_price = row["close"]
                else:
                    if row["high"] >= entry["stop"]:
                        should_exit = True
                        exit_reason = "stop"
                        exit_price = entry["stop"]
                    elif row["low"] <= entry["target"]:
                        should_exit = True
                        exit_reason = "target"
                        exit_price = entry["target"]
                    elif current_rsi <= 50:
                        should_exit = True
                        exit_reason = "rsi_neutral"
                        exit_price = row["close"]
                
                if should_exit:
                    pnl = (exit_price - entry["entry_price"]) / entry["entry_price"] * 100
                    if entry["direction"] == "short":
                        pnl = -pnl
                    
                    trades.append({
                        **entry,
                        "exit_idx": i,
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "bars_held": i - entry["entry_idx"]
                    })
                    
                    in_position = False
                    entry = None
        
        return trades
    
    def _calculate_stats(self, trades: List[Dict]) -> Dict:
        """Calculate aggregate statistics"""
        if not trades:
            return {"error": "No trades"}
        
        df = pd.DataFrame(trades)
        
        return {
            "total_trades": len(df),
            "win_rate": (df["pnl"] > 0).mean() * 100,
            "avg_pnl": df["pnl"].mean(),
            "total_pnl": df["pnl"].sum(),
            "max_win": df["pnl"].max(),
            "max_loss": df["pnl"].min(),
            "avg_bars_held": df["bars_held"].mean(),
            "profit_factor": abs(df[df["pnl"] > 0]["pnl"].sum() / df[df["pnl"] < 0]["pnl"].sum()) if df[df["pnl"] < 0]["pnl"].sum() != 0 else np.inf,
            "sharpe": df["pnl"].mean() / df["pnl"].std() * np.sqrt(252) if df["pnl"].std() > 0 else 0,
            "by_direction": {
                "long": {
                    "count": len(df[df["direction"] == "long"]),
                    "win_rate": (df[df["direction"] == "long"]["pnl"] > 0).mean() * 100 if len(df[df["direction"] == "long"]) > 0 else 0,
                    "avg_pnl": df[df["direction"] == "long"]["pnl"].mean() if len(df[df["direction"] == "long"]) > 0 else 0
                },
                "short": {
                    "count": len(df[df["direction"] == "short"]),
                    "win_rate": (df[df["direction"] == "short"]["pnl"] > 0).mean() * 100 if len(df[df["direction"] == "short"]) > 0 else 0,
                    "avg_pnl": df[df["direction"] == "short"]["pnl"].mean() if len(df[df["direction"] == "short"]) > 0 else 0
                }
            },
            "by_exit_reason": df.groupby("exit_reason")["pnl"].agg(["count", "mean", "sum"]).to_dict()
        }


# ============================================================================
# ENSEMBLE SIGNAL SYSTEM
# ============================================================================

class EnsembleSignalSystem:
    """
    Combine multiple signals with confidence weighting.
    
    Signals only fire when multiple conditions align.
    Each signal has a weight based on historical reliability.
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.feature_eng = FeatureEngineering(config)
        
        # Signal weights (would be calibrated from backtest)
        self.signal_weights = {
            "rsi_extreme": 0.25,
            "volume_confirmed": 0.20,
            "vwap_deviation": 0.15,
            "bollinger_extreme": 0.15,
            "divergence": 0.10,
            "support_resistance": 0.10,
            "trend_context": 0.05
        }
    
    def calculate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all component signals"""
        signals = pd.DataFrame(index=df.index)
        
        # 1. RSI Signal (-1 to +1)
        rsi = self.feature_eng.calc_rsi(df['close'])
        signals['rsi_signal'] = np.where(
            rsi < 30, (30 - rsi) / 30,  # Oversold: positive
            np.where(rsi > 70, (70 - rsi) / 30, 0)  # Overbought: negative
        )
        
        # 2. Volume Confirmation (0 to 1)
        rel_vol = self.feature_eng.calc_relative_volume(df)
        signals['volume_signal'] = np.clip((rel_vol - 1) / 2, 0, 1)  # Boost when vol > 1x
        
        # 3. VWAP Deviation (-1 to +1)
        vwap_dev = self.feature_eng.calc_distance_from_vwap(df)
        signals['vwap_signal'] = -np.clip(vwap_dev / 2, -1, 1)  # Contrarian
        
        # 4. Bollinger Position (-1 to +1)
        bb_pos = self.feature_eng.calc_bollinger_position(df)
        signals['bb_signal'] = -np.clip(bb_pos, -1, 1)  # Contrarian
        
        # 5. RSI Divergence (-1 to +1)
        divergence = self.feature_eng.calc_rsi_divergence(df)
        signals['divergence_signal'] = np.clip(divergence, -1, 1)
        
        # 6. Support/Resistance (closer to support = bullish)
        dist_res, dist_sup = self.feature_eng.calc_support_resistance_distance(df)
        signals['sr_signal'] = (dist_res - dist_sup) / (dist_res + dist_sup + 0.0001)
        
        # 7. Trend Context (trend alignment)
        trend = self.feature_eng.calc_higher_highs_lower_lows(df)
        signals['trend_signal'] = np.clip(trend / 5, -1, 1)
        
        return signals
    
    def calculate_composite_score(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score.
        Positive = bullish, Negative = bearish
        """
        score = pd.Series(0.0, index=signals.index)
        
        score += signals['rsi_signal'] * self.signal_weights['rsi_extreme']
        score += signals['vwap_signal'] * self.signal_weights['vwap_deviation']
        score += signals['bb_signal'] * self.signal_weights['bollinger_extreme']
        score += signals['divergence_signal'] * self.signal_weights['divergence']
        score += signals['sr_signal'] * self.signal_weights['support_resistance']
        score += signals['trend_signal'] * self.signal_weights['trend_context']
        
        # Volume confirmation multiplier
        vol_mult = 1 + signals['volume_signal'] * self.signal_weights['volume_confirmed']
        score = score * vol_mult
        
        return score
    
    def calculate_confidence(self, signals: pd.DataFrame) -> pd.Series:
        """
        Calculate confidence level (0-100%).
        Based on agreement between signals.
        """
        # Count how many signals agree on direction
        signal_cols = [col for col in signals.columns if col != 'volume_signal']
        
        n_bullish = (signals[signal_cols] > 0.1).sum(axis=1)
        n_bearish = (signals[signal_cols] < -0.1).sum(axis=1)
        n_signals = len(signal_cols)
        
        # Confidence = max agreement / total signals
        agreement = np.maximum(n_bullish, n_bearish) / n_signals
        
        # Boost confidence with volume
        volume_boost = signals['volume_signal'] * 0.1
        
        return np.clip((agreement + volume_boost) * 100, 0, 100)
    
    def generate_signals(
        self, 
        df: pd.DataFrame,
        score_threshold: float = 0.3,
        confidence_threshold: float = 50
    ) -> pd.DataFrame:
        """
        Generate trading signals with confidence scores.
        """
        signals = self.calculate_all_signals(df)
        composite = self.calculate_composite_score(signals)
        confidence = self.calculate_confidence(signals)
        
        result = pd.DataFrame({
            'composite_score': composite,
            'confidence': confidence,
            'signal': np.where(
                (composite > score_threshold) & (confidence > confidence_threshold), 'LONG',
                np.where(
                    (composite < -score_threshold) & (confidence > confidence_threshold), 'SHORT',
                    'NEUTRAL'
                )
            )
        }, index=df.index)
        
        # Add component signals for analysis
        for col in signals.columns:
            result[col] = signals[col]
        
        return result


# ============================================================================
# REGIME DETECTOR
# ============================================================================

class RegimeDetector:
    """
    Detect market regime to adjust strategy.
    
    Regimes:
    - High Vol / Mean Reverting
    - High Vol / Trending
    - Low Vol / Mean Reverting
    - Low Vol / Trending
    """
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.feature_eng = FeatureEngineering(config)
    
    def detect_regime(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Detect current regime"""
        # Volatility regime
        atr = self.feature_eng.calc_atr(df)
        atr_pct = self.feature_eng.calc_atr_percentile(df)
        high_vol = atr_pct > 60
        
        # Trend vs mean reversion
        # Use autocorrelation of returns
        returns = df['close'].pct_change()
        autocorr = returns.rolling(lookback).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        # Positive autocorr = trending, negative = mean reverting
        trending = autocorr > 0.1
        
        # Combine
        regime = pd.DataFrame({
            'high_vol': high_vol,
            'trending': trending,
            'regime': np.where(
                high_vol & trending, 'high_vol_trend',
                np.where(
                    high_vol & ~trending, 'high_vol_mr',
                    np.where(
                        ~high_vol & trending, 'low_vol_trend',
                        'low_vol_mr'
                    )
                )
            ),
            'atr_percentile': atr_pct,
            'autocorrelation': autocorr
        }, index=df.index)
        
        return regime
    
    def get_regime_adjustments(self, regime: str) -> Dict:
        """
        Get strategy adjustments for current regime.
        """
        adjustments = {
            'high_vol_mr': {
                'description': 'High volatility, mean reverting - BEST regime for strategy',
                'position_size_mult': 1.0,
                'rsi_threshold_adjustment': 5,  # More extreme thresholds
                'profit_target_mult': 1.5,
                'stop_loss_mult': 1.5,
                'confidence_boost': 10
            },
            'high_vol_trend': {
                'description': 'High volatility, trending - REDUCE mean reversion',
                'position_size_mult': 0.5,
                'rsi_threshold_adjustment': 10,  # Much more extreme needed
                'profit_target_mult': 1.0,
                'stop_loss_mult': 2.0,  # Wider stops
                'confidence_boost': -20
            },
            'low_vol_mr': {
                'description': 'Low volatility, mean reverting - Good but smaller moves',
                'position_size_mult': 0.75,
                'rsi_threshold_adjustment': 0,
                'profit_target_mult': 0.75,  # Smaller targets
                'stop_loss_mult': 1.0,
                'confidence_boost': 0
            },
            'low_vol_trend': {
                'description': 'Low volatility, trending - AVOID mean reversion',
                'position_size_mult': 0.25,
                'rsi_threshold_adjustment': 5,
                'profit_target_mult': 0.5,
                'stop_loss_mult': 1.5,
                'confidence_boost': -30
            }
        }
        
        return adjustments.get(regime, adjustments['low_vol_mr'])


# ============================================================================
# MAIN ENHANCED ANALYSIS
# ============================================================================

class EnhancedMeanReversionSystem:
    """Main orchestrator for enhanced system"""
    
    def __init__(self, config: EnhancedConfig = None):
        self.config = config or EnhancedConfig()
        self.feature_eng = FeatureEngineering(self.config)
        self.feature_selector = FeatureSelector(self.config)
        self.backtester = WalkForwardBacktester(self.config)
        self.ensemble = EnsembleSignalSystem(self.config)
        self.regime_detector = RegimeDetector(self.config)
    
    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """Run complete enhanced analysis"""
        
        print("=" * 60)
        print("ENHANCED MEAN REVERSION ANALYSIS")
        print("=" * 60)
        
        results = {}
        
        # 1. Feature Engineering
        print("\n1. GENERATING FEATURES...")
        features_df = self._generate_all_features(df)
        results['features'] = features_df
        print(f"   Generated {len(features_df.columns)} features")
        
        # 2. Feature Selection
        print("\n2. FEATURE SELECTION...")
        feature_cols = [col for col in features_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'forward_return']]
        feature_importance = self.feature_selector.calculate_feature_importance(
            features_df, 
            feature_cols,
            'forward_return'  # Target
        )
        results['feature_importance'] = feature_importance
        
        top_features = feature_importance[feature_importance['importance_score'] > self.config.MIN_FEATURE_IMPORTANCE]
        print(f"   Top features ({len(top_features)}):")
        for _, row in top_features.head(10).iterrows():
            stable = "✓" if row['stable'] else "✗"
            print(f"   {stable} {row['feature']}: corr={row['correlation']:.3f}, importance={row['importance_score']:.3f}")
        
        # 3. Regime Detection
        print("\n3. REGIME DETECTION...")
        regime_df = self.regime_detector.detect_regime(df)
        results['regime'] = regime_df
        
        current_regime = regime_df['regime'].iloc[-1]
        regime_adj = self.regime_detector.get_regime_adjustments(current_regime)
        print(f"   Current regime: {current_regime}")
        print(f"   {regime_adj['description']}")
        print(f"   Position size multiplier: {regime_adj['position_size_mult']}")
        
        # 4. Walk-Forward Backtest
        print("\n4. WALK-FORWARD BACKTEST...")
        wf_results = self.backtester.run_walk_forward(df)
        results['walk_forward'] = wf_results
        
        if 'aggregate_stats' in wf_results and 'total_trades' in wf_results.get('aggregate_stats', {}):
            stats = wf_results['aggregate_stats']
            print(f"   Total trades: {stats['total_trades']}")
            print(f"   Win rate: {stats['win_rate']:.1f}%")
            print(f"   Avg P&L: {stats['avg_pnl']:.2f}%")
            print(f"   Sharpe: {stats['sharpe']:.2f}")
            print(f"   Profit factor: {stats['profit_factor']:.2f}")
        else:
            print(f"   Walk-forward results: {wf_results.get('error', 'No trades generated')}")
        
        # 5. Ensemble Signals
        print("\n5. ENSEMBLE SIGNAL ANALYSIS...")
        ensemble_signals = self.ensemble.generate_signals(df)
        results['ensemble_signals'] = ensemble_signals
        
        # Current signal
        latest = ensemble_signals.iloc[-1]
        print(f"   Current signal: {latest['signal']}")
        print(f"   Composite score: {latest['composite_score']:.3f}")
        print(f"   Confidence: {latest['confidence']:.1f}%")
        print(f"   Component signals:")
        for col in ['rsi_signal', 'vwap_signal', 'bb_signal', 'divergence_signal']:
            print(f"     {col}: {latest[col]:.3f}")
        
        # 6. Summary & Recommendations
        print("\n" + "=" * 60)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 60)
        
        self._print_recommendations(results, regime_adj, latest)
        
        return results
    
    def _generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features for analysis"""
        features = df.copy()
        
        # Forward return (target)
        features['forward_return'] = features['close'].pct_change().shift(-1) * 100
        
        # Volume features
        features['rel_volume'] = self.feature_eng.calc_relative_volume(df)
        features['up_down_vol_ratio'] = self.feature_eng.calc_up_down_volume_ratio(df)
        
        # Volatility features
        features['atr'] = self.feature_eng.calc_atr(df)
        features['atr_percentile'] = self.feature_eng.calc_atr_percentile(df)
        features['range_ratio'] = self.feature_eng.calc_intraday_range_ratio(df)
        features['close_position'] = self.feature_eng.calc_close_position_in_range(df)
        
        # Price structure
        features['vwap_deviation'] = self.feature_eng.calc_distance_from_vwap(df)
        features['ma20_deviation'] = self.feature_eng.calc_distance_from_ma(df, 20)
        features['prior_close_position'] = self.feature_eng.calc_prior_day_close_position(df)
        features['hh_ll_count'] = self.feature_eng.calc_higher_highs_lower_lows(df)
        
        # Mean reversion indicators
        features['rsi'] = self.feature_eng.calc_rsi(df['close'])
        features['stoch_rsi'] = self.feature_eng.calc_stochastic_rsi(df['close'])
        features['bb_position'] = self.feature_eng.calc_bollinger_position(df)
        features['zscore'] = self.feature_eng.calc_mean_reversion_zscore(df)
        features['rsi_divergence'] = self.feature_eng.calc_rsi_divergence(df)
        features['vol_confirmed_rsi'] = self.feature_eng.calc_volume_confirmed_rsi(df)
        
        # Time features
        features['day_of_week'] = features.index.dayofweek
        
        return features
    
    def _print_recommendations(self, results: Dict, regime_adj: Dict, latest_signal: pd.Series):
        """Print actionable recommendations"""
        
        print(f"\n📊 CURRENT MARKET STATE:")
        print(f"   Regime: {results['regime']['regime'].iloc[-1]}")
        print(f"   Signal: {latest_signal['signal']}")
        print(f"   Confidence: {latest_signal['confidence']:.1f}%")
        
        adjusted_confidence = latest_signal['confidence'] + regime_adj['confidence_boost']
        print(f"   Regime-adjusted confidence: {adjusted_confidence:.1f}%")
        
        if latest_signal['signal'] == 'LONG' and adjusted_confidence > 50:
            print(f"\n🟢 ACTIONABLE SIGNAL: LONG")
            print(f"   Position size: {regime_adj['position_size_mult']*100:.0f}% of normal")
            print(f"   Target multiplier: {regime_adj['profit_target_mult']}x ATR")
            print(f"   Stop multiplier: {regime_adj['stop_loss_mult']}x ATR")
            print(f"\n   Options suggestion: Buy ATM call, 0DTE")
            
        elif latest_signal['signal'] == 'SHORT' and adjusted_confidence > 50:
            print(f"\n🔴 ACTIONABLE SIGNAL: SHORT")
            print(f"   Position size: {regime_adj['position_size_mult']*100:.0f}% of normal")
            print(f"   Target multiplier: {regime_adj['profit_target_mult']}x ATR")
            print(f"   Stop multiplier: {regime_adj['stop_loss_mult']}x ATR")
            print(f"\n   Options suggestion: Buy ATM put, 0DTE")
            
        else:
            print(f"\n⏸️ NO ACTIONABLE SIGNAL")
            if adjusted_confidence < 50:
                print(f"   Reason: Confidence too low ({adjusted_confidence:.1f}%)")
            else:
                print(f"   Reason: No clear directional signal")
            print(f"\n   Waiting for:")
            print(f"   - RSI < 30 or > 70")
            print(f"   - Volume confirmation")
            print(f"   - Multiple signal alignment")
        
        # Top features driving signal
        print(f"\n📈 TOP PREDICTIVE FEATURES:")
        top_features = results['feature_importance'].head(5)
        for _, row in top_features.iterrows():
            direction = "+" if row['correlation'] > 0 else "-"
            print(f"   {direction} {row['feature']}: {row['correlation']:.3f}")


# ============================================================================
# DEMO / TEST
# ============================================================================

def generate_sample_data(n_days: int = 120) -> pd.DataFrame:
    """Generate realistic sample data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    # Generate price with some mean reversion
    returns = np.random.normal(0, 0.02, n_days)
    
    # Add mean reversion tendency
    price = 250
    prices = [price]
    for i, r in enumerate(returns[1:]):
        # Mean revert if too far from 250
        deviation = (prices[-1] - 250) / 250
        mean_reversion = -deviation * 0.1
        prices.append(prices[-1] * (1 + r + mean_reversion))
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': prices * (1 + np.random.uniform(0.005, 0.02, n_days)),
        'low': prices * (1 - np.random.uniform(0.005, 0.02, n_days)),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_days)
    }, index=dates)
    
    # Calculate VWAP
    typical = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = typical  # Simplified
    
    return df


if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data(120)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    config = EnhancedConfig()
    system = EnhancedMeanReversionSystem(config)
    
    results = system.run_analysis(df)