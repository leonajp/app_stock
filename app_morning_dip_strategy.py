"""
APP Morning Dip Strategy Backtester v2

Strategy Logic:
1. CONDITIONS (checked before market open):
   - Gap up: premarket/open > previous EOD by X% (configurable)
   
2. ENTRY (9:30 AM - 10:10 AM):
   - APP drops consistently during this window (>1% from open)
   - Volume above average during dip
   - WAIT for momentum confirmation: 2-3 green candles after the low
   - Buy CALL when recovery confirmed
   
3. EXIT:
   - Time-based exit (configurable, default 11:20 AM)
   - Or EOD if time-based disabled

Uses:
- ThetaData for option prices
- Polygon for extended hours stock data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from theta_data_fetcher import ThetaDataFetcher
from io import StringIO
import requests
import os


class MorningDipBacktester:
    """Backtest the APP morning dip strategy with real option prices."""
    
    def __init__(self, fetcher: ThetaDataFetcher = None, polygon_api_key: str = None,
                 # Configurable parameters
                 min_gap_pct: float = 0.8,
                 min_dip_pct: float = 1.0,
                 confirm_candles: int = 2,
                 exit_time: str = "11:20",
                 use_volume_filter: bool = True,
                 require_pm_bullish: bool = True):
        """
        Args:
            min_gap_pct: Minimum gap up % required (premarket vs prev EOD)
            min_dip_pct: Minimum dip % from open required
            confirm_candles: Number of green candles to confirm recovery
            exit_time: Time to exit (HH:MM format) or "EOD"
            use_volume_filter: Require above-average volume on dip
            require_pm_bullish: Require previous day's post-market to be bullish
        """
        self.fetcher = fetcher or ThetaDataFetcher()
        self.theta_url = self.fetcher.base_url
        self.polygon_api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        
        # Strategy parameters
        self.min_gap_pct = min_gap_pct
        self.min_dip_pct = min_dip_pct
        self.confirm_candles = confirm_candles
        self.exit_time = exit_time
        self.use_volume_filter = use_volume_filter
        self.require_pm_bullish = require_pm_bullish
        
    def get_polygon_bars(self, symbol: str, date: str, timespan: str = "minute") -> pd.DataFrame:
        """Get intraday data from Polygon including extended hours."""
        if not self.polygon_api_key:
            return pd.DataFrame()
        
        if len(date) == 8:
            date_formatted = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            date_formatted = date
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/{date_formatted}/{date_formatted}"
        
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
        
        return pd.DataFrame()
    
    def get_theta_intraday(self, symbol: str, date: str) -> pd.DataFrame:
        """Get intraday stock data from ThetaData."""
        response = requests.get(
            f"{self.theta_url}/v3/stock/history/ohlc",
            params={"symbol": symbol, "date": date, "interval": "1m"},
            timeout=60
        )
        
        if response.status_code == 200 and response.text:
            df = pd.read_csv(StringIO(response.text))
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    
    def get_daily_data(self, symbol: str, date: str) -> Dict:
        """Get daily data with extended hours."""
        result = {
            'date': date,
            'premarket_close': None,
            'premarket_high': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'postmarket_close': None,
            'df': None  # Store full dataframe for analysis
        }
        
        df = self.get_polygon_bars(symbol, date)
        if df.empty:
            df = self.get_theta_intraday(symbol, date)
        
        if df.empty:
            return result
        
        # Make timestamp timezone-naive for filtering
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        result['df'] = df
        
        # Pre-market
        premarket = df[(df['hour'] >= 4) & (df['hour'] < 9) | 
                       ((df['hour'] == 9) & (df['minute'] < 30))]
        
        # Regular hours
        regular = df[((df['hour'] == 9) & (df['minute'] >= 30)) | 
                     ((df['hour'] >= 10) & (df['hour'] < 16))]
        
        # Post-market
        postmarket = df[(df['hour'] >= 16) & (df['hour'] < 20)]
        
        if not premarket.empty:
            result['premarket_close'] = premarket['close'].iloc[-1]
            result['premarket_high'] = premarket['high'].max()
        
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
        """
        Check if entry conditions are met.
        
        STRICT CONDITIONS:
        1. Previous day's post-market must be above previous day's EOD (if require_pm_bullish)
        2. Today's premarket must be above previous day's EOD by min_gap_pct
        3. Combined: both post-market and premarket showing bullish sentiment
        
        Returns: (bool, reason_string, gap_pct)
        """
        prev_eod = prev_day.get('close')
        prev_postmarket = prev_day.get('postmarket_close')
        today_premarket = today.get('premarket_close')
        today_open = today.get('open')
        
        if not prev_eod:
            return False, "no_prev_eod", 0
        
        # Calculate gaps
        pm_gap_pct = 0
        premarket_gap_pct = 0
        
        # Check post-market vs EOD (if required)
        if self.require_pm_bullish and prev_postmarket and prev_postmarket > 0:
            pm_gap_pct = (prev_postmarket - prev_eod) / prev_eod * 100
            # Post-market must not be significantly negative (allow small noise of -0.3%)
            if pm_gap_pct < -0.3:
                return False, f"postmarket_weak_{pm_gap_pct:+.1f}%", pm_gap_pct
        elif prev_postmarket and prev_postmarket > 0:
            pm_gap_pct = (prev_postmarket - prev_eod) / prev_eod * 100
        
        # Check premarket vs prev EOD
        if today_premarket and today_premarket > 0:
            premarket_gap_pct = (today_premarket - prev_eod) / prev_eod * 100
        elif today_open:
            premarket_gap_pct = (today_open - prev_eod) / prev_eod * 100
        else:
            return False, "no_premarket_or_open", 0
        
        # Premarket must be elevated by at least min_gap_pct
        if premarket_gap_pct < self.min_gap_pct:
            return False, f"premarket_gap_{premarket_gap_pct:+.1f}%_below_{self.min_gap_pct}%", premarket_gap_pct
        
        # If require_pm_bullish and we have post-market data, check combined condition
        if self.require_pm_bullish and prev_postmarket and prev_postmarket > 0:
            if pm_gap_pct < 0:
                # Post-market was down - need extra strong premarket to compensate
                if premarket_gap_pct < self.min_gap_pct + abs(pm_gap_pct):
                    return False, f"postmarket_down_{pm_gap_pct:+.1f}%_premarket_not_enough", premarket_gap_pct
        
        # Conditions met
        return True, f"pm_{pm_gap_pct:+.1f}%_pre_{premarket_gap_pct:+.1f}%", premarket_gap_pct
    
    def find_morning_dip_with_confirmation(self, df: pd.DataFrame, avg_volume: float = None) -> Optional[Dict]:
        """
        Find morning dip with momentum confirmation.
        
        Requirements:
        1. Dip > min_dip_pct from open
        2. Above-average volume during dip (if volume filter enabled)
        3. N green candles after the low (momentum confirmation)
        """
        # Filter to 9:30-10:30 window (extended for confirmation)
        morning = df[
            ((df['hour'] == 9) & (df['minute'] >= 30)) |
            ((df['hour'] == 10) & (df['minute'] <= 30))
        ].copy()
        
        if len(morning) < 10:
            return None
        
        morning = morning.reset_index(drop=True)
        
        # Find the low point in 9:30-10:10 window
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
        
        # Check minimum dip requirement
        dip_pct = (open_price - low_price) / open_price * 100
        if dip_pct < self.min_dip_pct:
            return {'rejected': True, 'reason': f'dip_too_small_{dip_pct:.1f}%'}
        
        # Volume filter: Check if volume during dip is above average
        if self.use_volume_filter and avg_volume:
            dip_volume = dip_window['volume'].mean()
            if dip_volume < avg_volume * 0.8:  # Allow 20% below avg
                return {'rejected': True, 'reason': 'low_volume_on_dip'}
        
        # Find the position of the low in the full morning dataframe
        low_idx_in_morning = morning[morning['timestamp'] == low_time].index[0]
        
        # Momentum confirmation: Check for N green candles after the low
        candles_after_low = morning.iloc[low_idx_in_morning + 1:low_idx_in_morning + 1 + self.confirm_candles + 2]
        
        if len(candles_after_low) < self.confirm_candles:
            return {'rejected': True, 'reason': 'not_enough_candles_after_low'}
        
        # Count green candles
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
                green_count = 0  # Reset on red candle
        
        if green_count < self.confirm_candles:
            return {'rejected': True, 'reason': f'no_momentum_confirmation_{green_count}_green'}
        
        # Entry is after confirmation candles
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
        
        # Make sure timestamp is comparable
        if df['timestamp'].dt.tz is not None:
            df = df.copy()
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        if self.exit_time.upper() == "EOD":
            # Use last regular hours candle
            regular = df[((df['hour'] == 9) & (df['minute'] >= 30)) | 
                        ((df['hour'] >= 10) & (df['hour'] < 16))]
            if not regular.empty:
                return regular['close'].iloc[-1], regular['timestamp'].iloc[-1]
            return None, None
        
        # Parse exit time
        try:
            exit_hour, exit_minute = map(int, self.exit_time.split(':'))
        except:
            exit_hour, exit_minute = 11, 20
        
        # Find candle at or after exit time
        exit_candles = df[(df['hour'] == exit_hour) & (df['minute'] >= exit_minute)]
        
        if not exit_candles.empty:
            return exit_candles['close'].iloc[0], exit_candles['timestamp'].iloc[0]
        
        # If exact time not found, get the nearest after
        after_entry = df[df['timestamp'] > entry_time]
        target_time = entry_time.replace(hour=exit_hour, minute=exit_minute)
        
        closest = after_entry[after_entry['timestamp'] >= target_time]
        if not closest.empty:
            return closest['close'].iloc[0], closest['timestamp'].iloc[0]
        
        # Fallback to last available
        if not after_entry.empty:
            return after_entry['close'].iloc[-1], after_entry['timestamp'].iloc[-1]
        
        return None, None
    
    def get_option_price_at_time(self, symbol: str, expiration: str, strike: float,
                                  right: str, date: str, target_time, get_last: bool = False) -> Optional[float]:
        """Get option price closest to a specific time, handling sparse data."""
        response = requests.get(
            f"{self.theta_url}/v3/option/history/ohlc",
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
        
        # Filter to non-zero prices only (sparse data)
        df_valid = df[df['close'] > 0].copy()
        
        if df_valid.empty:
            df_valid = df[(df['open'] > 0) | (df['high'] > 0) | (df['low'] > 0)].copy()
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
        elif row['open'] > 0:
            return float(row['open'])
        return None
    
    def find_best_expiration(self, symbol: str, trade_date: str) -> Optional[str]:
        """Find best expiration (0-7 DTE preferred)."""
        expirations = self.fetcher.get_expirations(symbol)
        trade_dt = datetime.strptime(trade_date, "%Y%m%d")
        
        for exp in expirations:
            try:
                if '-' in exp:
                    exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                else:
                    exp_dt = datetime.strptime(exp, "%Y%m%d")
                
                days_to_exp = (exp_dt - trade_dt).days
                if 0 <= days_to_exp <= 7:
                    return exp
            except:
                continue
        
        return expirations[0] if expirations else None
    
    def find_atm_strike(self, symbol: str, expiration: str, spot_price: float) -> float:
        """Find ATM strike."""
        strikes = self.fetcher.get_strikes(symbol, expiration)
        if not strikes:
            return round(spot_price / 5) * 5
        return min(strikes, key=lambda x: abs(x - spot_price))
    
    def backtest_day(self, symbol: str, date: str, prev_day: Dict) -> Dict:
        """Run backtest for a single day."""
        result = {'date': date, 'signal': False}
        
        today = self.get_daily_data(symbol, date)
        
        if not today.get('open'):
            result['reason'] = 'no_data'
            return result
        
        # Check entry conditions (gap up)
        conditions_met, condition_reason, gap_pct = self.check_entry_conditions(prev_day, today)
        result['gap_pct'] = gap_pct
        
        if not conditions_met:
            result['reason'] = condition_reason
            return result
        
        # Check morning dip with momentum confirmation
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
        
        # We have a confirmed signal!
        result['signal'] = True
        result['condition'] = condition_reason
        result['entry_time'] = str(dip_result['entry_time'])
        result['entry_stock_price'] = dip_result['entry_price']
        result['low_price'] = dip_result['low_price']
        result['low_time'] = str(dip_result['low_time'])
        result['morning_open'] = dip_result['open_price']
        result['dip_pct'] = dip_result['dip_pct']
        result['green_candles'] = dip_result['green_candles']
        
        # Get exit price
        exit_price, exit_time = self.get_exit_price_and_time(df, dip_result['entry_time'])
        result['exit_stock_price'] = exit_price
        result['exit_time'] = str(exit_time) if exit_time else None
        
        if exit_price and dip_result['entry_price']:
            result['stock_pnl_pct'] = (exit_price - dip_result['entry_price']) / dip_result['entry_price'] * 100
        
        # Initialize option columns (will be filled if data available)
        result['strike'] = None
        result['expiration'] = None
        result['entry_option_price'] = None
        result['exit_option_price'] = None
        result['option_pnl_pct'] = None
        result['option_pnl_dollar'] = None
        
        # Find option
        expiration = self.find_best_expiration(symbol, date)
        if not expiration:
            result['option_error'] = 'no_expiration'
            return result
        
        entry_price = dip_result['entry_price']
        
        # Try strikes near ATM
        all_strikes = self.fetcher.get_strikes(symbol, expiration)
        if all_strikes:
            nearby = [s for s in all_strikes if abs(s - entry_price) <= 15]
            strikes_to_try = sorted(nearby, key=lambda x: abs(x - entry_price))[:3]
        else:
            strikes_to_try = [round(entry_price / 5) * 5]
        
        result['expiration'] = expiration
        
        # Get option prices
        entry_time = dip_result['entry_time']
        
        for strike in strikes_to_try:
            # Entry price at confirmation time
            entry_opt_price = self.get_option_price_at_time(
                symbol, expiration, strike, "CALL", date, entry_time, get_last=False
            )
            
            # Exit price at exit time (or last of day if exit_time is EOD)
            if exit_time:
                exit_opt_price = self.get_option_price_at_time(
                    symbol, expiration, strike, "CALL", date, exit_time, get_last=False
                )
            else:
                exit_opt_price = self.get_option_price_at_time(
                    symbol, expiration, strike, "CALL", date, entry_time, get_last=True
                )
            
            if entry_opt_price and exit_opt_price and entry_opt_price > 0:
                result['strike'] = strike
                result['entry_option_price'] = entry_opt_price
                result['exit_option_price'] = exit_opt_price
                result['option_pnl_pct'] = (exit_opt_price - entry_opt_price) / entry_opt_price * 100
                result['option_pnl_dollar'] = (exit_opt_price - entry_opt_price) * 100
                break
        else:
            result['strike'] = strikes_to_try[0] if strikes_to_try else round(entry_price / 5) * 5
            result['option_error'] = 'no_option_prices'
        
        return result
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Run full backtest."""
        results = []
        
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        prev_day = None
        current = start_dt
        
        print(f"\nRunning backtest from {start_date} to {end_date}...")
        print(f"Parameters: gap>={self.min_gap_pct}%, dip>={self.min_dip_pct}%, "
              f"confirm={self.confirm_candles} candles, exit={self.exit_time}, "
              f"pm_bullish={self.require_pm_bullish}")
        print("=" * 70)
        
        while current <= end_dt:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue
            
            date_str = current.strftime("%Y%m%d")
            
            today_data = self.get_daily_data(symbol, date_str)
            
            if prev_day and today_data.get('open'):
                result = self.backtest_day(symbol, date_str, prev_day)
                results.append(result)
                
                if result.get('signal'):
                    stock_pnl = result.get('stock_pnl_pct', 0)
                    opt_pnl = result.get('option_pnl_pct')
                    opt_str = f"Opt: {opt_pnl:+.1f}%" if opt_pnl else "Opt: N/A"
                    print(f"  {date_str}: ✅ SIGNAL | Stock: {stock_pnl:+.2f}% | {opt_str}")
                else:
                    print(f"  {date_str}: ❌ {result.get('reason', 'no signal')}")
            else:
                print(f"  {date_str}: (baseline)")
            
            prev_day = today_data
            current += timedelta(days=1)
        
        return pd.DataFrame(results)
    
    def print_summary(self, df: pd.DataFrame):
        """Print backtest summary."""
        if df.empty:
            print("\nNo results.")
            return
        
        signals = df[df['signal'] == True]
        
        print("\n" + "=" * 70)
        print("📊 BACKTEST SUMMARY: APP Morning Dip Strategy")
        print("=" * 70)
        print(f"\nParameters: gap>={self.min_gap_pct}%, dip>={self.min_dip_pct}%, "
              f"confirm={self.confirm_candles}, exit={self.exit_time}, pm_bullish={self.require_pm_bullish}")
        
        print(f"\nTotal days: {len(df)}")
        print(f"Signal days: {len(signals)}")
        
        if len(signals) == 0:
            print("No signals generated.")
            return
        
        # Stock performance
        print(f"\n📈 STOCK PERFORMANCE:")
        print(f"  Win rate: {(signals['stock_pnl_pct'] > 0).mean()*100:.1f}%")
        print(f"  Avg P&L: {signals['stock_pnl_pct'].mean():+.2f}%")
        print(f"  Total P&L: {signals['stock_pnl_pct'].sum():+.2f}%")
        print(f"  Best: {signals['stock_pnl_pct'].max():+.2f}%")
        print(f"  Worst: {signals['stock_pnl_pct'].min():+.2f}%")
        
        # Option performance
        if 'option_pnl_pct' in signals.columns:
            opt_signals = signals[signals['option_pnl_pct'].notna()]
            if len(opt_signals) > 0:
                print(f"\n📊 OPTION PERFORMANCE:")
                print(f"  Trades with data: {len(opt_signals)}")
                print(f"  Win rate: {(opt_signals['option_pnl_pct'] > 0).mean()*100:.1f}%")
                print(f"  Avg P&L: {opt_signals['option_pnl_pct'].mean():+.2f}%")
                print(f"  Total P&L: {opt_signals['option_pnl_pct'].sum():+.2f}%")
                if 'option_pnl_dollar' in opt_signals.columns:
                    print(f"  Avg $ per contract: ${opt_signals['option_pnl_dollar'].mean():+.0f}")
        
        # Trade log
        print(f"\n📋 TRADE LOG:")
        print("-" * 100)
        print(f"{'Date':<10} {'Entry':<8} {'Exit':<8} {'Stock%':<8} {'Strike':<7} {'OptIn':<8} {'OptOut':<8} {'Opt%':<8} {'ExitTime':<8}")
        print("-" * 100)
        
        for _, row in signals.iterrows():
            date = row['date']
            entry = f"${row.get('entry_stock_price', 0):.2f}"
            exit_p = f"${row.get('exit_stock_price', 0):.2f}"
            stock_pnl = f"{row.get('stock_pnl_pct', 0):+.1f}%"
            strike = f"${row.get('strike', 0):.0f}C" if row.get('strike') else "N/A"
            
            entry_opt = f"${row.get('entry_option_price', 0):.2f}" if row.get('entry_option_price') else "N/A"
            exit_opt = f"${row.get('exit_option_price', 0):.2f}" if row.get('exit_option_price') else "N/A"
            opt_pnl = f"{row.get('option_pnl_pct', 0):+.1f}%" if row.get('option_pnl_pct') else "N/A"
            
            exit_time = str(row.get('exit_time', ''))[-8:-3] if row.get('exit_time') else "N/A"
            
            print(f"{date:<10} {entry:<8} {exit_p:<8} {stock_pnl:<8} {strike:<7} {entry_opt:<8} {exit_opt:<8} {opt_pnl:<8} {exit_time:<8}")


if __name__ == "__main__":
    print("=" * 70)
    print("APP Morning Dip Strategy Backtester v2")
    print("=" * 70)
    
    fetcher = ThetaDataFetcher()
    
    if not fetcher.check_connection():
        print("❌ Cannot connect to Theta Terminal")
        exit(1)
    
    print("✅ Connected to Theta Terminal")
    
    polygon_key = os.environ.get('POLYGON_API_KEY')
    if polygon_key:
        print("✅ Polygon API key found")
    else:
        print("⚠️ No Polygon API key - extended hours data limited")
    
    # Create backtester with parameters
    backtester = MorningDipBacktester(
        fetcher=fetcher,
        polygon_api_key=polygon_key,
        min_gap_pct=0.8,         # Require 0.8% gap up (premarket vs prev EOD)
        min_dip_pct=1.0,         # Require 1% dip
        confirm_candles=2,        # Wait for 2 green candles
        exit_time="11:20",        # Exit at 11:20 AM
        use_volume_filter=True,   # Require above-avg volume
        require_pm_bullish=True   # Require post-market bullish
    )
    
    # Test last 14 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    
    results = backtester.run_backtest(
        symbol="APP",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d")
    )
    
    backtester.print_summary(results)
    
    if not results.empty:
        results.to_csv("app_morning_dip_results.csv", index=False)
        print(f"\n💾 Saved to app_morning_dip_results.csv")