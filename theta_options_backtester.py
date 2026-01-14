"""
Real Options Backtester using ThetaData V3

Uses actual historical options prices and Greeks for realistic backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from theta_data_fetcher import ThetaDataFetcher


class RealOptionsBacktester:
    """
    Backtester using real options data from ThetaData V3.
    
    Flow:
    1. Get stock signal (RSI reversal)
    2. Fetch actual option chain at that moment
    3. Select strike based on confidence
    4. Track actual premium paid
    5. Calculate P&L based on actual premium at exit
    """
    
    def __init__(
        self,
        theta_fetcher: ThetaDataFetcher,
        oversold: float = 30,
        overbought: float = 70
    ):
        self.fetcher = theta_fetcher
        self.oversold = oversold
        self.overbought = overbought
        
        # Strike selection parameters
        self.strike_offsets = {
            'ATM': 0,
            'OTM': 0.015,
            'DEEP_OTM': 0.03
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'OTM': 2,
            'DEEP_OTM': 4
        }
        
        # Cache for options data
        self._options_cache = {}
        self._expirations_cache = {}
    
    def assess_confidence(self, rsi: float, rel_volume: float, direction: str) -> int:
        """Score trade confidence (0-6)"""
        score = 0
        
        if direction == 'long':
            if rsi < self.oversold - 8: score += 3
            elif rsi < self.oversold - 4: score += 2
            elif rsi < self.oversold: score += 1
        else:
            if rsi > self.overbought + 8: score += 3
            elif rsi > self.overbought + 4: score += 2
            elif rsi > self.overbought: score += 1
        
        if rel_volume > 2.5: score += 2
        elif rel_volume > 1.5: score += 1
        
        return score
    
    def select_strike_type(self, confidence: int) -> str:
        """Select strike type based on confidence"""
        if confidence >= self.confidence_thresholds['DEEP_OTM']:
            return 'DEEP_OTM'
        elif confidence >= self.confidence_thresholds['OTM']:
            return 'OTM'
        else:
            return 'ATM'
    
    def get_historical_option_price(
        self,
        symbol: str,
        trade_date: str,
        spot_price: float,
        direction: str,
        strike_type: str = 'ATM'
    ) -> Optional[Dict]:
        """
        Get historical option price for a specific date.
        
        Uses EOD data for historical backtesting.
        """
        # Cache key
        cache_key = f"{symbol}_{trade_date}_{direction}_{strike_type}"
        if cache_key in self._options_cache:
            return self._options_cache[cache_key]
        
        try:
            # Find nearest expiration to trade date
            # For intraday, we want 0-1 DTE
            trade_dt = datetime.strptime(trade_date, "%Y%m%d")
            
            # Get expirations
            if symbol not in self._expirations_cache:
                self._expirations_cache[symbol] = self.fetcher.get_expirations(symbol)
            
            expirations = self._expirations_cache[symbol]
            
            # Find nearest expiration >= trade_date
            valid_exps = []
            for exp in expirations:
                try:
                    exp_dt = datetime.strptime(str(exp), "%Y%m%d")
                    if exp_dt >= trade_dt:
                        valid_exps.append((exp, (exp_dt - trade_dt).days))
                except:
                    continue
            
            if not valid_exps:
                return None
            
            # Get nearest (prefer 0-1 DTE for intraday)
            valid_exps.sort(key=lambda x: x[1])
            expiration = valid_exps[0][0]
            
            # Calculate target strike
            offset = self.strike_offsets.get(strike_type, 0)
            if direction == 'long':
                target_strike = spot_price * (1 + offset)
                right = "CALL"
            else:
                target_strike = spot_price * (1 - offset)
                right = "PUT"
            
            # Round to nearest $5
            strike = round(target_strike / 5) * 5
            
            # Get historical EOD data
            eod_df = self.fetcher.get_option_history_eod(
                symbol=symbol,
                expiration=expiration,
                strike=strike,
                right=right,
                start_date=trade_date,
                end_date=trade_date
            )
            
            if eod_df.empty:
                # Try OHLC if EOD not available
                ohlc_df = self.fetcher.get_option_history_ohlc(
                    symbol=symbol,
                    expiration=expiration,
                    strike=strike,
                    right=right,
                    start_date=trade_date,
                    end_date=trade_date,
                    interval="1d"
                )
                
                if not ohlc_df.empty:
                    row = ohlc_df.iloc[-1]
                    result = {
                        "symbol": symbol,
                        "expiration": expiration,
                        "strike": strike,
                        "right": right,
                        "close": row.get("close", 0),
                        "open": row.get("open", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                        "volume": row.get("volume", 0),
                        "date": trade_date,
                        "strike_type": strike_type
                    }
                    self._options_cache[cache_key] = result
                    return result
                return None
            
            row = eod_df.iloc[-1]
            result = {
                "symbol": symbol,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "close": row.get("close", 0),
                "bid": row.get("bid", 0),
                "ask": row.get("ask", 0),
                "volume": row.get("volume", 0),
                "open_interest": row.get("open_interest", 0),
                "iv": row.get("implied_volatility", 0),
                "delta": row.get("delta", 0),
                "gamma": row.get("gamma", 0),
                "theta": row.get("theta", 0),
                "vega": row.get("vega", 0),
                "underlying_price": row.get("underlying_price", spot_price),
                "date": trade_date,
                "strike_type": strike_type
            }
            
            self._options_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Error fetching option: {e}")
            return None
    
    def calculate_option_pnl(
        self,
        entry_premium: float,
        exit_premium: float,
        contracts: int = 1
    ) -> Dict:
        """Calculate actual option P&L"""
        entry_cost = entry_premium * 100 * contracts
        exit_value = exit_premium * 100 * contracts
        
        dollar_pnl = exit_value - entry_cost
        pct_pnl = (dollar_pnl / entry_cost * 100) if entry_cost > 0 else 0
        
        return {
            'entry_cost': entry_cost,
            'exit_value': exit_value,
            'dollar_pnl': dollar_pnl,
            'pct_pnl': pct_pnl
        }
    
    def backtest_eod(
        self,
        stock_df: pd.DataFrame,
        symbol: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Run EOD backtest using historical options data.
        
        Args:
            stock_df: DataFrame with daily stock OHLC and RSI
            symbol: Options symbol
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            
        Returns:
            DataFrame with trade results
        """
        df = stock_df.copy()
        
        # Ensure RSI is calculated
        if 'rsi' not in df.columns:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate relative volume
        if 'rel_volume' not in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['rel_volume'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
        trades = []
        in_position = False
        entry_data = None
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            if pd.isna(row['rsi']):
                continue
            
            # Get trade date
            if isinstance(row.name, pd.Timestamp):
                trade_date = row.name.strftime('%Y%m%d')
            else:
                trade_date = str(row.name)
            
            # Filter by date range
            if start_date and trade_date < start_date:
                continue
            if end_date and trade_date > end_date:
                continue
            
            curr_rsi = row['rsi']
            prev_rsi = prev_row['rsi']
            rel_volume = row.get('rel_volume', 1.0)
            
            # Entry logic
            if not in_position:
                direction = None
                
                # Oversold bounce (long)
                if prev_rsi < self.oversold and curr_rsi >= self.oversold:
                    direction = 'long'
                
                # Overbought reversal (short)
                elif prev_rsi > self.overbought and curr_rsi <= self.overbought:
                    direction = 'short'
                
                if direction:
                    # Assess confidence
                    confidence = self.assess_confidence(curr_rsi, rel_volume, direction)
                    strike_type = self.select_strike_type(confidence)
                    
                    # Get actual option data
                    option = self.get_historical_option_price(
                        symbol=symbol,
                        trade_date=trade_date,
                        spot_price=row['close'],
                        direction=direction,
                        strike_type=strike_type
                    )
                    
                    if option and option.get('close', 0) > 0:
                        entry_premium = option['close']
                        
                        entry_data = {
                            'entry_date': trade_date,
                            'entry_stock_price': row['close'],
                            'entry_rsi': curr_rsi,
                            'direction': direction,
                            'strike': option['strike'],
                            'expiration': option['expiration'],
                            'right': option['right'],
                            'strike_type': strike_type,
                            'confidence': confidence,
                            'entry_premium': entry_premium,
                            'entry_delta': option.get('delta', 0.5),
                            'entry_iv': option.get('iv', 0),
                            'entry_theta': option.get('theta', 0)
                        }
                        in_position = True
            
            # Exit logic (next day or RSI reversal)
            else:
                should_exit = False
                exit_reason = None
                
                # Hold period (1-3 days for swing trades)
                entry_dt = datetime.strptime(entry_data['entry_date'], '%Y%m%d')
                current_dt = datetime.strptime(trade_date, '%Y%m%d')
                days_held = (current_dt - entry_dt).days
                
                # RSI reversal
                if entry_data['direction'] == 'long' and curr_rsi >= 50:
                    should_exit = True
                    exit_reason = 'rsi_neutral'
                elif entry_data['direction'] == 'short' and curr_rsi <= 50:
                    should_exit = True
                    exit_reason = 'rsi_neutral'
                
                # Stop loss
                if entry_data['direction'] == 'long' and curr_rsi < self.oversold - 5:
                    should_exit = True
                    exit_reason = 'stop_loss'
                elif entry_data['direction'] == 'short' and curr_rsi > self.overbought + 5:
                    should_exit = True
                    exit_reason = 'stop_loss'
                
                # Max hold period (5 days)
                if days_held >= 5:
                    should_exit = True
                    exit_reason = 'time_exit'
                
                if should_exit:
                    # Get exit option price
                    exit_option = self.get_historical_option_price(
                        symbol=symbol,
                        trade_date=trade_date,
                        spot_price=row['close'],
                        direction=entry_data['direction'],
                        strike_type=entry_data['strike_type']
                    )
                    
                    if exit_option and exit_option.get('close', 0) > 0:
                        exit_premium = exit_option['close']
                    else:
                        # Estimate using delta
                        stock_move = row['close'] - entry_data['entry_stock_price']
                        if entry_data['direction'] == 'short':
                            stock_move = -stock_move
                        delta = abs(entry_data.get('entry_delta', 0.5))
                        exit_premium = max(0.01, entry_data['entry_premium'] + (stock_move * delta))
                    
                    # Calculate P&L
                    pnl = self.calculate_option_pnl(entry_data['entry_premium'], exit_premium)
                    
                    # Stock P&L for comparison
                    stock_pnl = (row['close'] - entry_data['entry_stock_price']) / entry_data['entry_stock_price'] * 100
                    if entry_data['direction'] == 'short':
                        stock_pnl = -stock_pnl
                    
                    trades.append({
                        **entry_data,
                        'exit_date': trade_date,
                        'exit_stock_price': row['close'],
                        'exit_premium': exit_premium,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'option_pnl_pct': pnl['pct_pnl'],
                        'option_pnl_dollar': pnl['dollar_pnl'],
                        'stock_pnl_pct': stock_pnl
                    })
                    
                    in_position = False
                    entry_data = None
        
        return pd.DataFrame(trades)
    
    def calculate_stats(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate backtest statistics"""
        if len(trades_df) == 0:
            return {'total_trades': 0}
        
        df = trades_df
        
        stats = {
            'total_trades': len(df),
            'win_rate': (df['option_pnl_pct'] > 0).mean() * 100,
            'avg_pnl': df['option_pnl_pct'].mean(),
            'total_pnl': df['option_pnl_pct'].sum(),
            'max_win': df['option_pnl_pct'].max(),
            'max_loss': df['option_pnl_pct'].min(),
            'avg_days_held': df['days_held'].mean() if 'days_held' in df.columns else 0,
            
            # Compare to stock
            'stock_total_pnl': df['stock_pnl_pct'].sum(),
            'options_vs_stock': df['option_pnl_pct'].sum() - df['stock_pnl_pct'].sum()
        }
        
        # Sharpe
        if df['option_pnl_pct'].std() > 0:
            stats['sharpe'] = (stats['avg_pnl'] / df['option_pnl_pct'].std()) * np.sqrt(252)
        else:
            stats['sharpe'] = 0
        
        # Profit factor
        wins = df[df['option_pnl_pct'] > 0]['option_pnl_pct'].sum()
        losses = abs(df[df['option_pnl_pct'] <= 0]['option_pnl_pct'].sum())
        stats['profit_factor'] = wins / losses if losses > 0 else float('inf')
        
        # By strike type
        for st in ['ATM', 'OTM', 'DEEP_OTM']:
            subset = df[df['strike_type'] == st]
            if len(subset) > 0:
                stats[f'{st}_trades'] = len(subset)
                stats[f'{st}_win_rate'] = (subset['option_pnl_pct'] > 0).mean() * 100
                stats[f'{st}_total_pnl'] = subset['option_pnl_pct'].sum()
        
        # By direction
        for direction in ['long', 'short']:
            subset = df[df['direction'] == direction]
            if len(subset) > 0:
                stats[f'{direction}_trades'] = len(subset)
                stats[f'{direction}_win_rate'] = (subset['option_pnl_pct'] > 0).mean() * 100
                stats[f'{direction}_total_pnl'] = subset['option_pnl_pct'].sum()
        
        return stats


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Real Options Backtester (ThetaData V3)")
    print("=" * 60)
    print("Requires Theta Terminal running on localhost:25503")
    print("=" * 60)
    
    # Quick connection test
    fetcher = ThetaDataFetcher()
    if fetcher.check_connection():
        print("✅ Connected to Theta Terminal")
    else:
        print("❌ Cannot connect - make sure Theta Terminal is running")