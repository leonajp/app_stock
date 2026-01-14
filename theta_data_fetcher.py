"""
ThetaData V3 API Integration for Options Data

V3 API Changes from V2:
- Endpoints: /v3/... instead of /v2/...
- Strike prices in dollars (not cents×1000)
- right: "call"/"put" instead of "C"/"P"
- No separate list endpoints - use wildcard (*) in queries
- Response formats: csv (default), json, ndjson
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from io import StringIO


class ThetaDataFetcher:
    """
    Fetches options data from ThetaData V3 REST API.
    
    Requires Theta Terminal V3 running (default port 25503)
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:25503"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: dict = None, format: str = "json") -> dict:
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        # Add format parameter
        if params is None:
            params = {}
        params["format"] = format
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            
            if format == "json":
                return response.json()
            else:
                return {"raw": response.text}
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Theta Terminal V3. "
                "Make sure it's running on localhost:25503. "
                "Download from https://www.thetadata.net"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Request timed out")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")
    
    def _csv_to_dataframe(self, csv_text: str) -> pd.DataFrame:
        """Convert CSV response to DataFrame"""
        if not csv_text or csv_text.strip() == "":
            return pd.DataFrame()
        return pd.read_csv(StringIO(csv_text))
    
    def check_connection(self) -> bool:
        """Check if Theta Terminal is running"""
        try:
            response = self.session.get(f"{self.base_url}/v3/stock/snapshot/quote", 
                                        params={"symbol": "AAPL"}, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    # =========================================================================
    # STOCK DATA
    # =========================================================================
    
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get current stock quote"""
        response = self.session.get(
            f"{self.base_url}/v3/stock/snapshot/quote",
            params={"symbol": symbol},
            timeout=10
        )
        if response.status_code == 200 and response.text:
            try:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty:
                    row = df.iloc[0]
                    bid = float(row["bid"]) if "bid" in row.index else 0
                    ask = float(row["ask"]) if "ask" in row.index else 0
                    return {
                        "symbol": symbol,
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "timestamp": str(row["timestamp"]) if "timestamp" in row.index else ""
                    }
            except Exception as e:
                print(f"Parse error: {e}")
        return {}
    
    def get_stock_history(self, symbol: str, start_date: str, end_date: str, 
                          interval: str = "1d") -> pd.DataFrame:
        """
        Get historical stock OHLC data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            interval: "1m", "5m", "1h", "1d"
        """
        response = self.session.get(
            f"{self.base_url}/v3/stock/history/ohlc",
            params={
                "symbol": symbol,
                "start": start_date,
                "end": end_date,
                "interval": interval
            },
            timeout=60
        )
        
        if response.status_code == 200:
            df = self._csv_to_dataframe(response.text)
            if not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            return df
        return pd.DataFrame()
    
    # =========================================================================
    # OPTIONS DATA - SNAPSHOTS
    # =========================================================================
    
    def get_option_chain(self, symbol: str, expiration: str = "*") -> pd.DataFrame:
        """
        Get full option chain snapshot.
        
        Args:
            symbol: Underlying symbol (e.g., 'APP', 'AAPL')
            expiration: Specific expiration (YYYYMMDD) or '*' for all
            
        Returns:
            DataFrame with all options quotes
        """
        response = self.session.get(
            f"{self.base_url}/v3/option/snapshot/quote",
            params={"symbol": symbol, "expiration": expiration},
            timeout=60
        )
        
        if response.status_code == 200:
            return self._csv_to_dataframe(response.text)
        return pd.DataFrame()
    
    def get_option_quote(self, symbol: str, expiration: str, strike: float, 
                         right: str) -> Dict:
        """
        Get quote for a specific option contract.
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration date (YYYY-MM-DD or YYYYMMDD)
            strike: Strike price in dollars (e.g., 670.0)
            right: "call"/"put" or "CALL"/"PUT"
        """
        # Normalize right to uppercase
        right_upper = right.upper()
        
        response = self.session.get(
            f"{self.base_url}/v3/option/snapshot/quote",
            params={
                "symbol": symbol,
                "expiration": expiration,
                "strike": str(strike),
                "right": right_upper
            },
            timeout=10
        )
        
        if response.status_code == 200 and response.text:
            try:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty:
                    row = df.iloc[0]
                    # Use direct indexing for pandas Series
                    bid = float(row["bid"]) if "bid" in row.index else 0
                    ask = float(row["ask"]) if "ask" in row.index else 0
                    return {
                        "symbol": symbol,
                        "expiration": expiration,
                        "strike": strike,
                        "right": right_upper,
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "bid_size": int(row["bid_size"]) if "bid_size" in row.index else 0,
                        "ask_size": int(row["ask_size"]) if "ask_size" in row.index else 0,
                        "timestamp": str(row["timestamp"]) if "timestamp" in row.index else ""
                    }
            except Exception as e:
                print(f"Parse error: {e}")
        return {}
    
    def get_option_greeks(self, symbol: str, expiration: str, strike: float,
                          right: str) -> Dict:
        """
        Get Greeks for a specific option contract.
        
        Returns: Dict with delta, gamma, theta, vega, rho, iv
        """
        # Normalize right to uppercase
        right_upper = right.upper()
        
        response = self.session.get(
            f"{self.base_url}/v3/option/snapshot/greeks/all",
            params={
                "symbol": symbol,
                "expiration": expiration,
                "strike": str(strike),
                "right": right_upper
            },
            timeout=10
        )
        
        if response.status_code == 200 and response.text:
            try:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty:
                    row = df.iloc[0]
                    # Use direct indexing for pandas Series
                    return {
                        "symbol": symbol,
                        "expiration": expiration,
                        "strike": strike,
                        "right": right_upper,
                        "iv": float(row["implied_vol"]) if "implied_vol" in row.index else 0,
                        "delta": float(row["delta"]) if "delta" in row.index else 0,
                        "gamma": float(row["gamma"]) if "gamma" in row.index else 0,
                        "theta": float(row["theta"]) if "theta" in row.index else 0,
                        "vega": float(row["vega"]) if "vega" in row.index else 0,
                        "rho": float(row["rho"]) if "rho" in row.index else 0,
                        "underlying_price": float(row["underlying_price"]) if "underlying_price" in row.index else 0,
                        "bid": float(row["bid"]) if "bid" in row.index else 0,
                        "ask": float(row["ask"]) if "ask" in row.index else 0
                    }
            except Exception as e:
                print(f"Parse error: {e}")
        return {}
    
    def get_option_chain_greeks(self, symbol: str, expiration: str = "*") -> pd.DataFrame:
        """Get Greeks for entire option chain"""
        response = self.session.get(
            f"{self.base_url}/v3/option/snapshot/greeks/all",
            params={"symbol": symbol, "expiration": expiration},
            timeout=60
        )
        
        if response.status_code == 200:
            return self._csv_to_dataframe(response.text)
        return pd.DataFrame()
    
    # =========================================================================
    # OPTIONS DATA - HISTORICAL
    # =========================================================================
    
    def get_option_history_ohlc(self, symbol: str, expiration: str, strike: float,
                                 right: str, date: str = None, 
                                 start_date: str = None, end_date: str = None,
                                 interval: str = None) -> pd.DataFrame:
        """
        Get historical OHLC data for a specific option.
        
        V3 API uses single 'date' parameter. For date ranges, makes multiple calls.
        
        Args:
            symbol: Underlying symbol
            expiration: Expiration (YYYY-MM-DD or YYYYMMDD)
            strike: Strike price in dollars
            right: "call"/"put" or "CALL"/"PUT"
            date: Single date (YYYYMMDD) - preferred for V3
            start_date: Start date (YYYYMMDD) - will loop through dates
            end_date: End date (YYYYMMDD) - will loop through dates
            interval: "1m", "5m", "1h", "1d" (optional)
        """
        right_upper = right.upper()
        
        # If single date provided, use it directly
        if date:
            params = {
                "symbol": symbol,
                "expiration": expiration,
                "strike": str(strike),
                "right": right_upper,
                "date": date
            }
            if interval:
                params["interval"] = interval
                
            response = self.session.get(
                f"{self.base_url}/v3/option/history/ohlc",
                params=params,
                timeout=60
            )
            
            if response.status_code == 200:
                return self._csv_to_dataframe(response.text)
            return pd.DataFrame()
        
        # If date range, loop through dates
        if start_date and end_date:
            all_data = []
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            current = start_dt
            while current <= end_dt:
                # Skip weekends
                if current.weekday() < 5:
                    date_str = current.strftime("%Y%m%d")
                    params = {
                        "symbol": symbol,
                        "expiration": expiration,
                        "strike": str(strike),
                        "right": right_upper,
                        "date": date_str
                    }
                    if interval:
                        params["interval"] = interval
                    
                    try:
                        response = self.session.get(
                            f"{self.base_url}/v3/option/history/ohlc",
                            params=params,
                            timeout=30
                        )
                        
                        if response.status_code == 200 and response.text:
                            df = self._csv_to_dataframe(response.text)
                            if not df.empty:
                                all_data.append(df)
                    except:
                        pass
                
                current += timedelta(days=1)
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    def get_option_history_greeks(self, symbol: str, expiration: str, strike: float,
                                   right: str, start_date: str, end_date: str,
                                   interval: str = "1d") -> pd.DataFrame:
        """Get historical Greeks for an option"""
        # Normalize right to uppercase
        right_upper = right.upper()
        
        response = self.session.get(
            f"{self.base_url}/v3/option/history/greeks/all",
            params={
                "symbol": symbol,
                "expiration": expiration,
                "strike": str(strike),
                "right": right_upper,
                "start": start_date,
                "end": end_date,
                "interval": interval
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return self._csv_to_dataframe(response.text)
        return pd.DataFrame()
    
    def get_option_history_eod(self, symbol: str, expiration: str, strike: float,
                                right: str, date: str = None,
                                start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get End-of-Day historical data for an option.
        Includes OHLC, volume, open interest, and Greeks.
        
        V3 API uses single 'date' parameter.
        """
        right_upper = right.upper()
        
        if date:
            response = self.session.get(
                f"{self.base_url}/v3/option/history/eod",
                params={
                    "symbol": symbol,
                    "expiration": expiration,
                    "strike": str(strike),
                    "right": right_upper,
                    "date": date
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return self._csv_to_dataframe(response.text)
            return pd.DataFrame()
        
        # Date range - loop through dates
        if start_date and end_date:
            all_data = []
            start_dt = datetime.strptime(start_date, "%Y%m%d")
            end_dt = datetime.strptime(end_date, "%Y%m%d")
            
            current = start_dt
            while current <= end_dt:
                if current.weekday() < 5:
                    date_str = current.strftime("%Y%m%d")
                    try:
                        response = self.session.get(
                            f"{self.base_url}/v3/option/history/eod",
                            params={
                                "symbol": symbol,
                                "expiration": expiration,
                                "strike": str(strike),
                                "right": right_upper,
                                "date": date_str
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200 and response.text:
                            df = self._csv_to_dataframe(response.text)
                            if not df.empty:
                                all_data.append(df)
                    except:
                        pass
                
                current += timedelta(days=1)
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
        
        return pd.DataFrame()
    
    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    
    def get_expirations(self, symbol: str) -> List[str]:
        """
        Get available expiration dates for a symbol, sorted chronologically.
        Uses option chain snapshot with wildcard and extracts unique expirations.
        """
        df = self.get_option_chain(symbol, "*")
        if not df.empty and "expiration" in df.columns:
            expirations = df["expiration"].unique().tolist()
            # Sort chronologically (works for both YYYY-MM-DD and YYYYMMDD)
            expirations = sorted([str(e) for e in expirations])
            return expirations
        return []
    
    def get_strikes(self, symbol: str, expiration: str) -> List[float]:
        """Get available strikes for an expiration"""
        df = self.get_option_chain(symbol, expiration)
        if not df.empty and "strike" in df.columns:
            strikes = df["strike"].unique().tolist()
            return sorted(strikes)
        return []
    
    def find_atm_strike(self, symbol: str, expiration: str, spot_price: float) -> Optional[float]:
        """Find the ATM strike closest to spot price"""
        strikes = self.get_strikes(symbol, expiration)
        if not strikes:
            return None
        return min(strikes, key=lambda x: abs(x - spot_price))
    
    def find_nearest_expiration(self, symbol: str, target_dte: int = 0) -> Optional[str]:
        """
        Find expiration closest to target DTE.
        
        Args:
            symbol: Underlying symbol
            target_dte: Target days to expiration (0 = nearest)
        """
        expirations = self.get_expirations(symbol)
        if not expirations:
            return None
        
        today = datetime.now()
        target_date = today + timedelta(days=target_dte)
        
        # Filter to valid expirations (>= today)
        valid_exps = []
        for exp in expirations:
            try:
                # Handle both formats: 2025-12-19 and 20251219
                exp_str = str(exp)
                if '-' in exp_str:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                else:
                    exp_date = datetime.strptime(exp_str, "%Y%m%d")
                
                if exp_date >= today:
                    valid_exps.append((exp_str, abs((exp_date - target_date).days)))
            except:
                continue
        
        if not valid_exps:
            return None
        
        # Sort by distance to target
        valid_exps.sort(key=lambda x: x[1])
        return valid_exps[0][0]
    
    def get_option_data_for_trade(self, symbol: str, spot_price: float,
                                   direction: str, strike_type: str = "ATM",
                                   target_dte: int = 0) -> Dict:
        """
        Get complete option data for a trade setup.
        
        Args:
            symbol: Underlying symbol
            spot_price: Current stock price
            direction: "long" (call) or "short" (put)
            strike_type: "ATM", "OTM", "DEEP_OTM"
            target_dte: Target days to expiration
            
        Returns:
            Dict with quote and Greeks
        """
        # Find expiration
        expiration = self.find_nearest_expiration(symbol, target_dte)
        if not expiration:
            return {"error": "No expiration found"}
        
        # Calculate target strike
        offset_map = {"ATM": 0, "OTM": 0.015, "DEEP_OTM": 0.03}
        offset = offset_map.get(strike_type, 0)
        
        if direction == "long":
            target_strike = spot_price * (1 + offset)
            right = "CALL"
        else:
            target_strike = spot_price * (1 - offset)
            right = "PUT"
        
        # Find closest actual strike
        strike = self.find_atm_strike(symbol, expiration, target_strike)
        if not strike:
            return {"error": "No strikes found"}
        
        # Get quote
        quote = self.get_option_quote(symbol, expiration, strike, right)
        if not quote:
            return {"error": "No quote available"}
        
        # Get Greeks
        greeks = self.get_option_greeks(symbol, expiration, strike, right)
        
        return {
            **quote,
            **greeks,
            "strike_type": strike_type,
            "direction": direction,
            "spot_price": spot_price
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("ThetaData V3 Options Fetcher")
    print("=" * 60)
    
    fetcher = ThetaDataFetcher()
    
    # Check connection
    if not fetcher.check_connection():
        print("❌ Cannot connect to Theta Terminal!")
        print("   Make sure it's running on localhost:25503")
        exit(1)
    
    print("✅ Connected to Theta Terminal V3")
    
    # Test with AAPL (more liquid)
    symbol = "AAPL"
    
    # Get stock quote
    print(f"\n📈 Stock quote for {symbol}:")
    quote = fetcher.get_stock_quote(symbol)
    print(f"   {quote}")
    
    # Get expirations
    print(f"\n📅 Getting expirations for {symbol}...")
    expirations = fetcher.get_expirations(symbol)
    print(f"   Found {len(expirations)} expirations")
    if expirations:
        print(f"   Next 5: {expirations[:5]}")
    
    # Get strikes for nearest expiration
    if expirations:
        exp = expirations[0]
        print(f"\n💰 Getting strikes for {exp}...")
        strikes = fetcher.get_strikes(symbol, exp)
        print(f"   Found {len(strikes)} strikes")
        if strikes:
            print(f"   Range: ${min(strikes):.0f} - ${max(strikes):.0f}")
        
        # Get specific option quote
        atm_strike = fetcher.find_atm_strike(symbol, exp, 230)
        if atm_strike:
            print(f"\n📊 Option quote for ${atm_strike} Call:")
            opt_quote = fetcher.get_option_quote(symbol, exp, atm_strike, "call")
            print(f"   Bid: ${opt_quote.get('bid', 0):.2f}")
            print(f"   Ask: ${opt_quote.get('ask', 0):.2f}")
            
            # Get Greeks
            print(f"\n📐 Greeks:")
            greeks = fetcher.get_option_greeks(symbol, exp, atm_strike, "call")
            print(f"   Delta: {greeks.get('delta', 0):.3f}")
            print(f"   Gamma: {greeks.get('gamma', 0):.4f}")
            print(f"   Theta: {greeks.get('theta', 0):.3f}")
            print(f"   IV: {greeks.get('iv', 0):.2%}")
    
    # Test APP
    print(f"\n{'='*60}")
    print("Testing APP...")
    app_exps = fetcher.get_expirations("APP")
    print(f"APP expirations: {len(app_exps)}")
    if app_exps:
        print(f"Next: {app_exps[:3]}")
    
    print("\n" + "=" * 60)
    print("✅ ThetaData V3 Fetcher Ready!")