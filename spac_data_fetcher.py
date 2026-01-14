"""
SPAC Data Fetcher Module
=========================
Fetches real SPAC data from multiple sources:
- Polygon.io (adjusted & unadjusted prices)
- Bloomberg API (if available)
- SEC EDGAR (filings, announcements)
- SPAC Research / SPACInsider APIs (reference data)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from io import StringIO
import time


class PolygonDataFetcher:
    """Fetch data from Polygon.io API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def get_daily_bars(self, ticker: str, start_date: str, end_date: str,
                       adjusted: bool = True) -> pd.DataFrame:
        """
        Get daily OHLCV bars.
        
        Args:
            ticker: Stock ticker
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            adjusted: If True, use split/dividend adjusted prices
        """
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        
        try:
            response = requests.get(url, params={
                "apiKey": self.api_key,
                "adjusted": str(adjusted).lower(),
                "sort": "asc",
                "limit": 50000
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low',
                        'c': 'close', 'v': 'volume', 'vw': 'vwap',
                        'n': 'transactions'
                    })
                    df['ticker'] = ticker
                    df['adjusted'] = adjusted
                    return df[['date', 'datetime', 'ticker', 'open', 'high', 'low', 
                              'close', 'volume', 'vwap', 'transactions', 'adjusted']]
            elif response.status_code == 429:
                print(f"Rate limited on {ticker}, waiting...")
                time.sleep(12)  # Free tier: 5 calls/minute
                return self.get_daily_bars(ticker, start_date, end_date, adjusted)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        
        return pd.DataFrame()
    
    def get_intraday_bars(self, ticker: str, date: str, 
                          timespan: str = "minute") -> pd.DataFrame:
        """Get intraday bars (1-min, 5-min, etc.)."""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{date}/{date}"
        
        try:
            response = requests.get(url, params={
                "apiKey": self.api_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low',
                        'c': 'close', 'v': 'volume', 'vw': 'vwap'
                    })
                    df['ticker'] = ticker
                    return df
        except Exception as e:
            print(f"Error: {e}")
        
        return pd.DataFrame()
    
    def get_ticker_details(self, ticker: str) -> Dict:
        """Get ticker details (company info, market cap, etc.)."""
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"
        
        try:
            response = requests.get(url, params={"apiKey": self.api_key}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', {})
        except Exception as e:
            print(f"Error: {e}")
        
        return {}
    
    def get_stock_splits(self, ticker: str) -> pd.DataFrame:
        """Get stock split history (important for adjusted vs unadjusted)."""
        url = f"{self.base_url}/v3/reference/splits"
        
        try:
            response = requests.get(url, params={
                "apiKey": self.api_key,
                "ticker": ticker,
                "limit": 100
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    return pd.DataFrame(data['results'])
        except Exception as e:
            print(f"Error: {e}")
        
        return pd.DataFrame()
    
    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Get dividend history."""
        url = f"{self.base_url}/v3/reference/dividends"
        
        try:
            response = requests.get(url, params={
                "apiKey": self.api_key,
                "ticker": ticker,
                "limit": 100
            }, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    return pd.DataFrame(data['results'])
        except Exception as e:
            print(f"Error: {e}")
        
        return pd.DataFrame()
    
    def search_tickers(self, search: str, type: str = None) -> pd.DataFrame:
        """Search for tickers."""
        url = f"{self.base_url}/v3/reference/tickers"
        
        params = {
            "apiKey": self.api_key,
            "search": search,
            "active": "true",
            "limit": 100
        }
        if type:
            params['type'] = type
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    return pd.DataFrame(data['results'])
        except Exception as e:
            print(f"Error: {e}")
        
        return pd.DataFrame()


class BloombergDataFetcher:
    """Fetch data from Bloomberg API (requires blpapi and Terminal)."""
    
    def __init__(self, host: str = "localhost", port: int = 8194):
        self.host = host
        self.port = port
        self.session = None
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Bloomberg API is available."""
        try:
            import blpapi
            return True
        except ImportError:
            return False
    
    def connect(self) -> bool:
        """Connect to Bloomberg session."""
        if not self.available:
            return False
        
        try:
            import blpapi
            
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost(self.host)
            sessionOptions.setServerPort(self.port)
            
            self.session = blpapi.Session(sessionOptions)
            if not self.session.start():
                return False
            
            if not self.session.openService("//blp/refdata"):
                return False
            
            return True
        except Exception as e:
            print(f"Bloomberg connection error: {e}")
            return False
    
    def get_historical_data(self, ticker: str, start_date: str, end_date: str,
                            fields: List[str] = None, adjusted: bool = True) -> pd.DataFrame:
        """
        Get historical data from Bloomberg.
        
        Args:
            ticker: Ticker (will append " US Equity" if needed)
            start_date: YYYYMMDD
            end_date: YYYYMMDD
            fields: List of Bloomberg fields
            adjusted: Use adjusted prices
        """
        if not self.available or not self.session:
            if not self.connect():
                return pd.DataFrame()
        
        try:
            import blpapi
            
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Format ticker
            bbg_ticker = ticker if " " in ticker else f"{ticker} US Equity"
            request.getElement("securities").appendValue(bbg_ticker)
            
            # Default fields
            if fields is None:
                if adjusted:
                    fields = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"]
                else:
                    # Unadjusted prices
                    fields = ["PX_LAST_RAW", "PX_OPEN_RAW", "PX_HIGH_RAW", "PX_LOW_RAW", "PX_VOLUME"]
            
            for field in fields:
                request.getElement("fields").appendValue(field)
            
            # Date range
            request.set("startDate", start_date.replace("-", ""))
            request.set("endDate", end_date.replace("-", ""))
            request.set("periodicitySelection", "DAILY")
            
            self.session.sendRequest(request)
            
            data = []
            while True:
                ev = self.session.nextEvent(500)
                for msg in ev:
                    if msg.hasElement("securityData"):
                        secData = msg.getElement("securityData")
                        fieldData = secData.getElement("fieldData")
                        for i in range(fieldData.numValues()):
                            field = fieldData.getValueAsElement(i)
                            row = {'date': field.getElementAsDatetime("date")}
                            for f in fields:
                                try:
                                    row[f] = field.getElementAsFloat(f)
                                except:
                                    row[f] = None
                            data.append(row)
                
                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df['ticker'] = ticker
                df['adjusted'] = adjusted
                df['source'] = 'bloomberg'
                
                # Rename columns to standard format
                rename_map = {
                    'PX_LAST': 'close', 'PX_LAST_RAW': 'close',
                    'PX_OPEN': 'open', 'PX_OPEN_RAW': 'open',
                    'PX_HIGH': 'high', 'PX_HIGH_RAW': 'high',
                    'PX_LOW': 'low', 'PX_LOW_RAW': 'low',
                    'PX_VOLUME': 'volume'
                }
                df = df.rename(columns=rename_map)
                return df
                
        except Exception as e:
            print(f"Bloomberg error: {e}")
        
        return pd.DataFrame()
    
    def get_realtime_quote(self, ticker: str) -> Dict:
        """Get real-time quote."""
        if not self.available or not self.session:
            if not self.connect():
                return {}
        
        try:
            import blpapi
            
            refDataService = self.session.getService("//blp/refdata")
            request = refDataService.createRequest("ReferenceDataRequest")
            
            bbg_ticker = ticker if " " in ticker else f"{ticker} US Equity"
            request.getElement("securities").appendValue(bbg_ticker)
            
            fields = ["LAST_PRICE", "BID", "ASK", "VOLUME", "PX_CHANGE_PCT_1D"]
            for field in fields:
                request.getElement("fields").appendValue(field)
            
            self.session.sendRequest(request)
            
            result = {}
            while True:
                ev = self.session.nextEvent(500)
                for msg in ev:
                    if msg.hasElement("securityData"):
                        secData = msg.getElement("securityData")
                        for i in range(secData.numValues()):
                            sec = secData.getValueAsElement(i)
                            fieldData = sec.getElement("fieldData")
                            for field in fields:
                                try:
                                    result[field.lower()] = fieldData.getElementAsFloat(field)
                                except:
                                    pass
                
                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            
            return result
            
        except Exception as e:
            print(f"Bloomberg error: {e}")
        
        return {}
    
    def disconnect(self):
        """Disconnect from Bloomberg."""
        if self.session:
            self.session.stop()
            self.session = None


class SECEdgarFetcher:
    """Fetch SEC filings from EDGAR."""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov"
        self.headers = {
            "User-Agent": "SPAC Analysis Tool/1.0 (contact@example.com)"
        }
    
    def get_company_filings(self, cik: str, form_type: str = None) -> pd.DataFrame:
        """Get company filings from EDGAR."""
        # Pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)
        url = f"{self.base_url}/submissions/CIK{cik_padded}.json"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                
                filings = data.get('filings', {}).get('recent', {})
                if filings:
                    df = pd.DataFrame(filings)
                    
                    if form_type:
                        df = df[df['form'] == form_type]
                    
                    return df
        except Exception as e:
            print(f"SEC EDGAR error: {e}")
        
        return pd.DataFrame()
    
    def search_spac_filings(self, days_back: int = 30) -> pd.DataFrame:
        """Search for recent SPAC-related filings."""
        # Search for S-1, DEFM14A (merger proxy), 8-K filings
        # This is a simplified version - full implementation would use EDGAR full-text search
        
        # For now, return sample data structure
        print("Note: Full SEC EDGAR search requires additional API access")
        return pd.DataFrame()


class SPACDataAggregator:
    """
    Aggregates data from multiple sources and provides unified interface.
    """
    
    def __init__(self, polygon_api_key: str = None, use_bloomberg: bool = False):
        self.polygon = PolygonDataFetcher(polygon_api_key) if polygon_api_key else None
        self.bloomberg = BloombergDataFetcher() if use_bloomberg else None
        self.sec = SECEdgarFetcher()
        
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
    
    def get_price_data(self, ticker: str, start_date: str, end_date: str,
                       adjusted: bool = True, source: str = "auto") -> pd.DataFrame:
        """
        Get price data from best available source.
        
        Args:
            ticker: Stock ticker
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            adjusted: Use adjusted prices
            source: "polygon", "bloomberg", or "auto" (try both)
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{adjusted}_{source}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data
        
        df = pd.DataFrame()
        
        # Try Bloomberg first if requested
        if source in ["bloomberg", "auto"] and self.bloomberg and self.bloomberg.available:
            df = self.bloomberg.get_historical_data(ticker, start_date, end_date, adjusted=adjusted)
            if not df.empty:
                df['source'] = 'bloomberg'
        
        # Fall back to Polygon
        if df.empty and source in ["polygon", "auto"] and self.polygon:
            df = self.polygon.get_daily_bars(ticker, start_date, end_date, adjusted)
            if not df.empty:
                df['source'] = 'polygon'
        
        # Cache result
        if not df.empty:
            self.cache[cache_key] = (datetime.now(), df)
        
        return df
    
    def get_both_price_types(self, ticker: str, start_date: str, end_date: str,
                             source: str = "polygon") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get both adjusted and unadjusted prices for comparison.
        
        Returns:
            Tuple of (adjusted_df, unadjusted_df)
        """
        adjusted = self.get_price_data(ticker, start_date, end_date, adjusted=True, source=source)
        unadjusted = self.get_price_data(ticker, start_date, end_date, adjusted=False, source=source)
        
        return adjusted, unadjusted
    
    def compare_price_sources(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Compare prices from different sources (useful for data validation).
        """
        results = []
        
        if self.polygon:
            poly_adj = self.polygon.get_daily_bars(ticker, start_date, end_date, adjusted=True)
            if not poly_adj.empty:
                poly_adj['source'] = 'polygon_adjusted'
                results.append(poly_adj)
            
            poly_unadj = self.polygon.get_daily_bars(ticker, start_date, end_date, adjusted=False)
            if not poly_unadj.empty:
                poly_unadj['source'] = 'polygon_unadjusted'
                results.append(poly_unadj)
        
        if self.bloomberg and self.bloomberg.available:
            bbg_adj = self.bloomberg.get_historical_data(ticker, start_date, end_date, adjusted=True)
            if not bbg_adj.empty:
                bbg_adj['source'] = 'bloomberg_adjusted'
                results.append(bbg_adj)
            
            bbg_unadj = self.bloomberg.get_historical_data(ticker, start_date, end_date, adjusted=False)
            if not bbg_unadj.empty:
                bbg_unadj['source'] = 'bloomberg_unadjusted'
                results.append(bbg_unadj)
        
        if results:
            return pd.concat(results, ignore_index=True)
        
        return pd.DataFrame()


# =============================================================================
# SPAC-SPECIFIC DATA FUNCTIONS
# =============================================================================

def fetch_active_spacs_from_sec() -> pd.DataFrame:
    """
    Fetch list of active SPACs from SEC filings.
    
    Note: This is a simplified version. Full implementation would:
    1. Query EDGAR for recent S-1 filings with SPAC keywords
    2. Parse filing details for trust value, deadline, etc.
    3. Cross-reference with exchange listings
    """
    # For demo, return curated list
    # In production, this would scrape SEC EDGAR
    
    print("Fetching active SPACs from SEC EDGAR...")
    print("Note: Full implementation requires SEC EDGAR API access")
    
    return pd.DataFrame()


def calculate_spac_metrics(price_df: pd.DataFrame, ipo_date: str, 
                           announcement_date: str = None,
                           merger_date: str = None) -> Dict:
    """
    Calculate SPAC-specific metrics from price data.
    
    Returns:
        Dict with metrics like:
        - days_to_announcement
        - announcement_return
        - merger_return
        - peak_return
        - current_vs_trust
    """
    if price_df.empty:
        return {}
    
    metrics = {}
    
    # Ensure date column is datetime
    price_df = price_df.copy()
    if 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
    
    ipo_dt = pd.to_datetime(ipo_date)
    
    # IPO price (should be ~$10 for SPACs)
    ipo_prices = price_df[price_df['date'] >= ipo_dt].head(5)
    if not ipo_prices.empty:
        metrics['ipo_price'] = ipo_prices['close'].iloc[0]
    else:
        metrics['ipo_price'] = 10.0  # Default SPAC IPO price
    
    # Announcement metrics
    if announcement_date:
        ann_dt = pd.to_datetime(announcement_date)
        metrics['days_to_announcement'] = (ann_dt - ipo_dt).days
        
        ann_prices = price_df[price_df['date'] >= ann_dt].head(5)
        if not ann_prices.empty:
            metrics['announcement_price'] = ann_prices['close'].iloc[0]
            metrics['announcement_return'] = (
                (metrics['announcement_price'] - metrics['ipo_price']) / 
                metrics['ipo_price'] * 100
            )
    
    # Merger metrics
    if merger_date:
        merger_dt = pd.to_datetime(merger_date)
        
        merger_prices = price_df[price_df['date'] >= merger_dt].head(5)
        if not merger_prices.empty:
            metrics['merger_price'] = merger_prices['close'].iloc[0]
            metrics['merger_return'] = (
                (metrics['merger_price'] - metrics['ipo_price']) / 
                metrics['ipo_price'] * 100
            )
    
    # Peak and current
    metrics['peak_price'] = price_df['high'].max()
    metrics['peak_return'] = (metrics['peak_price'] - metrics['ipo_price']) / metrics['ipo_price'] * 100
    
    metrics['current_price'] = price_df['close'].iloc[-1]
    metrics['current_return'] = (metrics['current_price'] - metrics['ipo_price']) / metrics['ipo_price'] * 100
    
    # Trust premium/discount (assuming $10 trust value)
    metrics['trust_premium'] = (metrics['current_price'] - 10.0) / 10.0 * 100
    
    return metrics


def identify_spac_events(price_df: pd.DataFrame) -> List[Dict]:
    """
    Identify potential SPAC events from price/volume patterns.
    
    Looks for:
    - Announcement pops (sudden price + volume spike)
    - Pre-merger runups
    - Post-merger dumps
    """
    if price_df.empty or len(price_df) < 20:
        return []
    
    events = []
    price_df = price_df.copy()
    
    # Calculate metrics
    price_df['return'] = price_df['close'].pct_change() * 100
    price_df['volume_ratio'] = price_df['volume'] / price_df['volume'].rolling(20).mean()
    
    # Find significant moves (>10% with high volume)
    significant = price_df[
        (abs(price_df['return']) > 10) & 
        (price_df['volume_ratio'] > 3)
    ]
    
    for _, row in significant.iterrows():
        event_type = "Announcement Pop" if row['return'] > 0 else "Selloff"
        events.append({
            'date': row['date'],
            'type': event_type,
            'return': row['return'],
            'volume_ratio': row['volume_ratio'],
            'price': row['close']
        })
    
    return events


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Get API key
    api_key = os.environ.get('POLYGON_API_KEY')
    
    if not api_key:
        print("Set POLYGON_API_KEY environment variable")
        print("Example: export POLYGON_API_KEY=your_key_here")
    else:
        # Initialize aggregator
        aggregator = SPACDataAggregator(
            polygon_api_key=api_key,
            use_bloomberg=False  # Set True if Bloomberg available
        )
        
        # Fetch DKNG data (successful SPAC)
        print("\n" + "="*60)
        print("FETCHING DKNG (DraftKings) DATA")
        print("="*60)
        
        # Get both adjusted and unadjusted
        adj_df, unadj_df = aggregator.get_both_price_types(
            "DKNG",
            "2020-04-24",  # Merger date
            "2024-12-15"
        )
        
        if not adj_df.empty:
            print(f"\nAdjusted prices: {len(adj_df)} rows")
            print(adj_df.head())
            
            # Calculate metrics
            metrics = calculate_spac_metrics(
                adj_df,
                ipo_date="2020-04-24",
                announcement_date="2019-12-23",
                merger_date="2020-04-24"
            )
            
            print("\nSPAC Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
        
        # Compare adjusted vs unadjusted
        if not adj_df.empty and not unadj_df.empty:
            print("\n" + "="*60)
            print("ADJUSTED vs UNADJUSTED COMPARISON")
            print("="*60)
            
            # Merge on date
            comparison = adj_df[['date', 'close']].merge(
                unadj_df[['date', 'close']], 
                on='date', 
                suffixes=('_adj', '_unadj')
            )
            
            comparison['diff_pct'] = (
                (comparison['close_adj'] - comparison['close_unadj']) / 
                comparison['close_unadj'] * 100
            )
            
            print(f"\nPrice difference stats:")
            print(f"  Mean diff: {comparison['diff_pct'].mean():.2f}%")
            print(f"  Max diff: {comparison['diff_pct'].max():.2f}%")
            print(f"  Min diff: {comparison['diff_pct'].min():.2f}%")
