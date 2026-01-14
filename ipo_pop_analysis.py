"""
IPO Pop Analysis Framework
==========================
Test factors affecting D1/D2 IPO returns:
1. IPO Price level ($30+ threshold)
2. Offering size ($1B+ threshold)
3. Hot sector (AI, Tech, Biotech)
4. Top underwriter (GS, MS, JPM)
5. Oversubscription ratio

Data sources: EODHD, Bloomberg, Polygon
Storage: ClickHouse
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    EODHD_KEY: str = os.getenv('EODHD_API_KEY', '67cb1a00acd811.08902624')
    ORATS_KEY: str = os.getenv('ORATS_API_KEY', '647512a9-15a4-4571-b089-44ec424d8ff8')
    POLYGON_KEY: str = os.getenv('POLYGON_API_KEY', 'XwKz5sDplukJRPvbdRtSjADlnWtmxedH')

    # ClickHouse connection
    CLICKHOUSE_HOST: str = 'i35q8zrtq4.us-east-2.aws.clickhouse.cloud'
    CLICKHOUSE_USER: str = 'default'
    CLICKHOUSE_PASSWORD: str = '~AiDc7hJ7m1Bv'
    CLICKHOUSE_DATABASE: str = 'market_data'
    
    # Analysis thresholds (adjustable)
    HIGH_PRICE_THRESHOLD: float = 30.0
    LARGE_OFFERING_THRESHOLD: float = 1e9  # $1B
    POP_THRESHOLD: float = 20.0  # 20% return
    
    # Hot sectors
    HOT_SECTORS: List[str] = None
    
    # Top underwriters
    TOP_UNDERWRITERS: List[str] = None
    
    def __post_init__(self):
        self.HOT_SECTORS = [
            'Artificial Intelligence', 'AI', 'Machine Learning',
            'Cloud', 'SaaS', 'Cybersecurity',
            'Semiconductor', 'Biotech', 'Biotechnology',
            'Electric Vehicle', 'EV', 'Clean Energy',
            'Fintech', 'Blockchain', 'Crypto'
        ]
        self.TOP_UNDERWRITERS = [
            'Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'JP Morgan',
            'Bank of America', 'BofA', 'Citigroup', 'Citi',
            'Barclays', 'Credit Suisse', 'Deutsche Bank', 'UBS'
        ]

config = Config()


# ============================================================
# DATA COLLECTION
# ============================================================

class IPODataCollector:
    """Collect IPO data from multiple sources"""

    def __init__(self, config: Config):
        self.config = config
        self._clickhouse_client = None

    @property
    def clickhouse_client(self):
        """Lazy-load ClickHouse client"""
        if self._clickhouse_client is None:
            try:
                import clickhouse_connect
                self._clickhouse_client = clickhouse_connect.get_client(
                    host=self.config.CLICKHOUSE_HOST,
                    user=self.config.CLICKHOUSE_USER,
                    password=self.config.CLICKHOUSE_PASSWORD,
                    database=self.config.CLICKHOUSE_DATABASE,
                    secure=True,
                    port=8443
                )
            except Exception as e:
                print(f"Warning: Could not connect to ClickHouse: {e}")
                self._clickhouse_client = None
        return self._clickhouse_client

    def get_daily_prices_clickhouse(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get daily OHLCV from ClickHouse daily_prepost table (primary source)"""
        if self.clickhouse_client is None:
            return pd.DataFrame()

        try:
            query = f"""
            SELECT
                trade_dt as date,
                adjO as open,
                adjH as high,
                adjL as low,
                adjC as close,
                rawO as raw_open,
                rawH as raw_high,
                rawL as raw_low,
                rawC as raw_close,
                volume
            FROM daily_prepost
            WHERE symbol = '{ticker}'
                AND trade_dt BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_dt
            """
            result = self.clickhouse_client.query(query)

            if not result.result_rows:
                return pd.DataFrame()

            df = pd.DataFrame(result.result_rows, columns=[
                'date', 'open', 'high', 'low', 'close',
                'raw_open', 'raw_high', 'raw_low', 'raw_close', 'volume'
            ])
            df['date'] = pd.to_datetime(df['date'])

            # Convert numeric columns to float (ClickHouse returns Decimal)
            numeric_cols = ['open', 'high', 'low', 'close', 'raw_open', 'raw_high', 'raw_low', 'raw_close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            print(f"Error fetching ClickHouse prices for {ticker}: {e}")
            return pd.DataFrame()

    def get_daily_prices_polygon(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fallback: Get daily OHLCV from Polygon/Massive API"""
        if not self.config.POLYGON_KEY:
            return pd.DataFrame()

        # Use Massive.com API (same format as Polygon)
        url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {'apiKey': self.config.POLYGON_KEY, 'adjusted': 'true'}

        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()

            if data.get('resultsCount', 0) == 0:
                return pd.DataFrame()

            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            print(f"Error fetching Polygon prices for {ticker}: {e}")
            return pd.DataFrame()

    def get_daily_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get daily prices - ClickHouse first, then Polygon fallback"""
        # Try ClickHouse first
        df = self.get_daily_prices_clickhouse(ticker, start_date, end_date)
        if not df.empty:
            return df

        # Fallback to Polygon/Massive API
        df = self.get_daily_prices_polygon(ticker, start_date, end_date)
        return df
    
    def get_ipo_calendar_clickhouse(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get IPO calendar from ClickHouse ipo_master table (primary source)
        """
        if self.clickhouse_client is None:
            print("Warning: ClickHouse not connected")
            return pd.DataFrame()

        try:
            query = f"""
            SELECT
                ticker,
                Name as name,
                ipo_date,
                ipo_price,
                ipo_shares_offered as shares_offered,
                offer_size_m * 1000000 as offering_size,
                exchange,
                underwriter,
                sector,
                industry,
                d1_open,
                d1_high,
                d1_low,
                d1_close,
                d1_volume,
                d2_open,
                d2_high,
                d2_close,
                d2_volume,
                d5_close,
                ret_d1,
                ret_d5,
                is_spac,
                is_biotech,
                bookrunner,
                all_underwriters
            FROM ipo_master
            WHERE ipo_date BETWEEN '{start_date}' AND '{end_date}'
                AND ipo_price > 0
                AND (exchange LIKE 'NASDAQ%' OR exchange LIKE 'NYSE%' OR exchange IN ('AMEX', 'ARCA', 'New York', 'NYSEAmerican'))
            ORDER BY ipo_date
            """
            result = self.clickhouse_client.query(query)

            if not result.result_rows:
                print(f"No IPOs found between {start_date} and {end_date}")
                return pd.DataFrame()

            columns = [
                'ticker', 'name', 'ipo_date', 'ipo_price', 'shares_offered',
                'offering_size', 'exchange', 'underwriter', 'sector', 'industry',
                'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
                'd2_open', 'd2_high', 'd2_close', 'd2_volume',
                'd5_close', 'ret_d1', 'ret_d5', 'is_spac', 'is_biotech',
                'bookrunner', 'all_underwriters'
            ]
            df = pd.DataFrame(result.result_rows, columns=columns)

            # Convert types
            df['ipo_date'] = pd.to_datetime(df['ipo_date'])

            # Convert all numeric columns to float (ClickHouse returns Decimal)
            numeric_cols = [
                'ipo_price', 'shares_offered', 'offering_size',
                'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
                'd2_open', 'd2_high', 'd2_close', 'd2_volume',
                'd5_close', 'ret_d1', 'ret_d5'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"  Loaded {len(df)} IPOs from ClickHouse ipo_master table")
            return df

        except Exception as e:
            print(f"Error fetching ClickHouse IPO data: {e}")
            return pd.DataFrame()

    def get_ipo_calendar_eodhd(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fallback: Get IPO calendar from EODHD API
        """
        url = "https://eodhd.com/api/calendar/ipos"
        params = {
            'api_token': self.config.EODHD_KEY,
            'from': start_date,
            'to': end_date,
            'fmt': 'json'
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            ipos = data.get('ipos', [])
            if not ipos:
                print(f"No IPOs found from EODHD between {start_date} and {end_date}")
                return pd.DataFrame()

            df = pd.DataFrame(ipos)

            # Standardize columns
            df = df.rename(columns={
                'code': 'ticker',
                'date': 'ipo_date',
                'price': 'ipo_price',
                'shares': 'shares_offered',
                'amount': 'offering_size',
            })

            # Convert types
            df['ipo_price'] = pd.to_numeric(df['ipo_price'], errors='coerce')
            df['offering_size'] = pd.to_numeric(df['offering_size'], errors='coerce')
            df['shares_offered'] = pd.to_numeric(df['shares_offered'], errors='coerce')

            # Filter US only
            if 'exchange' in df.columns:
                df = df[df['exchange'].isin(['NYSE', 'NASDAQ', 'AMEX', 'US'])]

            return df

        except Exception as e:
            print(f"Error fetching EODHD IPO data: {e}")
            return pd.DataFrame()

    def get_ipo_calendar(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get IPO calendar - ClickHouse first, then EODHD fallback"""
        # Try ClickHouse first (ipo_master table)
        df = self.get_ipo_calendar_clickhouse(start_date, end_date)
        if not df.empty:
            return df

        # Fallback to EODHD API
        print("  Falling back to EODHD API...")
        df = self.get_ipo_calendar_eodhd(start_date, end_date)
        return df
    
    def get_company_profile_eodhd(self, ticker: str) -> Dict:
        """Get company fundamentals for sector info"""
        url = f"https://eodhd.com/api/fundamentals/{ticker}.US"
        params = {'api_token': self.config.EODHD_KEY, 'fmt': 'json'}
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code != 200:
                return {}
            return resp.json()
        except:
            return {}


# ============================================================
# IPO ENRICHMENT & FEATURE ENGINEERING
# ============================================================

class IPOFeatureEngineer:
    """Add features for analysis"""
    
    def __init__(self, config: Config, collector: IPODataCollector):
        self.config = config
        self.collector = collector
    
    def calculate_returns(
        self,
        ticker: str,
        ipo_date: str,
        ipo_price: float
    ) -> Dict:
        """Calculate D1, D2 returns from IPO price"""

        if pd.isna(ipo_price) or ipo_price <= 0:
            return None

        ipo_dt = pd.to_datetime(ipo_date)
        end_dt = ipo_dt + timedelta(days=14)  # Buffer for weekends/holidays

        # Get prices from ClickHouse (primary) or Polygon (fallback)
        prices = self.collector.get_daily_prices(
            ticker,
            ipo_dt.strftime('%Y-%m-%d'),
            end_dt.strftime('%Y-%m-%d')
        )
        
        if prices.empty or len(prices) < 1:
            return None
        
        prices = prices.sort_values('date').reset_index(drop=True)
        
        result = {
            'ticker': ticker,
            'ipo_date': ipo_date,
            'ipo_price': ipo_price,
        }
        
        # D1 metrics
        if len(prices) >= 1:
            d1 = prices.iloc[0]
            result['d1_open'] = d1['open']
            result['d1_high'] = d1['high']
            result['d1_low'] = d1['low']
            result['d1_close'] = d1['close']
            result['d1_volume'] = d1.get('volume', 0)
            
            # Returns from IPO price
            result['d1_open_ret'] = (d1['open'] / ipo_price - 1) * 100
            result['d1_close_ret'] = (d1['close'] / ipo_price - 1) * 100
            result['d1_high_ret'] = (d1['high'] / ipo_price - 1) * 100
            result['d1_intraday_range'] = (d1['high'] - d1['low']) / d1['open'] * 100
        
        # D2 metrics
        if len(prices) >= 2:
            d2 = prices.iloc[1]
            result['d2_open'] = d2['open']
            result['d2_high'] = d2['high']
            result['d2_close'] = d2['close']
            result['d2_close_ret'] = (d2['close'] / ipo_price - 1) * 100
            result['d2_high_ret'] = (d2['high'] / ipo_price - 1) * 100
        
        # D5 metrics (first week)
        if len(prices) >= 5:
            d5 = prices.iloc[4]
            result['d5_close'] = d5['close']
            result['d5_close_ret'] = (d5['close'] / ipo_price - 1) * 100
            result['d5_max_high'] = prices.iloc[:5]['high'].max()
            result['d5_max_ret'] = (result['d5_max_high'] / ipo_price - 1) * 100
        
        return result
    
    def classify_features(self, row: pd.Series) -> Dict:
        """Add categorical features for analysis"""
        features = {}
        
        # 1. Price tier
        ipo_price = row.get('ipo_price', 0) or 0
        features['price_tier'] = (
            'high_30plus' if ipo_price >= 30 else
            'mid_15_30' if ipo_price >= 15 else
            'low_under_15'
        )
        features['is_high_price'] = ipo_price >= self.config.HIGH_PRICE_THRESHOLD
        
        # 2. Offering size tier
        offering_size = row.get('offering_size', 0) or 0
        features['size_tier'] = (
            'mega_1B_plus' if offering_size >= 1e9 else
            'large_500M_1B' if offering_size >= 500e6 else
            'mid_100M_500M' if offering_size >= 100e6 else
            'small_under_100M'
        )
        features['is_large_offering'] = offering_size >= self.config.LARGE_OFFERING_THRESHOLD
        
        # 3. Hot sector detection
        name = str(row.get('name', '')).lower()
        sector = str(row.get('sector', '')).lower()
        description = str(row.get('description', '')).lower()
        combined_text = f"{name} {sector} {description}"
        
        features['is_hot_sector'] = any(
            hot.lower() in combined_text 
            for hot in self.config.HOT_SECTORS
        )
        
        # Specific sector tags
        features['is_ai'] = any(x in combined_text for x in ['artificial intelligence', ' ai ', 'machine learning'])
        features['is_biotech'] = any(x in combined_text for x in ['biotech', 'therapeutics', 'pharma', 'oncology'])
        features['is_fintech'] = any(x in combined_text for x in ['fintech', 'payment', 'crypto', 'blockchain'])
        features['is_ev_clean'] = any(x in combined_text for x in ['electric vehicle', ' ev ', 'clean energy', 'solar'])
        
        # 4. Underwriter quality
        underwriter = str(row.get('underwriter', '')).lower()
        features['has_top_underwriter'] = any(
            uw.lower() in underwriter 
            for uw in self.config.TOP_UNDERWRITERS
        )
        
        # 5. Timing features
        ipo_date = pd.to_datetime(row.get('ipo_date'))
        if pd.notna(ipo_date):
            features['ipo_year'] = ipo_date.year
            features['ipo_month'] = ipo_date.month
            features['ipo_quarter'] = (ipo_date.month - 1) // 3 + 1
            features['ipo_weekday'] = ipo_date.weekday()
        
        return features
    
    def enrich_ipo_data(self, ipo_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Full enrichment pipeline"""

        results = []
        total = len(ipo_df)

        for idx, row in ipo_df.iterrows():
            ticker = row.get('ticker', '')
            ipo_date = row.get('ipo_date', '')
            ipo_price = row.get('ipo_price', 0)

            if verbose and idx % 10 == 0:
                print(f"Processing {idx+1}/{total}: {ticker}")

            # Skip invalid entries
            if not ticker or pd.isna(ipo_price) or ipo_price <= 0:
                continue

            # Use price data from ipo_master if available (calculate returns ourselves)
            d1_close = row.get('d1_close', 0) or 0
            d1_open = row.get('d1_open', 0) or 0

            if d1_close > 0 and d1_open > 0:
                # Use prices from ipo_master, calculate returns ourselves
                d1_high = row.get('d1_high', 0) or 0
                d1_low = row.get('d1_low', 0) or 0
                d2_close = row.get('d2_close', 0) or 0
                d5_close = row.get('d5_close', 0) or 0

                returns = {
                    'ticker': ticker,
                    'ipo_date': ipo_date,
                    'ipo_price': ipo_price,
                    'd1_open': d1_open,
                    'd1_high': d1_high,
                    'd1_low': d1_low,
                    'd1_close': d1_close,
                    'd1_volume': row.get('d1_volume', 0),
                    'd1_open_ret': (d1_open / ipo_price - 1) * 100,
                    'd1_close_ret': (d1_close / ipo_price - 1) * 100,
                    'd1_high_ret': (d1_high / ipo_price - 1) * 100 if d1_high > 0 else 0,
                    'd1_intraday_range': (d1_high - d1_low) / d1_open * 100 if d1_low > 0 else 0,
                    'd2_open': row.get('d2_open', 0),
                    'd2_high': row.get('d2_high', 0),
                    'd2_close': d2_close,
                    'd2_close_ret': (d2_close / ipo_price - 1) * 100 if d2_close > 0 else 0,
                    'd5_close': d5_close,
                    'd5_close_ret': (d5_close / ipo_price - 1) * 100 if d5_close > 0 else 0,
                }
            else:
                # Fetch prices from database
                returns = self.calculate_returns(ticker, ipo_date, ipo_price)
                if not returns:
                    continue

            # Get features
            features = self.classify_features(row)

            # Combine
            record = {**row.to_dict(), **returns, **features}
            results.append(record)
        
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Filter out obviously bad data (D1 return > 500% or < -90% is likely data error)
        # Normal IPO D1 returns range from -90% to ~400% in extreme cases
        before_filter = len(df)
        df = df[(df['d1_close_ret'] >= -90) & (df['d1_close_ret'] <= 500)]
        after_filter = len(df)
        if before_filter != after_filter:
            print(f"  Filtered {before_filter - after_filter} IPOs with suspicious returns (outside -90% to +500%)")

        # Add pop flags
        threshold = self.config.POP_THRESHOLD
        df['d1_pop'] = df['d1_close_ret'] >= threshold
        df['d2_pop'] = df.get('d2_close_ret', 0) >= threshold
        df['any_d1d2_pop'] = df['d1_pop'] | df['d2_pop']

        return df


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

class IPOAnalyzer:
    """Statistical analysis of IPO factors"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic summary statistics"""
        return df[[
            'd1_open_ret', 'd1_close_ret', 'd1_high_ret',
            'd2_close_ret', 'd5_close_ret'
        ]].describe().round(2)
    
    def factor_analysis(
        self, 
        df: pd.DataFrame, 
        factor_col: str,
        return_col: str = 'd1_close_ret'
    ) -> pd.DataFrame:
        """Analyze returns by factor"""
        
        grouped = df.groupby(factor_col).agg({
            return_col: ['count', 'mean', 'median', 'std'],
            'd1_pop': ['sum', 'mean'],
        }).round(2)
        
        grouped.columns = ['n', 'mean_ret', 'median_ret', 'std_ret', 'pop_count', 'pop_rate']
        grouped['pop_rate'] = (grouped['pop_rate'] * 100).round(1)
        
        return grouped.sort_values('mean_ret', ascending=False)
    
    def run_all_factor_tests(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run analysis for all factors"""
        results = {}
        
        factors = [
            ('price_tier', 'Price Tier'),
            ('is_high_price', 'High Price ($30+)'),
            ('size_tier', 'Offering Size Tier'),
            ('is_large_offering', 'Large Offering ($1B+)'),
            ('is_hot_sector', 'Hot Sector'),
            ('has_top_underwriter', 'Top Underwriter'),
            ('ipo_year', 'Year'),
            ('ipo_quarter', 'Quarter'),
        ]
        
        for col, name in factors:
            if col in df.columns:
                results[name] = self.factor_analysis(df, col)
        
        return results
    
    def statistical_significance(
        self, 
        df: pd.DataFrame,
        group_col: str,
        return_col: str = 'd1_close_ret'
    ) -> Dict:
        """T-test and Chi-square for factor significance"""
        from scipy import stats
        
        if df[group_col].dtype == bool:
            group_true = df[df[group_col]][return_col].dropna()
            group_false = df[~df[group_col]][return_col].dropna()
        else:
            groups = df[group_col].unique()
            if len(groups) != 2:
                return {'error': 'Need exactly 2 groups for t-test'}
            group_true = df[df[group_col] == groups[0]][return_col].dropna()
            group_false = df[df[group_col] == groups[1]][return_col].dropna()
        
        # T-test for means
        t_stat, t_pval = stats.ttest_ind(group_true, group_false)
        
        # Mann-Whitney U (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(group_true, group_false, alternative='two-sided')
        
        return {
            'group_true_n': len(group_true),
            'group_true_mean': group_true.mean(),
            'group_false_n': len(group_false),
            'group_false_mean': group_false.mean(),
            'diff': group_true.mean() - group_false.mean(),
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'significant_t': t_pval < 0.05,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_pval,
            'significant_mw': u_pval < 0.05,
        }
    
    def multi_factor_regression(self, df: pd.DataFrame) -> Dict:
        """OLS regression with multiple factors"""
        try:
            import statsmodels.api as sm
        except ImportError:
            return {'error': 'statsmodels not installed'}
        
        # Prepare features
        feature_cols = [
            'is_high_price', 'is_large_offering', 'is_hot_sector',
            'has_top_underwriter', 'ipo_price', 'offering_size'
        ]
        
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols].copy()
        X['offering_size_log'] = np.log1p(X.get('offering_size', 0))
        X = X.drop(columns=['offering_size'], errors='ignore')
        
        y = df['d1_close_ret'].dropna()
        X = X.loc[y.index]
        
        # Handle missing values
        mask = ~X.isna().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        
        return {
            'summary': model.summary(),
            'r_squared': model.rsquared,
            'coefficients': model.params.to_dict(),
            'pvalues': model.pvalues.to_dict(),
        }


# ============================================================
# VISUALIZATION
# ============================================================

def create_analysis_report(df: pd.DataFrame, results: Dict) -> str:
    """Generate text report"""
    
    report = []
    report.append("=" * 70)
    report.append("IPO POP FACTOR ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nTotal IPOs analyzed: {len(df)}")
    report.append(f"Date range: {df['ipo_date'].min()} to {df['ipo_date'].max()}")
    
    # Overall stats
    report.append(f"\n{'-' * 50}")
    report.append("OVERALL RETURNS")
    report.append(f"{'-' * 50}")
    report.append(f"D1 Close Return: mean={df['d1_close_ret'].mean():.1f}%, median={df['d1_close_ret'].median():.1f}%")
    report.append(f"D1 Pop Rate (>={config.POP_THRESHOLD}%): {df['d1_pop'].mean()*100:.1f}%")

    # Factor results
    for factor_name, factor_df in results.items():
        report.append(f"\n{'-' * 50}")
        report.append(f"FACTOR: {factor_name}")
        report.append(f"{'-' * 50}")
        report.append(factor_df.to_string())
    
    return "\n".join(report)


# ============================================================
# MAIN EXECUTION
# ============================================================

def run_analysis(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31',
    save_to_csv: bool = True,
    verbose: bool = True
):
    """Main analysis pipeline"""
    
    print(f"\n{'='*60}")
    print("IPO POP FACTOR ANALYSIS")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*60}\n")
    
    # Initialize
    collector = IPODataCollector(config)
    engineer = IPOFeatureEngineer(config, collector)
    analyzer = IPOAnalyzer(config)
    
    # Step 1: Get IPO calendar (ClickHouse primary, EODHD fallback)
    print("Step 1: Fetching IPO calendar...")
    ipo_df = collector.get_ipo_calendar(start_date, end_date)
    print(f"  Found {len(ipo_df)} IPOs")
    
    if ipo_df.empty:
        print("No IPO data found. Exiting.")
        return None, None
    
    # Step 2: Enrich with returns and features
    print("\nStep 2: Calculating returns and features...")
    enriched_df = engineer.enrich_ipo_data(ipo_df, verbose=verbose)
    print(f"  Successfully processed {len(enriched_df)} IPOs")
    
    if enriched_df.empty:
        print("No valid IPO data after enrichment. Exiting.")
        return None, None
    
    # Step 3: Run factor analysis
    print("\nStep 3: Running factor analysis...")
    factor_results = analyzer.run_all_factor_tests(enriched_df)
    
    # Step 4: Statistical significance tests
    print("\nStep 4: Statistical significance tests...")
    significance_results = {}
    
    for factor in ['is_high_price', 'is_large_offering', 'is_hot_sector', 'has_top_underwriter']:
        if factor in enriched_df.columns:
            sig = analyzer.statistical_significance(enriched_df, factor)
            significance_results[factor] = sig
            
            print(f"\n  {factor}:")
            print(f"    True group (n={sig['group_true_n']}): mean={sig['group_true_mean']:.1f}%")
            print(f"    False group (n={sig['group_false_n']}): mean={sig['group_false_mean']:.1f}%")
            print(f"    Difference: {sig['diff']:.1f}%")
            print(f"    T-test p-value: {sig['t_pvalue']:.4f} {'***' if sig['significant_t'] else ''}")
    
    # Step 5: Generate report
    print("\nStep 5: Generating report...")
    report = create_analysis_report(enriched_df, factor_results)
    print(report)
    
    # Save results
    if save_to_csv:
        enriched_df.to_csv('ipo_analysis_results.csv', index=False)
        print("\nResults saved to ipo_analysis_results.csv")
    
    return enriched_df, factor_results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Run with live data from ClickHouse (primary) and Polygon (fallback)
    print("Running IPO analysis with LIVE DATA from ClickHouse/Polygon...\n")

    df, results = run_analysis(
        start_date='2020-01-01',
        end_date='2024-12-31',
        verbose=True
    )

    if df is not None:
        print(f"\nSuccessfully analyzed {len(df)} IPOs with live data!")
        print("Results saved to ipo_analysis_results.csv")
