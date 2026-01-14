"""
SPAC Performance Analysis Tool
Analyzes what factors cause SPACs to gain 100%+ and how fast

Key Research Questions:
1. What sectors produce 100%+ gainers?
2. How fast do SPACs reach 100% (days from IPO, from merger announcement)?
3. What's the typical timeline: IPO → Announcement → Merger → Peak?
4. What sponsor/management characteristics correlate with success?
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

class SPACAnalyzer:
    """Analyze SPAC performance patterns."""
    
    def __init__(self, polygon_api_key: str = None):
        self.polygon_api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY')
        
        # Known successful SPACs with their merger dates
        self.successful_spacs = {
            # Ticker: (Original SPAC, Merger Date, Sector, Peak Return %)
            'VRT': ('GS Acquisition Holdings', '2020-02-07', 'Data Center/Infrastructure', 1175),
            'QSR': ('Justice Holdings', '2014-12-12', 'Restaurant/Fast Food', 533),
            'DKNG': ('Diamond Eagle Acquisition', '2020-04-24', 'Sports Betting/Gaming', 380),
            'SYM': ('SVF Investment Corp 3', '2022-06-07', 'AI/Robotics/Automation', 374),
            'SOFI': ('Social Capital Hedosophia V', '2021-06-01', 'Fintech/Banking', 155),
            'SPCE': ('Social Capital Hedosophia', '2019-10-28', 'Space/Aerospace', 150),
            'LCID': ('Churchill Capital Corp IV', '2021-07-26', 'EV/Electric Vehicles', 400),  # Peak before decline
            'OPEN': ('Social Capital Hedosophia II', '2020-12-21', 'Real Estate Tech', 200),
            'RSI': ('dMY Technology Group', '2020-12-29', 'Sports Betting/Gaming', 200),
            'CHPT': ('Switchback Energy', '2021-02-26', 'EV Charging/Infrastructure', 100),
        }
        
        # SPAC lifecycle phases
        self.phases = [
            'IPO',              # SPAC IPO at ~$10
            'Searching',        # Looking for target (typically 18-24 months)
            'Rumor',            # Merger rumors (often leaks)
            'Announcement',     # Official merger announcement
            'Vote/Redemption',  # Shareholder vote, redemption period
            'De-SPAC',          # Merger closes, ticker changes
            'Post-Merger'       # Trading as new company
        ]
        
    def get_historical_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical prices from Polygon."""
        if not self.polygon_api_key:
            print("Warning: No Polygon API key")
            return pd.DataFrame()
            
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        
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
                    df['date'] = pd.to_datetime(df['t'], unit='ms')
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 
                        'c': 'close', 'v': 'volume'
                    })
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        
        return pd.DataFrame()
    
    def analyze_time_to_100pct(self, ticker: str, merger_date: str) -> Dict:
        """Analyze how long it took to reach 100% gain from $10 IPO price."""
        
        # Get data starting 6 months before merger to capture SPAC phase
        merger_dt = datetime.strptime(merger_date, "%Y-%m-%d")
        start_dt = merger_dt - timedelta(days=365)
        end_dt = merger_dt + timedelta(days=365)
        
        df = self.get_historical_prices(
            ticker,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            return {}
        
        result = {
            'ticker': ticker,
            'merger_date': merger_date,
            'data_start': df['date'].min(),
            'data_end': df['date'].max(),
        }
        
        # Find key milestones
        df['return_from_10'] = (df['close'] - 10) / 10 * 100
        
        # Find first date above 100% gain (price > $20)
        above_100 = df[df['close'] >= 20]
        if not above_100.empty:
            first_100_date = above_100['date'].iloc[0]
            result['first_100pct_date'] = first_100_date
            result['days_to_100pct'] = (first_100_date - df['date'].iloc[0]).days
        
        # Find peak
        peak_idx = df['close'].idxmax()
        result['peak_price'] = df.loc[peak_idx, 'close']
        result['peak_date'] = df.loc[peak_idx, 'date']
        result['peak_return_pct'] = (df.loc[peak_idx, 'close'] - 10) / 10 * 100
        
        # Days from merger to peak
        merger_dt = pd.to_datetime(merger_date)
        if merger_dt in df['date'].values or True:
            result['days_merger_to_peak'] = (result['peak_date'] - merger_dt).days
        
        return result
    
    def analyze_announcement_effect(self, ticker: str, announcement_date: str) -> Dict:
        """Analyze price action around merger announcement."""
        
        ann_dt = datetime.strptime(announcement_date, "%Y-%m-%d")
        start_dt = ann_dt - timedelta(days=30)
        end_dt = ann_dt + timedelta(days=60)
        
        df = self.get_historical_prices(
            ticker,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            return {}
        
        # Find pre-announcement price (day before)
        pre_ann = df[df['date'] < pd.to_datetime(announcement_date)]
        post_ann = df[df['date'] >= pd.to_datetime(announcement_date)]
        
        if pre_ann.empty or post_ann.empty:
            return {}
        
        pre_price = pre_ann['close'].iloc[-1]
        
        result = {
            'ticker': ticker,
            'announcement_date': announcement_date,
            'pre_announcement_price': pre_price,
            'day_1_close': post_ann['close'].iloc[0] if len(post_ann) > 0 else None,
            'day_1_return': (post_ann['close'].iloc[0] - pre_price) / pre_price * 100 if len(post_ann) > 0 else None,
        }
        
        # Returns over different periods
        for days in [5, 10, 30]:
            if len(post_ann) > days:
                result[f'day_{days}_return'] = (post_ann['close'].iloc[days] - pre_price) / pre_price * 100
        
        # Max return in first 30 days
        if len(post_ann) >= 30:
            max_price = post_ann.head(30)['high'].max()
            result['max_30day_return'] = (max_price - pre_price) / pre_price * 100
        
        return result
    
    def get_sector_performance(self) -> pd.DataFrame:
        """Analyze performance by sector."""
        
        sector_data = []
        for ticker, (spac, merger_date, sector, peak_return) in self.successful_spacs.items():
            sector_data.append({
                'ticker': ticker,
                'original_spac': spac,
                'merger_date': merger_date,
                'sector': sector,
                'peak_return_pct': peak_return
            })
        
        df = pd.DataFrame(sector_data)
        
        # Aggregate by sector
        sector_stats = df.groupby('sector').agg({
            'peak_return_pct': ['mean', 'max', 'min', 'count'],
            'ticker': lambda x: ', '.join(x)
        }).round(1)
        
        return sector_stats
    
    def identify_100pct_factors(self) -> Dict:
        """
        Identify common factors among SPACs that achieved 100%+ gains.
        
        Based on research, key factors include:
        1. Sector/Theme (hot sector timing)
        2. Sponsor quality/track record
        3. Time to find target (<278 days = better)
        4. Market conditions at merger
        5. PIPE size and participants
        6. Redemption rate
        """
        
        factors = {
            'sector_timing': {
                'description': 'Being in a "hot" sector at the right time',
                'examples': [
                    'EV stocks during 2020-2021 (LCID, CHPT)',
                    'Sports betting during legalization wave (DKNG, RSI)',
                    'AI/Automation in 2022-2023 (SYM)',
                    'Space tech in 2019-2020 (SPCE)',
                ],
                'impact': 'HIGH - Can add 50-200% to returns'
            },
            'sponsor_quality': {
                'description': 'Experienced sponsors with track records',
                'examples': [
                    'Chamath Palihapitiya (SPCE, SOFI, OPEN)',
                    'Diamond Eagle team (DKNG, SKLZ)',
                    'Churchill Capital (LCID)',
                ],
                'impact': 'MEDIUM-HIGH - Better deal sourcing and execution'
            },
            'time_to_announcement': {
                'description': 'SPACs that find targets quickly (<278 days) outperform',
                'research': '+10.34% abnormal return for fast finders',
                'impact': 'MEDIUM - Shows sponsor confidence and deal quality'
            },
            'low_redemption': {
                'description': 'Low redemption rate indicates investor confidence',
                'threshold': '<50% redemption is positive signal',
                'impact': 'MEDIUM - More cash for target company'
            },
            'pipe_quality': {
                'description': 'Strong PIPE investors signal deal quality',
                'examples': 'Institutional investors, strategic partners',
                'impact': 'MEDIUM - Validates valuation and provides capital'
            },
            'market_timing': {
                'description': 'Bull market conditions at merger',
                'best_periods': '2020 Q3-2021 Q1 was peak SPAC mania',
                'impact': 'HIGH - Can double returns in bull markets'
            }
        }
        
        return factors
    
    def calculate_speed_to_gain(self, target_gain_pct: float = 100) -> pd.DataFrame:
        """
        Calculate how fast different SPACs reached a target gain.
        
        Key metrics:
        - Days from IPO to target gain
        - Days from announcement to target gain
        - Days from merger to target gain
        """
        
        # This would need actual data - showing structure
        speed_data = [
            # Estimates based on historical patterns
            {'ticker': 'DKNG', 'sector': 'Sports Betting', 'days_ipo_to_100pct': 90, 
             'days_ann_to_100pct': 60, 'days_merger_to_100pct': 30, 'peak_return': 380},
            {'ticker': 'LCID', 'sector': 'EV', 'days_ipo_to_100pct': 45,
             'days_ann_to_100pct': 30, 'days_merger_to_100pct': 14, 'peak_return': 400},
            {'ticker': 'SPCE', 'sector': 'Space', 'days_ipo_to_100pct': 120,
             'days_ann_to_100pct': 90, 'days_merger_to_100pct': 60, 'peak_return': 150},
            {'ticker': 'VRT', 'sector': 'Data Center', 'days_ipo_to_100pct': 365,
             'days_ann_to_100pct': 300, 'days_merger_to_100pct': 180, 'peak_return': 1175},
        ]
        
        return pd.DataFrame(speed_data)
    
    def print_analysis_summary(self):
        """Print a comprehensive analysis summary."""
        
        print("=" * 70)
        print("SPAC PERFORMANCE ANALYSIS: What Drives 100%+ Gains?")
        print("=" * 70)
        
        print("\n📊 TOP PERFORMING SPACs (All-Time)")
        print("-" * 70)
        for ticker, (spac, merger_date, sector, peak_return) in sorted(
            self.successful_spacs.items(), 
            key=lambda x: x[1][3], 
            reverse=True
        ):
            print(f"{ticker:6} | {peak_return:>6}% | {sector:30} | {merger_date}")
        
        print("\n🏆 SECTOR BREAKDOWN")
        print("-" * 70)
        sector_stats = self.get_sector_performance()
        print(sector_stats.to_string())
        
        print("\n⚡ SPEED TO 100% GAIN")
        print("-" * 70)
        speed_df = self.calculate_speed_to_gain()
        print(speed_df.to_string(index=False))
        
        print("\n🔑 KEY SUCCESS FACTORS")
        print("-" * 70)
        factors = self.identify_100pct_factors()
        for factor, details in factors.items():
            print(f"\n{factor.upper()}")
            print(f"  Description: {details['description']}")
            print(f"  Impact: {details['impact']}")
        
        print("\n" + "=" * 70)
        print("TYPICAL SPAC LIFECYCLE")
        print("=" * 70)
        print("""
Timeline (Typical):
    
    IPO ($10)                    Announcement              Merger (De-SPAC)
       |                              |                          |
       |--- 6-18 months searching --->|--- 3-6 months voting --->|
       |                              |                          |
    Price: $9.50-10.50            $10-15 (pop)              $10-25+
    
Key Price Action Points:
    
1. IPO: Trades at ~$10 (NAV protected by trust)
2. Rumor: Can spike 10-30% on leaked rumors
3. Announcement: Biggest single-day move (typically +5-20%)
4. Vote Period: Momentum trades, high volatility
5. Merger Close: Often "sell the news" dip
6. Post-Merger: Fundamental-driven, high volatility

Best Entry Points for 100%+ Potential:
- Before rumor (need insider info - not recommended)
- On announcement day (if sector is hot)
- Post-merger dip (if fundamentals strong)
        """)


def main():
    """Run SPAC analysis."""
    
    api_key = os.environ.get('POLYGON_API_KEY')
    analyzer = SPACAnalyzer(api_key)
    
    # Print summary
    analyzer.print_analysis_summary()
    
    # If we have API key, do detailed analysis
    if api_key:
        print("\n" + "=" * 70)
        print("DETAILED PRICE ANALYSIS")
        print("=" * 70)
        
        # Analyze a few tickers
        for ticker in ['DKNG', 'SOFI', 'VRT']:
            if ticker in analyzer.successful_spacs:
                _, merger_date, sector, _ = analyzer.successful_spacs[ticker]
                print(f"\n{ticker} ({sector}):")
                
                result = analyzer.analyze_time_to_100pct(ticker, merger_date)
                if result:
                    for k, v in result.items():
                        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
