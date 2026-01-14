"""
SPAC Analysis Dashboard
========================
Comprehensive SPAC analysis with 4 integrated modules:
1. SPAC Screener - Track active SPACs, filter by sector, days to deadline, trust value
2. Announcement Alerts - Monitor for new merger announcements  
3. Historical Backtester - Test strategies (buy announcement, sell merger, etc.)
4. Sector Rotation - Identify hot sectors for SPAC mergers

Data Sources:
- Polygon.io (adjusted & unadjusted prices)
- Bloomberg API (if available)
- SEC EDGAR (filings)
- SPAC Research / SPACInsider (reference data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from io import StringIO
import time

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SPAC Analysis Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Financial Terminal Style
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a2332;
        --accent-green: #00ff88;
        --accent-red: #ff4757;
        --accent-blue: #00d4ff;
        --accent-yellow: #ffd93d;
        --accent-purple: #a855f7;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border-color: #2d3748;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-green), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .module-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--accent-blue);
        border-bottom: 2px solid var(--accent-blue);
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .positive { color: var(--accent-green) !important; }
    .negative { color: var(--accent-red) !important; }
    .neutral { color: var(--accent-blue) !important; }
    .warning { color: var(--accent-yellow) !important; }
    
    .spac-table {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    .alert-box {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(0, 212, 255, 0.2));
        border: 1px solid var(--accent-purple);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .hot-sector {
        background: linear-gradient(90deg, rgba(0, 255, 136, 0.2), transparent);
        border-left: 4px solid var(--accent-green);
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: var(--text-secondary);
        background: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        color: white !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace;
    }
    
    .sidebar .stSelectbox label, .sidebar .stMultiSelect label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA PROVIDER CLASS
# =============================================================================
class SPACDataProvider:
    """Unified data provider for Polygon and Bloomberg."""
    
    def __init__(self, polygon_api_key: str = None, bloomberg_available: bool = False):
        self.polygon_api_key = polygon_api_key
        self.bloomberg_available = bloomberg_available
        self.cache = {}
        
    def get_price_data(self, ticker: str, start_date: str, end_date: str, 
                       adjusted: bool = True, source: str = "polygon") -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            adjusted: Use adjusted prices (default True)
            source: "polygon" or "bloomberg"
        """
        cache_key = f"{ticker}_{start_date}_{end_date}_{adjusted}_{source}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if source == "bloomberg" and self.bloomberg_available:
            df = self._get_bloomberg_data(ticker, start_date, end_date, adjusted)
        else:
            df = self._get_polygon_data(ticker, start_date, end_date, adjusted)
        
        if not df.empty:
            self.cache[cache_key] = df
        return df
    
    def _get_polygon_data(self, ticker: str, start_date: str, end_date: str, 
                          adjusted: bool = True) -> pd.DataFrame:
        """Get data from Polygon.io."""
        if not self.polygon_api_key:
            return pd.DataFrame()
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        
        try:
            response = requests.get(url, params={
                "apiKey": self.polygon_api_key,
                "adjusted": str(adjusted).lower(),
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
                        'c': 'close', 'v': 'volume', 'vw': 'vwap'
                    })
                    df['adjusted'] = adjusted
                    df['source'] = 'polygon'
                    return df[['date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'adjusted', 'source']]
        except Exception as e:
            st.warning(f"Polygon API error for {ticker}: {e}")
        
        return pd.DataFrame()
    
    def _get_bloomberg_data(self, ticker: str, start_date: str, end_date: str,
                            adjusted: bool = True) -> pd.DataFrame:
        """Get data from Bloomberg API."""
        try:
            # Bloomberg API integration
            # This assumes you have blpapi installed and configured
            import blpapi
            
            sessionOptions = blpapi.SessionOptions()
            sessionOptions.setServerHost("localhost")
            sessionOptions.setServerPort(8194)
            
            session = blpapi.Session(sessionOptions)
            if not session.start():
                return pd.DataFrame()
            
            if not session.openService("//blp/refdata"):
                return pd.DataFrame()
            
            refDataService = session.getService("//blp/refdata")
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Add security
            bbg_ticker = f"{ticker} US Equity"
            request.getElement("securities").appendValue(bbg_ticker)
            
            # Add fields - use adjusted or unadjusted
            if adjusted:
                request.getElement("fields").appendValue("PX_LAST")
                request.getElement("fields").appendValue("PX_OPEN")
                request.getElement("fields").appendValue("PX_HIGH")
                request.getElement("fields").appendValue("PX_LOW")
            else:
                request.getElement("fields").appendValue("PX_LAST_RAW")
                request.getElement("fields").appendValue("PX_OPEN_RAW")
                request.getElement("fields").appendValue("PX_HIGH_RAW")
                request.getElement("fields").appendValue("PX_LOW_RAW")
            
            request.getElement("fields").appendValue("PX_VOLUME")
            
            request.set("startDate", start_date.replace("-", ""))
            request.set("endDate", end_date.replace("-", ""))
            
            session.sendRequest(request)
            
            data = []
            while True:
                ev = session.nextEvent(500)
                for msg in ev:
                    if msg.hasElement("securityData"):
                        secData = msg.getElement("securityData")
                        fieldData = secData.getElement("fieldData")
                        for i in range(fieldData.numValues()):
                            field = fieldData.getValueAsElement(i)
                            row = {
                                'date': field.getElementAsDatetime("date"),
                                'close': field.getElementAsFloat("PX_LAST" if adjusted else "PX_LAST_RAW"),
                                'open': field.getElementAsFloat("PX_OPEN" if adjusted else "PX_OPEN_RAW"),
                                'high': field.getElementAsFloat("PX_HIGH" if adjusted else "PX_HIGH_RAW"),
                                'low': field.getElementAsFloat("PX_LOW" if adjusted else "PX_LOW_RAW"),
                                'volume': field.getElementAsFloat("PX_VOLUME"),
                            }
                            data.append(row)
                
                if ev.eventType() == blpapi.Event.RESPONSE:
                    break
            
            session.stop()
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                df['adjusted'] = adjusted
                df['source'] = 'bloomberg'
                return df
                
        except ImportError:
            st.info("Bloomberg API (blpapi) not installed")
        except Exception as e:
            st.warning(f"Bloomberg API error: {e}")
        
        return pd.DataFrame()
    
    def get_intraday_data(self, ticker: str, date: str, 
                          source: str = "polygon") -> pd.DataFrame:
        """Get intraday minute data."""
        if source == "polygon" and self.polygon_api_key:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
            
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
                        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
                        df = df.rename(columns={
                            'o': 'open', 'h': 'high', 'l': 'low',
                            'c': 'close', 'v': 'volume'
                        })
                        return df
            except Exception as e:
                pass
        
        return pd.DataFrame()


# =============================================================================
# SPAC DATABASE
# =============================================================================
class SPACDatabase:
    """Database of SPAC information."""
    
    def __init__(self):
        # Sample SPAC data - in production, this would come from SEC/SPAC Research API
        self.active_spacs = self._load_active_spacs()
        self.completed_spacs = self._load_completed_spacs()
        self.sector_mapping = self._load_sector_mapping()
    
    def _load_active_spacs(self) -> pd.DataFrame:
        """Load active (searching) SPACs."""
        # Sample data - would be fetched from API in production
        data = [
            {"ticker": "PSNY", "name": "Polestar Automotive", "ipo_date": "2024-06-15", 
             "deadline": "2026-06-15", "trust_value": 10.25, "sector_focus": "EV/Automotive",
             "sponsor": "Gores Holdings", "status": "Searching"},
            {"ticker": "GPAC", "name": "Global Partner Acquisition II", "ipo_date": "2024-03-01",
             "deadline": "2026-03-01", "trust_value": 10.18, "sector_focus": "Technology",
             "sponsor": "Global Partner", "status": "Searching"},
            {"ticker": "NLIT", "name": "Northern Lights Acquisition", "ipo_date": "2024-07-20",
             "deadline": "2026-07-20", "trust_value": 10.30, "sector_focus": "Clean Energy",
             "sponsor": "Northern Lights", "status": "Searching"},
            {"ticker": "FEXD", "name": "Finnovate Acquisition", "ipo_date": "2024-01-10",
             "deadline": "2026-01-10", "trust_value": 10.12, "sector_focus": "Fintech",
             "sponsor": "Finnovate Partners", "status": "Searching"},
            {"ticker": "AISP", "name": "AI SPAC Corp", "ipo_date": "2024-09-01",
             "deadline": "2026-09-01", "trust_value": 10.35, "sector_focus": "AI/Technology",
             "sponsor": "AI Ventures", "status": "Announced", "target": "TechAI Inc"},
            {"ticker": "EVGO", "name": "EV Growth Acquisition", "ipo_date": "2024-04-15",
             "deadline": "2026-04-15", "trust_value": 10.22, "sector_focus": "EV/Infrastructure",
             "sponsor": "EV Growth Partners", "status": "Announced", "target": "ChargeNet"},
            {"ticker": "HLTH", "name": "Health Innovation SPAC", "ipo_date": "2024-02-28",
             "deadline": "2026-02-28", "trust_value": 10.28, "sector_focus": "Healthcare/Biotech",
             "sponsor": "Health Innovations", "status": "Searching"},
            {"ticker": "SPCE2", "name": "Space Ventures II", "ipo_date": "2024-08-10",
             "deadline": "2026-08-10", "trust_value": 10.40, "sector_focus": "Aerospace/Defense",
             "sponsor": "Space Ventures", "status": "Searching"},
        ]
        df = pd.DataFrame(data)
        df['ipo_date'] = pd.to_datetime(df['ipo_date'])
        df['deadline'] = pd.to_datetime(df['deadline'])
        df['days_to_deadline'] = (df['deadline'] - datetime.now()).dt.days
        return df
    
    def _load_completed_spacs(self) -> pd.DataFrame:
        """Load completed (merged) SPACs with performance data."""
        data = [
            {"original_spac": "Diamond Eagle", "ticker": "DKNG", "name": "DraftKings",
             "announcement_date": "2019-12-23", "merger_date": "2020-04-24",
             "sector": "Sports Betting", "ipo_price": 10.0, "announcement_price": 10.50,
             "merger_price": 18.50, "peak_price": 72.00, "current_price": 42.00,
             "sponsor": "Diamond Eagle", "days_to_announcement": 180, "days_to_peak": 330},
            {"original_spac": "Churchill Capital IV", "ticker": "LCID", "name": "Lucid Motors",
             "announcement_date": "2021-02-22", "merger_date": "2021-07-26",
             "sector": "EV", "ipo_price": 10.0, "announcement_price": 11.00,
             "merger_price": 25.00, "peak_price": 57.00, "current_price": 3.50,
             "sponsor": "Churchill Capital", "days_to_announcement": 120, "days_to_peak": 180},
            {"original_spac": "Social Capital Hedosophia", "ticker": "SPCE", "name": "Virgin Galactic",
             "announcement_date": "2019-07-09", "merger_date": "2019-10-28",
             "sector": "Space", "ipo_price": 10.0, "announcement_price": 10.20,
             "merger_price": 12.00, "peak_price": 62.00, "current_price": 5.00,
             "sponsor": "Chamath Palihapitiya", "days_to_announcement": 90, "days_to_peak": 600},
            {"original_spac": "Social Capital Hedosophia V", "ticker": "SOFI", "name": "SoFi Technologies",
             "announcement_date": "2021-01-07", "merger_date": "2021-06-01",
             "sector": "Fintech", "ipo_price": 10.0, "announcement_price": 10.50,
             "merger_price": 22.00, "peak_price": 28.00, "current_price": 15.00,
             "sponsor": "Chamath Palihapitiya", "days_to_announcement": 150, "days_to_peak": 210},
            {"original_spac": "GS Acquisition Holdings", "ticker": "VRT", "name": "Vertiv Holdings",
             "announcement_date": "2019-12-11", "merger_date": "2020-02-07",
             "sector": "Data Center", "ipo_price": 10.0, "announcement_price": 10.30,
             "merger_price": 12.00, "peak_price": 115.00, "current_price": 95.00,
             "sponsor": "Goldman Sachs", "days_to_announcement": 200, "days_to_peak": 1400},
            {"original_spac": "SVF Investment Corp 3", "ticker": "SYM", "name": "Symbotic",
             "announcement_date": "2021-12-12", "merger_date": "2022-06-07",
             "sector": "AI/Robotics", "ipo_price": 10.0, "announcement_price": 10.20,
             "merger_price": 12.50, "peak_price": 64.00, "current_price": 25.00,
             "sponsor": "SoftBank", "days_to_announcement": 180, "days_to_peak": 400},
            {"original_spac": "dMY Technology Group", "ticker": "RSI", "name": "Rush Street Interactive",
             "announcement_date": "2020-10-01", "merger_date": "2020-12-29",
             "sector": "Sports Betting", "ipo_price": 10.0, "announcement_price": 10.80,
             "merger_price": 18.00, "peak_price": 26.00, "current_price": 12.00,
             "sponsor": "dMY Technology", "days_to_announcement": 150, "days_to_peak": 90},
            {"original_spac": "Switchback Energy", "ticker": "CHPT", "name": "ChargePoint",
             "announcement_date": "2020-09-24", "merger_date": "2021-02-26",
             "sector": "EV Charging", "ipo_price": 10.0, "announcement_price": 11.50,
             "merger_price": 30.00, "peak_price": 50.00, "current_price": 1.50,
             "sponsor": "Switchback", "days_to_announcement": 120, "days_to_peak": 60},
            {"original_spac": "Pershing Square Tontine", "ticker": "UMG", "name": "Universal Music (failed)",
             "announcement_date": "2021-06-04", "merger_date": None,
             "sector": "Entertainment", "ipo_price": 10.0, "announcement_price": 10.50,
             "merger_price": None, "peak_price": 34.00, "current_price": None,
             "sponsor": "Bill Ackman", "days_to_announcement": 365, "days_to_peak": 30},
            {"original_spac": "Reinvent Technology Partners", "ticker": "JOBY", "name": "Joby Aviation",
             "announcement_date": "2021-02-24", "merger_date": "2021-08-10",
             "sector": "eVTOL/Aviation", "ipo_price": 10.0, "announcement_price": 10.30,
             "merger_price": 10.50, "peak_price": 16.00, "current_price": 6.00,
             "sponsor": "Reid Hoffman", "days_to_announcement": 200, "days_to_peak": 30},
        ]
        df = pd.DataFrame(data)
        df['announcement_date'] = pd.to_datetime(df['announcement_date'])
        df['merger_date'] = pd.to_datetime(df['merger_date'])
        
        # Calculate returns
        df['announcement_return'] = (df['announcement_price'] - df['ipo_price']) / df['ipo_price'] * 100
        df['merger_return'] = (df['merger_price'] - df['ipo_price']) / df['ipo_price'] * 100
        df['peak_return'] = (df['peak_price'] - df['ipo_price']) / df['ipo_price'] * 100
        df['current_return'] = (df['current_price'] - df['ipo_price']) / df['ipo_price'] * 100
        
        return df
    
    def _load_sector_mapping(self) -> Dict:
        """Load sector performance data."""
        return {
            "EV/Automotive": {"2020": 150, "2021": 80, "2022": -40, "2023": -20, "2024": 10},
            "Sports Betting": {"2020": 200, "2021": 50, "2022": -30, "2023": 40, "2024": 60},
            "Fintech": {"2020": 80, "2021": 100, "2022": -50, "2023": 20, "2024": 40},
            "Space/Aerospace": {"2020": 120, "2021": 60, "2022": -60, "2023": -30, "2024": 20},
            "AI/Technology": {"2020": 50, "2021": 70, "2022": -20, "2023": 80, "2024": 150},
            "Healthcare/Biotech": {"2020": 40, "2021": 30, "2022": -40, "2023": 10, "2024": 30},
            "Clean Energy": {"2020": 100, "2021": 40, "2022": -50, "2023": 0, "2024": 25},
            "Data Center": {"2020": 60, "2021": 80, "2022": 20, "2023": 100, "2024": 200},
        }


# =============================================================================
# MODULE 1: SPAC SCREENER
# =============================================================================
def render_spac_screener(db: SPACDatabase, data_provider: SPACDataProvider):
    """Render the SPAC Screener module."""
    
    st.markdown('<p class="module-header">🔍 SPAC SCREENER</p>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_filter = st.multiselect(
            "Status",
            options=["Searching", "Announced", "Voting"],
            default=["Searching", "Announced"]
        )
    
    with col2:
        sectors = db.active_spacs['sector_focus'].unique().tolist()
        sector_filter = st.multiselect(
            "Sector Focus",
            options=sectors,
            default=[]
        )
    
    with col3:
        days_filter = st.slider(
            "Days to Deadline",
            min_value=0,
            max_value=730,
            value=(0, 730)
        )
    
    with col4:
        trust_filter = st.slider(
            "Trust Value ($)",
            min_value=9.0,
            max_value=12.0,
            value=(9.5, 11.5),
            step=0.1
        )
    
    # Apply filters
    filtered = db.active_spacs.copy()
    if status_filter:
        filtered = filtered[filtered['status'].isin(status_filter)]
    if sector_filter:
        filtered = filtered[filtered['sector_focus'].isin(sector_filter)]
    filtered = filtered[
        (filtered['days_to_deadline'] >= days_filter[0]) &
        (filtered['days_to_deadline'] <= days_filter[1])
    ]
    filtered = filtered[
        (filtered['trust_value'] >= trust_filter[0]) &
        (filtered['trust_value'] <= trust_filter[1])
    ]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active SPACs", len(filtered))
    with col2:
        announced = len(filtered[filtered['status'] == 'Announced'])
        st.metric("With Targets", announced)
    with col3:
        avg_trust = filtered['trust_value'].mean()
        st.metric("Avg Trust Value", f"${avg_trust:.2f}")
    with col4:
        near_deadline = len(filtered[filtered['days_to_deadline'] < 180])
        st.metric("Near Deadline (<6mo)", near_deadline)
    
    # Display table
    display_cols = ['ticker', 'name', 'status', 'sector_focus', 'trust_value', 
                    'days_to_deadline', 'sponsor', 'ipo_date', 'deadline']
    
    if 'target' in filtered.columns:
        display_cols.insert(3, 'target')
    
    st.dataframe(
        filtered[display_cols].sort_values('days_to_deadline'),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "sector_focus": st.column_config.TextColumn("Sector", width="medium"),
            "trust_value": st.column_config.NumberColumn("Trust $", format="$%.2f"),
            "days_to_deadline": st.column_config.NumberColumn("Days Left", width="small"),
            "sponsor": st.column_config.TextColumn("Sponsor", width="medium"),
        }
    )
    
    # Deadline timeline chart
    st.markdown("#### 📅 Deadline Timeline")
    fig = px.scatter(
        filtered,
        x='deadline',
        y='trust_value',
        size='days_to_deadline',
        color='status',
        hover_data=['ticker', 'name', 'sector_focus'],
        color_discrete_map={'Searching': '#00d4ff', 'Announced': '#00ff88', 'Voting': '#ffd93d'}
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Deadline Date",
        yaxis_title="Trust Value ($)"
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MODULE 2: ANNOUNCEMENT ALERTS
# =============================================================================
def render_announcement_alerts(db: SPACDatabase, data_provider: SPACDataProvider):
    """Render the Announcement Alerts module."""
    
    st.markdown('<p class="module-header">🔔 ANNOUNCEMENT ALERTS</p>', unsafe_allow_html=True)
    
    # Recent announcements (simulated - would be real-time in production)
    st.markdown("#### 📢 Recent Merger Announcements")
    
    # Get announced SPACs
    announced = db.active_spacs[db.active_spacs['status'] == 'Announced'].copy()
    
    if not announced.empty:
        for _, row in announced.iterrows():
            st.markdown(f"""
            <div class="alert-box">
                <strong style="color: var(--accent-green);">🎯 NEW TARGET ANNOUNCED</strong><br>
                <span style="font-family: 'JetBrains Mono'; font-size: 1.2rem;">{row['ticker']}</span> → 
                <span style="color: var(--accent-blue);">{row.get('target', 'Target TBD')}</span><br>
                <span style="color: var(--text-secondary);">Sector: {row['sector_focus']} | Sponsor: {row['sponsor']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent announcements")
    
    # Price movement alerts
    st.markdown("#### 📈 Unusual Price Movement")
    
    # Simulated alerts - would be real-time in production
    alerts = [
        {"ticker": "AISP", "change": 15.2, "volume_ratio": 5.2, "reason": "Rumored target in AI sector"},
        {"ticker": "EVGO", "change": 8.5, "volume_ratio": 3.1, "reason": "Vote date approaching"},
        {"ticker": "HLTH", "change": -5.2, "volume_ratio": 2.8, "reason": "Deadline extension filed"},
    ]
    
    for alert in alerts:
        color = "positive" if alert['change'] > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-family: 'JetBrains Mono'; font-size: 1.2rem;">{alert['ticker']}</span>
                    <span class="{color}" style="font-size: 1.1rem; margin-left: 1rem;">
                        {'+' if alert['change'] > 0 else ''}{alert['change']:.1f}%
                    </span>
                </div>
                <div style="text-align: right;">
                    <span style="color: var(--accent-yellow);">Vol: {alert['volume_ratio']:.1f}x avg</span>
                </div>
            </div>
            <div style="color: var(--text-secondary); margin-top: 0.5rem; font-size: 0.9rem;">
                {alert['reason']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert configuration
    st.markdown("#### ⚙️ Configure Alerts")
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox("🎯 New Merger Announcements", value=True)
        st.checkbox("📊 Price Movement > 5%", value=True)
        st.checkbox("📅 Deadline < 30 days", value=True)
    with col2:
        st.checkbox("📰 SEC Filing Alerts", value=False)
        st.checkbox("🔊 Volume Spike > 3x", value=True)
        st.checkbox("💰 Trust Value Change", value=False)
    
    st.text_input("Email for alerts", placeholder="your@email.com")
    st.button("💾 Save Alert Settings", type="secondary")


# =============================================================================
# MODULE 3: HISTORICAL BACKTESTER
# =============================================================================
def render_backtester(db: SPACDatabase, data_provider: SPACDataProvider):
    """Render the Historical Backtester module."""
    
    st.markdown('<p class="module-header">📊 HISTORICAL BACKTESTER</p>', unsafe_allow_html=True)
    
    # Strategy selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy = st.selectbox(
            "Strategy",
            options=[
                "Buy Announcement → Sell Merger",
                "Buy IPO → Sell Announcement",
                "Buy Merger → Hold 30 Days",
                "Buy Announcement → Hold to Peak",
                "Buy Rumor → Sell Announcement",
                "Custom"
            ]
        )
    
    with col2:
        price_type = st.selectbox(
            "Price Type",
            options=["Adjusted", "Unadjusted"],
            help="Adjusted prices account for splits and dividends"
        )
    
    with col3:
        data_source = st.selectbox(
            "Data Source",
            options=["Polygon", "Bloomberg"] if data_provider.bloomberg_available else ["Polygon"]
        )
    
    # Sector filter for backtest
    sectors = db.completed_spacs['sector'].unique().tolist()
    selected_sectors = st.multiselect(
        "Filter by Sector",
        options=sectors,
        default=sectors
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.selectbox("Start Year", [2019, 2020, 2021, 2022, 2023, 2024], index=1)
    with col2:
        end_year = st.selectbox("End Year", [2020, 2021, 2022, 2023, 2024, 2025], index=4)
    
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        
        # Filter completed SPACs
        backtest_data = db.completed_spacs[
            db.completed_spacs['sector'].isin(selected_sectors)
        ].copy()
        
        # Calculate returns based on strategy
        if strategy == "Buy Announcement → Sell Merger":
            backtest_data['strategy_return'] = (
                (backtest_data['merger_price'] - backtest_data['announcement_price']) / 
                backtest_data['announcement_price'] * 100
            )
            entry_col = 'announcement_price'
            exit_col = 'merger_price'
        elif strategy == "Buy IPO → Sell Announcement":
            backtest_data['strategy_return'] = backtest_data['announcement_return']
            entry_col = 'ipo_price'
            exit_col = 'announcement_price'
        elif strategy == "Buy Merger → Hold 30 Days":
            # Approximate with current price for demo
            backtest_data['strategy_return'] = (
                (backtest_data['current_price'] - backtest_data['merger_price']) / 
                backtest_data['merger_price'] * 100
            ).clip(-50, 100)  # Cap for demo
            entry_col = 'merger_price'
            exit_col = 'current_price'
        else:
            backtest_data['strategy_return'] = backtest_data['peak_return']
            entry_col = 'ipo_price'
            exit_col = 'peak_price'
        
        # Filter out invalid data
        backtest_data = backtest_data.dropna(subset=['strategy_return'])
        
        # Results summary
        st.markdown("### 📈 Backtest Results")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_return = backtest_data['strategy_return'].sum()
        avg_return = backtest_data['strategy_return'].mean()
        win_rate = (backtest_data['strategy_return'] > 0).mean() * 100
        max_gain = backtest_data['strategy_return'].max()
        max_loss = backtest_data['strategy_return'].min()
        
        with col1:
            color = "positive" if total_return > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value {color}">{total_return:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            color = "positive" if avg_return > 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Return</div>
                <div class="metric-value {color}">{avg_return:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value neutral">{win_rate:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Best Trade</div>
                <div class="metric-value positive">+{max_gain:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Worst Trade</div>
                <div class="metric-value negative">{max_loss:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Individual trades table
        st.markdown("### 📋 Individual Trades")
        
        display_df = backtest_data[[
            'ticker', 'name', 'sector', 'announcement_date', 
            entry_col, exit_col, 'strategy_return', 'days_to_peak'
        ]].copy()
        display_df.columns = ['Ticker', 'Name', 'Sector', 'Ann. Date', 
                              'Entry $', 'Exit $', 'Return %', 'Days']
        
        st.dataframe(
            display_df.sort_values('Return %', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Entry $": st.column_config.NumberColumn(format="$%.2f"),
                "Exit $": st.column_config.NumberColumn(format="$%.2f"),
                "Return %": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
        
        # Return distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Return Distribution")
            fig = px.histogram(
                backtest_data,
                x='strategy_return',
                nbins=20,
                color_discrete_sequence=['#00d4ff']
            )
            fig.add_vline(x=0, line_dash="dash", line_color="white")
            fig.add_vline(x=avg_return, line_dash="dot", line_color="#00ff88",
                         annotation_text=f"Avg: {avg_return:.1f}%")
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Return (%)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Returns by Sector")
            sector_returns = backtest_data.groupby('sector')['strategy_return'].mean().sort_values()
            fig = px.bar(
                x=sector_returns.values,
                y=sector_returns.index,
                orientation='h',
                color=sector_returns.values,
                color_continuous_scale=['#ff4757', '#ffd93d', '#00ff88']
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Avg Return (%)",
                yaxis_title="",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative equity curve
        st.markdown("#### 📈 Cumulative Equity Curve")
        
        # Sort by announcement date
        equity_data = backtest_data.sort_values('announcement_date').copy()
        equity_data['cumulative_return'] = equity_data['strategy_return'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_data['announcement_date'],
            y=equity_data['cumulative_return'],
            mode='lines+markers',
            line=dict(color='#00ff88', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MODULE 4: SECTOR ROTATION
# =============================================================================
def render_sector_rotation(db: SPACDatabase, data_provider: SPACDataProvider):
    """Render the Sector Rotation Analysis module."""
    
    st.markdown('<p class="module-header">🔄 SECTOR ROTATION ANALYSIS</p>', unsafe_allow_html=True)
    
    # Current hot sectors
    st.markdown("#### 🔥 Current Hot Sectors (2024)")
    
    # Calculate current sector performance
    sector_perf = []
    for sector, years in db.sector_mapping.items():
        sector_perf.append({
            'sector': sector,
            '2024': years.get('2024', 0),
            '2023': years.get('2023', 0),
            '2022': years.get('2022', 0),
            '2021': years.get('2021', 0),
            '2020': years.get('2020', 0),
        })
    
    perf_df = pd.DataFrame(sector_perf).sort_values('2024', ascending=False)
    
    # Hot sector cards
    top_sectors = perf_df.head(3)
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_sectors.iterrows()):
        with cols[i]:
            yoy_change = row['2024'] - row['2023']
            st.markdown(f"""
            <div class="hot-sector">
                <div style="font-family: 'JetBrains Mono'; font-size: 1.1rem; color: var(--accent-green);">
                    #{i+1} {row['sector']}
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: white;">
                    +{row['2024']}%
                </div>
                <div style="color: var(--text-secondary); font-size: 0.85rem;">
                    YoY: {'+' if yoy_change > 0 else ''}{yoy_change}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Sector performance heatmap
    st.markdown("#### 📊 Sector Performance Heatmap (2020-2024)")
    
    heatmap_data = perf_df.set_index('sector')[['2020', '2021', '2022', '2023', '2024']]
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Sector", color="Return %"),
        color_continuous_scale=['#ff4757', '#1a2332', '#00ff88'],
        color_continuous_midpoint=0,
        aspect="auto"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    # Add text annotations
    for i, sector in enumerate(heatmap_data.index):
        for j, year in enumerate(heatmap_data.columns):
            val = heatmap_data.loc[sector, year]
            fig.add_annotation(
                x=j, y=i,
                text=f"{val:+.0f}%",
                showarrow=False,
                font=dict(color='white', size=10)
            )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector rotation timeline
    st.markdown("#### 📈 Sector Performance Over Time")
    
    fig = go.Figure()
    
    colors = ['#00ff88', '#00d4ff', '#a855f7', '#ffd93d', '#ff4757', '#ff6b9d', '#00ffff', '#ff8c00']
    
    for i, (_, row) in enumerate(perf_df.iterrows()):
        years = ['2020', '2021', '2022', '2023', '2024']
        values = [row[y] for y in years]
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=row['sector'],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Year",
        yaxis_title="Avg SPAC Return (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector correlation matrix
    st.markdown("#### 🔗 Sector Correlation Matrix")
    
    # Create correlation data from historical returns
    corr_data = heatmap_data.T.corr()
    
    fig = px.imshow(
        corr_data,
        labels=dict(color="Correlation"),
        color_continuous_scale=['#ff4757', 'white', '#00ff88'],
        color_continuous_midpoint=0,
        aspect="equal"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sector recommendations
    st.markdown("#### 💡 Sector Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--accent-green); font-weight: 600; margin-bottom: 0.5rem;">
                🟢 BULLISH SECTORS
            </div>
            <ul style="color: var(--text-primary); margin: 0; padding-left: 1.2rem;">
                <li><strong>Data Center/AI</strong> - Riding AI infrastructure wave</li>
                <li><strong>AI/Technology</strong> - Strong momentum, institutional interest</li>
                <li><strong>Sports Betting</strong> - Continued legalization tailwinds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--accent-red); font-weight: 600; margin-bottom: 0.5rem;">
                🔴 CAUTIOUS SECTORS
            </div>
            <ul style="color: var(--text-primary); margin: 0; padding-left: 1.2rem;">
                <li><strong>EV/Automotive</strong> - Oversupply, competition pressures</li>
                <li><strong>Space/Aerospace</strong> - Long development timelines</li>
                <li><strong>Clean Energy</strong> - Policy uncertainty</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SETTINGS
# =============================================================================
def render_settings():
    """Render settings page."""
    
    st.markdown('<p class="module-header">⚙️ SETTINGS</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔑 API Configuration")
        
        polygon_key = st.text_input(
            "Polygon API Key",
            type="password",
            value=st.session_state.get('polygon_api_key', ''),
            help="Get your free API key at polygon.io"
        )
        if polygon_key:
            st.session_state['polygon_api_key'] = polygon_key
        
        bloomberg_enabled = st.checkbox(
            "Enable Bloomberg API",
            value=st.session_state.get('bloomberg_enabled', False),
            help="Requires Bloomberg Terminal with API access"
        )
        st.session_state['bloomberg_enabled'] = bloomberg_enabled
        
        if bloomberg_enabled:
            st.info("Bloomberg API requires blpapi package and active Terminal session")
    
    with col2:
        st.markdown("#### 📊 Data Preferences")
        
        default_price = st.selectbox(
            "Default Price Type",
            options=["Adjusted", "Unadjusted"],
            index=0 if st.session_state.get('default_adjusted', True) else 1
        )
        st.session_state['default_adjusted'] = (default_price == "Adjusted")
        
        default_source = st.selectbox(
            "Default Data Source",
            options=["Polygon", "Bloomberg"],
            index=0
        )
        st.session_state['default_source'] = default_source.lower()
        
        cache_duration = st.slider(
            "Cache Duration (minutes)",
            min_value=5,
            max_value=60,
            value=15
        )
        st.session_state['cache_duration'] = cache_duration
    
    st.markdown("#### 🔔 Notification Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Email Notifications", value=False)
        st.checkbox("Browser Notifications", value=True)
        st.checkbox("Mobile Push (if app installed)", value=False)
    
    with col2:
        st.selectbox("Alert Frequency", ["Real-time", "Hourly", "Daily"])
        st.multiselect(
            "Alert Types",
            ["Announcements", "Price Moves", "Deadlines", "SEC Filings"],
            default=["Announcements", "Price Moves"]
        )
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("Settings saved!")


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application entry point."""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 SPAC ANALYSIS DASHBOARD</h1>', unsafe_allow_html=True)
    
    # Initialize data
    api_key = st.session_state.get('polygon_api_key', os.environ.get('POLYGON_API_KEY', ''))
    bloomberg_available = st.session_state.get('bloomberg_enabled', False)
    
    data_provider = SPACDataProvider(
        polygon_api_key=api_key,
        bloomberg_available=bloomberg_available
    )
    db = SPACDatabase()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Screener",
        "🔔 Alerts", 
        "📊 Backtester",
        "🔄 Sector Rotation",
        "⚙️ Settings"
    ])
    
    with tab1:
        render_spac_screener(db, data_provider)
    
    with tab2:
        render_announcement_alerts(db, data_provider)
    
    with tab3:
        render_backtester(db, data_provider)
    
    with tab4:
        render_sector_rotation(db, data_provider)
    
    with tab5:
        render_settings()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-secondary); font-size: 0.85rem;">
        📊 Data Sources: Polygon.io, Bloomberg, SEC EDGAR | 
        🔄 Last Updated: {now} |
        ⚠️ Not Financial Advice
    </div>
    """.format(now=datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
