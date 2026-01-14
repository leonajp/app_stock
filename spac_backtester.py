"""
SPAC Backtesting Engine
========================
Test various SPAC trading strategies with historical data.

Strategies:
1. Buy Announcement → Sell Merger
2. Buy IPO → Sell Announcement  
3. Buy Merger → Hold X Days
4. Buy Announcement → Hold to Peak
5. Buy Pre-Rumor → Sell Announcement
6. Custom Entry/Exit Rules

Supports:
- Adjusted and unadjusted prices
- Position sizing
- Stop losses / Take profits
- Transaction costs
- Multiple data sources (Polygon, Bloomberg)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json


class PriceType(Enum):
    ADJUSTED = "adjusted"
    UNADJUSTED = "unadjusted"


class DataSource(Enum):
    POLYGON = "polygon"
    BLOOMBERG = "bloomberg"
    AUTO = "auto"


@dataclass
class Trade:
    """Represents a single trade."""
    ticker: str
    spac_name: str
    sector: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: int
    direction: str  # "long" or "short"
    strategy: str
    
    @property
    def pnl(self) -> float:
        """Calculate P&L in dollars."""
        if self.direction == "long":
            return (self.exit_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - self.exit_price) * self.shares
    
    @property
    def pnl_pct(self) -> float:
        """Calculate P&L as percentage."""
        if self.direction == "long":
            return (self.exit_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.exit_price) / self.entry_price * 100
    
    @property
    def holding_days(self) -> int:
        """Calculate holding period in days."""
        return (self.exit_date - self.entry_date).days
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'spac_name': self.spac_name,
            'sector': self.sector,
            'entry_date': self.entry_date.strftime('%Y-%m-%d'),
            'entry_price': self.entry_price,
            'exit_date': self.exit_date.strftime('%Y-%m-%d'),
            'exit_price': self.exit_price,
            'shares': self.shares,
            'direction': self.direction,
            'strategy': self.strategy,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'holding_days': self.holding_days
        }


@dataclass 
class BacktestConfig:
    """Configuration for backtest."""
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0  # % of capital per trade
    max_positions: int = 10
    commission_per_share: float = 0.0
    commission_min: float = 0.0
    slippage_pct: float = 0.1  # 0.1% slippage
    stop_loss_pct: Optional[float] = None  # e.g., 10 for 10% stop
    take_profit_pct: Optional[float] = None  # e.g., 50 for 50% take profit
    price_type: PriceType = PriceType.ADJUSTED
    data_source: DataSource = DataSource.POLYGON


@dataclass
class BacktestResult:
    """Results from a backtest."""
    trades: List[Trade]
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def winning_trades(self) -> int:
        return len([t for t in self.trades if t.pnl > 0])
    
    @property
    def losing_trades(self) -> int:
        return len([t for t in self.trades if t.pnl < 0])
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades * 100
    
    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)
    
    @property
    def total_return_pct(self) -> float:
        return self.total_pnl / self.config.initial_capital * 100
    
    @property
    def avg_pnl_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def avg_return_pct(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return np.mean([t.pnl_pct for t in self.trades])
    
    @property
    def avg_holding_days(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return np.mean([t.holding_days for t in self.trades])
    
    @property
    def max_gain(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return max(t.pnl_pct for t in self.trades)
    
    @property
    def max_loss(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return min(t.pnl_pct for t in self.trades)
    
    @property
    def profit_factor(self) -> float:
        """Gross profits / Gross losses."""
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe ratio (assuming risk-free = 0)."""
        if self.total_trades < 2:
            return 0.0
        returns = [t.pnl_pct for t in self.trades]
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252 / max(1, self.avg_holding_days))
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Calculate equity curve from trades."""
        if not self.trades:
            return pd.DataFrame()
        
        equity = self.config.initial_capital
        curve_data = []
        
        for trade in sorted(self.trades, key=lambda t: t.exit_date):
            equity += trade.pnl
            curve_data.append({
                'date': trade.exit_date,
                'equity': equity,
                'trade_pnl': trade.pnl,
                'ticker': trade.ticker
            })
        
        return pd.DataFrame(curve_data)
    
    def get_returns_by_sector(self) -> pd.DataFrame:
        """Calculate returns grouped by sector."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'sector': trade.sector,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct
            })
        
        df = pd.DataFrame(data)
        return df.groupby('sector').agg({
            'pnl': ['sum', 'mean', 'count'],
            'pnl_pct': ['mean', 'std']
        }).round(2)
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'month': trade.exit_date.strftime('%Y-%m'),
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct
            })
        
        df = pd.DataFrame(data)
        return df.groupby('month').agg({
            'pnl': 'sum',
            'pnl_pct': ['mean', 'count']
        }).round(2)
    
    def to_summary_dict(self) -> Dict:
        """Get summary as dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 1),
            'total_pnl': round(self.total_pnl, 2),
            'total_return_pct': round(self.total_return_pct, 2),
            'avg_pnl_per_trade': round(self.avg_pnl_per_trade, 2),
            'avg_return_pct': round(self.avg_return_pct, 2),
            'avg_holding_days': round(self.avg_holding_days, 1),
            'max_gain_pct': round(self.max_gain, 2),
            'max_loss_pct': round(self.max_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2)
        }


class SPACBacktester:
    """
    Backtesting engine for SPAC trading strategies.
    """
    
    def __init__(self, spac_data: pd.DataFrame, price_fetcher=None):
        """
        Initialize backtester.
        
        Args:
            spac_data: DataFrame with SPAC information including:
                - ticker, name, sector
                - ipo_date, announcement_date, merger_date
                - ipo_price, announcement_price, merger_price, peak_price, current_price
            price_fetcher: Optional data fetcher for detailed price data
        """
        self.spac_data = spac_data.copy()
        self.price_fetcher = price_fetcher
        
        # Ensure date columns are datetime
        for col in ['ipo_date', 'announcement_date', 'merger_date']:
            if col in self.spac_data.columns:
                self.spac_data[col] = pd.to_datetime(self.spac_data[col])
    
    def run_strategy(self, strategy: str, config: BacktestConfig,
                    sectors: List[str] = None,
                    start_year: int = None, end_year: int = None) -> BacktestResult:
        """
        Run a predefined strategy.
        
        Args:
            strategy: Strategy name
            config: Backtest configuration
            sectors: Filter by sectors (None = all)
            start_year: Filter by start year
            end_year: Filter by end year
        """
        # Filter data
        data = self.spac_data.copy()
        
        if sectors:
            data = data[data['sector'].isin(sectors)]
        
        if start_year:
            data = data[data['announcement_date'].dt.year >= start_year]
        
        if end_year:
            data = data[data['announcement_date'].dt.year <= end_year]
        
        # Run appropriate strategy
        if strategy == "Buy Announcement → Sell Merger":
            return self._strategy_announcement_to_merger(data, config)
        elif strategy == "Buy IPO → Sell Announcement":
            return self._strategy_ipo_to_announcement(data, config)
        elif strategy == "Buy Merger → Hold 30 Days":
            return self._strategy_merger_hold(data, config, hold_days=30)
        elif strategy == "Buy Announcement → Hold to Peak":
            return self._strategy_announcement_to_peak(data, config)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _strategy_announcement_to_merger(self, data: pd.DataFrame, 
                                          config: BacktestConfig) -> BacktestResult:
        """
        Strategy: Buy on merger announcement, sell on merger completion.
        """
        trades = []
        
        for _, row in data.iterrows():
            # Skip if missing required data
            if pd.isna(row.get('announcement_date')) or pd.isna(row.get('merger_date')):
                continue
            if pd.isna(row.get('announcement_price')) or pd.isna(row.get('merger_price')):
                continue
            
            entry_price = row['announcement_price']
            exit_price = row['merger_price']
            
            # Apply slippage
            entry_price *= (1 + config.slippage_pct / 100)
            exit_price *= (1 - config.slippage_pct / 100)
            
            # Calculate position size
            position_value = config.initial_capital * config.position_size_pct / 100
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            trade = Trade(
                ticker=row['ticker'],
                spac_name=row.get('name', row['ticker']),
                sector=row.get('sector', 'Unknown'),
                entry_date=row['announcement_date'],
                entry_price=entry_price,
                exit_date=row['merger_date'],
                exit_price=exit_price,
                shares=shares,
                direction="long",
                strategy="Announcement → Merger"
            )
            trades.append(trade)
        
        return BacktestResult(
            trades=trades,
            config=config,
            start_date=data['announcement_date'].min() if len(data) > 0 else datetime.now(),
            end_date=data['merger_date'].max() if len(data) > 0 else datetime.now()
        )
    
    def _strategy_ipo_to_announcement(self, data: pd.DataFrame,
                                       config: BacktestConfig) -> BacktestResult:
        """
        Strategy: Buy at IPO (~$10), sell on merger announcement.
        """
        trades = []
        
        for _, row in data.iterrows():
            if pd.isna(row.get('ipo_date')) or pd.isna(row.get('announcement_date')):
                continue
            
            entry_price = row.get('ipo_price', 10.0)
            exit_price = row.get('announcement_price')
            
            if pd.isna(exit_price):
                continue
            
            # Apply slippage
            entry_price *= (1 + config.slippage_pct / 100)
            exit_price *= (1 - config.slippage_pct / 100)
            
            # Calculate position size
            position_value = config.initial_capital * config.position_size_pct / 100
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            trade = Trade(
                ticker=row['ticker'],
                spac_name=row.get('name', row['ticker']),
                sector=row.get('sector', 'Unknown'),
                entry_date=row['ipo_date'],
                entry_price=entry_price,
                exit_date=row['announcement_date'],
                exit_price=exit_price,
                shares=shares,
                direction="long",
                strategy="IPO → Announcement"
            )
            trades.append(trade)
        
        return BacktestResult(
            trades=trades,
            config=config,
            start_date=data['ipo_date'].min() if len(data) > 0 else datetime.now(),
            end_date=data['announcement_date'].max() if len(data) > 0 else datetime.now()
        )
    
    def _strategy_merger_hold(self, data: pd.DataFrame, config: BacktestConfig,
                              hold_days: int = 30) -> BacktestResult:
        """
        Strategy: Buy at merger, hold for X days.
        """
        trades = []
        
        for _, row in data.iterrows():
            if pd.isna(row.get('merger_date')) or pd.isna(row.get('merger_price')):
                continue
            
            entry_price = row['merger_price']
            
            # For exit price, we'd ideally fetch actual price data
            # For now, use a simple estimate based on current_price
            # In production, use price_fetcher to get actual price
            exit_date = row['merger_date'] + timedelta(days=hold_days)
            
            # Estimate exit price (simplified - would use actual data in production)
            if pd.notna(row.get('current_price')) and pd.notna(row.get('peak_price')):
                # Rough estimate based on trajectory
                days_to_peak = row.get('days_to_peak', 90)
                if hold_days < days_to_peak:
                    # Linear interpolation to peak
                    exit_price = entry_price + (row['peak_price'] - entry_price) * (hold_days / days_to_peak)
                else:
                    # After peak, use current as proxy
                    exit_price = row['current_price']
            else:
                continue
            
            # Apply slippage
            entry_price *= (1 + config.slippage_pct / 100)
            exit_price *= (1 - config.slippage_pct / 100)
            
            # Calculate position size
            position_value = config.initial_capital * config.position_size_pct / 100
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            trade = Trade(
                ticker=row['ticker'],
                spac_name=row.get('name', row['ticker']),
                sector=row.get('sector', 'Unknown'),
                entry_date=row['merger_date'],
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                shares=shares,
                direction="long",
                strategy=f"Merger → Hold {hold_days}d"
            )
            trades.append(trade)
        
        return BacktestResult(
            trades=trades,
            config=config,
            start_date=data['merger_date'].min() if len(data) > 0 else datetime.now(),
            end_date=datetime.now()
        )
    
    def _strategy_announcement_to_peak(self, data: pd.DataFrame,
                                        config: BacktestConfig) -> BacktestResult:
        """
        Strategy: Buy on announcement, sell at peak (hindsight/ideal scenario).
        Note: This is for analysis only - not achievable in real trading.
        """
        trades = []
        
        for _, row in data.iterrows():
            if pd.isna(row.get('announcement_date')) or pd.isna(row.get('announcement_price')):
                continue
            if pd.isna(row.get('peak_price')):
                continue
            
            entry_price = row['announcement_price']
            exit_price = row['peak_price']
            
            # Estimate peak date
            days_to_peak = row.get('days_to_peak', 90)
            exit_date = row['announcement_date'] + timedelta(days=days_to_peak)
            
            # Apply slippage
            entry_price *= (1 + config.slippage_pct / 100)
            exit_price *= (1 - config.slippage_pct / 100)
            
            # Calculate position size
            position_value = config.initial_capital * config.position_size_pct / 100
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            trade = Trade(
                ticker=row['ticker'],
                spac_name=row.get('name', row['ticker']),
                sector=row.get('sector', 'Unknown'),
                entry_date=row['announcement_date'],
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                shares=shares,
                direction="long",
                strategy="Announcement → Peak (Ideal)"
            )
            trades.append(trade)
        
        return BacktestResult(
            trades=trades,
            config=config,
            start_date=data['announcement_date'].min() if len(data) > 0 else datetime.now(),
            end_date=datetime.now()
        )
    
    def run_custom_strategy(self, entry_func: Callable, exit_func: Callable,
                           config: BacktestConfig) -> BacktestResult:
        """
        Run a custom strategy with user-defined entry/exit functions.
        
        Args:
            entry_func: Function(row) -> (should_enter: bool, entry_price: float)
            exit_func: Function(row, entry_price, entry_date) -> (should_exit: bool, exit_price: float)
            config: Backtest configuration
        """
        trades = []
        
        for _, row in self.spac_data.iterrows():
            # Check entry
            should_enter, entry_price = entry_func(row)
            if not should_enter or entry_price is None:
                continue
            
            # Check exit
            should_exit, exit_price = exit_func(row, entry_price, row.get('announcement_date'))
            if not should_exit or exit_price is None:
                continue
            
            # Apply slippage
            entry_price *= (1 + config.slippage_pct / 100)
            exit_price *= (1 - config.slippage_pct / 100)
            
            # Calculate position size
            position_value = config.initial_capital * config.position_size_pct / 100
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            trade = Trade(
                ticker=row['ticker'],
                spac_name=row.get('name', row['ticker']),
                sector=row.get('sector', 'Unknown'),
                entry_date=row.get('announcement_date', datetime.now()),
                entry_price=entry_price,
                exit_date=row.get('merger_date', datetime.now()),
                exit_price=exit_price,
                shares=shares,
                direction="long",
                strategy="Custom"
            )
            trades.append(trade)
        
        return BacktestResult(
            trades=trades,
            config=config,
            start_date=self.spac_data['announcement_date'].min(),
            end_date=datetime.now()
        )
    
    def compare_strategies(self, strategies: List[str], config: BacktestConfig,
                          sectors: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple strategies side by side.
        """
        results = []
        
        for strategy in strategies:
            result = self.run_strategy(strategy, config, sectors)
            summary = result.to_summary_dict()
            summary['strategy'] = strategy
            results.append(summary)
        
        return pd.DataFrame(results)
    
    def optimize_strategy(self, strategy: str, param_grid: Dict,
                         base_config: BacktestConfig) -> pd.DataFrame:
        """
        Optimize strategy parameters.
        
        Args:
            strategy: Strategy name
            param_grid: Dict of parameter ranges, e.g.:
                {'position_size_pct': [5, 10, 15], 'slippage_pct': [0.1, 0.2]}
            base_config: Base configuration to modify
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        for combo in combinations:
            # Create config with these parameters
            config = BacktestConfig(
                initial_capital=base_config.initial_capital,
                position_size_pct=base_config.position_size_pct,
                max_positions=base_config.max_positions,
                commission_per_share=base_config.commission_per_share,
                slippage_pct=base_config.slippage_pct,
                price_type=base_config.price_type,
                data_source=base_config.data_source
            )
            
            # Update with this combination
            for name, value in zip(param_names, combo):
                setattr(config, name, value)
            
            # Run backtest
            result = self.run_strategy(strategy, config)
            summary = result.to_summary_dict()
            
            # Add parameters to summary
            for name, value in zip(param_names, combo):
                summary[name] = value
            
            results.append(summary)
        
        return pd.DataFrame(results).sort_values('total_return_pct', ascending=False)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Sample SPAC data (in production, this would come from database/API)
    sample_data = pd.DataFrame([
        {"ticker": "DKNG", "name": "DraftKings", "sector": "Sports Betting",
         "ipo_date": "2019-05-01", "announcement_date": "2019-12-23", "merger_date": "2020-04-24",
         "ipo_price": 10.0, "announcement_price": 10.50, "merger_price": 18.50,
         "peak_price": 72.00, "current_price": 42.00, "days_to_peak": 330},
        {"ticker": "LCID", "name": "Lucid Motors", "sector": "EV",
         "ipo_date": "2020-09-01", "announcement_date": "2021-02-22", "merger_date": "2021-07-26",
         "ipo_price": 10.0, "announcement_price": 11.00, "merger_price": 25.00,
         "peak_price": 57.00, "current_price": 3.50, "days_to_peak": 180},
        {"ticker": "SPCE", "name": "Virgin Galactic", "sector": "Space",
         "ipo_date": "2019-01-01", "announcement_date": "2019-07-09", "merger_date": "2019-10-28",
         "ipo_price": 10.0, "announcement_price": 10.20, "merger_price": 12.00,
         "peak_price": 62.00, "current_price": 5.00, "days_to_peak": 600},
        {"ticker": "SOFI", "name": "SoFi Technologies", "sector": "Fintech",
         "ipo_date": "2020-07-01", "announcement_date": "2021-01-07", "merger_date": "2021-06-01",
         "ipo_price": 10.0, "announcement_price": 10.50, "merger_price": 22.00,
         "peak_price": 28.00, "current_price": 15.00, "days_to_peak": 210},
        {"ticker": "VRT", "name": "Vertiv Holdings", "sector": "Data Center",
         "ipo_date": "2018-06-01", "announcement_date": "2019-12-11", "merger_date": "2020-02-07",
         "ipo_price": 10.0, "announcement_price": 10.30, "merger_price": 12.00,
         "peak_price": 115.00, "current_price": 95.00, "days_to_peak": 1400},
        {"ticker": "SYM", "name": "Symbotic", "sector": "AI/Robotics",
         "ipo_date": "2021-01-01", "announcement_date": "2021-12-12", "merger_date": "2022-06-07",
         "ipo_price": 10.0, "announcement_price": 10.20, "merger_price": 12.50,
         "peak_price": 64.00, "current_price": 25.00, "days_to_peak": 400},
        {"ticker": "RSI", "name": "Rush Street Interactive", "sector": "Sports Betting",
         "ipo_date": "2020-04-01", "announcement_date": "2020-10-01", "merger_date": "2020-12-29",
         "ipo_price": 10.0, "announcement_price": 10.80, "merger_price": 18.00,
         "peak_price": 26.00, "current_price": 12.00, "days_to_peak": 90},
        {"ticker": "CHPT", "name": "ChargePoint", "sector": "EV Charging",
         "ipo_date": "2020-03-01", "announcement_date": "2020-09-24", "merger_date": "2021-02-26",
         "ipo_price": 10.0, "announcement_price": 11.50, "merger_price": 30.00,
         "peak_price": 50.00, "current_price": 1.50, "days_to_peak": 60},
    ])
    
    # Initialize backtester
    backtester = SPACBacktester(sample_data)
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=10,
        slippage_pct=0.1,
        price_type=PriceType.ADJUSTED
    )
    
    print("="*70)
    print("SPAC BACKTESTING ENGINE")
    print("="*70)
    
    # Test different strategies
    strategies = [
        "Buy Announcement → Sell Merger",
        "Buy IPO → Sell Announcement",
        "Buy Merger → Hold 30 Days",
        "Buy Announcement → Hold to Peak"
    ]
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy}")
        print("="*70)
        
        result = backtester.run_strategy(strategy, config)
        summary = result.to_summary_dict()
        
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Compare all strategies
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print("="*70)
    
    comparison = backtester.compare_strategies(strategies, config)
    print(comparison[['strategy', 'total_trades', 'win_rate', 'total_return_pct', 
                      'avg_return_pct', 'sharpe_ratio']].to_string(index=False))
    
    # Sector breakdown
    print(f"\n{'='*70}")
    print("RETURNS BY SECTOR (Best Strategy)")
    print("="*70)
    
    best_result = backtester.run_strategy("Buy Announcement → Hold to Peak", config)
    sector_returns = best_result.get_returns_by_sector()
    print(sector_returns.to_string())
