"""
APP Options Trading Strategy - Clear Explanation
=================================================

This document explains how to interpret the backtest analysis
and translate signals into actual options trades.

Author: Limestone Hill Capital
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# WHAT THE CORRELATION CHART MEANS
# =============================================================================

"""
SESSION PREDICTOR CORRELATION EXPLAINED
---------------------------------------

The chart shows: How well does each session's return PREDICT the first 5 minutes?

Example output:
    gap_return:       correlation = -0.15 (CONTRARIAN)
    premarket_return: correlation = +0.08 (MOMENTUM)

INTERPRETATION:

1. NEGATIVE correlation (CONTRARIAN signal):
   - When predictor is POSITIVE → first 5 min tends to be NEGATIVE
   - When predictor is NEGATIVE → first 5 min tends to be POSITIVE
   
   Trading implication:
   - Gap UP +2% → Expect pullback → BUY PUTS at open
   - Gap DOWN -2% → Expect bounce → BUY CALLS at open

2. POSITIVE correlation (MOMENTUM signal):
   - When predictor is POSITIVE → first 5 min tends to be POSITIVE
   - When predictor is NEGATIVE → first 5 min tends to be NEGATIVE
   
   Trading implication:
   - Premarket UP +1.5% → Expect continuation → BUY CALLS at open
   - Premarket DOWN -1.5% → Expect continuation → BUY PUTS at open

3. CORRELATION STRENGTH:
   - |corr| > 0.20: Strong signal, tradeable
   - |corr| 0.10-0.20: Moderate signal, use with confirmation
   - |corr| < 0.10: Weak signal, probably noise

4. P-VALUE:
   - p < 0.05: Statistically significant (not random)
   - p > 0.05: Could be random noise, be careful
"""

# =============================================================================
# CONCRETE OPTIONS TRADING RULES
# =============================================================================

TRADING_RULES = """
============================================================
APP OPTIONS TRADING PLAYBOOK
============================================================

SETUP: Check pre-market data by 9:25 AM ET

STEP 1: Calculate today's signals
---------------------------------
- Gap Return = (Pre-market price / Yesterday close - 1) × 100
- Premarket Return = (9:25 price / 4:00 AM price - 1) × 100
- Total Extended = Postmarket + Overnight + Premarket returns

STEP 2: Determine signal direction
----------------------------------

┌─────────────────────────────────────────────────────────────┐
│ IF Gap > +2% (Large gap up)                                 │
│ AND Gap correlation is NEGATIVE (contrarian)                │
│ THEN: Expect pullback in first 5 min                        │
│ ACTION: BUY ATM PUT at 9:30, sell by 9:45                   │
│ TARGET: 30-50% premium gain                                 │
│ STOP: 40% premium loss                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ IF Gap < -2% (Large gap down)                               │
│ AND Gap correlation is NEGATIVE (contrarian)                │
│ THEN: Expect bounce in first 5 min                          │
│ ACTION: BUY ATM CALL at 9:30, sell by 9:45                  │
│ TARGET: 30-50% premium gain                                 │
│ STOP: 40% premium loss                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ IF Premarket > +1.5% AND Premarket corr is POSITIVE         │
│ THEN: Expect momentum continuation                          │
│ ACTION: BUY ATM CALL at 9:30                                │
│ TARGET: RSI hits 70 or +1% move                             │
│ STOP: Price breaks below VWAP                               │
└─────────────────────────────────────────────────────────────┘

STEP 3: Position sizing
-----------------------
- Max 1-2% of account per trade
- Use 0DTE options for first 5-min trades
- Use weekly options for RSI mean reversion trades

STEP 4: RSI mean reversion (during the day)
-------------------------------------------

┌─────────────────────────────────────────────────────────────┐
│ IF RSI < 30 (oversold) AND volume > 1.5x average            │
│ THEN: Expect bounce                                         │
│ ACTION: BUY ATM CALL                                        │
│ TARGET: RSI crosses 50 or +1.5% move                        │
│ STOP: RSI drops below 20 or -1% move                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ IF RSI > 70 (overbought) AND volume > 1.5x average          │
│ THEN: Expect pullback                                       │
│ ACTION: BUY ATM PUT                                         │
│ TARGET: RSI crosses 50 or -1.5% move                        │
│ STOP: RSI rises above 80 or +1% move                        │
└─────────────────────────────────────────────────────────────┘

============================================================
"""

# =============================================================================
# EXAMPLE TRADE WALKTHROUGH
# =============================================================================

EXAMPLE_TRADE = """
============================================================
EXAMPLE: December 11, 2024 - APP Trade
============================================================

PRE-MARKET DATA (9:25 AM):
- Yesterday close: $710.00
- Pre-market price: $725.00
- Gap: +2.11%

HISTORICAL ANALYSIS SAYS:
- Gap correlation with first 5 min: -0.15 (CONTRARIAN)
- When gap > +2%, first 5 min averages: -0.3%
- Hit rate for pullback: 62%

TRADE DECISION:
- Signal: CONTRARIAN (gap up → expect pullback)
- Action: BUY $725 PUT at 9:30

EXECUTION:
- 9:30:00 - Buy 2x $725 Put @ $3.50 = $700 risk
- 9:32:00 - APP drops to $722, put now $4.80
- 9:35:00 - APP at $720, put now $5.90
- 9:36:00 - SELL @ $5.90 = $1,180

RESULT:
- Profit: $480 (68% gain on premium)
- Duration: 6 minutes
- Risk/Reward achieved: 1:1.7

============================================================
"""


def create_signal_explanation_chart():
    """Create visual explanation of correlation signals"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Negative Correlation (Contrarian)",
            "Positive Correlation (Momentum)",
            "How to Trade: Gap Up + Contrarian",
            "How to Trade: Oversold RSI"
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "table"}, {"type": "table"}]]
    )
    
    # Contrarian example
    gap = np.array([-2, -1, 0, 1, 2, 3])
    first5_contrarian = -0.15 * gap + np.random.normal(0, 0.3, len(gap))
    
    fig.add_trace(
        go.Scatter(x=gap, y=first5_contrarian, mode='markers', 
                   marker=dict(size=12, color='#ff6b6b'),
                   name="Data points"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=gap, y=-0.15 * gap, mode='lines',
                   line=dict(color='#00d4aa', width=2, dash='dash'),
                   name="Trend (negative slope)"),
        row=1, col=1
    )
    
    # Momentum example  
    first5_momentum = 0.12 * gap + np.random.normal(0, 0.3, len(gap))
    
    fig.add_trace(
        go.Scatter(x=gap, y=first5_momentum, mode='markers',
                   marker=dict(size=12, color='#00d4aa'),
                   name="Data points"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=gap, y=0.12 * gap, mode='lines',
                   line=dict(color='#ffd93d', width=2, dash='dash'),
                   name="Trend (positive slope)"),
        row=1, col=2
    )
    
    # Trade table 1
    fig.add_trace(
        go.Table(
            header=dict(values=['Step', 'Action'],
                       fill_color='#1a1a2e',
                       font=dict(color='white')),
            cells=dict(values=[
                ['1. Gap Up > 2%', '2. Check Correlation', '3. Signal', '4. Trade'],
                ['APP gaps up +2.5%', 'Gap corr = -0.15 (contrarian)', 
                 'Expect pullback', 'BUY ATM PUT at open']
            ],
            fill_color='#2a2a4a',
            font=dict(color='white'))
        ),
        row=2, col=1
    )
    
    # Trade table 2
    fig.add_trace(
        go.Table(
            header=dict(values=['Step', 'Action'],
                       fill_color='#1a1a2e',
                       font=dict(color='white')),
            cells=dict(values=[
                ['1. RSI Reading', '2. Volume Check', '3. Signal', '4. Trade'],
                ['RSI = 28 (oversold)', 'Volume = 1.8x avg ✓',
                 'Expect bounce', 'BUY ATM CALL, exit at RSI 50']
            ],
            fill_color='#2a2a4a',
            font=dict(color='white'))
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Gap Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="First 5min Return (%)", row=1, col=1)
    fig.update_xaxes(title_text="Gap Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="First 5min Return (%)", row=1, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        height=600,
        showlegend=False,
        title="Understanding Session Predictor Correlations"
    )
    
    return fig


def create_decision_flowchart():
    """Create a decision tree for options trading"""
    
    flowchart = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PRE-MARKET CHECKLIST (9:25 AM)               │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Calculate: Gap = (Current Price / Prev Close - 1) × 100        │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Gap > +2%    │ │ Gap < -2%    │ │ |Gap| < 2%   │
            │ (Big gap up) │ │ (Big gap dn) │ │ (Small gap)  │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │ If contrarian│ │ If contrarian│ │ Wait for RSI │
            │ correlation: │ │ correlation: │ │ signal during│
            │ BUY PUTS     │ │ BUY CALLS    │ │ the day      │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │               │               │
                    ▼               ▼               ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  DURING TRADING HOURS: Monitor RSI                              │
    │                                                                 │
    │  RSI < 30 + High Volume → BUY CALLS (expect bounce)            │
    │  RSI > 70 + High Volume → BUY PUTS (expect pullback)           │
    │                                                                 │
    │  Exit when: RSI crosses 50 OR hit profit target OR stop loss   │
    └─────────────────────────────────────────────────────────────────┘
    """
    return flowchart


def generate_todays_trade_plan(
    prev_close: float,
    current_price: float,
    premarket_high: float,
    premarket_low: float,
    gap_correlation: float,
    premarket_correlation: float
) -> dict:
    """
    Generate specific trade plan for today based on signals.
    
    Parameters:
    -----------
    prev_close : float - Yesterday's closing price
    current_price : float - Current pre-market price
    premarket_high : float - Pre-market high
    premarket_low : float - Pre-market low
    gap_correlation : float - Historical correlation of gap with first 5min
    premarket_correlation : float - Historical correlation of premarket with first 5min
    
    Returns:
    --------
    dict with trade plan details
    """
    
    gap_pct = (current_price / prev_close - 1) * 100
    
    plan = {
        "pre_market_data": {
            "prev_close": prev_close,
            "current_price": current_price,
            "gap_pct": gap_pct,
            "premarket_range": f"${premarket_low:.2f} - ${premarket_high:.2f}"
        },
        "signals": [],
        "primary_trade": None,
        "alternative_trade": None,
        "key_levels": {
            "resistance": max(premarket_high, prev_close * 1.01),
            "support": min(premarket_low, prev_close * 0.99),
            "vwap_estimate": current_price
        },
        "risk_management": {
            "max_position": "1-2% of account",
            "stop_loss": "40-50% of premium",
            "profit_target": "50-100% of premium"
        }
    }
    
    # Determine signal type
    gap_signal = "CONTRARIAN" if gap_correlation < 0 else "MOMENTUM"
    
    # Large gap up scenario
    if gap_pct > 2.0:
        if gap_signal == "CONTRARIAN":
            plan["signals"].append({
                "signal": "GAP FADE",
                "direction": "BEARISH",
                "strength": "STRONG" if gap_pct > 3 else "MODERATE",
                "reasoning": f"Gap +{gap_pct:.1f}% with negative correlation ({gap_correlation:.2f}) suggests pullback"
            })
            plan["primary_trade"] = {
                "action": "BUY PUT",
                "strike": f"ATM (${round(current_price, 0):.0f})",
                "expiry": "0DTE",
                "entry": "9:30-9:32 AM",
                "exit": "9:40-9:45 AM or RSI > 70",
                "target": "+50% premium",
                "stop": "-40% premium"
            }
            plan["alternative_trade"] = {
                "action": "BEAR PUT SPREAD",
                "strikes": f"${round(current_price, 0):.0f}/${round(current_price - 5, 0):.0f}",
                "rationale": "Lower cost, defined risk"
            }
        else:
            plan["signals"].append({
                "signal": "GAP CONTINUATION",
                "direction": "BULLISH",
                "strength": "MODERATE",
                "reasoning": f"Gap +{gap_pct:.1f}% with positive correlation suggests momentum"
            })
            plan["primary_trade"] = {
                "action": "BUY CALL",
                "strike": f"ATM (${round(current_price, 0):.0f})",
                "expiry": "0DTE",
                "entry": "On first pullback to VWAP",
                "exit": "New high or RSI > 75",
                "target": "+50% premium",
                "stop": "Break below premarket low"
            }
    
    # Large gap down scenario
    elif gap_pct < -2.0:
        if gap_signal == "CONTRARIAN":
            plan["signals"].append({
                "signal": "GAP FADE",
                "direction": "BULLISH", 
                "strength": "STRONG" if gap_pct < -3 else "MODERATE",
                "reasoning": f"Gap {gap_pct:.1f}% with negative correlation suggests bounce"
            })
            plan["primary_trade"] = {
                "action": "BUY CALL",
                "strike": f"ATM (${round(current_price, 0):.0f})",
                "expiry": "0DTE",
                "entry": "9:30-9:32 AM",
                "exit": "9:40-9:45 AM or RSI < 30 bounces",
                "target": "+50% premium",
                "stop": "-40% premium"
            }
    
    # Small gap - wait for RSI
    else:
        plan["signals"].append({
            "signal": "NO GAP TRADE",
            "direction": "NEUTRAL",
            "strength": "N/A",
            "reasoning": f"Gap {gap_pct:+.1f}% is too small for gap trade. Wait for RSI extremes."
        })
        plan["primary_trade"] = {
            "action": "WAIT FOR RSI SIGNAL",
            "condition": "RSI < 30 or RSI > 70 with volume confirmation",
            "entry": "When RSI crosses threshold",
            "notes": "Monitor throughout the day"
        }
    
    return plan


# =============================================================================
# PRINT FUNCTIONS
# =============================================================================

def print_trade_plan(plan: dict):
    """Pretty print the trade plan"""
    
    print("\n" + "=" * 60)
    print("TODAY'S APP OPTIONS TRADE PLAN")
    print("=" * 60)
    
    print("\n📊 PRE-MARKET DATA:")
    print(f"  Previous Close: ${plan['pre_market_data']['prev_close']:.2f}")
    print(f"  Current Price:  ${plan['pre_market_data']['current_price']:.2f}")
    print(f"  Gap:            {plan['pre_market_data']['gap_pct']:+.2f}%")
    print(f"  PM Range:       {plan['pre_market_data']['premarket_range']}")
    
    print("\n📈 SIGNALS:")
    for sig in plan['signals']:
        print(f"  {sig['signal']}: {sig['direction']} ({sig['strength']})")
        print(f"    → {sig['reasoning']}")
    
    if plan['primary_trade']:
        print("\n🎯 PRIMARY TRADE:")
        for key, value in plan['primary_trade'].items():
            print(f"  {key.title()}: {value}")
    
    if plan['alternative_trade']:
        print("\n🔄 ALTERNATIVE:")
        for key, value in plan['alternative_trade'].items():
            print(f"  {key.title()}: {value}")
    
    print("\n📍 KEY LEVELS:")
    print(f"  Resistance: ${plan['key_levels']['resistance']:.2f}")
    print(f"  Support:    ${plan['key_levels']['support']:.2f}")
    
    print("\n⚠️ RISK MANAGEMENT:")
    for key, value in plan['risk_management'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print(TRADING_RULES)
    print(EXAMPLE_TRADE)
    
    # Generate sample trade plan
    plan = generate_todays_trade_plan(
        prev_close=710.00,
        current_price=728.50,
        premarket_high=730.00,
        premarket_low=722.00,
        gap_correlation=-0.15,  # Contrarian
        premarket_correlation=0.08  # Momentum
    )
    
    print_trade_plan(plan)
