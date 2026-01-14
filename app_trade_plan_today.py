"""
APP Trade Plan - December 12, 2025
===================================
APP dropped with the broader tech selloff (Nasdaq -1.6%)

Based on mean reversion analysis from our backtest.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# TODAY'S MARKET DATA (Dec 12, 2025)
# ============================================================================

MARKET_DATA = {
    # Previous day (Dec 11)
    "prev_close": 716.98,
    "prev_high": 723.00,
    "prev_low": 696.00,
    
    # Today's estimates (Dec 12) - with tech selloff
    # Nasdaq down 1.6%, if APP follows similar move
    "estimated_current": 695.00,  # ~3% drop estimate
    "day_low_estimate": 680.00,   # Possible intraday low
    "day_high_estimate": 710.00,  # Resistance
    
    # Key levels from analysis
    "support_1": 700.00,
    "support_2": 680.00,
    "support_3": 650.00,  # Strong support from models
    "resistance_1": 720.00,
    "resistance_2": 728.00,  # Major resistance
    "all_time_high": 745.61,
    "52_week_low": 200.50,
}

# ============================================================================
# SIGNAL ANALYSIS
# ============================================================================

def analyze_todays_setup():
    """Analyze today's trading setup for APP"""
    
    prev_close = MARKET_DATA["prev_close"]
    est_current = MARKET_DATA["estimated_current"]
    
    # Calculate gap/drop
    drop_pct = (est_current / prev_close - 1) * 100
    
    print("=" * 70)
    print("APP TRADE PLAN - December 12, 2025")
    print("=" * 70)
    
    print(f"""
📊 MARKET CONTEXT:
   • Nasdaq: -1.6% (tech selloff, Broadcom -10%)
   • S&P 500: -1.1%
   • Sentiment: Risk-off rotation out of AI/tech
   
📈 APP PRICE DATA:
   • Yesterday Close: ${prev_close:.2f}
   • Estimated Current: ${est_current:.2f}
   • Today's Move: {drop_pct:+.1f}%
   • Day Range Est: ${MARKET_DATA['day_low_estimate']:.0f} - ${MARKET_DATA['day_high_estimate']:.0f}
""")

    print("=" * 70)
    print("🎯 MEAN REVERSION SIGNAL ANALYSIS")
    print("=" * 70)
    
    # Signal determination based on our backtest findings
    if drop_pct < -2.0:
        signal = "BULLISH"
        confidence = "HIGH" if drop_pct < -3.0 else "MODERATE"
        
        print(f"""
   SIGNAL: 🟢 {signal} (Confidence: {confidence})
   
   REASONING:
   • Large drop ({drop_pct:.1f}%) triggers CONTRARIAN signal
   • Our backtest showed: After DOWN gaps > 2%, expect bounce
   • Historical win rate for oversold bounces: ~65%
   • Strong support at $680-$700 zone
   
   ⚠️ CAUTION FACTORS:
   • Broader tech selling may continue
   • RSI likely approaching oversold (watch for < 30)
   • Volume needed to confirm reversal
""")
    elif drop_pct < -1.0:
        signal = "LEAN BULLISH"
        confidence = "LOW"
        print(f"""
   SIGNAL: 🟡 {signal} (Confidence: {confidence})
   
   REASONING:
   • Moderate drop ({drop_pct:.1f}%) - waiting for better entry
   • Watch for RSI < 30 or test of $680 support
""")
    else:
        signal = "NEUTRAL"
        print(f"""
   SIGNAL: ⚪ {signal}
   • Drop not significant enough for mean reversion trade
""")

    return signal, drop_pct


def generate_trade_plan(signal, drop_pct):
    """Generate specific trade recommendations"""
    
    print("=" * 70)
    print("📋 TRADE RECOMMENDATIONS")
    print("=" * 70)
    
    if "BULLISH" in signal:
        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  PRIMARY TRADE: BUY CALLS (Mean Reversion Bounce)                   │
└─────────────────────────────────────────────────────────────────────┘

   OPTION DETAILS:
   ├── Strike: ATM ($695-$700)
   ├── Expiry: 0DTE or Dec 13 (Friday expiry)
   ├── Type: BUY CALL
   └── Position Size: 1-2% of account

   ENTRY TRIGGERS (wait for confirmation):
   ├── 1. RSI crosses above 30 (oversold bounce)
   ├── 2. Price holds $680 support with volume
   ├── 3. First green 5-min candle after testing low
   └── 4. VIX/market stabilization

   PROFIT TARGETS:
   ├── Target 1: $705-$710 (1.5-2% move) → Take 50% off
   ├── Target 2: $715-$720 (return to prev close) → Take remaining
   └── Max Target: $728 (if momentum continues)

   STOP LOSS:
   ├── Price: Below $675 (or -40% on premium)
   └── Time: Exit by 3:30 PM if target not hit

   EXPECTED P&L:
   ├── Win: +40% to +80% on premium (if hits $710+)
   └── Loss: -30% to -40% if stopped out

┌─────────────────────────────────────────────────────────────────────┐
│  ALTERNATIVE: BULL CALL SPREAD (Lower Risk)                         │
└─────────────────────────────────────────────────────────────────────┘

   SPREAD DETAILS:
   ├── Buy: $695 Call
   ├── Sell: $710 Call
   ├── Max Profit: $15 spread width minus premium
   └── Max Loss: Premium paid

   WHY SPREAD:
   ├── Lower cost than naked call
   ├── Defined risk
   └── Works if price bounces to $710+ by EOD
""")

        print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️ DO NOT TRADE IF:                                                │
└─────────────────────────────────────────────────────────────────────┘

   ❌ Nasdaq continues falling (watch for -2.5%+ day)
   ❌ APP breaks below $675 with volume
   ❌ VIX spikes above 20
   ❌ More negative tech news emerges
   ❌ You're already at max daily risk
""")

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  📊 KEY LEVELS TO WATCH                                             │
└─────────────────────────────────────────────────────────────────────┘

   SUPPORT:
   ├── $700 - Psychological level
   ├── $680 - Strong support (model target)
   ├── $650 - Major support if selloff accelerates
   └── $632 - Deep support

   RESISTANCE:
   ├── $710 - First target
   ├── $720 - Yesterday's range high
   ├── $728 - Major resistance (consolidation zone)
   └── $745 - All-time high

┌─────────────────────────────────────────────────────────────────────┐
│  ⏰ TIMING                                                          │
└─────────────────────────────────────────────────────────────────────┘

   BEST ENTRY WINDOWS:
   ├── 9:35-9:45 AM - After initial volatility settles
   ├── 10:00-10:30 AM - If testing support with reversal
   └── 2:00-2:30 PM - Afternoon reversal opportunity

   AVOID:
   ├── 9:30-9:35 AM - Too volatile
   ├── 12:00-1:00 PM - Low volume lunch
   └── After 3:30 PM - Gamma risk on 0DTE
""")


def print_risk_management():
    """Print risk management rules"""
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  💰 RISK MANAGEMENT                                                 │
└─────────────────────────────────────────────────────────────────────┘

   POSITION SIZING:
   ├── Max per trade: 1-2% of account
   ├── Max daily loss: 5% of account
   └── If down 3% on day, STOP trading

   0DTE OPTIONS RULES:
   ├── Never hold into close
   ├── Take profits at 50%+ gain
   ├── Cut losses at 40% quickly
   └── Time decay accelerates after 2 PM

   WHAT SUCCESS LOOKS LIKE:
   ├── Win Rate Target: 55-60%
   ├── Avg Win/Loss Ratio: 1.3:1
   └── Monthly Goal: 5-10% account growth

""")


def print_alternative_scenarios():
    """Print what to do in different scenarios"""
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  🔄 SCENARIO PLANNING                                               │
└─────────────────────────────────────────────────────────────────────┘

   SCENARIO A: APP bounces from $680-$690 to $710+
   └── ✅ Take the call trade, ride to target

   SCENARIO B: APP continues falling to $650
   └── ⏸️ Wait - don't catch falling knife
   └── Look for RSI < 25 or capitulation volume

   SCENARIO C: APP chops between $690-$710 all day
   └── 😐 Small gains possible, manage theta decay
   └── Consider closing early if no momentum

   SCENARIO D: Market reverses, Nasdaq goes green
   └── 🚀 APP likely to outperform - aggressive calls
   └── Target $720+ quickly

   SCENARIO E: More bad tech news hits
   └── 🛑 Stay out or buy puts for hedge
   └── Wait for next day setup
""")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    signal, drop_pct = analyze_todays_setup()
    generate_trade_plan(signal, drop_pct)
    print_risk_management()
    print_alternative_scenarios()
    
    print("=" * 70)
    print("📱 EXECUTION CHECKLIST")
    print("=" * 70)
    print("""
   □ Check current APP price and RSI
   □ Confirm Nasdaq/SPY direction
   □ Set alerts at $680, $700, $710
   □ Calculate position size (1-2% max)
   □ Have stop loss ready BEFORE entry
   □ Know your exit: target OR stop OR time
   
   🎯 BOTTOM LINE:
   
   If APP is down 3%+ and holding $680 support with RSI < 35:
   → BUY ATM CALL, target $710, stop below $675
   
   Expected: 55-65% win rate, 1.5:1 reward/risk
""")
    print("=" * 70)
