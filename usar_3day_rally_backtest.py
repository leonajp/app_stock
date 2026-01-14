#!/usr/bin/env python3
"""
Backtest: After USAR is up 3 days in a row, what happens the next day?

- Fetches daily OHLC from Polygon.io
- Identifies all points where close increased 3 days in a row
- Measures next-day return:
    * How often it goes up vs down vs flat
    * Average next-day return

Usage:
    POLYGON_API_KEY="YOUR_REAL_KEY" python usar_3day_rally_backtest.py
"""

import os
import requests
import pandas as pd
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
TICKER = "USAR"
START_DATE = "2015-01-01"   # adjust if you want shorter history
END_DATE = datetime.today().strftime("%Y-%m-%d")

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise RuntimeError("Please set POLYGON_API_KEY environment variable.")


# -----------------------------
# Data fetcher from Polygon
# -----------------------------
def fetch_daily_ohlc(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLC data from Polygon v2 aggregates.

    Returns a DataFrame with columns:
    ['date', 'open', 'high', 'low', 'close', 'volume']
    """
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start_date}/{end_date}"
    )
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    results = []
    url = base_url
    while True:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "OK":
            raise RuntimeError(f"Polygon error: {data}")

        results.extend(data.get("results", []))

        # Polygon v2 aggs can return a next_url for pagination
        next_url = data.get("next_url")
        if not next_url:
            break

        # After first request, use the next_url directly (it already contains params)
        url = next_url
        params = {}  # params already embedded in next_url

    if not results:
        raise RuntimeError("No data returned from Polygon. Check ticker/date range.")

    df = pd.DataFrame(results)

    # Map Polygon fields to cleaner names
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )

    df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
    df = df.reset_index(drop=True)
    return df


# -----------------------------
# Backtest logic
# -----------------------------
def analyze_three_day_rallies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a daily OHLC DataFrame, find all events where:
        close_t > close_{t-1} > close_{t-2}
    and compute next-day return (t+1 vs t).

    Returns a DataFrame of events with:
        ['signal_date', 'day0_close', 'next_date', 'next_close', 'next_ret']
    """
    events = []

    # We need t-2, t-1, t, and t+1, so start at index 2 and stop at len-2
    for i in range(2, len(df) - 1):
        c0 = df.loc[i - 2, "close"]
        c1 = df.loc[i - 1, "close"]
        c2 = df.loc[i, "close"]      # day of the 3rd up close
        c3 = df.loc[i + 1, "close"]  # next day's close

        # Strictly higher 3 days in a row
        if c0 < c1 < c2:
            next_ret = (c3 / c2) - 1.0
            events.append(
                {
                    "signal_date": df.loc[i, "date"],
                    "day0_close": c2,
                    "next_date": df.loc[i + 1, "date"],
                    "next_close": c3,
                    "next_ret": next_ret,
                }
            )

    if not events:
        raise RuntimeError("No 3-day-up sequences found in the data.")

    events_df = pd.DataFrame(events)
    return events_df


def summarize_events(events_df: pd.DataFrame) -> None:
    """
    Print summary stats: how often next day is up/down/flat,
    and basic return statistics.
    """
    n = len(events_df)
    up = (events_df["next_ret"] > 0).sum()
    down = (events_df["next_ret"] < 0).sum()
    flat = (events_df["next_ret"] == 0).sum()

    pct_up = up / n * 100
    pct_down = down / n * 100
    pct_flat = flat / n * 100

    mean_ret = events_df["next_ret"].mean() * 100
    median_ret = events_df["next_ret"].median() * 100
    max_ret = events_df["next_ret"].max() * 100
    min_ret = events_df["next_ret"].min() * 100

    print(f"=== 3-Day Up Streak Analysis for {TICKER} ===")
    print(f"Sample window: {events_df['signal_date'].min().date()} "
          f"to {events_df['signal_date'].max().date()}")
    print(f"Number of 3-up signals: {n}")
    print()
    print("Next-day direction after 3 up days:")
    print(f"  Up   : {up:3d} ({pct_up:5.1f}%)")
    print(f"  Down : {down:3d} ({pct_down:5.1f}%)")
    print(f"  Flat : {flat:3d} ({pct_flat:5.1f}%)")
    print()
    print("Next-day return statistics (close_{t+1} / close_t - 1):")
    print(f"  Mean   : {mean_ret:6.3f}%")
    print(f"  Median : {median_ret:6.3f}%")
    print(f"  Max    : {max_ret:6.3f}%")
    print(f"  Min    : {min_ret:6.3f}%")


# -----------------------------
# Main
# -----------------------------
def main():
    print(f"Fetching daily data for {TICKER} from Polygon...")
    df = fetch_daily_ohlc(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        api_key=POLYGON_API_KEY,
        adjusted=True,
    )
    print(f"Got {len(df)} trading days of data.")

    events_df = analyze_three_day_rallies(df)
    summarize_events(events_df)

    # Optional: save the events to CSV for further analysis
    out_file = f"{TICKER}_3day_rally_events.csv"
    events_df.to_csv(out_file, index=False)
    print(f"\nEvent details saved to {out_file}")


if __name__ == "__main__":
    main()
