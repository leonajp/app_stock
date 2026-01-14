"""Test if MorningDipBacktester can connect to ThetaData"""
import requests
import os

# First, direct test
print("=" * 60)
print("1. Direct ThetaData test:")
print("=" * 60)
try:
    response = requests.get(
        "http://127.0.0.1:25503/v3/stock/snapshot/quote",
        params={"symbol": "AAPL"},
        timeout=5
    )
    print(f"   Status: {response.status_code}")
    print(f"   ✅ Direct connection works!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Now test via the backtester class logic
print("\n" + "=" * 60)
print("2. Testing MorningDipBacktester logic:")
print("=" * 60)

theta_available = False
theta_url = "http://127.0.0.1:25503"
theta_error = None

try:
    response = requests.get(
        f"{theta_url}/v3/stock/snapshot/quote", 
        params={"symbol": "AAPL"}, 
        timeout=5
    )
    print(f"   Response status: {response.status_code}")
    if response.status_code == 200:
        theta_available = True
        print(f"   ✅ theta_available = True")
    else:
        theta_error = f"Status {response.status_code}"
        print(f"   ❌ theta_error = {theta_error}")
except requests.exceptions.ConnectionError:
    theta_error = "Connection refused - ThetaData Terminal not running"
    print(f"   ❌ theta_error = {theta_error}")
except requests.exceptions.Timeout:
    theta_error = "Connection timeout"
    print(f"   ❌ theta_error = {theta_error}")
except Exception as e:
    theta_error = str(e)
    print(f"   ❌ theta_error = {theta_error}")

print(f"\n   Final: theta_available = {theta_available}")

# Test option data fetch
if theta_available:
    print("\n" + "=" * 60)
    print("3. Testing option data fetch:")
    print("=" * 60)
    
    from io import StringIO
    import pandas as pd
    
    response = requests.get(
        f"{theta_url}/v3/option/history/ohlc",
        params={
            "symbol": "APP",
            "expiration": "2025-12-19",
            "strike": "665",
            "right": "CALL",
            "date": "20251215"
        },
        timeout=30
    )
    
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text))
        df_valid = df[(df['close'] > 0) | (df['volume'] > 0)]
        print(f"   Total rows: {len(df)}")
        print(f"   Rows with trades: {len(df_valid)}")
        if len(df_valid) > 0:
            print(f"   ✅ Option data available!")
            print(f"   First price: ${df_valid['close'].iloc[0]:.2f}")
            print(f"   Last price: ${df_valid['close'].iloc[-1]:.2f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)