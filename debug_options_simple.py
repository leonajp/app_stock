"""Simple debug - run this and share output"""
import requests
import pandas as pd
from io import StringIO

THETA_URL = "http://127.0.0.1:25503"

print("=" * 60)
print("DEBUG: Option Data for Morning Dip Dashboard")
print("=" * 60)

# Step 1: Check ThetaData
print("\n[1] ThetaData Connection Test:")
try:
    r = requests.get(f"{THETA_URL}/v3/stock/snapshot/quote", 
                     params={"symbol": "APP"}, timeout=5)
    print(f"    Status: {r.status_code}")
    if r.status_code == 200:
        print(f"    ✅ Connected!")
        df = pd.read_csv(StringIO(r.text))
        if not df.empty:
            print(f"    APP price: ${df['last'].iloc[0] if 'last' in df.columns else 'N/A'}")
    else:
        print(f"    Response: {r.text[:200]}")
except Exception as e:
    print(f"    ❌ Error: {e}")
    print("\n    >>> ThetaData Terminal is NOT running!")
    print("    >>> Start it first, then re-run this script")
    exit()

# Step 2: Get expirations
print("\n[2] Getting APP Option Expirations:")
try:
    r = requests.get(f"{THETA_URL}/v3/option/snapshot/quote",
                     params={"symbol": "APP", "expiration": "*"}, timeout=30)
    print(f"    Status: {r.status_code}")
    if r.status_code == 200:
        df = pd.read_csv(StringIO(r.text))
        print(f"    Rows: {len(df)}")
        print(f"    Columns: {list(df.columns)}")
        if 'expiration' in df.columns:
            exps = sorted(df['expiration'].unique())
            print(f"    Expirations: {exps[:5]}...")
        else:
            print("    ❌ No 'expiration' column!")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Step 3: Get strikes for first expiration
print("\n[3] Getting Strikes for First Expiration:")
try:
    # Use a specific recent expiration
    test_exp = "20251220"  # Try Dec 20, 2025
    r = requests.get(f"{THETA_URL}/v3/option/snapshot/quote",
                     params={"symbol": "APP", "expiration": test_exp}, timeout=30)
    print(f"    Expiration: {test_exp}")
    print(f"    Status: {r.status_code}")
    if r.status_code == 200:
        df = pd.read_csv(StringIO(r.text))
        print(f"    Rows: {len(df)}")
        if 'strike' in df.columns:
            strikes = sorted(df['strike'].unique())
            print(f"    Strikes near $665: {[s for s in strikes if 650 <= s <= 680]}")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Step 4: Get historical option data
print("\n[4] Getting Historical Option OHLC (the key test!):")
try:
    test_params = {
        "symbol": "APP",
        "expiration": "20251220",
        "strike": "665",
        "right": "CALL",
        "date": "20251215"
    }
    print(f"    Params: {test_params}")
    r = requests.get(f"{THETA_URL}/v3/option/history/ohlc",
                     params=test_params, timeout=60)
    print(f"    Status: {r.status_code}")
    print(f"    Response length: {len(r.text)} chars")
    
    if r.status_code == 200 and r.text.strip():
        df = pd.read_csv(StringIO(r.text))
        print(f"    Rows: {len(df)}")
        print(f"    Columns: {list(df.columns)}")
        if not df.empty:
            print(f"    First row: {df.iloc[0].to_dict()}")
            df_valid = df[df['close'] > 0] if 'close' in df.columns else df
            print(f"    Rows with close > 0: {len(df_valid)}")
            if not df_valid.empty:
                print(f"\n    ✅ OPTION DATA AVAILABLE!")
                print(f"    Sample prices: {df_valid['close'].head().tolist()}")
            else:
                print(f"\n    ❌ All close prices are 0 or missing")
    else:
        print(f"    Response: {r.text[:500]}")
except Exception as e:
    print(f"    ❌ Error: {e}")

print("\n" + "=" * 60)
print("END DEBUG")
print("=" * 60)
