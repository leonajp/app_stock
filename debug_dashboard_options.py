"""Debug why option columns are empty in dashboard"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO
import os

THETA_URL = "http://127.0.0.1:25503"
POLYGON_KEY = os.environ.get('POLYGON_API_KEY')

print("=" * 70)
print("Debugging Morning Dip Dashboard Option Data")
print("=" * 70)

# 1. Check ThetaData connection
print("\n1. ThetaData Connection:")
try:
    response = requests.get(f"{THETA_URL}/v3/stock/snapshot/quote", 
                           params={"symbol": "AAPL"}, timeout=2)
    if response.status_code == 200:
        print(f"   ✅ ThetaData connected (status={response.status_code})")
        theta_available = True
    else:
        print(f"   ❌ ThetaData returned status {response.status_code}")
        theta_available = False
except Exception as e:
    print(f"   ❌ ThetaData not reachable: {e}")
    theta_available = False

# 2. Check Polygon connection
print("\n2. Polygon Connection:")
if POLYGON_KEY:
    print(f"   ✅ Polygon API key found: {POLYGON_KEY[:8]}...")
else:
    print("   ❌ No POLYGON_API_KEY environment variable")

# 3. Test a specific date (12/15)
print("\n3. Testing option data for 12/15/2025:")
test_date = "20251215"
symbol = "APP"

if theta_available:
    # Get expirations
    print("\n   a. Getting expirations...")
    try:
        response = requests.get(
            f"{THETA_URL}/v3/option/snapshot/quote",
            params={"symbol": symbol, "expiration": "*"},
            timeout=30
        )
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if not df.empty and 'expiration' in df.columns:
                expirations = sorted(df['expiration'].unique().tolist())
                print(f"      Found {len(expirations)} expirations")
                print(f"      First 5: {expirations[:5]}")
            else:
                print(f"      ❌ Empty or no expiration column")
                expirations = []
        else:
            print(f"      ❌ Status {response.status_code}: {response.text[:100]}")
            expirations = []
    except Exception as e:
        print(f"      ❌ Error: {e}")
        expirations = []
    
    # Find valid expiration for 12/15
    print("\n   b. Finding valid expiration for 12/15...")
    date_dt = datetime.strptime(test_date, "%Y%m%d")
    valid_exp = None
    for exp in expirations:
        try:
            if '-' in exp:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            else:
                exp_dt = datetime.strptime(exp, "%Y%m%d")
            days_to_exp = (exp_dt - date_dt).days
            print(f"      {exp}: {days_to_exp} days to expiration")
            if 0 <= days_to_exp <= 7:
                valid_exp = exp
                print(f"      ✅ Using {exp} ({days_to_exp} DTE)")
                break
        except Exception as e:
            print(f"      Error parsing {exp}: {e}")
    
    if not valid_exp:
        print("      ❌ No valid expiration found (0-7 DTE)")
    
    # Get strikes
    if valid_exp:
        print(f"\n   c. Getting strikes for {valid_exp}...")
        try:
            response = requests.get(
                f"{THETA_URL}/v3/option/snapshot/quote",
                params={"symbol": symbol, "expiration": valid_exp},
                timeout=30
            )
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty and 'strike' in df.columns:
                    strikes = sorted(df['strike'].unique().tolist())
                    print(f"      Found {len(strikes)} strikes")
                    # Find ATM around $665 (entry price on 12/15)
                    entry_price = 665
                    nearby = [s for s in strikes if abs(s - entry_price) <= 15]
                    print(f"      Strikes near ${entry_price}: {nearby}")
                else:
                    print(f"      ❌ Empty or no strike column")
                    strikes = []
            else:
                print(f"      ❌ Status {response.status_code}")
                strikes = []
        except Exception as e:
            print(f"      ❌ Error: {e}")
            strikes = []
        
        # Get option price
        if strikes:
            test_strike = min(strikes, key=lambda x: abs(x - 665))
            print(f"\n   d. Getting option history for ${test_strike}C on {test_date}...")
            try:
                response = requests.get(
                    f"{THETA_URL}/v3/option/history/ohlc",
                    params={
                        "symbol": symbol,
                        "expiration": valid_exp,
                        "strike": str(test_strike),
                        "right": "CALL",
                        "date": test_date
                    },
                    timeout=60
                )
                print(f"      Status: {response.status_code}")
                print(f"      Response length: {len(response.text)}")
                
                if response.status_code == 200 and response.text:
                    df = pd.read_csv(StringIO(response.text))
                    print(f"      Rows: {len(df)}")
                    
                    if not df.empty:
                        df_valid = df[df['close'] > 0]
                        print(f"      Rows with prices: {len(df_valid)}")
                        
                        if not df_valid.empty:
                            print(f"      First price: ${df_valid['close'].iloc[0]:.2f}")
                            print(f"      Last price: ${df_valid['close'].iloc[-1]:.2f}")
                            print(f"\n      ✅ Option data IS available!")
                        else:
                            print(f"      ❌ All prices are zero")
                else:
                    print(f"      Response: {response.text[:200]}")
            except Exception as e:
                print(f"      ❌ Error: {e}")

# 4. Check what the dashboard class is doing
print("\n" + "=" * 70)
print("4. Testing Dashboard MorningDipBacktester class...")
print("=" * 70)

# Import from streamlit_app
import sys
sys.path.insert(0, '/mnt/user-data/outputs')

try:
    # We need to mock streamlit
    class MockSt:
        def set_page_config(self, **kwargs): pass
        def markdown(self, *args, **kwargs): pass
        session_state = {}
    
    import sys
    sys.modules['streamlit'] = MockSt()
    
    # Now try to test the class
    print("\n   Testing MorningDipBacktester...")
    print(f"   theta_available in class would be: {theta_available}")
    
except Exception as e:
    print(f"   Error importing: {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if theta_available:
    print("✅ ThetaData is running")
    print("   Option prices should be available for CURRENT date")
    print("   For HISTORICAL dates, ThetaData needs historical data access")
else:
    print("❌ ThetaData is NOT running")
    print("   Start ThetaData Terminal to get option prices")
