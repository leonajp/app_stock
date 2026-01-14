"""
Simple debug - what's actually tradeable for 2025-12-19?
"""
import requests
from io import StringIO
import pandas as pd

THETA_URL = "http://127.0.0.1:25503"

print("Getting APP option chain...")
response = requests.get(
    f"{THETA_URL}/v3/option/snapshot/quote",
    params={"symbol": "APP", "expiration": "*"},
    timeout=60
)
df = pd.read_csv(StringIO(response.text))

print(f"\nTotal contracts: {len(df)}")
print(f"Columns: {list(df.columns)}")

# What expirations have active CALLs?
print("\n" + "=" * 70)
print("ACTIVE CALLS BY EXPIRATION (bid > 0):")
print("=" * 70)

calls = df[(df['right'] == 'CALL') & (df['bid'] > 0)]
for exp in sorted(calls['expiration'].unique())[:8]:
    exp_calls = calls[calls['expiration'] == exp]
    strikes = sorted(exp_calls['strike'].unique())
    
    # Find one near $670
    near_atm = exp_calls.iloc[(exp_calls['strike'] - 670).abs().argsort().iloc[0]]
    
    print(f"\n{exp}: {len(exp_calls)} active calls")
    print(f"  Strikes: ${min(strikes):.0f} - ${max(strikes):.0f}")
    print(f"  Near ATM: ${near_atm['strike']:.0f} @ ${near_atm['bid']:.2f}/${near_atm['ask']:.2f}")

# Check 2025-12-19 specifically
print("\n" + "=" * 70)
print("2025-12-19 DETAILS:")
print("=" * 70)

exp_1219 = df[df['expiration'] == '2025-12-19']
print(f"Total contracts for 2025-12-19: {len(exp_1219)}")

if len(exp_1219) > 0:
    # Check if any have non-zero bids
    active = exp_1219[exp_1219['bid'] > 0]
    print(f"Active (bid > 0): {len(active)}")
    
    if len(active) > 0:
        print("\nSample active contracts:")
        print(active[['strike', 'right', 'bid', 'ask']].head(10))
    else:
        print("\n⚠️ NO ACTIVE CONTRACTS FOR 2025-12-19!")
        print("All bids are $0 - this expiration may not be trading yet")
        print("\nRaw sample (first 5 rows):")
        print(exp_1219[['strike', 'right', 'bid', 'ask']].head())

# Recommend which expiration to use
print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

best_exp = None
for exp in sorted(calls['expiration'].unique()):
    exp_calls = calls[calls['expiration'] == exp]
    # Check for ATM activity
    near_atm = exp_calls[(exp_calls['strike'] >= 650) & (exp_calls['strike'] <= 700)]
    if len(near_atm) > 0:
        best_exp = exp
        break

if best_exp:
    print(f"Use expiration: {best_exp}")
    best_calls = calls[calls['expiration'] == best_exp]
    atm = best_calls.iloc[(best_calls['strike'] - 670).abs().argsort().iloc[0]]
    print(f"ATM Call: ${atm['strike']:.0f} @ ${atm['bid']:.2f}/${atm['ask']:.2f}")
else:
    print("No good expiration found!")