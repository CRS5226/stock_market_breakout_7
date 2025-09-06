# personal_test.py

import os
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

api_key = os.getenv("KITE_API_KEY")
access_token = os.getenv("KITE_ACCESS_TOKEN")

# Initialize KiteConnect client
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

try:
    # ✅ 1. Get user profile
    profile = kite.profile()
    print("\n👤 User Profile:")
    print(f"Name     : {profile.get('user_name')}")
    print(f"User ID  : {profile.get('user_id')}")
    print(f"Email    : {profile.get('email')}")
    print(f"Broker   : {profile.get('broker')}")
    print(f"Product  : {profile.get('products')}")

    # ✅ 2. Get holdings (stocks you hold)
    holdings = kite.holdings()
    print(f"\n📦 Holdings ({len(holdings)} items):")
    for h in holdings:
        print(
            f"{h['tradingsymbol']} - Qty: {h['quantity']} @ AvgPrice: ₹{h['average_price']}"
        )

    # ✅ 3. Get margins
    margins = kite.margins()
    print("\n💰 Available Margins:")
    print(f"Equity    : ₹{margins['equity']['available']['cash']}")
    print(f"Commodity : ₹{margins['commodity']['available']['cash']}")

    # ✅ 4. Get quote for a stock
    quote = kite.quote(["NSE:RELIANCE"])
    print("\n📈 RELIANCE Quote:")
    q = quote["NSE:RELIANCE"]
    print(f"Last Price   : ₹{q['last_price']}")
    print(f"Day High     : ₹{q['ohlc']['high']}")
    print(f"Day Low      : ₹{q['ohlc']['low']}")
    print(f"Volume       : {q['volume']}")

except Exception as e:
    print(f"\n❌ API Error: {e}")
