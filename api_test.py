# kite_test.py

import logging
import os
from kiteconnect import KiteTicker
from dotenv import load_dotenv

# Load credentials
load_dotenv()
API_KEY = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

# RELIANCE instrument token
INSTRUMENT_TOKEN = 738561

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create KiteTicker instance
kws = KiteTicker(API_KEY, ACCESS_TOKEN)


def on_ticks(ws, ticks):
    print("📈 Ticks received:")
    for tick in ticks:
        print("-" * 40)
        for key, val in tick.items():
            print(f"{key}: {val}")


def on_connect(ws, response):
    print("✅ Connected to Kite WebSocket")
    ws.subscribe([INSTRUMENT_TOKEN])
    ws.set_mode(ws.MODE_FULL, [INSTRUMENT_TOKEN])


def on_close(ws, code, reason):
    print(f"❌ Disconnected: {code} - {reason}")
    ws.stop()


# Attach callbacks
kws.on_ticks = on_ticks
kws.on_connect = on_connect
kws.on_close = on_close

# Start connection
print("🚀 Connecting to Kite WebSocket...")
kws.connect(threaded=False)
