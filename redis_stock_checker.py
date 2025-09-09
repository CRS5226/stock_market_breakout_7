import os
import json
import pandas as pd

# Import the utility functions from your redis_utils.py file
from redis_utils import (
    get_redis,
    get_recent_candles_tf,
    get_recent_indicators_tf,  # <--- NEW import
)

# Match the TFs you defined in collector.py
TF_DEFS = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "45min": 45,
    "1hour": 60,
    "4hour": 240,
}


def fetch_candles_tf(stock_code: str, tf: str, n: int = 1):
    """Fetch last N candles for a given timeframe (chronological order)."""
    r = get_redis()
    items = get_recent_candles_tf(r, stock_code, tf, n)
    if not items:
        return None

    df = pd.DataFrame(items)
    if "bucket" in df.columns:
        df["bucket"] = pd.to_datetime(df["bucket"])
        df = df.sort_values("bucket").reset_index(drop=True)
    return df


def fetch_indicators_tf(stock_code: str, tf: str, n: int = 1):
    """Fetch last N indicators for a given timeframe (newest first)."""
    r = get_redis()
    items = get_recent_indicators_tf(r, stock_code, tf, n)
    if not items:
        return None
    return pd.DataFrame(items)


def fetch_all_timeframes(stock_code: str):
    """
    Fetch the latest candle + indicator for all timeframes.
    Returns dict: {tf: {"candle": DataFrame (1 row) or None,
                        "indicator": DataFrame (1 row) or None}}
    """
    out = {}
    for tf in TF_DEFS.keys():
        candle_df = fetch_candles_tf(stock_code, tf, n=1)
        indicator_df = fetch_indicators_tf(stock_code, tf, n=1)

        out[tf] = {
            "candle": (
                candle_df if candle_df is not None and not candle_df.empty else None
            ),
            "indicator": (
                indicator_df
                if indicator_df is not None and not indicator_df.empty
                else None
            ),
        }
    return out


def fetch_historical(stock_code: str, folder="historical_data_candles"):
    """Load the latest historical CSV for a stock."""
    if not os.path.isdir(folder):
        print(f"[⚠️] No historical_data_candles folder found: {folder}")
        return None

    hist_csv = None
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(stock_code.upper()) and file.endswith(".csv"):
                hist_csv = os.path.join(root, file)
                break

    if hist_csv and os.path.exists(hist_csv):
        try:
            df = pd.read_csv(hist_csv)
            return df
        except Exception as e:
            print(f"[❌] Error loading historical CSV {hist_csv}: {e}")
    else:
        print(f"[⚠️] No historical file found for {stock_code} in {folder}")
    return None


if __name__ == "__main__":
    stock = "GMDCLTD"

    # 🔹 Fetch latest candle + indicator for each TF
    tf_data = fetch_all_timeframes(stock)
    for tf, data in tf_data.items():
        print(f"\n=== {tf} ===")
        if data["candle"] is not None:
            print("[🕒] Candle:", data["candle"].iloc[-1].to_dict())
        else:
            print("[⏳] Candle: None")

        if data["indicator"] is not None:
            print("[📈] Indicator:", data["indicator"].iloc[-1].to_dict())
        else:
            print("[⏳] Indicator: None")
