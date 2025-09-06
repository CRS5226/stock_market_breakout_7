import os
import json
import pandas as pd

# Import the utility functions from your redis_utils.py file
from redis_utils import get_redis, get_recent_candles, get_recent_indicators


def fetch_candles(stock_code: str, n: int = 2):
    """Fetch last N candles from Redis (chronological order)."""
    r = get_redis()
    items = get_recent_candles(r, stock_code, n)  # Use the function from redis_utils
    if not items:
        print(f"[⚠️] No candles found in Redis for {stock_code}")
        return None

    df = pd.DataFrame(items)
    df["minute"] = pd.to_datetime(df["minute"])
    # The get_recent_candles function returns newest-first, so sort it for chronological order
    return df.sort_values("minute").reset_index(drop=True)


def fetch_indicators(stock_code: str, n: int = 2):
    """Fetch last N indicators from Redis (newest first)."""
    r = get_redis()
    items = get_recent_indicators(r, stock_code, n)  # Use the function from redis_utils
    if not items:
        print(f"[⚠️] No indicators found in Redis for {stock_code}")
        return None

    return pd.DataFrame(items)


def fetch_historical(stock_code: str, folder="historical_data"):
    """Load the latest historical CSV for a stock."""
    if not os.path.isdir(folder):
        print(f"[⚠️] No historical_data folder found: {folder}")
        return None

    hist_csv = None
    # Use a more robust way to find the file
    file_prefix = f"{stock_code.upper()}_historical_"
    for file in os.listdir(folder):
        if file.startswith(file_prefix) and file.endswith(".csv"):
            hist_csv = os.path.join(folder, file)
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
    stock = "RTNINDIA"

    # You can now uncomment and use the functions
    df_candles = fetch_candles(stock, n=2)
    if df_candles is not None:
        print(f"[🕒] Redis Candles columns for {stock}: {list(df_candles.columns)}")

    df_indicators = fetch_indicators(stock, n=2)
    if df_indicators is not None:
        print(
            f"[📈] Redis Indicators columns for {stock}: {list(df_indicators.columns)}",
            df_indicators.head(1),
        )

    # df_hist = fetch_historical(stock)
    # if df_hist is not None:
    #     print(f"[📚] Historical CSV columns for {stock}: {list(df_hist.columns)}")
