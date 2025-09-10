# dataset_checker.py
import os
import json
import pandas as pd

# Import your Redis + helpers
from redis_utils import get_redis, get_recent_indicators_tf

# from your_module import load_candles  # adjust import if needed

# Define your TF plan (same as in your main code)
TF_PLAN = [
    ("1min", 0, None),
    ("5min", 1, None),
    ("15min", 2, None),
    ("30min", 3, None),
    ("45min", 4, None),
    ("1hour", 5, None),
    ("4hour", 6, None),
    ("1day", 7, None),
    ("1month", 8, None),
]

BASE_DIR = os.path.join(os.getcwd(), "historical_data_candles")


def load_candles(
    stock_code: str,
    tf: str,
    n: int = 5,
    base_dir: str = "historical_data_candles",
    source: str = "redis",
) -> pd.DataFrame:
    """
    Load last N candles for a stock and timeframe.
    - source="redis": get from Redis (via get_recent_indicators_tf)
    - source="disk": get from historical_data_candles/<tf>/ CSVs
    Returns normalized DataFrame with at least:
      Timestamp, Open, High, Low, Close, Volume, + any indicators.
    """
    import glob

    if source == "redis":
        try:
            rows = get_recent_indicators_tf(get_redis(), stock_code, tf, n=n)
            if rows:
                df = pd.DataFrame(rows)
                if "Timestamp" not in df.columns:
                    for c in ("minute", "timestamp", "datetime"):
                        if c in df.columns:
                            df["Timestamp"] = pd.to_datetime(df[c], errors="coerce")
                            break
                return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").tail(n)
        except Exception:
            return pd.DataFrame()

    if source == "disk":
        tf_dir = os.path.join(base_dir, tf)
        if os.path.isdir(tf_dir):
            patt = os.path.join(tf_dir, f"{stock_code}_{tf}_*.csv")
            files = glob.glob(patt)
            if files:
                latest_path = max(files, key=os.path.getmtime)
                try:
                    df = pd.read_csv(latest_path)
                    if "Timestamp" not in df.columns:
                        for c in ("minute", "timestamp", "datetime"):
                            if c in df.columns:
                                df = df.rename(columns={c: "Timestamp"})
                                break
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                    return (
                        df.dropna(subset=["Timestamp"]).sort_values("Timestamp").tail(n)
                    )
                except Exception:
                    return pd.DataFrame()

    return pd.DataFrame()


def check_datasets(stock_code: str, n: int = 5):
    r = get_redis()
    results = {}

    for tf, idx, _ in TF_PLAN:
        print(f"\n=== {stock_code} | {tf} ===")

        # Redis data
        df_r = pd.DataFrame()
        if tf in ["1min", "5min", "15min", "30min", "45min", "1hour", "4hour"]:
            try:
                rows = get_recent_indicators_tf(r, stock_code, tf, n=n)
                if rows:
                    df_r = pd.DataFrame(rows)
            except Exception as e:
                print(f"Redis fetch error: {e}")

        if not df_r.empty:
            print(f"Redis {tf} -> {len(df_r)} rows, columns: {list(df_r.columns)}")
        else:
            print(f"Redis {tf} -> NO DATA")

        # Disk data
        df_h = load_candles(stock_code, tf, n=n, base_dir=BASE_DIR, source="disk")
        if not df_h.empty:
            print(f"Disk {tf} -> {len(df_h)} rows, columns: {list(df_h.columns)}")
        else:
            print(f"Disk {tf} -> NO DATA")

        results[tf] = {
            "redis_cols": list(df_r.columns) if not df_r.empty else [],
            "disk_cols": list(df_h.columns) if not df_h.empty else [],
        }

    return results


if __name__ == "__main__":
    # Pick a stock you know exists in Redis + Disk
    stock_code = "CDSL"  # change as needed
    res = check_datasets(stock_code, n=5)

    print(json.dumps(res, indent=2))

    # Save to JSON for easier inspection if needed
    # with open("dataset_columns.json", "w") as f:
    #     json.dump(res, f, indent=2)
    # print("\n[✔] Column info saved to dataset_columns.json")
