import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, KiteException
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = "config30a.json"
SAVE_FOLDER = "historical_data"
INTERVAL = "day"

# ===================== helpers =====================


def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder smoothing."""
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def _adx_wilder(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """+DI, -DI, ADX with Wilder smoothing."""
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    tr_ema = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_ema = (
        pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    )
    minus_dm_ema = (
        pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    )

    plus_di = 100.0 * (plus_dm_ema / tr_ema.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_ema / tr_ema.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out["+DI"] = plus_di
    out["-DI"] = minus_di
    out["ADX"] = adx
    return out


def _rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling VWAP for daily bars."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"].clip(lower=0)
    num = pv.rolling(window=window, min_periods=1).sum()
    den = df["Volume"].rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return num / den


# ===================== indicators =====================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Moving Averages
    df["MA_Fast"] = df["Close"].rolling(window=9, min_periods=1).mean()
    df["MA_Slow"] = df["Close"].rolling(window=20, min_periods=1).mean()

    # Bollinger Bands
    bb_period = 20
    bb_mid = df["Close"].rolling(window=bb_period, min_periods=1).mean()
    bb_std = df["Close"].rolling(window=bb_period, min_periods=1).std(ddof=0)
    df["BB_Mid"] = bb_mid
    df["BB_Upper"] = bb_mid + (bb_std * 2.0)
    df["BB_Lower"] = bb_mid - (bb_std * 2.0)

    # MACD
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ADX
    adx_df = _adx_wilder(df, period=14)
    df["+DI"] = adx_df["+DI"]
    df["-DI"] = adx_df["-DI"]
    df["ADX"] = adx_df["ADX"]

    # HH20/LL20
    lookback = 20
    df["HH20"] = df["High"].rolling(window=lookback, min_periods=1).max()
    df["LL20"] = df["Low"].rolling(window=lookback, min_periods=1).min()

    # dist_hh20_bps
    df["dist_hh20_bps"] = (
        (df["HH20"] - df["Close"]) / df["Close"].replace(0, np.nan)
    ) * 10000.0

    # bb_width_bps
    df["bb_width_bps"] = (
        (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"].replace(0, np.nan)
    ) * 10000.0

    # bb_squeeze
    roll_min_width = df["bb_width_bps"].rolling(window=lookback, min_periods=1).min()
    df["bb_squeeze"] = (df["bb_width_bps"] < (roll_min_width * 1.10)).astype(int)

    # EMA slopes
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema20_slope_bps"] = ema20.pct_change() * 10000.0
    df["ema50_slope_bps"] = ema50.pct_change() * 10000.0

    # adx14 alias
    df["adx14"] = df["ADX"]

    # macd_hist_delta
    df["macd_hist_delta"] = df["MACD_Hist"].diff()

    # VWAP
    df["VWAP"] = _rolling_vwap(df, window=20)
    df["vwap_diff_bps"] = (
        (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)
    ) * 10000.0

    # ATR14 + atr_pct
    df["ATR14"] = _atr_wilder(df, period=14)
    df["atr_pct"] = (df["ATR14"] / df["Close"].replace(0, np.nan)) * 100.0

    # vol_z
    vol_roll = df["Volume"].rolling(window=20, min_periods=5)
    df["vol_z"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std(ddof=0)

    # ✅ RSI (14-period Wilder’s)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # ✅ Stochastic Oscillator (14,3,3)
    low14 = df["Low"].rolling(window=14, min_periods=1).min()
    high14 = df["High"].rolling(window=14, min_periods=1).max()
    df["StochK"] = ((df["Close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100
    df["StochD"] = df["StochK"].rolling(window=3, min_periods=1).mean()

    # ✅ CCI (Commodity Channel Index, 20-period)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma_tp = tp.rolling(window=20, min_periods=1).mean()
    md = (tp - ma_tp).abs().rolling(window=20, min_periods=1).mean()
    df["CCI20"] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

    return df


# ===================== KiteConnect =====================


def make_kite():
    API_KEY = os.getenv("KITE_API_KEY")
    ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing creds: set KITE_API_KEY + KITE_ACCESS_TOKEN")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


def fetch_historical_from_config(config_path: str, interval: str = "day"):
    kite = make_kite()
    today = datetime.today()
    to_date = today - timedelta(days=1)
    from_date = to_date.replace(year=to_date.year - 1)

    print(f"📅 Fetching data from {from_date.date()} to {to_date.date()} [{interval}]")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {config_path}: {e}")
        return

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for stock in cfg.get("stocks", []):
        stock_code = stock.get("stock_code")
        token = stock.get("instrument_token")
        if not stock_code or not token:
            print(f"⚠️ Skipping invalid entry: {stock}")
            continue

        print(f"📥 Fetching {stock_code} ({token})...")
        try:
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d"),
                interval=interval,
                continuous=False,
                oi=False,
            )
        except TokenException as te:
            print(f"❌ Auth error for {stock_code}: {te}")
            continue
        except KiteException as ke:
            print(f"❌ Kite error for {stock_code}: {ke}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error for {stock_code}: {e}")
            continue

        if not candles:
            print(f"⚠️ No data returned for {stock_code}")
            continue

        df = pd.DataFrame(candles)

        # 🔹 Rename to match Redis naming
        rename_map = {
            "date": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns=rename_map, inplace=True)

        df = calculate_indicators(df)
        df = df.round(4)

        out = os.path.join(
            SAVE_FOLDER,
            f"{stock_code}_historical_{from_date.date()}_to_{to_date.date()}.csv",
        )
        df.to_csv(out, index=False)
        print(f"✅ Saved {len(df)} candles with indicators → {out}")


if __name__ == "__main__":
    fetch_historical_from_config(CONFIG_PATH, INTERVAL)
