import os
import shutil
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, KiteException
from dotenv import load_dotenv
import pytz

load_dotenv()

# ===================== CONFIG =====================
CONFIG_PATH = "config30a.json"
CANDLES_ROOT = "historical_data_candles"
SUBFOLDERS = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1hour": "1hour",
    "4hour": "4hour",
    "1day": "1day",
    "1month": "1month",
}

# Base fetch choices
BASE_5M_INTERVAL = "5minute"  # for 1y coverage and all frames >= 5m
BASE_1M_INTERVAL = "minute"  # 1min (we'll fetch only last 7 days)

INDIA_TZ = pytz.timezone("Asia/Kolkata")


# ===================== helpers =====================
def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = (df["High"] - df["Low"]).abs()
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx_wilder(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
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
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = tp * df["Volume"].clip(lower=0)
    num = pv.rolling(window=window, min_periods=1).sum()
    den = df["Volume"].rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return num / den


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # MAs
    df["MA_Fast"] = df["Close"].rolling(window=9, min_periods=1).mean()
    df["MA_Slow"] = df["Close"].rolling(window=20, min_periods=1).mean()

    # BB
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

    # widths / distances
    df["dist_hh20_bps"] = (
        (df["HH20"] - df["Close"]) / df["Close"].replace(0, np.nan)
    ) * 10000.0
    df["bb_width_bps"] = (
        (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"].replace(0, np.nan)
    ) * 10000.0
    roll_min_width = df["bb_width_bps"].rolling(window=lookback, min_periods=1).min()
    df["bb_squeeze"] = (df["bb_width_bps"] < (roll_min_width * 1.10)).astype(int)

    # EMA slopes
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema20_slope_bps"] = ema20.pct_change() * 10000.0
    df["ema50_slope_bps"] = ema50.pct_change() * 10000.0

    # alias & delta
    df["adx14"] = df["ADX"]
    df["macd_hist_delta"] = df["MACD_Hist"].diff()

    # VWAP
    df["VWAP"] = _rolling_vwap(df, window=20)
    df["vwap_diff_bps"] = (
        (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)
    ) * 10000.0

    # ATR & pct
    df["ATR14"] = _atr_wilder(df, period=14)
    df["atr_pct"] = (df["ATR14"] / df["Close"].replace(0, np.nan)) * 100.0

    # Vol z-score
    vol_roll = df["Volume"].rolling(window=20, min_periods=5)
    df["vol_z"] = (df["Volume"] - vol_roll.mean()) / vol_roll.std(ddof=0)

    # RSI 14
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # Stoch 14,3
    low14 = df["Low"].rolling(window=14, min_periods=1).min()
    high14 = df["High"].rolling(window=14, min_periods=1).max()
    df["StochK"] = ((df["Close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100
    df["StochD"] = df["StochK"].rolling(window=3, min_periods=1).mean()

    # CCI 20
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma_tp = tp.rolling(window=20, min_periods=1).mean()
    md = (tp - ma_tp).abs().rolling(window=20, min_periods=1).mean()
    df["CCI20"] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

    return df


def make_kite():
    API_KEY = os.getenv("KITE_API_KEY")
    ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
    if not API_KEY or not ACCESS_TOKEN:
        raise RuntimeError("Missing creds: set KITE_API_KEY + KITE_ACCESS_TOKEN")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite


def _wipe_and_prepare_dirs():
    os.makedirs(CANDLES_ROOT, exist_ok=True)
    # Remove all existing subfolders inside root
    for name in os.listdir(CANDLES_ROOT):
        p = os.path.join(CANDLES_ROOT, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    # Re-create fresh subfolders
    for sub in SUBFOLDERS.values():
        os.makedirs(os.path.join(CANDLES_ROOT, sub), exist_ok=True)


def _chunked_ranges_for_minute_data(
    from_date: datetime, to_date: datetime, chunk_days: int = 60
):
    start = from_date
    while start <= to_date:
        end = min(start + timedelta(days=chunk_days - 1), to_date)
        yield (start, end)
        start = end + timedelta(days=1)


def _kite_fetch_hist(
    kite, token: int, from_date: datetime, to_date: datetime, interval: str
):
    all_rows = []
    if interval in (
        "minute",
        "3minute",
        "5minute",
        "10minute",
        "15minute",
        "30minute",
        "60minute",
    ):
        for s, e in _chunked_ranges_for_minute_data(from_date, to_date, chunk_days=60):
            candles = kite.historical_data(
                instrument_token=token,
                from_date=s.strftime("%Y-%m-%d"),
                to_date=e.strftime("%Y-%m-%d"),
                interval=interval,
                continuous=False,
                oi=False,
            )
            all_rows.extend(candles)
    else:
        candles = kite.historical_data(
            instrument_token=token,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval=interval,
            continuous=False,
            oi=False,
        )
        all_rows.extend(candles)
    return all_rows


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date": "Timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert(INDIA_TZ)
    df = (
        df.sort_values("Timestamp")
        .drop_duplicates(subset=["Timestamp"])
        .reset_index(drop=True)
    )
    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = df.set_index("Timestamp")
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = (
        df.resample(rule, label="right", closed="right").agg(agg).dropna().reset_index()
    )
    return out


def _save_df(
    df: pd.DataFrame,
    subfolder_key: str,
    stock_code: str,
    suffix: str,
    from_date: datetime,
    to_date: datetime,
):
    path = os.path.join(CANDLES_ROOT, SUBFOLDERS[subfolder_key])
    fname = f"{stock_code}_{suffix}_{from_date.date()}_to_{to_date.date()}.csv"
    out_path = os.path.join(path, fname)
    df.round(4).to_csv(out_path, index=False)
    print(f"✅ Saved {len(df)} rows → {out_path}")


def _derive_and_save_from_5m(
    df_5m_raw: pd.DataFrame, stock_code: str, from_date: datetime, to_date: datetime
):
    # 5m indicators & save
    df_5m = calculate_indicators(df_5m_raw)
    _save_df(df_5m, "5min", stock_code, "5min", from_date, to_date)

    # Build higher frames from 5m
    frames = {
        "15min": "15T",
        "30min": "30T",
        "45min": "45T",
        "1hour": "1H",
        "4hour": "4H",
        "1day": "1D",
        "1month": "M",
    }
    for key, rule in frames.items():
        df_tf = _resample_ohlcv(df_5m_raw, rule)
        df_tf = calculate_indicators(df_tf)
        _save_df(df_tf, key, stock_code, key, from_date, to_date)


def _fetch_and_save_1min(kite, token: int, stock_code: str, to_date_ist: datetime):
    """Fetch last 7 days of 1-minute data and save with indicators."""
    from_date_1m = (to_date_ist - timedelta(days=7)).astimezone(INDIA_TZ)
    rows_1m = _kite_fetch_hist(kite, token, from_date_1m, to_date_ist, BASE_1M_INTERVAL)
    if not rows_1m:
        print(f"⚠️ No 1-min data returned for {stock_code}")
        return
    df_1m = _standardize_df(pd.DataFrame(rows_1m))
    df_1m = calculate_indicators(df_1m)
    _save_df(df_1m, "1min", stock_code, "1min", from_date_1m, to_date_ist)


# ===================== main =====================
def fetch_all_frames(config_path: str):
    kite = make_kite()
    today = datetime.now(INDIA_TZ)
    to_date = (today - timedelta(days=1)).astimezone(INDIA_TZ)
    from_date_5m = (to_date - timedelta(days=365)).astimezone(INDIA_TZ)

    print(f"🧹 Resetting output folders under {CANDLES_ROOT} …")
    _wipe_and_prepare_dirs()

    print(f"📅 5m base range: {from_date_5m.date()} → {to_date.date()} (IST)")
    print(f"⏱ 1m base range: last 7 days ending {to_date.date()} (IST)")

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {config_path}: {e}")
        return

    for stock in cfg.get("stocks", []):
        stock_code = stock.get("stock_code")
        token = stock.get("instrument_token")
        if not stock_code or not token:
            print(f"⚠️ Skipping invalid entry: {stock}")
            continue

        print(f"\n📥 {stock_code} ({token}) — fetching 5m (1y)…")
        try:
            rows_5m = _kite_fetch_hist(
                kite, token, from_date_5m, to_date, BASE_5M_INTERVAL
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

        if not rows_5m:
            print(f"⚠️ No 5m data returned for {stock_code}")
            # Still try 1m (7d) so at least something is saved
            _fetch_and_save_1min(kite, token, stock_code, to_date)
            continue

        df_5m = _standardize_df(pd.DataFrame(rows_5m))

        # Save all frames built from 5m (includes 5m itself)
        _derive_and_save_from_5m(df_5m, stock_code, from_date_5m, to_date)

        # Also fetch & save 1-minute (last 7d only)
        _fetch_and_save_1min(kite, token, stock_code, to_date)


if __name__ == "__main__":
    fetch_all_frames(CONFIG_PATH)
