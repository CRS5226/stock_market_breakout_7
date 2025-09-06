# basic_algo_forecaster.py
import os
import copy
import math
import pandas as pd
from datetime import datetime

# -------------------- utils --------------------


def safe_number(val, default=0):
    """Replace NaN/None/inf with a safe default number."""
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            if math.isnan(val) or val in (float("inf"), float("-inf")):
                return default
        return val
    except Exception:
        return default


def _get(df: pd.DataFrame, *candidates, default=None):
    """Fetch a column by trying multiple candidate names (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name is None:
            continue
        key = str(name).lower()
        if key in cols:
            return df[cols[key]]
    return pd.Series([default] * len(df), index=df.index)


def _last_val(row, *candidates, default=None):
    for name in candidates:
        if name in row and pd.notna(row[name]):
            return row[name]
        # also try case-insensitive
    for name in candidates:
        lname = str(name).lower()
        for c in row.index:
            if c.lower() == lname and pd.notna(row[c]):
                return row[c]
    return default


# -------------------- support / resistance --------------------


def compute_support_resistance(df: pd.DataFrame):
    """
    Compute support/resistance with preference:
    1) HH20/LL20 if present
    2) swing (last 20 highs/lows)
    3) pivot/BB fallback
    """
    # standardize core series (try both cases)
    high_s = _get(df, "High", "high")
    low_s = _get(df, "Low", "low")
    close_s = _get(df, "Close", "close")
    bb_upper_s = _get(df, "BB_Upper", "bb_upper")
    bb_lower_s = _get(df, "BB_Lower", "bb_lower")

    # latest
    latest = df.iloc[-1]
    high = _last_val(latest, "High", "high", default=None)
    low = _last_val(latest, "Low", "low", default=None)
    close = _last_val(latest, "Close", "close", default=None)

    # HH/LL preferred
    hh20_s = _get(df, "HH20", "hh20")
    ll20_s = _get(df, "LL20", "ll20")

    swing_high = safe_number(
        hh20_s.iloc[-1] if pd.notna(hh20_s.iloc[-1]) else high_s.tail(20).max(),
        default=None,
    )
    swing_low = safe_number(
        ll20_s.iloc[-1] if pd.notna(ll20_s.iloc[-1]) else low_s.tail(20).min(),
        default=None,
    )

    # pivot
    if all(v is not None for v in (high, low, close)):
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
    else:
        r1 = swing_high
        s1 = swing_low

    bb_upper = safe_number(bb_upper_s.iloc[-1], default=r1)
    bb_lower = safe_number(bb_lower_s.iloc[-1], default=s1)

    # final S/R preference: max of (r1, bb_upper, swing_high), min of (s1, bb_lower, swing_low)
    res = max([v for v in [r1, bb_upper, swing_high] if v is not None])
    sup = min([v for v in [s1, bb_lower, swing_low] if v is not None])

    return round(sup, 2), round(res, 2), round(bb_upper, 2), round(bb_lower, 2)


# -------------------- signal logic --------------------


def determine_signal(
    latest: pd.Series, support: float, resistance: float, volume_threshold: float
):
    """
    Upgraded rule set using new features when available.
    Returns (signal, reasons[])
    """
    # core values (case-insensitive)
    close = _last_val(latest, "Close", "close", default=None)
    volume = _last_val(latest, "Volume", "volume", default=0)

    # existing simple features (optional)
    rsi = _last_val(latest, "RSI", "rsi", default=None)
    ema_fast = _last_val(latest, "EMA_9", "ema_9", "MA_Fast", "ma_fast", default=None)
    ema_slow = _last_val(latest, "EMA_21", "ema_21", "MA_Slow", "ma_slow", default=None)

    # new features (all optional)
    dist_hh20_bps = _last_val(latest, "dist_hh20_bps", default=None)
    bb_width_bps = _last_val(latest, "bb_width_bps", default=None)
    bb_squeeze = _last_val(latest, "bb_squeeze", default=0)
    ema20_slope = _last_val(latest, "ema20_slope_bps", default=None)
    ema50_slope = _last_val(latest, "ema50_slope_bps", default=None)
    adx14 = _last_val(latest, "adx14", "ADX", "adx", default=None)
    macd_hist_delta = _last_val(latest, "macd_hist_delta", default=None)
    vwap_diff_bps = _last_val(latest, "vwap_diff_bps", default=None)
    atr_pct = _last_val(latest, "atr_pct", default=None)
    vol_z = _last_val(latest, "vol_z", default=None)
    bb_upper = _last_val(latest, "BB_Upper", "bb_upper", default=None)
    bb_lower = _last_val(latest, "BB_Lower", "bb_lower", default=None)

    # thresholds (tune as needed)
    NEAR_BARRIER_BPS = 15  # within 15 bps of S/R counts as "near"
    STRONG_ADX = 20
    VOL_SPIKE_Z = 1.5
    SLOPE_OK = 2.0  # ema slope in bps
    MACD_DELTA_OK = 0.0  # histogram rising if > 0
    ATR_MIN = 0.25  # min movement potential (% of price)
    VWAP_NEAR_BPS = 15  # within 15 bps of VWAP is "near fair value"

    reasons = []
    signal = "No Action"

    if close is None or support is None or resistance is None:
        return signal, ["Missing core fields"]

    # helpers
    near_resistance = abs(close - resistance) / resistance * 10000.0 < NEAR_BARRIER_BPS
    near_support = abs(close - support) / support * 10000.0 < NEAR_BARRIER_BPS
    volume_spike = (
        volume is not None
        and volume_threshold is not None
        and volume > volume_threshold
    )
    adx_good = (adx14 is not None) and (adx14 >= STRONG_ADX)
    slopes_bull = ((ema20_slope or 0) > SLOPE_OK) and ((ema50_slope or 0) > 0)
    slopes_bear = ((ema20_slope or 0) < -SLOPE_OK) and ((ema50_slope or 0) < 0)
    macd_up = (macd_hist_delta is not None) and (macd_hist_delta > MACD_DELTA_OK)
    macd_down = (macd_hist_delta is not None) and (macd_hist_delta < -MACD_DELTA_OK)
    atr_ok = (atr_pct is not None) and (atr_pct >= ATR_MIN)
    vwap_ok_long = (vwap_diff_bps is None) or (
        -VWAP_NEAR_BPS <= vwap_diff_bps <= VWAP_NEAR_BPS or vwap_diff_bps > 0
    )
    vwap_ok_short = (vwap_diff_bps is None) or (
        -VWAP_NEAR_BPS <= vwap_diff_bps <= VWAP_NEAR_BPS or vwap_diff_bps < 0
    )
    bb_conf_long = bb_upper is not None and close > bb_upper
    bb_conf_short = bb_lower is not None and close < bb_lower

    # --- Long / breakout bias ---
    if near_resistance and volume_spike:
        reasons.append(
            f"Price near resistance {round(resistance,2)} with volume spike."
        )
        if adx_good:
            reasons.append(f"ADX {round(adx14,1)} strong.")
        if slopes_bull:
            reasons.append("EMA20/50 slopes positive.")
        if macd_up:
            reasons.append("MACD histogram rising.")
        if bb_squeeze:
            reasons.append("BB squeeze → potential expansion.")
        if bb_conf_long:
            reasons.append("Price above upper BB.")
        if atr_ok:
            reasons.append(f"ATR {round(atr_pct,2)}% supports move.")
        if vwap_ok_long:
            reasons.append("VWAP alignment OK.")

        strong_checks = sum([adx_good, slopes_bull, macd_up, bb_conf_long, atr_ok])
        if strong_checks >= 3:
            signal = "Strong Breakout"
        else:
            signal = "Potential Breakout"

    # --- Short / breakdown bias ---
    elif near_support and volume_spike:
        reasons.append(f"Price near support {round(support,2)} with volume spike.")
        if adx_good:
            reasons.append(f"ADX {round(adx14,1)} strong.")
        if slopes_bear:
            reasons.append("EMA20/50 slopes negative.")
        if macd_down:
            reasons.append("MACD histogram falling.")
        if bb_squeeze:
            reasons.append("BB squeeze → potential expansion.")
        if bb_conf_short:
            reasons.append("Price below lower BB.")
        if atr_ok:
            reasons.append(f"ATR {round(atr_pct,2)}% supports move.")
        if vwap_ok_short:
            reasons.append("VWAP alignment OK.")

        strong_checks = sum([adx_good, slopes_bear, macd_down, bb_conf_short, atr_ok])
        if strong_checks >= 3:
            signal = "Strong Breakdown"
        else:
            signal = "Potential Breakdown"

    return signal, reasons


# -------------------- main update --------------------


def basic_forecast_update(stock_cfg, recent_df, historical_folder="historical_data"):
    """
    Basic algo forecaster using Redis indicators + historical data.
    Avoids updating config if only last_updated would change.
    """

    stock_cfg = copy.deepcopy(stock_cfg)
    stock_code = stock_cfg.get("stock_code")

    # --- Load historical ---
    df_hist = pd.DataFrame()
    hist_csv = None
    if os.path.isdir(historical_folder):
        for file in os.listdir(historical_folder):
            if file.startswith(f"{stock_code}_historical_") and file.endswith(".csv"):
                hist_csv = os.path.join(historical_folder, file)
                break

    if hist_csv and os.path.exists(hist_csv):
        try:
            df_hist = pd.read_csv(hist_csv).tail(10).reset_index(drop=True)
            if "Timestamp" in df_hist.columns:
                df_hist = df_hist.drop_duplicates(subset=["Timestamp"])
        except Exception:
            df_hist = pd.DataFrame()

    # --- Recent (from Redis) ---
    df_recent = (
        recent_df.copy()
        if recent_df is not None and not recent_df.empty
        else pd.DataFrame()
    )
    if not df_recent.empty:
        df_recent = df_recent.reset_index(drop=True)
        if "Timestamp" in df_recent.columns:
            df_recent = df_recent.drop_duplicates(subset=["Timestamp"])

    # --- No data case ---
    if df_recent.empty:
        return {
            **stock_cfg,
            "forecast": "basic_algo",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # --- Merge for SR ---
    df_all = (
        pd.concat([df_hist, df_recent], ignore_index=True)
        if not df_hist.empty
        else df_recent.copy()
    )

    # --- Compute support/resistance + threshold ---
    support, resistance, bb_upper, bb_lower = compute_support_resistance(df_all)
    vol_series = _get(df_recent, "Volume", "volume", default=0)
    volume_threshold = (
        float(vol_series.tail(50).mean() * 1.5) if len(vol_series) else 0.0
    )

    # --- Decision from latest row ---
    latest = df_recent.iloc[-1]
    signal, reasons = determine_signal(latest, support, resistance, volume_threshold)

    # --- New forecast dict ---
    new_cfg = {
        "stock_code": stock_cfg.get("stock_code"),
        "instrument_token": stock_cfg.get("instrument_token"),
        "support": safe_number(support),
        "resistance": safe_number(resistance),
        "volume_threshold": int(safe_number(volume_threshold)),
        "bollinger": {
            "mid_price": safe_number(round((support + resistance) / 2, 2)),
            "upper_band": safe_number(bb_upper),
            "lower_band": safe_number(bb_lower),
        },
        "macd": stock_cfg.get("macd", {}),
        "adx": stock_cfg.get("adx", {}),
        "moving_averages": stock_cfg.get("moving_averages", {}),
        "inside_bar": stock_cfg.get("inside_bar", {}),
        "candle": stock_cfg.get("candle", {}),
        "reason": reasons,
        "signal": signal,
        "forecast": "basic_algo",
    }

    # --- Compare with old config (ignore last_updated) ---
    compare_keys = [
        "support",
        "resistance",
        "volume_threshold",
        "bollinger",
        "signal",
        "reason",
    ]
    changed = any(new_cfg.get(k) != stock_cfg.get(k) for k in compare_keys)

    if not changed:
        # Only refresh timestamp if no meaningful change
        return {
            **stock_cfg,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        return {**new_cfg, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
