# basic_algo_forecaster.py
import os
import copy
import math
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Optional, Tuple
from redis_utils import get_recent_candles, get_redis


# -------------------- utils --------------------
def safe_str(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)


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


def enrich_with_targets(
    df: pd.DataFrame,
    time_col: str = "Timestamp",
    close_col: str = "Close",
    horizons: dict | None = None,
    use_business_days: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    # 🔧 normalize to IST, tz-aware
    out[time_col] = to_ist(out[time_col], keep_tz=True)
    out = out.sort_values(time_col).reset_index(drop=True)

    if horizons is None:
        horizons = (
            {"1d": 1, "1w": 5, "3m": 63}
            if use_business_days
            else {"1d": "1D", "1w": "7D", "3m": "90D"}
        )

    ref = out[[time_col, close_col]].rename(
        columns={time_col: "_ref_time", close_col: "_ref_close"}
    )
    ref = ref.sort_values("_ref_time")

    for key, step in horizons.items():
        target_times = out[[time_col]].copy()
        if use_business_days:
            target_times[f"target_time_{key}"] = out[time_col] + pd.offsets.BDay(
                int(step)
            )
        else:
            target_times[f"target_time_{key}"] = out[time_col] + pd.to_timedelta(step)

        merged = pd.merge_asof(
            target_times.sort_values(f"target_time_{key}"),
            ref,
            left_on=f"target_time_{key}",
            right_on="_ref_time",
            direction="forward",
        ).sort_index()

        out[f"target_close_{key}"] = merged["_ref_close"].values
        out[f"target_ret_{key}"] = (out[f"target_close_{key}"] / out[close_col]) - 1

    return out


def _sr_width_pct(
    sup: Optional[float], res: Optional[float], base_close: Optional[float]
) -> Optional[float]:
    try:
        if sup is None or res is None or base_close is None or base_close <= 0:
            return None
        return round(((res - sup) / sup) * 100.0, 4)
    except Exception:
        return None


def _donchian_sr_from_daily(
    df_daily: pd.DataFrame, lookback: int
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (resistance, support) using High/Low over the last `lookback` rows of a *daily* dataframe.
    Expects columns 'High','Low' (case-sensitive, so coerce if needed).
    """
    if df_daily is None or df_daily.empty:
        return None, None
    cols = {c.lower(): c for c in df_daily.columns}
    H = cols.get("high")
    L = cols.get("low")
    if not H or not L:
        return None, None
    tail = df_daily.tail(max(1, lookback))
    try:
        res = float(tail[H].max())
        sup = float(tail[L].min())
        return res, sup
    except Exception:
        return None, None


# ======================= SANITY HELPERS (NEW) =======================


def _clamp(x, lo=None, hi=None):
    if x is None:
        return None
    if (lo is not None) and (x < lo):
        return lo
    if (hi is not None) and (x > hi):
        return hi
    return x


def _round_opt(x, nd=3):
    try:
        return round(float(x), nd)
    except Exception:
        return None


def _ensure_sr_order(support, resistance):
    """Swap if support > resistance; return (support, resistance)."""
    if support is None or resistance is None:
        return support, resistance
    if support > resistance:
        return resistance, support
    return support, resistance


def _sanitize_levels(
    entry, target, stoploss, side, support, resistance, buf_frac, atr_frac
):
    tiny = max(atr_frac, 0.0008)  # ≥ 8 bps

    if side == "long":
        # Clamp entry/target inside band
        if resistance is not None:
            target = _clamp(target, hi=resistance * (1.0 - buf_frac))
            if entry is not None:
                entry = _clamp(entry, hi=resistance * (1.0 - buf_frac))
        if support is not None:
            entry = _clamp(entry, lo=support * (1.0 + buf_frac))
            # >>> enforce stop strictly below support, but not too far
            lo_stop = support * (1.0 - 1.25 * atr_frac)  # deepest allowed (ATR-bounded)
            hi_stop = support * (1.0 - tiny)  # just below support
            stoploss = _clamp(
                stoploss if stoploss is not None else lo_stop, lo=lo_stop, hi=hi_stop
            )

        # Ensure ordering: stop < entry < target
        if entry is not None and stoploss is not None and stoploss >= entry:
            # keep stop below entry and below support cap if we had it
            stoploss = entry * (1.0 - tiny)
        if entry is not None and target is not None and target <= entry:
            target = entry * (1.0 + tiny)

    else:  # short
        # Clamp entry/target inside band
        if support is not None:
            target = _clamp(target, lo=support * (1.0 + buf_frac))
            if entry is not None:
                entry = _clamp(entry, lo=support * (1.0 + buf_frac))
        if resistance is not None:
            entry = _clamp(entry, hi=resistance * (1.0 - buf_frac))
            # >>> enforce stop strictly above resistance, but not too far
            lo_stop = resistance * (1.0 + tiny)  # just above resistance
            hi_stop = resistance * (
                1.0 + 1.25 * atr_frac
            )  # highest allowed (ATR-bounded)
            stoploss = _clamp(
                stoploss if stoploss is not None else hi_stop, lo=lo_stop, hi=hi_stop
            )

        # Ensure ordering: target < entry < stop
        if entry is not None and stoploss is not None and stoploss <= entry:
            stoploss = entry * (1.0 + tiny)
        if entry is not None and target is not None and target >= entry:
            target = entry * (1.0 - tiny)

    return _round_opt(entry), _round_opt(target), _round_opt(stoploss)


# def _decide_side_and_mode(
#     signal, trending, bias_up, current_price, support, resistance, mode
# ):
#     """
#     Returns (side, eff_mode) where side in {'long','short'} and eff_mode in {'range','breakout','breakdown'}.
#     - Default eff_mode is whatever you pass; if it's 'auto', we infer from signal/trend/bias.
#     - We also pick side.
#     """
#     text = (signal or "").lower()
#     eff_mode = (mode or "range").lower()

#     # Side decision
#     if "breakdown" in text:
#         side = "short"
#         eff_mode = "breakdown"
#     elif "breakout" in text:
#         side = "long"
#         eff_mode = "breakout"
#     else:
#         # Range mode or bias-based
#         if eff_mode == "range":
#             side = "long" if bias_up else "short"
#         elif eff_mode == "breakout":
#             side = "long"
#         elif eff_mode == "breakdown":
#             side = "short"
#         else:
#             # auto fallback
#             if trending and bias_up:
#                 side = "long"
#                 eff_mode = "breakout"
#             elif trending and not bias_up:
#                 side = "short"
#                 eff_mode = "breakdown"
#             else:
#                 side = "long" if bias_up else "short"
#                 eff_mode = "range"
#     # If support/resistance missing, fall back to range
#     if support is None or resistance is None:
#         eff_mode = "range"
#     return side, eff_mode


def _decide_side_and_mode(
    signal, trending, bias_up, current_price, support, resistance, mode
):
    """
    LONG-ONLY: ignore signal/bias for side selection.
    Always trade the range inside [support, resistance].
    """
    return "long", "range"


# ---- Timezone helpers ----
LOCAL_TZ = "Asia/Kolkata"  # IST


def to_ist(series: pd.Series, keep_tz=True) -> pd.Series:
    """
    Normalize a datetime-like Series to IST.
    - If values are tz-aware (UTC or anything), convert to IST.
    - If values are tz-naive, *assume they are IST* and localize.
    - keep_tz=True => return tz-aware IST
      keep_tz=False => return tz-naive timestamps that *represent IST wall time*.
    """
    s = pd.to_datetime(series, errors="coerce")  # don't force utc here

    # If any tz-aware values exist, convert them all to IST
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(LOCAL_TZ)
    else:
        # Naive -> interpret as IST local time and localize
        s = s.dt.tz_localize(LOCAL_TZ)

    if not keep_tz:
        # Drop tz info but keep IST wall time
        s = s.dt.tz_localize(None)
    return s


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


def horizon_sr_from_daily(
    df_daily: pd.DataFrame, lookback: int, use_quantile: bool = False, q: float = 0.05
) -> Tuple[Optional[float], Optional[float]]:
    """
    Composite S/R over a daily window:
      - swing = max(High), min(Low) in window (or quantiles to reduce spikes)
      - pivot R1/S1 from the *last day* in the window (if Close available)
      - optional BB bounds if present in df (BB_Upper / BB_Lower)
      - final resistance = max(R1, swing_high, bb_upper_window_max)
        final support    = min(S1, swing_low,  bb_lower_window_min)
    """
    if df_daily is None or df_daily.empty:
        return None, None

    cols = {c.lower(): c for c in df_daily.columns}
    H = cols.get("high")
    L = cols.get("low")
    C = cols.get("close")
    U = cols.get("bb_upper")
    D = cols.get("bb_lower")

    tail = df_daily.tail(max(1, lookback)).copy()
    if H is None or L is None:
        return None, None

    # swing extremes (optionally robust to outliers via quantiles)
    if use_quantile:
        swing_high = float(tail[H].quantile(1.0 - q))
        swing_low = float(tail[L].quantile(q))
    else:
        swing_high = float(tail[H].max())
        swing_low = float(tail[L].min())

    # pivot from the last day in window if Close is available
    r1 = swing_high
    s1 = swing_low  # fallback
    if C is not None and tail[C].notna().any():
        h = float(tail[H].iloc[-1])
        l = float(tail[L].iloc[-1])
        c = float(tail[C].iloc[-1])
        pivot = (h + l + c) / 3.0
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h

    # Bollinger bounds over the window if available
    bb_u = float(tail[U].max()) if U and tail[U].notna().any() else None
    bb_l = float(tail[D].min()) if D and tail[D].notna().any() else None

    # final composite
    cand_res = [v for v in (r1, swing_high, bb_u) if v is not None]
    cand_sup = [v for v in (s1, swing_low, bb_l) if v is not None]
    if not cand_res or not cand_sup:
        return None, None
    return max(cand_res), min(cand_sup)


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


# predictor using heuristic model (row/Series fetcher)
def _get_row(row: pd.Series, *names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
        # case-insensitive fallback
        ln = str(n).lower()
        for c in row.index:
            if c.lower() == ln and pd.notna(row[c]):
                return row[c]
    return default


def _safe(v, d=0.0):
    try:
        return float(v) if v is not None and not pd.isna(v) else d
    except Exception:
        return d


def predict_forward_returns(df_all: pd.DataFrame, horizons=("1d", "1w", "3m")) -> dict:
    """
    Heuristic, regime-aware predictor using features already coming from Redis.
    Returns {H: {"ret": r_hat, "price": P_t*(1+r_hat), "explain": [...]}}
    """
    preds = {h: {"ret": 0.0, "price": None, "explain": ["no data"]} for h in horizons}
    if df_all is None or df_all.empty:
        return preds

    last = df_all.iloc[-1]
    P_t = _get_row(last, "Close", "close")
    if P_t is None or pd.isna(P_t):
        return preds

    # === pull features from Redis columns (already present) ===
    adx = _get_row(last, "adx14", "ADX")
    mhd = _get_row(last, "macd_hist_delta", "MACD_Hist")
    s20 = _get_row(last, "ema20_slope_bps")
    s50 = _get_row(last, "ema50_slope_bps")
    vwapd = _get_row(last, "vwap_diff_bps")
    atrp = _get_row(last, "atr_pct")
    volz = _get_row(last, "vol_z")
    distH = _get_row(last, "dist_hh20_bps")
    bbw = _get_row(last, "bb_width_bps")
    bb_sq = _get_row(last, "bb_squeeze")
    above_ubb = _get_row(last, "above_upper_bb_flag")
    below_lbb = _get_row(last, "below_lower_bb_flag")

    # recent realized momentum and volatility
    df = df_all.tail(400).copy()
    if "Close" not in df and "close" in df:
        df["Close"] = df["close"]
    df["ret1"] = df["Close"].pct_change()
    mom = (
        df["ret1"].ewm(span=10, min_periods=5).mean().iloc[-1]
        if df["ret1"].notna().sum() >= 5
        else 0.0
    )
    vol = (
        df["ret1"].rolling(20).std().iloc[-1]
        if df["ret1"].notna().sum() >= 20
        else 0.01
    )

    # === regime detection ===
    adx_val = _safe(adx)
    s20_val = _safe(s20)
    s50_val = _safe(s50)
    trending_up = (adx_val >= 20) and (s20_val > 0) and (s50_val > 0)
    trending_down = (adx_val >= 20) and (s20_val < 0) and (s50_val < 0)
    choppy = adx_val < 15

    # === core score (basis points per horizon then convert to returns) ===
    # components are small and additive; they’re clipped by recent vol.
    base = 0.0
    explain = []

    # momentum via EMA returns and MACD histogram delta
    base += _safe(mom) * 100.0  # amplify small EMA to bps scale
    explain.append(f"mom={_safe(mom):.5f}")

    if mhd is not None and not pd.isna(mhd):
        base += _safe(mhd) * 10.0
        explain.append(f"macdΔ*10={_safe(mhd):.3f}")

    # trend slopes add directional bias
    base += 0.5 * s20_val + 0.25 * s50_val
    explain.append(f"s20*0.5+s50*0.25={0.5*s20_val+0.25*s50_val:.2f}bps")

    # VWAP tilt: if above VWAP (vwap_diff_bps > 0), slight positive bias
    if vwapd is not None:
        base += 0.10 * _safe(vwapd)  # 10% of bps deviation
        explain.append(f"VWAP tilt={0.10*_safe(vwapd):.2f}bps")

    # proximity to HH20: near resistance → fade a bit; far → no drag
    if distH is not None:
        # Negative contribution when very close to HH20 (harder to break)
        drag = max(0.0, 15.0 - abs(_safe(distH)))  # within 15 bps penalize
        base -= drag
        explain.append(f"HH20 drag=-{drag:.2f}bps")

    # squeeze breakout flags
    if _safe(bb_sq) > 0 and trending_up:
        base += 10.0
        explain.append("squeeze+trend=+10bps")
    if _safe(bb_sq) > 0 and trending_down:
        base -= 10.0
        explain.append("squeeze+downtrend=-10bps")
    if _safe(above_ubb) > 0:
        base += 5.0
        explain.append("above UBB=+5bps")
    if _safe(below_lbb) > 0:
        base -= 5.0
        explain.append("below LBB=-5bps")

    # ATR% gates: if too low volatility, shrink; if healthy, keep
    atrp_val = _safe(atrp)
    scale = 1.0
    if atrp_val < 0.25:
        scale *= 0.5
        explain.append("low ATR% → x0.5")
    elif atrp_val > 1.0:
        scale *= 1.2
        explain.append("high ATR% → x1.2")
    base *= scale

    # choppy: dampen signal
    if choppy:
        base *= 0.6
        explain.append("choppy regime → x0.6")

    # cap by recent volatility (convert vol to bps-ish guard)
    cap_bps = max(20.0, _safe(vol, 0.01) * 10000.0 * 0.4)  # 0.4 * daily vol (bps)
    base = float(np.clip(base, -cap_bps, cap_bps))
    explain.append(f"cap±{cap_bps:.1f}bps → base={base:.1f}bps")

    # map to horizons; compounding small edge by √time for short-term, linear for long
    scalers = {"1d": 1.0, "1w": np.sqrt(5.0), "3m": 5.0 * np.sqrt(5.0)}  # tune later
    out = {}
    for h in horizons:
        bps = base * scalers[h]
        r_hat = float(bps / 10000.0)
        # extra cap per horizon vs vol
        hcap = max(0.03, _safe(vol, 0.01) * {"1d": 3, "1w": 6, "3m": 12}[h])
        r_hat = float(np.clip(r_hat, -hcap, hcap))
        out[h] = {
            "ret": r_hat,
            "price": float(P_t * (1.0 + r_hat)),
            "explain": explain + [f"{h}: {bps:.1f}bps→r={r_hat:.4f}"],
        }
    return out


def predict_forward_returns_v2(df_all, horizons=("1d", "1w", "3m")) -> dict:
    import numpy as np

    preds = {h: {"ret": 0.0, "price": None, "explain": ["no data"]} for h in horizons}
    if df_all is None or df_all.empty:
        return preds

    last = df_all.iloc[-1]
    P_t = _get_row(last, "Close", "close")
    if P_t is None or pd.isna(P_t):
        return preds

    # --- features from Redis row ---
    adx = _get_row(last, "adx14", "ADX")
    mhd = _get_row(last, "macd_hist_delta", "MACD_Hist")
    s20 = _get_row(last, "ema20_slope_bps")
    s50 = _get_row(last, "ema50_slope_bps")
    vwapd = _get_row(last, "vwap_diff_bps")
    atrp = _get_row(last, "atr_pct")
    distH = _get_row(last, "dist_hh20_bps")
    bb_sq = _get_row(last, "bb_squeeze")
    aubb = _get_row(last, "above_upper_bb_flag")
    blbb = _get_row(last, "below_lower_bb_flag")

    # --- realized momentum/vol (from recent returns) ---
    df = df_all.tail(400).copy()
    if "Close" not in df and "close" in df:
        df["Close"] = df["close"]
    df["ret1"] = df["Close"].pct_change()
    mom = (
        df["ret1"].ewm(span=15, min_periods=5).mean().iloc[-1]
        if df["ret1"].notna().sum() >= 5
        else 0.0
    )
    vol = (
        df["ret1"].rolling(20).std().iloc[-1]
        if df["ret1"].notna().sum() >= 20
        else 0.01
    )

    # --- regimes (tweaked from v1) ---
    adx_val = _safe(adx)
    s20_val = _safe(s20)
    s50_val = _safe(s50)
    trending_up = (adx_val >= 18) and (s20_val > 0) and (s50_val > 0)
    trending_down = (adx_val >= 18) and (s20_val < 0) and (s50_val < 0)
    choppy = adx_val < 12

    # --- base score (bps) ---
    base = 0.0
    explain = []
    # momentum ↑, macdΔ ↓
    base += _safe(mom) * 120.0
    explain.append(f"mom*120={_safe(mom)*120:.2f}bps")
    if mhd is not None and not pd.isna(mhd):
        base += _safe(mhd) * 6.0
        explain.append(f"macdΔ*6={_safe(mhd)*6:.2f}bps")
    # slope weights
    base += 0.6 * s20_val + 0.2 * s50_val
    explain.append(f"s20*0.6+s50*0.2={0.6*s20_val+0.2*s50_val:.2f}bps")

    # VWAP: tilt + mean-reversion when far
    vd = _safe(vwapd)
    base += 0.08 * vd
    explain.append(f"VWAP tilt={0.08*vd:.2f}bps")
    if abs(vd) > 30:  # far from VWAP → partial MR
        base += -0.04 * vd
        explain.append(f"VWAP MR={-0.04*vd:.2f}bps")

    # HH20 drag (softer than v1)
    if distH is not None:
        drag = max(0.0, 12.0 - abs(_safe(distH)))  # v1 used 15
        base -= drag
        explain.append(f"HH20 drag=-{drag:.2f}bps")

    # squeeze/bands + regime
    if _safe(bb_sq) > 0 and trending_up:
        base += 12.0
        explain.append("squeeze+up=+12")
    if _safe(bb_sq) > 0 and trending_down:
        base -= 12.0
        explain.append("squeeze+down=-12")
    if _safe(aubb) > 0:
        base += 4.0
        explain.append("above UBB=+4")
    if _safe(blbb) > 0:
        base -= 4.0
        explain.append("below LBB=-4")

    # ATR scaling (slightly different) + choppy damp
    atrp_val = _safe(atrp)
    scale = 1.0
    if atrp_val < 0.25:
        scale *= 0.55
        explain.append("low ATR% x0.55")
    elif atrp_val > 1.2:
        scale *= 1.25
        explain.append("high ATR% x1.25")
    base *= scale
    if choppy:
        base *= 0.55
        explain.append("choppy x0.55")

    # vol-based cap (slightly higher)
    cap_bps = max(18.0, _safe(vol, 0.01) * 10000.0 * 0.45)
    base = float(np.clip(base, -cap_bps, cap_bps))
    explain.append(f"cap±{cap_bps:.1f} → base={base:.1f}")

    # horizon scalers/caps (different from v1)
    scalers = {"1d": 1.1, "1w": 2.5, "3m": 10.0}
    out = {}
    for h in horizons:
        bps = base * scalers[h]
        r_hat = float(bps / 10000.0)
        hcap = max(0.025, _safe(vol, 0.01) * {"1d": 3, "1w": 7, "3m": 14}[h])
        r_hat = float(np.clip(r_hat, -hcap, hcap))
        out[h] = {
            "ret": r_hat,
            "price": float(P_t * (1.0 + r_hat)),
            "explain": explain + [f"{h}: {bps:.1f}bps→r={r_hat:.4f}"],
        }
    return out


# ---------------------- entry target stoploss ----------------------
# ===================== MODULAR TRADE LEVELS (DROP-IN) =====================


def _safe_float(x, d=None):
    try:
        return float(x) if x is not None and not pd.isna(x) else d
    except Exception:
        return d


def _round3(x):
    try:
        return round(float(x), 3)
    except Exception:
        return None


def _last_float_ci(row, *names, default=None):
    """Case-insensitive row getter that returns float."""
    for n in names:
        if n in row and pd.notna(row[n]):
            return _safe_float(row[n], default)
        ln = str(n).lower()
        for c in row.index:
            if c.lower() == ln and pd.notna(row[c]):
                return _safe_float(row[c], default)
    return default


def _get_floor_pivots_from_last_daily(df_daily: pd.DataFrame):
    """Return floor pivots (P,R1,S1,R2,S2,R3,S3) from the last daily bar if available."""
    if df_daily is None or df_daily.empty:
        return None
    cols = {c.lower(): c for c in df_daily.columns}
    H, L, C = cols.get("high"), cols.get("low"), cols.get("close")
    if not (H and L and C):
        return None
    last = df_daily.dropna(subset=[H, L, C]).tail(1)
    if last.empty:
        return None
    h = float(last[H].iloc[-1])
    l = float(last[L].iloc[-1])
    c = float(last[C].iloc[-1])
    P = (h + l + c) / 3.0
    R1 = 2 * P - l
    S1 = 2 * P - h
    R2 = P + (h - l)
    S2 = P - (h - l)
    R3 = h + 2 * (P - l)  # = R1 + (h - l)
    S3 = l - 2 * (h - P)  # = S1 - (h - l)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2, "R3": R3, "S3": S3}


def _horizon_params(horizon: str, atr_pct_fallback=0.35):
    """
    Parameter schedule by horizon:
      - buf_bps:   entry buffer (in basis points) relative to level (converted from ATR%)
      - stop_mult: how many *ATR%* beyond the level for stop buffer when structure absent
      - rr_normal/rr_strong: risk-reward for neutral vs strong regimes
    """
    # Default ATR% fallback (if your latest['atr_pct'] missing): 0.35%
    # We dynamically build the buffer from ATR% but also clamp per horizon.
    horizon = (horizon or "1d").lower()
    if horizon == "1d":
        return {
            "buf_min_bps": 12,
            "buf_max_bps": 45,
            "stop_mult": 1.1,
            "rr_normal": 1.15,
            "rr_strong": 1.6,
            "atr_pct_fallback": atr_pct_fallback,
        }
    if horizon == "1w":
        return {
            "buf_min_bps": 15,
            "buf_max_bps": 60,
            "stop_mult": 1.2,
            "rr_normal": 1.25,
            "rr_strong": 1.8,
            "atr_pct_fallback": atr_pct_fallback,
        }
    # 3m+
    return {
        "buf_min_bps": 18,
        "buf_max_bps": 75,
        "stop_mult": 1.35,
        "rr_normal": 1.35,
        "rr_strong": 2.0,
        "atr_pct_fallback": atr_pct_fallback,
    }


# def _compute_pair_levels(
#     support: float,
#     resistance: float,
#     current_price: float,
#     horizon: str,
#     adx_val: float,
#     atr_pct: float,
#     pred_1d_ret: float,
#     pred_1w_ret: float,
#     signal: str,
#     structural_floors: dict | None = None,
# ):
#     """
#     Core modular routine to compute entry/target/stoploss for ONE S/R pair, parameterized by horizon.
#     - Chooses breakout vs breakdown vs range using ADX, 1d/1w bias, and proximity.
#     - Uses structure (support/resistance/pivots) first, then ATR slack fallback.
#     """
#     out = {"entry": None, "target": None, "stoploss": None}
#     if support is None or resistance is None or current_price is None:
#         return out

#     params = _horizon_params(horizon)
#     # ADX / bias
#     adx_val = 15.0 if adx_val is None or adx_val <= 0 else float(adx_val)
#     trending = adx_val >= 20
#     bias_up = (float(pred_1d_ret or 0.0) + float(pred_1w_ret or 0.0)) > 0.0

#     # ATR%
#     atr_pct = (
#         params["atr_pct_fallback"]
#         if (atr_pct is None or atr_pct <= 0)
#         else float(atr_pct)
#     )
#     atr_frac = max(0.001, atr_pct / 100.0)  # % → fraction

#     # Entry buffer in FRACTION (derived from ATR%, clamped to horizon bps)
#     #  Example: 0.75*ATR% in frac → convert to bps clamp → back to frac.
#     buf_bps = max(
#         params["buf_min_bps"], min(params["buf_max_bps"], 0.75 * atr_frac * 10000.0)
#     )
#     buf_frac = buf_bps / 10000.0

#     stop_mult = params["stop_mult"]
#     rr_normal = params["rr_normal"]
#     rr_strong = params["rr_strong"]

#     # Proximity check (near barriers within 15 bps)
#     near_res = abs(current_price - resistance) / max(1e-9, resistance) < (15 / 10000)
#     near_sup = abs(current_price - support) / max(1e-9, support) < (15 / 10000)

#     # Decision
#     want_breakout_long = (
#         ("Breakout" in (signal or "")) or (trending and bias_up) or near_res
#     )
#     want_breakdown_short = (
#         ("Breakdown" in (signal or "")) or (trending and not bias_up) or near_sup
#     )

#     # Structural context (optional pivots to anchor stops/targets)
#     P = structural_floors.get("P") if structural_floors else None
#     R1 = structural_floors.get("R1") if structural_floors else None
#     R2 = structural_floors.get("R2") if structural_floors else None
#     R3 = structural_floors.get("R3") if structural_floors else None
#     S1 = structural_floors.get("S1") if structural_floors else None
#     S2 = structural_floors.get("S2") if structural_floors else None
#     S3 = structural_floors.get("S3") if structural_floors else None

#     # Compute
#     if want_breakout_long and (current_price >= support):
#         entry = resistance * (1.0 + buf_frac)
#         # prefer structural stop just below the breakout level; otherwise ATR slack
#         structural_stop = max(
#             x
#             for x in [support, P, R1, resistance * (1.0 - stop_mult * atr_frac)]
#             if x is not None
#         )
#         stoploss = structural_stop
#         risk = max(0.01, entry - stoploss)
#         rr = rr_strong if ("Strong" in (signal or "")) else rr_normal
#         # prefer next structural target beyond resistance, else RR
#         next_R = None
#         if resistance == (R1 or resistance):
#             next_R = R2 or R3
#         elif resistance == (R2 or resistance):
#             next_R = R3
#         target = (
#             (next_R * (1.0 - buf_frac)) if next_R is not None else (entry + rr * risk)
#         )

#     elif want_breakdown_short and (current_price <= resistance):
#         entry = support * (1.0 - buf_frac)
#         structural_stop = min(
#             x
#             for x in [resistance, P, S1, support * (1.0 + stop_mult * atr_frac)]
#             if x is not None
#         )
#         stoploss = structural_stop
#         risk = max(0.01, stoploss - entry)
#         rr = rr_strong if ("Strong" in (signal or "")) else rr_normal
#         next_S = None
#         if support == (S1 or support):
#             next_S = S2 or S3
#         elif support == (S2 or support):
#             next_S = S3
#         target = (
#             (next_S * (1.0 + buf_frac)) if next_S is not None else (entry - rr * risk)
#         )

#     else:
#         # Range logic: buy S→R if up-bias, else sell R→S
#         if bias_up:
#             entry = max(current_price, support * (1.0 + buf_frac))
#             stoploss = max(
#                 x
#                 for x in [support * (1.0 - stop_mult * atr_frac), S1, P]
#                 if x is not None
#             )
#             target = resistance * (1.0 - buf_frac)
#         else:
#             entry = min(current_price, resistance * (1.0 - buf_frac))
#             stoploss = min(
#                 x
#                 for x in [resistance * (1.0 + stop_mult * atr_frac), R1, P]
#                 if x is not None
#             )
#             target = support * (1.0 + buf_frac)

#     out["entry"] = _round3(entry)
#     out["target"] = _round3(target)
#     out["stoploss"] = _round3(stoploss)
#     return out


# OPTIONAL (if you want to ALSO use your horizon_SR from daily):
# sr_1d_set = _compute_pair_levels(support_1d, resistance_1d, current_px, "1d", adx_val, atr_pct, pred_1d_ret, pred_1w_ret, signal, _piv)
# sr_1w_set = _compute_pair_levels(support_1w, resistance_1w, current_px, "1w", adx_val, atr_pct, pred_1d_ret, pred_1w_ret, signal, _piv)
# sr_3m_set = _compute_pair_levels(support_3m, resistance_3m, current_px, "3m", adx_val, atr_pct, pred_1d_ret, pred_1w_ret, signal, _piv)

# ---------------- inject into new_cfg (do this just before you create compare_keys) -----
# new_cfg.update({
#     # Simple SR (default)
#     "entry":    sr_simple["entry"],
#     "target":   sr_simple["target"],
#     "stoploss": sr_simple["stoploss"],

#     # S1/R1
#     "entry1":   entry1_set["entry"],
#     "target1":  entry1_set["target"],
#     "stoploss1":entry1_set["stoploss"],

#     # S2/R2
#     "entry2":   entry2_set["entry"],
#     "target2":  entry2_set["target"],
#     "stoploss2":entry2_set["stoploss"],

#     # S3/R3
#     "entry3":   entry3_set["entry"],
#     "target3":  entry3_set["target"],
#     "stoploss3":entry3_set["stoploss"],
# })

# ===== Ensure your change detection includes these keys (after you define compare_keys) =====
# compare_keys += ["entry","target","stoploss","entry1","target1","stoploss1",
#                  "entry2","target2","stoploss2","entry3","target3","stoploss3"]
# ===========================================================================

# ======================= UPDATED CORE LEVELS =======================


# def _compute_pair_levels(
#     support: float,
#     resistance: float,
#     current_price: float,
#     horizon: str,
#     adx_val: float,
#     atr_pct: float,
#     pred_1d_ret: float,
#     pred_1w_ret: float,
#     signal: str,
#     structural_floors: dict | None = None,
#     mode: str = "range",  # NEW: 'range' (default), 'breakout', 'breakdown', or 'auto'
# ):
#     """
#     Returns dict(entry, target, stoploss) with sensible, sanitized levels.
#     - 'range' (default): longs enter near S, shorts near R; targets stay inside band.
#     - 'breakout': long entry above R; 'breakdown': short entry below S.
#     - 'auto': infer from signal + ADX/bias.
#     """
#     out = {"entry": None, "target": None, "stoploss": None}
#     support, resistance = _ensure_sr_order(support, resistance)
#     if current_price is None or (support is None and resistance is None):
#         return out

#     params = _horizon_params(horizon)
#     adx_val = 15.0 if adx_val is None or adx_val <= 0 else float(adx_val)
#     trending = adx_val >= 20
#     bias_up = (float(pred_1d_ret or 0.0) + float(pred_1w_ret or 0.0)) > 0.0

#     # ATR handling
#     atr_pct = (
#         params["atr_pct_fallback"]
#         if (atr_pct is None or atr_pct <= 0)
#         else float(atr_pct)
#     )
#     atr_frac = max(0.001, atr_pct / 100.0)  # % → fraction

#     # Buffer: 0.75 * ATR% in bps, clamped
#     buf_bps = max(
#         params["buf_min_bps"], min(params["buf_max_bps"], 0.75 * atr_frac * 10000.0)
#     )
#     buf_frac = buf_bps / 10000.0
#     stop_mult = params["stop_mult"]
#     rr_normal = params["rr_normal"]
#     rr_strong = params["rr_strong"]

#     # Structural anchors
#     P = structural_floors.get("P") if structural_floors else None
#     R1 = structural_floors.get("R1") if structural_floors else None
#     R2 = structural_floors.get("R2") if structural_floors else None
#     R3 = structural_floors.get("R3") if structural_floors else None
#     S1 = structural_floors.get("S1") if structural_floors else None
#     S2 = structural_floors.get("S2") if structural_floors else None
#     S3 = structural_floors.get("S3") if structural_floors else None

#     side, eff_mode = _decide_side_and_mode(
#         signal, trending, bias_up, current_price, support, resistance, mode
#     )

#     entry = target = stoploss = None
#     strong = "strong" in (signal or "").lower()
#     rr = rr_strong if strong else rr_normal

#     # ------------- LONG LOGIC -------------
#     if side == "long":
#         if eff_mode == "breakout" and resistance is not None:
#             # breakout (kept for optional use; sanitize will clamp if user prefers range)
#             entry = resistance * (1.0 + buf_frac)
#             # stop just below breakout region / nearest structure
#             structural_stop = max(
#                 x
#                 for x in [support, P, R1, resistance * (1.0 - stop_mult * atr_frac)]
#                 if x is not None
#             )
#             stoploss = structural_stop
#             risk = max(0.01, entry - stoploss)
#             # next structural target else RR
#             next_R = None
#             if resistance == (R1 or resistance):
#                 next_R = R2 or R3
#             elif resistance == (R2 or resistance):
#                 next_R = R3
#             target = (
#                 (next_R * (1.0 - buf_frac))
#                 if next_R is not None
#                 else (entry + rr * risk)
#             )

#         else:
#             # range (ENTER NEAR SUPPORT; TARGET INSIDE RESISTANCE)
#             if support is None or resistance is None:
#                 # fallback: small band around current price
#                 entry = current_price
#                 stoploss = current_price * (1.0 - stop_mult * atr_frac)
#                 target = current_price * (1.0 + rr * stop_mult * atr_frac)
#             else:
#                 entry = max(current_price, support * (1.0 + buf_frac))
#                 # stop anchored near support but below entry
#                 cand_stops = [support * (1.0 - stop_mult * atr_frac), S1, P]
#                 cand_stops = [x for x in cand_stops if x is not None]
#                 stoploss = (
#                     max(cand_stops)
#                     if cand_stops
#                     else support * (1.0 - stop_mult * atr_frac)
#                 )
#                 # RR target inside resistance
#                 risk = max(0.01, entry - stoploss)
#                 target_rr = entry + rr * risk
#                 target_cap = resistance * (1.0 - buf_frac)
#                 target = target_rr if target_cap is None else min(target_rr, target_cap)

#         # FINAL SANITIZE for long
#         entry, target, stoploss = _sanitize_levels(
#             entry, target, stoploss, "long", support, resistance, buf_frac, atr_frac
#         )

#     # ------------- SHORT LOGIC -------------
#     else:  # side == "short"
#         if eff_mode == "breakdown" and support is not None:
#             # breakdown (optional)
#             entry = support * (1.0 - buf_frac)
#             structural_stop = min(
#                 x
#                 for x in [resistance, P, S1, support * (1.0 + stop_mult * atr_frac)]
#                 if x is not None
#             )
#             stoploss = structural_stop
#             risk = max(0.01, stoploss - entry)
#             next_S = None
#             if support == (S1 or support):
#                 next_S = S2 or S3
#             elif support == (S2 or support):
#                 next_S = S3
#             target = (
#                 (next_S * (1.0 + buf_frac))
#                 if next_S is not None
#                 else (entry - rr * risk)
#             )

#         else:
#             # range (ENTER NEAR RESISTANCE; TARGET INSIDE SUPPORT)
#             if support is None or resistance is None:
#                 entry = current_price
#                 stoploss = current_price * (1.0 + stop_mult * atr_frac)
#                 target = current_price * (1.0 - rr * stop_mult * atr_frac)
#             else:
#                 entry = min(current_price, resistance * (1.0 - buf_frac))
#                 cand_stops = [resistance * (1.0 + stop_mult * atr_frac), R1, P]
#                 cand_stops = [x for x in cand_stops if x is not None]
#                 stoploss = (
#                     min(cand_stops)
#                     if cand_stops
#                     else resistance * (1.0 + stop_mult * atr_frac)
#                 )
#                 risk = max(0.01, stoploss - entry)
#                 target_rr = entry - rr * risk
#                 target_cap = support * (1.0 + buf_frac)
#                 target = target_rr if target_cap is None else max(target_rr, target_cap)

#         # FINAL SANITIZE for short
#         entry, target, stoploss = _sanitize_levels(
#             entry, target, stoploss, "short", support, resistance, buf_frac, atr_frac
#         )

#     out["entry"] = entry
#     out["target"] = target
#     out["stoploss"] = stoploss
#     return out

def _compute_pair_levels(
    support: float,
    resistance: float,
    current_price: float,
    horizon: str,
    adx_val: float,
    atr_pct: float,
    pred_1d_ret: float,
    pred_1w_ret: float,
    signal: str,
    structural_floors: dict | None = None,
    mode: str = "range",
):
    """
    LONG-ONLY levels with enforced ordering:
      stoploss < support < entry < target < resistance
    Entry is placed near support (ATR-buffered), target stays inside resistance.
    """
    out = {"entry": None, "target": None, "stoploss": None}

    support, resistance = _ensure_sr_order(support, resistance)
    if current_price is None or (support is None or resistance is None):
        return out  # need both S & R for long-only range logic

    params = _horizon_params(horizon)

    # Vol/ATR handling
    adx_val = 15.0 if adx_val is None or adx_val <= 0 else float(adx_val)
    atr_pct = params["atr_pct_fallback"] if (atr_pct is None or atr_pct <= 0) else float(atr_pct)
    atr_frac = max(0.001, atr_pct / 100.0)  # % → fraction

    # Buffer: 0.75*ATR% (in bps), clamped per horizon
    buf_bps  = max(params["buf_min_bps"], min(params["buf_max_bps"], 0.75 * atr_frac * 10000.0))
    buf_frac = buf_bps / 10000.0

    stop_mult = params["stop_mult"]
    strong = "strong" in (signal or "").lower()
    rr = params["rr_strong"] if strong else params["rr_normal"]

    # Structural anchors (optional)
    P  = structural_floors.get("P")  if structural_floors else None
    S1 = structural_floors.get("S1") if structural_floors else None

    # Draft levels (range long)
    entry = max(current_price, support * (1.0 + buf_frac))

    # Stop anchored below support (ATR-bounded), or structural if present
    cand_stops = [support * (1.0 - stop_mult * atr_frac), S1, P]
    cand_stops = [x for x in cand_stops if x is not None]
    stoploss = max(cand_stops) if cand_stops else support * (1.0 - stop_mult * atr_frac)

    # Target via RR, capped inside resistance
    risk = max(0.01, entry - stoploss)
    target_rr  = entry + rr * risk
    target_cap = resistance * (1.0 - buf_frac)
    target = min(target_rr, target_cap)

    # Final invariant enforcement (long)
    entry, target, stoploss = _sanitize_levels(
        entry, target, stoploss, "long", support, resistance, buf_frac, atr_frac
    )

    out["entry"] = entry
    out["target"] = target
    out["stoploss"] = stoploss
    return out



# Final belt & suspenders (keeps everything inside S/R band)
# def _final_sanitize_block(block, side_hint, sup, res, atr_pct):
#     if not block: return block
#     frac = max(0.001, (atr_pct or 0.35)/100.0)
#     buf = max(12/10000.0, min(60/10000.0, 0.75*frac))
#     e,t,s = _sanitize_levels(block.get("entry"), block.get("target"), block.get("stoploss"),
#                              side_hint, sup, res, buf, frac)
#     block["entry"], block["target"], block["stoploss"] = e,t,s
#     return block

# sr_simple  = _final_sanitize_block(sr_simple,  "long" if pred_1d_ret+pred_1w_ret>0 else "short", support, resistance, atr_pct)
# entry1_set = _final_sanitize_block(entry1_set, "long" if pred_1d_ret+pred_1w_ret>0 else "short",
#                                    _piv.get("S1") if _piv else support,
#                                    _piv.get("R1") if _piv else resistance,
#                                    atr_pct)
# entry2_set = _final_sanitize_block(entry2_set, "long" if pred_1d_ret+pred_1w_ret>0 else "short",
#                                    _piv.get("S2") if _piv else support,
#                                    _piv.get("R2") if _piv else resistance,
#                                    atr_pct)
# entry3_set = _final_sanitize_block(entry3_set, "long" if pred_1d_ret+pred_1w_ret>0 else "short",
#                                    _piv.get("S3") if _piv else support,
#                                    _piv.get("R3") if _piv else resistance,
#                                    atr_pct)


# -------------------- support/resistance counting --------------------
# ---------- helpers (keep or replace your existing ones) ----------


def _colmap(df):
    cols = {c.lower(): c for c in df.columns}
    return cols.get("high"), cols.get("low"), cols.get("close")


def _auto_tol_frac(
    df_tail, atr_col="atr_pct", default_pct=0.35, scale=0.25, min_bps=8, max_bps=60
):
    """
    ATR%-based tolerance (fraction). E.g., 0.0015 = 15 bps.
    scale defaults to 0.25*ATR%; clamped between [min_bps, max_bps].
    """
    tol_pct = default_pct
    if df_tail is not None and not df_tail.empty:
        cols = {c.lower(): c for c in df_tail.columns}
        A = cols.get(atr_col.lower())
        if A and df_tail[A].notna().any():
            try:
                tol_pct = float(df_tail[A].iloc[-1])
            except Exception:
                pass
    frac = (tol_pct / 100.0) * scale
    frac = max(min_bps / 10000.0, min(max_bps / 10000.0, frac))
    return frac


def _count_respects(df, level, side, tol_frac=0.0015, min_sep=3):
    """
    Count 'respect' events for a single level over a dataframe of candles.
    side: 'support' or 'resistance'
    tol_frac: proximity band around level (fraction of price)
    min_sep: min bars between events (debounce)
    Logic:
      - SUPPORT respected if:
          (a) near-touch: abs(Low - level)/level <= tol AND Close >= level
          OR
          (b) wick break & reclaim: Low < level AND Close > level
      - RESISTANCE respected if:
          (a) near-touch: abs(High - level)/level <= tol AND Close <= level
          OR
          (b) wick break & reject: High > level AND Close < level
    """
    if level is None or df is None or df.empty:
        return 0

    H, L, C = _colmap(df)
    if not (H and L and C):
        return 0

    cnt = 0
    last_i = -(10**9)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            hi = float(row[H])
            lo = float(row[L])
            cl = float(row[C])
        except Exception:
            continue

        if side == "support":
            near = abs(lo - level) / max(1e-9, level) <= tol_frac and cl >= level
            wick = (lo < level) and (cl > level)
            hit = near or wick
        else:
            near = abs(hi - level) / max(1e-9, level) <= tol_frac and cl <= level
            wick = (hi > level) and (cl < level)
            hit = near or wick

        if hit and (i - last_i) >= min_sep:
            cnt += 1
            last_i = i
    return cnt


# ---------- main function (drop-in replacement) ----------


def compute_sr_respect_counts(
    df_recent: pd.DataFrame,
    df_hist: pd.DataFrame,
    support: float,
    resistance: float,
    pivots: dict | None = None,
    # windows
    lookback_hist_bars_for_SR: int = 180,  # base S/R => DAILY window
    lookback_hist_bars_for_pivots: int = 180,  # pivots => DAILY window
    # tolerances
    base_sr_fixed_tol_pct: float = 0.50,  # 0.50% wide band for base S/R
    pivot_auto_scale: float = 0.25,  # pivots use ATR% * 0.25 (clamped 8–60 bps)
    pivot_min_bps: int = 8,
    pivot_max_bps: int = 60,
    # debounce
    min_sep_bars: int = 2,
) -> dict:
    """
    Counts 'respect' events for:
      - Base S/R (support/resistance): uses DAILY candles with a WIDE fixed tolerance (default 0.50%).
      - Pivot pairs S1/R1, S2/R2, S3/R3: uses DAILY candles with ATR-based, narrower tolerance.

    Returns:
      {
        "respected_S": int, "respected_R": int,
        "respected_S1": int, "respected_R1": int,
        "respected_S2": int, "respected_R2": int,
        "respected_S3": int, "respected_R3": int
      }
    """
    out = {
        "respected_S": 0,
        "respected_R": 0,
        "respected_S1": 0,
        "respected_R1": 0,
        "respected_S2": 0,
        "respected_R2": 0,
        "respected_S3": 0,
        "respected_R3": 0,
    }

    # Ensure we have daily data for both base S/R and pivots
    dly = (
        df_hist.tail(max(1, lookback_hist_bars_for_SR))
        if (df_hist is not None and not df_hist.empty)
        else pd.DataFrame()
    )
    dly_piv = (
        df_hist.tail(max(1, lookback_hist_bars_for_pivots))
        if (df_hist is not None and not df_hist.empty)
        else pd.DataFrame()
    )

    # ---- 1) Base S/R on DAILY with WIDE fixed tolerance ----
    # Convert the fixed tolerance % to fraction
    base_tol_frac = max(
        0.0005, float(base_sr_fixed_tol_pct) / 100.0
    )  # e.g., 0.50% => 0.005
    if not dly.empty:
        if support is not None:
            out["respected_S"] = _count_respects(
                dly, support, "support", base_tol_frac, min_sep_bars
            )
        if resistance is not None:
            out["respected_R"] = _count_respects(
                dly, resistance, "resistance", base_tol_frac, min_sep_bars
            )

    # ---- 2) Pivot S/R on DAILY with ATR-based tolerance ----
    piv = pivots or {}
    if not dly_piv.empty and piv:
        # ATR-based tolerance (narrower, adaptive)
        pivot_tol_frac = _auto_tol_frac(
            dly_piv,
            atr_col="atr_pct",
            default_pct=0.35,
            scale=pivot_auto_scale,
            min_bps=pivot_min_bps,
            max_bps=pivot_max_bps,
        )

        s1, r1 = piv.get("S1"), piv.get("R1")
        s2, r2 = piv.get("S2"), piv.get("R2")
        s3, r3 = piv.get("S3"), piv.get("R3")

        if s1 is not None:
            out["respected_S1"] = _count_respects(
                dly_piv, s1, "support", pivot_tol_frac, min_sep_bars
            )
        if r1 is not None:
            out["respected_R1"] = _count_respects(
                dly_piv, r1, "resistance", pivot_tol_frac, min_sep_bars
            )
        if s2 is not None:
            out["respected_S2"] = _count_respects(
                dly_piv, s2, "support", pivot_tol_frac, min_sep_bars
            )
        if r2 is not None:
            out["respected_R2"] = _count_respects(
                dly_piv, r2, "resistance", pivot_tol_frac, min_sep_bars
            )
        if s3 is not None:
            out["respected_S3"] = _count_respects(
                dly_piv, s3, "support", pivot_tol_frac, min_sep_bars
            )
        if r3 is not None:
            out["respected_R3"] = _count_respects(
                dly_piv, r3, "resistance", pivot_tol_frac, min_sep_bars
            )

    return out


# -------------------- ohlv update --------------------


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


# -------------------- main update --------------------


def basic_forecast_update(stock_cfg, recent_df, historical_folder="historical_data"):

    stock_cfg = copy.deepcopy(stock_cfg)
    stock_code = stock_cfg.get("stock_code")

    # Historical (daily) — used for 1d/1w/3m S/R windows
    df_hist = pd.DataFrame()
    hist_csv = None
    if os.path.isdir(historical_folder):
        for file in os.listdir(historical_folder):
            if file.startswith(f"{stock_code}_historical_") and file.endswith(".csv"):
                hist_csv = os.path.join(historical_folder, file)
                break

    if hist_csv and os.path.exists(hist_csv):
        try:
            df_hist = pd.read_csv(hist_csv)
            # Be tolerant to casing; ensure 'Timestamp' exists for df_all merge later
            cols = {c.lower(): c for c in df_hist.columns}
            # If daily file has 'Date', rename to 'Timestamp'
            if "timestamp" not in cols and "date" in cols:
                df_hist = df_hist.rename(columns={cols["date"]: "Timestamp"})
            elif "timestamp" not in cols and "datetime" in cols:
                df_hist = df_hist.rename(columns={cols["datetime"]: "Timestamp"})

            if "Timestamp" in df_hist.columns:
                df_hist["Timestamp"] = to_ist(df_hist["Timestamp"], keep_tz=True)
                df_hist = df_hist.drop_duplicates(subset=["Timestamp"]).reset_index(
                    drop=True
                )
        except Exception:
            df_hist = pd.DataFrame()
    else:
        df_hist = pd.DataFrame()

    # Recent (from Redis)
    df_recent = (
        recent_df.copy()
        if recent_df is not None and not recent_df.empty
        else pd.DataFrame()
    )
    if not df_recent.empty:
        df_recent = df_recent.reset_index(drop=True)
        if "Timestamp" in df_recent.columns:
            df_recent["Timestamp"] = to_ist(df_recent["Timestamp"], keep_tz=True)
            df_recent = df_recent.drop_duplicates(subset=["Timestamp"])

    # Merge
    df_all = (
        pd.concat([df_hist, df_recent], ignore_index=True)
        if not df_hist.empty
        else df_recent.copy()
    )
    df_all = df_all.sort_values("Timestamp").reset_index(drop=True)

    # --- Compute SR + decision (unchanged user functions) ---
    support, resistance, bb_upper, bb_lower = compute_support_resistance(df_all)
    # Ensure S < R
    support, resistance = _ensure_sr_order(support, resistance)

    vol_series = _get(df_recent, "Volume", "volume", default=0)
    volume_threshold = (
        float(vol_series.tail(50).mean() * 1.5) if len(vol_series) else 0.0
    )

    latest = df_recent.iloc[-1]
    signal, reasons = determine_signal(latest, support, resistance, volume_threshold)

    # === Realized targets (labels) ===
    df_targets = enrich_with_targets(
        df_all,
        time_col="Timestamp",
        close_col="Close",
        horizons={"1d": 1, "1w": 5, "3m": 63},  # business-day horizons
        use_business_days=True,
    )

    # # Extract last row targets (if available)
    # last_row = df_targets.iloc[-1]
    # targets_block = {
    #     "1d": {
    #         "time": safe_str(last_row.get("target_time_1d")),
    #         "price": safe_number(last_row.get("target_close_1d")),
    #         "ret": safe_number(last_row.get("target_ret_1d")),
    #     },
    #     "1w": {
    #         "time": safe_str(last_row.get("target_time_1w")),
    #         "price": safe_number(last_row.get("target_close_1w")),
    #         "ret": safe_number(last_row.get("target_ret_1w")),
    #     },
    #     "3m": {
    #         "time": safe_str(last_row.get("target_time_3m")),
    #         "price": safe_number(last_row.get("target_close_3m")),
    #         "ret": safe_number(last_row.get("target_ret_3m")),
    #     },
    # }

    # === NEW: Predicted targets (real-time forecast) ===
    # Uses your existing predictor that reads Redis features from df_all
    preds = predict_forward_returns(df_all, horizons=("1d", "1w", "3m"))
    predicted_targets = {
        "1d": {
            "ret": preds["1d"]["ret"],
            "price": preds["1d"]["price"],
            "method": "heuristic_v1",
        },
        "1w": {
            "ret": preds["1w"]["ret"],
            "price": preds["1w"]["price"],
            "method": "heuristic_v1",
        },
        "3m": {
            "ret": preds["3m"]["ret"],
            "price": preds["3m"]["price"],
            "method": "heuristic_v1",
        },
    }

    # === NEW: multi-horizon S/R from *daily* data ===
    # Lookbacks: 1d = last day (1 row), 1w = 5 rows, 3m ≈ 60 rows
    # res_1d, sup_1d = _donchian_sr_from_daily(df_hist, lookback=1)
    # res_1w, sup_1w = _donchian_sr_from_daily(df_hist, lookback=5)
    # res_3m, sup_3m = _donchian_sr_from_daily(df_hist, lookback=60)
    res_1d, sup_1d = horizon_sr_from_daily(df_hist, 1)
    res_1w, sup_1w = horizon_sr_from_daily(df_hist, 5)
    res_3m, sup_3m = horizon_sr_from_daily(df_hist, 60)

    # Base close for % width = latest close we’ll have anyway
    base_close = None
    if not df_all.empty:
        base_close = _last_val(df_all.iloc[-1], "Close", "close", default=None)

    sr_range_pct = _sr_width_pct(support, resistance, base_close)
    sr_range_pct_1d = _sr_width_pct(sup_1d, res_1d, base_close)
    sr_range_pct_1w = _sr_width_pct(sup_1w, res_1w, base_close)
    sr_range_pct_3m = _sr_width_pct(sup_3m, res_3m, base_close)

    # Backward-compatibility: keep legacy 'support'/'resistance' mapped to 1d set if available
    if sup_1d is not None and res_1d is not None:
        support, resistance = round(sup_1d, 2), round(res_1d, 2)

    last_candles = fetch_candles(
        stock_code, n=1
    )  # newest single candle, chronological df
    if last_candles is not None and not last_candles.empty:
        row = last_candles.iloc[-1]
        ohlcv_block = {
            "time": str(row.get("minute")),
            "open": float(row.get("open", "")) if row.get("open") is not None else None,
            "high": float(row.get("high", "")) if row.get("high") is not None else None,
            "low": float(row.get("low", "")) if row.get("low") is not None else None,
            "close": (
                float(row.get("close", "")) if row.get("close") is not None else None
            ),
            "volume": (
                int(row.get("volume", 0)) if row.get("volume") is not None else None
            ),
        }
        current_price = ohlcv_block["close"]
    else:
        ohlcv_block = None
        # fallback to latest indicators row if candles missing
        current_price = (
            float(latest.get("Close"))
            if "Close" in latest
            else float(latest.get("close", float("nan")))
        )

    # ------------------ WIRE-UP: call once to compute all sets ------------------
    # Pull context from your existing variables:
    close_ = _last_float_ci(latest, "Close", "close")
    current_px = current_price if current_price is not None else close_
    adx_val = _last_float_ci(latest, "adx14", "ADX")
    atr_pct = _last_float_ci(latest, "atr_pct")  # % (e.g., 0.45 means 0.45%)
    pred_1d_ret = _safe_float(preds.get("1d", {}).get("ret"), 0.0)
    pred_1w_ret = _safe_float(preds.get("1w", {}).get("ret"), 0.0)

    entry_mode = (
        stock_cfg.get("entry_mode") or "range"
    ).lower()  # 'range' (default), 'breakout', 'breakdown', or 'auto'

    # entry_mode = "range"

    # Floor pivots from last daily bar (used as structural anchors for all horizons)
    _piv = _get_floor_pivots_from_last_daily(df_hist) or {}

    # 1) SIMPLE SR (you already map base SR to 1d later; we treat as 1d horizon)
    sr_simple = _compute_pair_levels(
        support=support,
        resistance=resistance,
        current_price=current_px,
        horizon="1d",
        adx_val=adx_val,
        atr_pct=atr_pct,
        pred_1d_ret=pred_1d_ret,
        pred_1w_ret=pred_1w_ret,
        signal=signal,
        structural_floors=_piv,
        mode=entry_mode,  # <--- NEW
    )

    entry1_set = _compute_pair_levels(
        support=_piv.get("S1") if _piv else None,
        resistance=_piv.get("R1") if _piv else None,
        current_price=current_px,
        horizon="1d",
        adx_val=adx_val,
        atr_pct=atr_pct,
        pred_1d_ret=pred_1d_ret,
        pred_1w_ret=pred_1w_ret,
        signal=signal,
        structural_floors=_piv,
        mode=entry_mode,  # <--- NEW
    )

    entry2_set = _compute_pair_levels(
        support=_piv.get("S2") if _piv else None,
        resistance=_piv.get("R2") if _piv else None,
        current_price=current_px,
        horizon="1w",
        adx_val=adx_val,
        atr_pct=atr_pct,
        pred_1d_ret=pred_1d_ret,
        pred_1w_ret=pred_1w_ret,
        signal=signal,
        structural_floors=_piv,
        mode=entry_mode,  # <--- NEW
    )

    entry3_set = _compute_pair_levels(
        support=_piv.get("S3") if _piv else None,
        resistance=_piv.get("R3") if _piv else None,
        current_price=current_px,
        horizon="3m",
        adx_val=adx_val,
        atr_pct=atr_pct,
        pred_1d_ret=pred_1d_ret,
        pred_1w_ret=pred_1w_ret,
        signal=signal,
        structural_floors=_piv,
        mode=entry_mode,  # <--- NEW
    )

    # Pivot anchors (optional but recommended)
    _piv = _get_floor_pivots_from_last_daily(df_hist) or {}

    # Count respect events
    # respect_counts = compute_sr_respect_counts(
    #     df_recent=df_recent,
    #     df_hist=df_hist,
    #     support=support,
    #     resistance=resistance,
    #     pivots=_piv,
    #     lookback_recent_bars=600,  # tune for your minute history depth
    #     lookback_hist_bars=120,  # ~6 months of dailies
    #     tol_mode="auto",  # ATR-based tolerance
    #     tol_bps_fixed=15,  # if you switch tol_mode="fixed"
    #     min_sep_bars=3,  # debounce consecutive taps
    # )

    respect_counts = compute_sr_respect_counts(
        df_recent=df_recent,
        df_hist=df_hist,
        support=support,
        resistance=resistance,
        pivots=_piv,
        # windows (daily)
        lookback_hist_bars_for_SR=180,  # base S/R on daily
        lookback_hist_bars_for_pivots=180,  # pivots on daily
        # tolerances
        base_sr_fixed_tol_pct=0.50,  # 0.50% wide band for base S/R
        pivot_auto_scale=0.25,  # pivots: ATR% * 0.25
        pivot_min_bps=8,
        pivot_max_bps=60,
        # debounce
        min_sep_bars=2,
    )

    new_cfg = {
        "stock_code": stock_cfg.get("stock_code"),
        "instrument_token": stock_cfg.get("instrument_token"),
        "ohlcv": ohlcv_block,  # {"time", "open","high","low","close","volume"} or None
        "support": safe_number(support),
        "resistance": safe_number(resistance),
        "sr_range_pct": sr_range_pct,
        "volume_threshold": int(safe_number(volume_threshold)),
        # Legacy S/R mapped to 1d (see override above). Also store multi-horizon S/R:
        "support_1d": safe_number(sup_1d),
        "resistance_1d": safe_number(res_1d),
        "support_1w": safe_number(sup_1w),
        "resistance_1w": safe_number(res_1w),
        "support_3m": safe_number(sup_3m),
        "resistance_3m": safe_number(res_3m),
        "sr_range_pct_1d": sr_range_pct_1d,
        "sr_range_pct_1w": sr_range_pct_1w,
        "sr_range_pct_3m": sr_range_pct_3m,
        "entry": sr_simple["entry"],
        "target": sr_simple["target"],
        "stoploss": sr_simple["stoploss"],
        # S1/R1
        "entry1": entry1_set["entry"],
        "target1": entry1_set["target"],
        "stoploss1": entry1_set["stoploss"],
        # S2/R2
        "entry2": entry2_set["entry"],
        "target2": entry2_set["target"],
        "stoploss2": entry2_set["stoploss"],
        # S3/R3
        "entry3": entry3_set["entry"],
        "target3": entry3_set["target"],
        "stoploss3": entry3_set["stoploss"],
        "respected_S": respect_counts.get("respected_S"),
        "respected_R": respect_counts.get("respected_R"),
        "respected_S1": respect_counts.get("respected_S1"),
        "respected_R1": respect_counts.get("respected_R1"),
        "respected_S2": respect_counts.get("respected_S2"),
        "respected_R2": respect_counts.get("respected_R2"),
        "respected_S3": respect_counts.get("respected_S3"),
        "respected_R3": respect_counts.get("respected_R3"),
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
        # snapshots
        # "targets": targets_block,  # realized labels (may be pending)
        "predicted_targets": predicted_targets,  # NEW: model forecast
    }

    compare_keys = [
        "support",
        "resistance",
        "support_1d",
        "resistance_1d",
        "support_1w",
        "resistance_1w",
        "support_3m",
        "resistance_3m",
        "sr_range_pct",
        "sr_range_pct_1d",
        "sr_range_pct_1w",
        "sr_range_pct_3m",
        "volume_threshold",
        "bollinger",
        "signal",
        "reason",
        # "targets",
        "predicted_targets",
    ]

    compare_keys += [
        "entry",
        "target",
        "stoploss",
        "entry1",
        "target1",
        "stoploss1",
        "entry2",
        "target2",
        "stoploss2",
        "entry3",
        "target3",
        "stoploss3",
    ]

    changed = any(new_cfg.get(k) != stock_cfg.get(k) for k in compare_keys)

    if not changed:
        return {
            **stock_cfg,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        return {**new_cfg, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
