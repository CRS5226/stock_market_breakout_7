# basic_algo_forecaster.py
import os
import glob
import copy
import math
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Optional, Tuple

from redis_utils import (
    get_recent_candles,  # legacy minute-candle fetch (unused by TF logic but kept)
    get_redis,
    get_recent_candles_tf
)

# if you already added these in redis_utils.py, import them;
# otherwise the function will skip TF-recent blending gracefully.
try:
    from redis_utils import get_recent_indicators_tf
except Exception:

    def get_recent_indicators_tf(*args, **kwargs):
        return []


# -------------------- utils (unchanged) --------------------
# def safe_str(x):
#     try:
#         if pd.isna(x):
#             return None
#     except Exception:
#         pass
#     return str(x)


def safe_number(val, default=0):
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
    for name in candidates:
        lname = str(name).lower()
        for c in row.index:
            if c.lower() == lname and pd.notna(row[c]):
                return row[c]
    return default


# ---- Timezone helpers ----
LOCAL_TZ = "Asia/Kolkata"  # IST


def to_ist(series: pd.Series, keep_tz=True) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(LOCAL_TZ)
    else:
        s = s.dt.tz_localize(LOCAL_TZ)
    if not keep_tz:
        s = s.dt.tz_localize(None)
    return s


# -------------------- S/R + helpers (mostly unchanged) --------------------
# def enrich_with_targets(
#     df: pd.DataFrame,
#     time_col: str = "Timestamp",
#     close_col: str = "Close",
#     horizons: dict | None = None,
#     use_business_days: bool = True,
# ) -> pd.DataFrame:
#     if df is None or df.empty:
#         return df
#     out = df.copy()
#     out[time_col] = to_ist(out[time_col], keep_tz=True)
#     out = out.sort_values(time_col).reset_index(drop=True)
#     if horizons is None:
#         horizons = (
#             {"1d": 1, "1w": 5, "3m": 63}
#             if use_business_days
#             else {"1d": "1D", "1w": "7D", "3m": "90D"}
#         )
#     ref = out[[time_col, close_col]].rename(
#         columns={time_col: "_ref_time", close_col: "_ref_close"}
#     )
#     ref = ref.sort_values("_ref_time")
#     for key, step in horizons.items():
#         target_times = out[[time_col]].copy()
#         if use_business_days:
#             target_times[f"target_time_{key}"] = out[time_col] + pd.offsets.BDay(
#                 int(step)
#             )
#         else:
#             target_times[f"target_time_{key}"] = out[time_col] + pd.to_timedelta(step)
#         merged = pd.merge_asof(
#             target_times.sort_values(f"target_time_{key}"),
#             ref,
#             left_on=f"target_time_{key}",
#             right_on="_ref_time",
#             direction="forward",
#         ).sort_index()
#         out[f"target_close_{key}"] = merged["_ref_close"].values
#         out[f"target_ret_{key}"] = (out[f"target_close_{key}"] / out[close_col]) - 1
#     return out


# def _sr_width_pct(
#     sup: Optional[float], res: Optional[float], base_close: Optional[float]
# ) -> Optional[float]:
#     try:
#         if sup is None or res is None or base_close is None or base_close <= 0:
#             return None
#         return round(((res - sup) / sup) * 100.0, 4)
#     except Exception:
#         return None


def compute_support_resistance(df: pd.DataFrame):
    high_s = _get(df, "High", "high")
    low_s = _get(df, "Low", "low")
    close_s = _get(df, "Close", "close")
    bb_upper_s = _get(df, "BB_Upper", "bb_upper")
    bb_lower_s = _get(df, "BB_Lower", "bb_lower")

    latest = df.iloc[-1]
    high = _last_val(latest, "High", "high", default=None)
    low = _last_val(latest, "Low", "low", default=None)
    close = _last_val(latest, "Close", "close", default=None)

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

    if all(v is not None for v in (high, low, close)):
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
    else:
        r1, s1 = swing_high, swing_low

    bb_upper = safe_number(bb_upper_s.iloc[-1], default=r1)
    bb_lower = safe_number(bb_lower_s.iloc[-1], default=s1)

    res = max([v for v in [r1, bb_upper, swing_high] if v is not None])
    sup = min([v for v in [s1, bb_lower, swing_low] if v is not None])

    return round(sup, 2), round(res, 2), round(bb_upper, 2), round(bb_lower, 2)


# def horizon_sr_from_daily(
#     df_daily: pd.DataFrame, lookback: int, use_quantile: bool = False, q: float = 0.05
# ) -> Tuple[Optional[float], Optional[float]]:
#     if df_daily is None or df_daily.empty:
#         return None, None
#     cols = {c.lower(): c for c in df_daily.columns}
#     H, L, C, U, D = (
#         cols.get("high"),
#         cols.get("low"),
#         cols.get("close"),
#         cols.get("bb_upper"),
#         cols.get("bb_lower"),
#     )
#     tail = df_daily.tail(max(1, lookback)).copy()
#     if H is None or L is None:
#         return None, None
#     if use_quantile:
#         swing_high = float(tail[H].quantile(1.0 - q))
#         swing_low = float(tail[L].quantile(q))
#     else:
#         swing_high = float(tail[H].max())
#         swing_low = float(tail[L].min())
#     r1, s1 = swing_high, swing_low
#     if C is not None and tail[C].notna().any():
#         h, l, c = (
#             float(tail[H].iloc[-1]),
#             float(tail[L].iloc[-1]),
#             float(tail[C].iloc[-1]),
#         )
#         pivot = (h + l + c) / 3.0
#         r1, s1 = 2 * pivot - l, 2 * pivot - h
#     bb_u = float(tail[U].max()) if U and tail[U].notna().any() else None
#     bb_l = float(tail[D].min()) if D and tail[D].notna().any() else None
#     cand_res = [v for v in (r1, swing_high, bb_u) if v is not None]
#     cand_sup = [v for v in (s1, swing_low, bb_l) if v is not None]
#     if not cand_res or not cand_sup:
#         return None, None
#     return max(cand_res), min(cand_sup)


# -------------------- signal logic (unchanged) --------------------
# def determine_signal(
#     latest: pd.Series, support: float, resistance: float, volume_threshold: float
# ):
#     close = _last_val(latest, "Close", "close", default=None)
#     volume = _last_val(latest, "Volume", "volume", default=0)
#     rsi = _last_val(latest, "RSI", "rsi", default=None)
#     ema_fast = _last_val(latest, "EMA_9", "ema_9", "MA_Fast", "ma_fast", default=None)
#     ema_slow = _last_val(latest, "EMA_21", "ema_21", "MA_Slow", "ma_slow", default=None)
#     dist_hh20_bps = _last_val(latest, "dist_hh20_bps", default=None)
#     bb_width_bps = _last_val(latest, "bb_width_bps", default=None)
#     bb_squeeze = _last_val(latest, "bb_squeeze", default=0)
#     ema20_slope = _last_val(latest, "ema20_slope_bps", default=None)
#     ema50_slope = _last_val(latest, "ema50_slope_bps", default=None)
#     adx14 = _last_val(latest, "adx14", "ADX", "adx", default=None)
#     macd_hist_delta = _last_val(latest, "macd_hist_delta", default=None)
#     vwap_diff_bps = _last_val(latest, "vwap_diff_bps", default=None)
#     atr_pct = _last_val(latest, "atr_pct", default=None)
#     vol_z = _last_val(latest, "vol_z", default=None)
#     bb_upper = _last_val(latest, "BB_Upper", "bb_upper", default=None)
#     bb_lower = _last_val(latest, "BB_Lower", "bb_lower", default=None)

#     NEAR_BARRIER_BPS = 15
#     STRONG_ADX = 20
#     VOL_SPIKE_Z = 1.5
#     SLOPE_OK = 2.0
#     MACD_DELTA_OK = 0.0
#     ATR_MIN = 0.25
#     VWAP_NEAR_BPS = 15

#     reasons = []
#     signal = "No Action"
#     if close is None or support is None or resistance is None:
#         return signal, ["Missing core fields"]

#     near_resistance = abs(close - resistance) / resistance * 10000.0 < NEAR_BARRIER_BPS
#     near_support = abs(close - support) / support * 10000.0 < NEAR_BARRIER_BPS
#     volume_spike = (
#         volume is not None
#         and volume_threshold is not None
#         and volume > volume_threshold
#     )
#     adx_good = (adx14 is not None) and (adx14 >= STRONG_ADX)
#     slopes_bull = ((ema20_slope or 0) > SLOPE_OK) and ((ema50_slope or 0) > 0)
#     slopes_bear = ((ema20_slope or 0) < -SLOPE_OK) and ((ema50_slope or 0) < 0)
#     macd_up = (macd_hist_delta is not None) and (macd_hist_delta > MACD_DELTA_OK)
#     macd_down = (macd_hist_delta is not None) and (macd_hist_delta < -MACD_DELTA_OK)
#     atr_ok = (atr_pct is not None) and (atr_pct >= ATR_MIN)
#     vwap_ok_long = (vwap_diff_bps is None) or (
#         -VWAP_NEAR_BPS <= vwap_diff_bps <= VWAP_NEAR_BPS or vwap_diff_bps > 0
#     )
#     vwap_ok_short = (vwap_diff_bps is None) or (
#         -VWAP_NEAR_BPS <= vwap_diff_bps <= VWAP_NEAR_BPS or vwap_diff_bps < 0
#     )
#     bb_conf_long = bb_upper is not None and close > bb_upper
#     bb_conf_short = bb_lower is not None and close < bb_lower

#     if near_resistance and volume_spike:
#         reasons.append(
#             f"Price near resistance {round(resistance,2)} with volume spike."
#         )
#         if adx_good:
#             reasons.append(f"ADX {round(adx14,1)} strong.")
#         if slopes_bull:
#             reasons.append("EMA20/50 slopes positive.")
#         if macd_up:
#             reasons.append("MACD histogram rising.")
#         if bb_squeeze:
#             reasons.append("BB squeeze → potential expansion.")
#         if bb_conf_long:
#             reasons.append("Price above upper BB.")
#         if atr_ok:
#             reasons.append(f"ATR {round(atr_pct,2)}% supports move.")
#         if vwap_ok_long:
#             reasons.append("VWAP alignment OK.")
#         strong_checks = sum([adx_good, slopes_bull, macd_up, bb_conf_long, atr_ok])
#         signal = "Strong Breakout" if strong_checks >= 3 else "Potential Breakout"

#     elif near_support and volume_spike:
#         reasons.append(f"Price near support {round(support,2)} with volume spike.")
#         if adx_good:
#             reasons.append(f"ADX {round(adx14,1)} strong.")
#         if slopes_bear:
#             reasons.append("EMA20/50 slopes negative.")
#         if macd_down:
#             reasons.append("MACD histogram falling.")
#         if bb_squeeze:
#             reasons.append("BB squeeze → potential expansion.")
#         if bb_conf_short:
#             reasons.append("Price below lower BB.")
#         if atr_ok:
#             reasons.append(f"ATR {round(atr_pct,2)}% supports move.")
#         if vwap_ok_short:
#             reasons.append("VWAP alignment OK.")
#         strong_checks = sum([adx_good, slopes_bear, macd_down, bb_conf_short, atr_ok])
#         signal = "Strong Breakdown" if strong_checks >= 3 else "Potential Breakdown"

#     return signal, reasons


def determine_signal(
    latest: pd.Series, support: float, resistance: float, volume_threshold: float
):
    close = _last_val(latest, "Close", "close", default=None)
    volume = _last_val(latest, "Volume", "volume", default=0)
    ema20_slope = _last_val(latest, "ema20_slope_bps", default=None)
    ema50_slope = _last_val(latest, "ema50_slope_bps", default=None)
    adx14 = _last_val(latest, "adx14", "ADX", "adx", default=None)
    macd_hist_delta = _last_val(latest, "macd_hist_delta", default=None)
    bb_squeeze = _last_val(latest, "bb_squeeze", default=0)
    vwap_diff_bps = _last_val(latest, "vwap_diff_bps", default=None)
    atr_pct = _last_val(latest, "atr_pct", default=None)
    bb_upper = _last_val(latest, "BB_Upper", "bb_upper", default=None)

    NEAR_BARRIER_BPS = 15
    STRONG_ADX = 20
    SLOPE_OK = 2.0
    MACD_DELTA_OK = 0.0
    ATR_MIN = 0.25
    VWAP_NEAR_BPS = 15

    reasons = []
    if close is None or support is None or resistance is None:
        return "No Action", ["Missing core fields"]

    # Long-window guard: only consider signals if S < close < R
    if not (support < close < resistance):
        return "No Action", [
            f"Invalid long window: support={support}, close={close}, resistance={resistance}"
        ]

    near_resistance = abs(close - resistance) / resistance * 10000.0 < NEAR_BARRIER_BPS
    near_support = abs(close - support) / support * 10000.0 < NEAR_BARRIER_BPS

    volume_spike = (
        volume is not None
        and volume_threshold is not None
        and volume > volume_threshold
    )
    adx_good = (adx14 is not None) and (adx14 >= STRONG_ADX)
    slopes_bull = ((ema20_slope or 0) > SLOPE_OK) and ((ema50_slope or 0) > 0)
    macd_up = (macd_hist_delta is not None) and (macd_hist_delta > MACD_DELTA_OK)
    atr_ok = (atr_pct is not None) and (atr_pct >= ATR_MIN)
    vwap_ok = (vwap_diff_bps is None) or (
        -VWAP_NEAR_BPS <= vwap_diff_bps <= VWAP_NEAR_BPS or vwap_diff_bps > 0
    )
    bb_conf = bb_upper is not None and close > bb_upper

    # Long-only outcomes
    if near_resistance and volume_spike:
        if adx_good:
            reasons.append(f"ADX {round(adx14,1)} strong.")
        if slopes_bull:
            reasons.append("EMA20/50 slopes positive.")
        if macd_up:
            reasons.append("MACD histogram rising.")
        if bb_squeeze:
            reasons.append("BB squeeze → expansion likely.")
        if bb_conf:
            reasons.append("Price above upper BB.")
        if atr_ok:
            reasons.append(f"ATR {round(atr_pct,2)}% supports move.")
        if vwap_ok:
            reasons.append("VWAP alignment OK.")
        strong_checks = sum([adx_good, slopes_bull, macd_up, bb_conf, atr_ok])
        return (
            "Strong Breakout" if strong_checks >= 3 else "Potential Breakout",
            reasons,
        )

    if near_support and volume_spike:
        if adx_good:
            reasons.append(f"ADX {round(adx14,1)} strong.")
        if slopes_bull:
            reasons.append("EMA20/50 slopes positive.")
        if macd_up:
            reasons.append("MACD histogram rising.")
        if bb_squeeze:
            reasons.append("BB squeeze → expansion likely.")
        if atr_ok:
            reasons.append(f"ATR {round(atr_pct,2)}% supports move.")
        if vwap_ok:
            reasons.append("VWAP alignment OK.")
        return "Potential Long Near Support", reasons

    return "No Action", reasons or ["No bullish setup"]


# predictor (unchanged)
def _get_row(row: pd.Series, *names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
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


def _strict_long_chain_ok(support, resistance, cp, entry, target, stop, eps=1e-9):
    if any(v is None for v in (support, resistance, cp, entry, target, stop)):
        return False
    return (
        stop < support - eps
        and support < cp - eps
        and cp < entry - eps
        and entry < target - eps
        and target < resistance - eps
    )


def predict_forward_returns(df_all: pd.DataFrame, horizons=("1d", "1w", "3m")) -> dict:
    preds = {h: {"ret": 0.0, "price": None, "explain": ["no data"]} for h in horizons}
    if df_all is None or df_all.empty:
        return preds
    last = df_all.iloc[-1]
    P_t = _get_row(last, "Close", "close")
    if P_t is None or pd.isna(P_t):
        return preds

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

    adx_val = _safe(adx)
    s20_val = _safe(s20)
    s50_val = _safe(s50)
    trending_up = (adx_val >= 20) and (s20_val > 0) and (s50_val > 0)
    trending_down = (adx_val >= 20) and (s20_val < 0) and (s50_val < 0)
    choppy = adx_val < 15

    base = 0.0
    explain = []
    base += _safe(mom) * 100.0
    explain.append(f"mom={_safe(mom):.5f}")
    if mhd is not None and not pd.isna(mhd):
        base += _safe(mhd) * 10.0
        explain.append(f"macdΔ*10={_safe(mhd):.3f}")
    base += 0.5 * s20_val + 0.25 * s50_val
    explain.append(f"s20*0.5+s50*0.25={0.5*s20_val+0.25*s50_val:.2f}bps")
    if vwapd is not None:
        base += 0.10 * _safe(vwapd)
        explain.append(f"VWAP tilt={0.10*_safe(vwapd):.2f}bps")
    if distH is not None:
        drag = max(0.0, 15.0 - abs(_safe(distH)))
        base -= drag
        explain.append(f"HH20 drag=-{drag:.2f}bps")
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

    atrp_val = _safe(atrp)
    scale = 1.0
    if atrp_val < 0.25:
        scale *= 0.5
        explain.append("low ATR% → x0.5")
    elif atrp_val > 1.0:
        scale *= 1.2
        explain.append("high ATR% → x1.2")
    base *= scale
    if choppy:
        base *= 0.6
        explain.append("choppy regime → x0.6")

    cap_bps = max(20.0, _safe(vol, 0.01) * 10000.0 * 0.4)
    base = float(np.clip(base, -cap_bps, cap_bps))
    explain.append(f"cap±{cap_bps:.1f}bps → base={base:.1f}bps")
    scalers = {"1d": 1.0, "1w": np.sqrt(5.0), "3m": 5.0 * np.sqrt(5.0)}
    out = {}
    for h in horizons:
        bps = base * scalers[h]
        r_hat = float(bps / 10000.0)
        hcap = max(0.03, _safe(vol, 0.01) * {"1d": 3, "1w": 6, "3m": 12}[h])
        r_hat = float(np.clip(r_hat, -hcap, hcap))
        out[h] = {
            "ret": r_hat,
            "price": float(P_t * (1.0 + r_hat)),
            "explain": explain + [f"{h}: {bps:.1f}bps→r={r_hat:.4f}"],
        }
    return out


# ---------------------- entry/target/stoploss (tweaked) ----------------------
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
    for n in names:
        if n in row and pd.notna(row[n]):
            return _safe_float(row[n], default)
        ln = str(n).lower()
        for c in row.index:
            if c.lower() == ln and pd.notna(row[c]):
                return _safe_float(row[c], default)
    return default


# def _get_floor_pivots_from_last_daily(df_daily: pd.DataFrame):
#     if df_daily is None or df_daily.empty:
#         return None
#     cols = {c.lower(): c for c in df_daily.columns}
#     H, L, C = cols.get("high"), cols.get("low"), cols.get("close")
#     if not (H and L and C):
#         return None
#     last = df_daily.dropna(subset=[H, L, C]).tail(1)
#     if last.empty:
#         return None
#     h, l, c = float(last[H].iloc[-1]), float(last[L].iloc[-1]), float(last[C].iloc[-1])
#     P = (h + l + c) / 3.0
#     R1, S1 = 2 * P - l, 2 * P - h
#     R2, S2 = P + (h - l), P - (h - l)
#     R3, S3 = h + 2 * (P - l), l - 2 * (h - P)
#     return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2, "R3": R3, "S3": S3}


def _ensure_sr_order(support, resistance):
    if support is None or resistance is None:
        return support, resistance
    if support > resistance:
        return resistance, support
    return support, resistance


def _sanitize_levels(
    entry,
    target,
    stoploss,
    side,
    support,
    resistance,
    buf_frac,
    atr_frac,
    current_price=None,
):
    """Enforce strict long-only ordering:
    stoploss < support < current_price < entry < target < resistance
    """
    tiny = max(atr_frac, 0.0008)  # ≥ 8 bps
    if side == "long":
        if resistance is not None:
            target = (
                min(target, resistance * (1.0 - buf_frac))
                if target is not None
                else resistance * (1.0 - buf_frac)
            )
            entry = (
                min(entry, resistance * (1.0 - buf_frac))
                if entry is not None
                else resistance * (1.0 - 2 * buf_frac)
            )
        if support is not None:
            entry = (
                max(entry, support * (1.0 + buf_frac))
                if entry is not None
                else support * (1.0 + buf_frac)
            )
            lo_stop = support * (1.0 - 1.25 * atr_frac)
            hi_stop = support * (1.0 - tiny)
            stoploss = max(
                lo_stop, min(hi_stop, stoploss if stoploss is not None else lo_stop)
            )

        # enforce current_price < entry strictly
        if current_price is not None and entry is not None and entry <= current_price:
            entry = current_price * (1.0 + tiny)

        # strict ordering: stop < support < cp < entry < target < resistance
        if target is not None and entry is not None and target <= entry:
            target = entry * (1.0 + tiny)
        if stoploss is not None and support is not None and stoploss >= support:
            stoploss = support * (1.0 - tiny)
        if entry is not None and support is not None and entry <= support:
            entry = support * (1.0 + tiny)

    return _round3(entry), _round3(target), _round3(stoploss)


def _horizon_params(horizon: str, atr_pct_fallback=0.35):
    horizon = (horizon or "1d").lower()
    if horizon in (
        "1min",
        "5min",
        "15min",
        "30min",
        "45min",
        "1hour",
        "4hour",
        "1d",
        "1day",
    ):
        return {
            "buf_min_bps": 12,
            "buf_max_bps": 45,
            "stop_mult": 1.1,
            "rr_normal": 1.15,
            "rr_strong": 1.6,
            "atr_pct_fallback": atr_pct_fallback,
        }
    if horizon in ("1w", "weekly"):
        return {
            "buf_min_bps": 15,
            "buf_max_bps": 60,
            "stop_mult": 1.2,
            "rr_normal": 1.25,
            "rr_strong": 1.8,
            "atr_pct_fallback": atr_pct_fallback,
        }
    return {
        "buf_min_bps": 18,
        "buf_max_bps": 75,
        "stop_mult": 1.35,
        "rr_normal": 1.35,
        "rr_strong": 2.0,
        "atr_pct_fallback": atr_pct_fallback,
    }


def _compute_pair_levels(
    support,
    resistance,
    current_price,
    horizon,
    adx_val,
    atr_pct,
    pred_1d_ret,
    pred_1w_ret,
    signal,
    structural_floors=None,
    mode="range",
):
    out = {"entry": None, "target": None, "stoploss": None}
    support, resistance = _ensure_sr_order(support, resistance)
    if current_price is None or (support is None or resistance is None):
        return out

    params = _horizon_params(horizon)
    adx_val = 15.0 if adx_val is None or adx_val <= 0 else float(adx_val)
    atr_pct = (
        params["atr_pct_fallback"]
        if (atr_pct is None or atr_pct <= 0)
        else float(atr_pct)
    )
    atr_frac = max(0.001, atr_pct / 100.0)

    buf_bps = max(
        params["buf_min_bps"], min(params["buf_max_bps"], 0.75 * atr_frac * 10000.0)
    )
    buf_frac = buf_bps / 10000.0

    stop_mult = params["stop_mult"]
    strong = "strong" in (signal or "").lower()
    rr = params["rr_strong"] if strong else params["rr_normal"]

    P = structural_floors.get("P") if structural_floors else None
    S1 = structural_floors.get("S1") if structural_floors else None

    entry = max(current_price * (1.0 + 1e-6), support * (1.0 + buf_frac))
    cand_stops = [support * (1.0 - stop_mult * atr_frac), S1, P]
    cand_stops = [x for x in cand_stops if x is not None]
    stoploss = max(cand_stops) if cand_stops else support * (1.0 - stop_mult * atr_frac)
    risk = max(0.01, entry - stoploss)
    target_rr = entry + rr * risk
    target_cap = resistance * (1.0 - buf_frac)
    target = min(target_rr, target_cap)

    entry, target, stoploss = _sanitize_levels(
        entry,
        target,
        stoploss,
        "long",
        support,
        resistance,
        buf_frac,
        atr_frac,
        current_price=current_price,
    )

    # hard clamp just in case:
    # if entry is not None and cp is not None:
    #     entry = max(entry, cp * (1.0 + 1e-6))
    # if target is not None and entry is not None:
    #     target = max(target, entry * (1.0 + 1e-6))
    # if stoploss is not None and support is not None:
    #     stoploss = min(stoploss, support * (1.0 - 1e-6))

    return out | {"entry": entry, "target": target, "stoploss": stoploss}


# -------------------- respect counts (unchanged) --------------------
def _colmap(df):
    cols = {c.lower(): c for c in df.columns}
    return cols.get("high"), cols.get("low"), cols.get("close")


def _auto_tol_frac(
    df_tail, atr_col="atr_pct", default_pct=0.35, scale=0.25, min_bps=8, max_bps=60
):
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
    if level is None or df is None or df.empty:
        return 0
    H, L, C = _colmap(df)
    if not (H and L and C):
        return 0
    cnt, last_i = 0, -(10**9)
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


def _enforce_long_invariants(
    levels: dict, support: float, resistance: float, current_price: float | None
):
    """Ensure strict long-only ordering:
    stoploss < support < current_price < entry < target < resistance
    """
    entry = levels.get("entry")
    target = levels.get("target")
    stop = levels.get("stoploss")
    if support is None or resistance is None:
        return levels
    cp = current_price

    # tiny = 8 bps min
    tiny = 0.0008

    # current_price < entry
    if cp is not None and entry is not None and entry <= cp:
        entry = cp * (1.0 + tiny)

    # entry > support
    if entry is not None and entry <= support:
        entry = support * (1.0 + tiny)

    # target < resistance and > entry
    if target is None:
        target = entry * (1.0 + tiny) if entry is not None else None
    if target is not None and target >= resistance:
        target = resistance * (1.0 - tiny)
    if entry is not None and target is not None and target <= entry:
        target = entry * (1.0 + tiny)

    # stop < support
    if stop is None:
        stop = support * (1.0 - tiny)
    if stop >= support:
        stop = support * (1.0 - tiny)

    return {
        "entry": _round3(entry),
        "target": _round3(target),
        "stoploss": _round3(stop),
    }


# --------------------------- helpers ---------------------------
# --- NEW: file-based TF history discovery ---
HIST_ROOTS_DEFAULT = [
    os.path.join("historical_data_candles"),
    os.path.join("p17", "historical_data_candles"),
]

# map TF → subfolder and filename token used in your files
TF_FILE_MAP = {
    "1min": ("1min", "1min"),
    "5min": ("5min", "5min"),
    "15min": ("15min", "15min"),
    "30min": ("30min", "30min"),
    "45min": ("45min", "45min"),
    "1hour": ("1hour", "1hour"),
    "4hour": ("4hour", "4hour"),
    "1day": ("1day", "1day"),  # daily
    "1month": ("1day", "1day"),  # we'll resample this from daily later
}


def _glob_hist_csv(
    stock_code: str, tf: str, roots: list[str] | None = None
) -> Optional[str]:
    """
    Find the latest CSV for stock_code & tf under given roots.
    We match files like ANANDRATHI_1day_YYYY-MM-DD_to_YYYY-MM-DD.csv.
    """
    roots = roots or HIST_ROOTS_DEFAULT
    if tf not in TF_FILE_MAP:
        return None
    subdir, token = TF_FILE_MAP[tf]
    patterns = []
    for root in roots:
        base = os.path.join(root, subdir)
        # windows/backslash-safe glob
        patterns.append(os.path.join(base, f"{stock_code}_{token}_*.csv"))
    # collect all matches
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    if not matches:
        return None
    # pick the most recent by mtime (robust) instead of parsing dates
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def _load_hist_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common columns
    cols = {c.lower(): c for c in df.columns}
    # Timestamp column
    if "timestamp" not in cols:
        for cand in ("date", "datetime", "time", "minute"):
            if cand in cols:
                df = df.rename(columns={cols[cand]: "Timestamp"})
                break
    else:
        df = df.rename(columns={cols["timestamp"]: "Timestamp"})

    # Core OHLCV
    renames = {}
    for raw, std in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ]:
        if raw in cols and std not in df.columns:
            renames[cols[raw]] = std
    if renames:
        df = df.rename(columns=renames)

    if "Timestamp" in df.columns:
        df["Timestamp"] = to_ist(df["Timestamp"], keep_tz=True)
        df = (
            df.dropna(subset=["Timestamp"])
            .drop_duplicates(subset=["Timestamp"])
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )
    return df


# -------------------- multi-timeframe SR + levels --------------------

# Map the user plan to TF keys and output suffixes
# base (1min) uses un-suffixed fields: support/resistance, entry/target/stoploss
TF_PLAN = [
    ("1min", 0, "1d"),  # base, in practice intra-day horizon
    ("5min", 1, "1d"),
    ("15min", 2, "1d"),
    ("30min", 3, "1d"),
    ("45min", 4, "1d"),
    ("1hour", 5, "1w"),
    ("4hour", 6, "1w"),
    ("1day", 7, "1w"),  # S7/R7
    ("1month", 8, "3m"),  # S8/R8
]


def _resample_daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or df_daily.empty or "Timestamp" not in df_daily.columns:
        return pd.DataFrame()
    df = df_daily.copy()
    df = df.set_index(pd.to_datetime(df["Timestamp"]))
    agg = {}
    for k, v in [
        ("Open", "first"),
        ("High", "max"),
        ("Low", "min"),
        ("Close", "last"),
        ("Volume", "sum"),
        ("atr_pct", "last"),
        ("BB_Upper", "last"),
        ("BB_Lower", "last"),
        ("HH20", "last"),
        ("LL20", "last"),
    ]:
        if k in df.columns:
            agg[k] = v
    out = df.resample("M").agg(agg).dropna(how="all")
    if out.empty:
        return pd.DataFrame()
    out = out.reset_index().rename(columns={"index": "Timestamp"})
    return out


def _tf_recent_from_redis(stock_code: str, tf: str, n: int = 200) -> pd.DataFrame:
    """Best-effort recent fetch per TF: indicators first; fallback to candles; return chronological df."""
    try:
        r = get_redis()
    except Exception:
        r = None

    rows = []
    if r is not None:
        # 1) indicators
        try:
            rows = get_recent_indicators_tf(r, stock_code, tf, n=n) or []
        except Exception:
            rows = []

        # 2) fallback to candles if indicators missing
        if not rows:
            try:
                # if you have this utility; otherwise remove this block
                raw = get_recent_candles_tf(r, stock_code, tf, n=n) or []
                # normalize candle rows to indicator-like dicts
                rows = raw
            except Exception:
                rows = []

    if not rows:
        print(f"[ℹ️ TF fetch] {stock_code} {tf}: no Redis rows (indicators/candles).")
        return pd.DataFrame()

    df = pd.DataFrame(reversed(rows))  # chronological

    # ---- normalize columns ----
    # Timestamp
    if "Timestamp" not in df.columns:
        for c in ("minute", "timestamp", "datetime", "time"):
            if c in df.columns:
                df["Timestamp"] = pd.to_datetime(df[c], errors="coerce")
                break
    if "Timestamp" not in df.columns:
        print(f"[⚠️] {stock_code} {tf}: no time column among {list(df.columns)}")
        df["Timestamp"] = pd.NaT

    # OHLCV standardization
    cols = {c.lower(): c for c in df.columns}
    for a, b in [
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ]:
        if a in cols and b not in df.columns:
            df[b] = df[cols[a]]

    # final clean
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        print(
            f"[ℹ️ TF fetch] {stock_code} {tf}: rows present but no valid Timestamp after parsing."
        )
    return df


def _tf_hist_source(
    df_hist_daily: pd.DataFrame,
    tf: str,
    stock_code: str,
    hist_roots: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prefer per-TF CSV from disk; for 1month we resample daily.
    If no TF CSV exists, fall back to df_hist_daily where sensible.
    """
    # daily as baseline
    if tf == "1day":
        # daily history from disk if available, else df_hist_daily
        p = _glob_hist_csv(stock_code, "1day", hist_roots)
        if p:
            return _load_hist_csv(p)
        return df_hist_daily.copy() if df_hist_daily is not None else pd.DataFrame()

    if tf == "1month":
        # build from daily
        daily = _tf_hist_source(df_hist_daily, "1day", stock_code, hist_roots)
        return _resample_daily_to_monthly(daily)

    # intraday buckets
    p = _glob_hist_csv(stock_code, tf, hist_roots)
    if p:
        return _load_hist_csv(p)

    # no intraday CSV → fall back to daily just to derive rough SR
    return df_hist_daily.copy() if df_hist_daily is not None else pd.DataFrame()


def _compute_sr_for_df(
    df_tf: pd.DataFrame,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if df_tf is None or df_tf.empty:
        return None, None, None, None
    try:
        s, r, u, l = compute_support_resistance(df_tf)
        return s, r, u, l
    except Exception:
        return None, None, None, None


def _atr_from_df(df_tf: pd.DataFrame, fallback_pct: float = 0.35) -> float:
    if df_tf is None or df_tf.empty:
        return fallback_pct
    cols = {c.lower(): c for c in df_tf.columns}
    A = cols.get("atr_pct")
    if A and df_tf[A].notna().any():
        try:
            return float(df_tf[A].iloc[-1])
        except Exception:
            pass
    return fallback_pct


def _adx_from_df(df_tf: pd.DataFrame, fallback: float = 15.0) -> float:
    if df_tf is None or df_tf.empty:
        return fallback
    for name in ("adx14", "ADX", "adx"):
        if name in df_tf.columns and df_tf[name].notna().any():
            try:
                return float(df_tf[name].iloc[-1])
            except Exception:
                continue
    return fallback


def _close_from_df(df_tf: pd.DataFrame) -> Optional[float]:
    if df_tf is None or df_tf.empty:
        return None
    for name in ("Close", "close"):
        if name in df_tf.columns and df_tf[name].notna().any():
            try:
                return float(df_tf[name].iloc[-1])
            except Exception:
                continue
    return None


def _respect_counts_for_df(
    df_tf: pd.DataFrame, support: float, resistance: float
) -> dict:
    """Count respects on the TF dataframe itself (same logic, ATR-based tolerance)."""
    if df_tf is None or df_tf.empty or support is None or resistance is None:
        return {"S": 0, "R": 0}
    tol = _auto_tol_frac(
        df_tf.tail(180),
        atr_col="atr_pct",
        default_pct=0.35,
        scale=0.25,
        min_bps=8,
        max_bps=60,
    )
    s_cnt = _count_respects(df_tf, support, "support", tol_frac=tol, min_sep=2)
    r_cnt = _count_respects(df_tf, resistance, "resistance", tol_frac=tol, min_sep=2)
    return {"S": s_cnt, "R": r_cnt}


def _rectify_sr_window(
    support, resistance, cp, atr_pct, *, min_bps=8, max_bps=60, scale=0.25, pad_mult=1.2
):
    """
    If S/R are on the wrong side of close (or inverted), expand them a tiny ATR-based
    buffer around close so S < close < R. Keeps changes minimal.
    """
    if cp is None:
        return support, resistance

    # derive a small fractional pad from ATR%
    try:
        atr_pct = float(atr_pct) if atr_pct is not None else 0.35
    except Exception:
        atr_pct = 0.35
    frac = max(min_bps / 10000.0, min(max_bps / 10000.0, (atr_pct / 100.0) * scale))

    s, r = support, resistance
    bad = (s is None) or (r is None) or (s >= r) or not (s < cp < r)
    if bad:
        s = cp * (1.0 - pad_mult * frac)
        r = cp * (1.0 + pad_mult * frac)
    return s, r


def build_all_timeframe_levels(
    stock_code: str,
    df_hist_daily: pd.DataFrame,
    df_recent_base: pd.DataFrame,
    current_px_hint: Optional[float],
    signal: str,
    hist_roots: list[str] | None = None,
) -> dict:
    out = {}
    pred_1d_ret = 0.0
    pred_1w_ret = 0.0

    tf_recent_map = {
        tf: _tf_recent_from_redis(stock_code, tf, n=200) for tf, _, _ in TF_PLAN
    }

    def do_one(tf: str, idx: int, horizon: str):
        nonlocal out
        df_recent_tf = tf_recent_map.get(tf)
        if df_recent_tf is None:
            df_recent_tf = pd.DataFrame()

        df_hist_tf = _tf_hist_source(df_hist_daily, tf, stock_code, hist_roots)

        # prefer recent TF (Redis); else historical CSV
        df_tf = (
            df_recent_tf
            if (isinstance(df_recent_tf, pd.DataFrame) and not df_recent_tf.empty)
            else df_hist_tf
        )
        if df_tf is None or df_tf.empty:
            print(f"[ℹ️ SR skip] {stock_code} {tf}: no recent or hist TF data.")
            return

        sup, res, _, _ = _compute_sr_for_df(df_tf)
        sup, res = _ensure_sr_order(sup, res)
        if sup is None or res is None:
            print(f"[ℹ️ SR skip] {stock_code} {tf}: could not compute S/R.")
            return

        cp = _close_from_df(df_tf) or current_px_hint
        atr = _atr_from_df(df_tf, fallback_pct=0.35)
        adx = _adx_from_df(df_tf, fallback=15.0)

        lv = _compute_pair_levels(
            support=sup,
            resistance=res,
            current_price=cp,
            horizon=horizon,
            adx_val=adx,
            atr_pct=atr,
            pred_1d_ret=pred_1d_ret,
            pred_1w_ret=pred_1w_ret,
            signal=signal,
            structural_floors=None,
            mode="range",
        )
        lv = _enforce_long_invariants(lv, sup, res, cp)

        # if not _strict_long_chain_ok(
        #     sup, res, cp, lv.get("entry"), lv.get("target"), lv.get("stoploss")
        # ):
        #     print(
        #         f"[⏭️ Long-chain fail] {stock_code} {tf}: S/R vs cp violates long-only chain; skipping TF."
        #     )
        #     return

        rc = _respect_counts_for_df(df_tf.tail(400), sup, res)

        if idx == 0:
            out["support"] = safe_number(sup)
            out["resistance"] = safe_number(res)
            out["entry"] = lv["entry"]
            out["target"] = lv["target"]
            out["stoploss"] = lv["stoploss"]
            out["respected_S"] = rc["S"]
            out["respected_R"] = rc["R"]
        else:
            out[f"support{idx}"] = safe_number(sup)
            out[f"resistance{idx}"] = safe_number(res)
            out[f"entry{idx}"] = lv["entry"]
            out[f"target{idx}"] = lv["target"]
            out[f"stoploss{idx}"] = lv["stoploss"]
            out[f"respected_S{idx}"] = rc["S"]
            out[f"respected_R{idx}"] = rc["R"]

    for tf, idx, horizon in TF_PLAN:
        do_one(tf, idx, horizon)

    return out


# ---------- Completeness gate helpers (put these near your other utils) ----------
REQUIRED_TF_INDEXES = list(range(0, 9))  # TF0..TF8


def _has_val(x) -> bool:
    return x is not None and str(x) != ""


def _tf_keys(i: int):
    # TF0 uses unsuffixed keys; TF1..8 use suffixed keys
    if i == 0:
        return ("support", "resistance", "entry", "target", "stoploss")
    return (f"support{i}", f"resistance{i}", f"entry{i}", f"target{i}", f"stoploss{i}")


def _all_tf_complete(
    cfg: dict, require_triplets: bool = True
) -> tuple[bool, list[str]]:
    """
    Returns (ok, missing_list). missing_list contains human-friendly hints per TF.
    If require_triplets=False, only S/R are required for each TF.
    """
    missing = []
    for i in REQUIRED_TF_INDEXES:
        s, r, e, t, sl = _tf_keys(i)
        if not (_has_val(cfg.get(s)) and _has_val(cfg.get(r))):
            missing.append(f"TF{i}: {s}/{r}")
            continue
        if require_triplets:
            if not (
                _has_val(cfg.get(e)) and _has_val(cfg.get(t)) and _has_val(cfg.get(sl))
            ):
                missing.append(f"TF{i}: {e}/{t}/{sl}")
    return (len(missing) == 0, missing)


def fetch_candles(stock_code: str, n: int = 2):
    """
    Fetch last N 1-minute candles (chronological).
    Prefers TF pipeline (candles:{code}:1min via get_recent_candles_tf).
    Falls back to legacy get_recent_candles if TF store is empty.
    """
    r = get_redis()

    rows = []
    # --- 1) Prefer TF list (newest first) ---
    try:
        # newest first per your helper; may return [] if no finalized bar yet
        tf_items = get_recent_candles_tf(r, stock_code, "1min", n=n)
        if tf_items:
            rows = list(reversed(tf_items))  # chronological: oldest -> newest
    except Exception as e:
        print(f"[⚠️] TF fetch error for {stock_code}: {e}")

    # --- 2) Fallback: legacy list (unknown freshness) ---
    if not rows:
        try:
            legacy = get_recent_candles(r, stock_code, n)
            if legacy:
                rows = legacy  # legacy assumed already chronological; if not, sort below
        except Exception as e:
            print(f"[⚠️] Legacy fetch error for {stock_code}: {e}")

    if not rows:
        print(f"[⚠️] No candles found (TF/legacy) for {stock_code}")
        return None

    # --- 3) Normalize columns & sort chronologically ---
    df = pd.DataFrame(rows)

    # unify time column → "minute"
    if "minute" not in df.columns:
        if "bucket" in df.columns:  # TF snapshot uses bucket start time
            df["minute"] = df["bucket"]
        elif "timestamp" in df.columns:
            df["minute"] = df["timestamp"]
        elif "datetime" in df.columns:
            df["minute"] = df["datetime"]

    # ensure datetime and sort
    if "minute" in df.columns:
        df["minute"] = pd.to_datetime(df["minute"], errors="coerce")
        df = df.dropna(subset=["minute"]).sort_values("minute").reset_index(drop=True)
    else:
        # last resort: just keep as-is but at least cap to last n and stable-sort
        df = df.tail(n).reset_index(drop=True)

    # normalize OHLCV field names (some TF/legacy payloads already match)
    def _copy_col(src, dst):
        nonlocal df
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    for src, dst in [("open", "open"), ("high", "high"), ("low", "low"),
                     ("close", "close"), ("volume", "volume")]:
        _copy_col(src, dst)

    # type coercion (safe)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    return df.tail(n).reset_index(drop=True)



def basic_forecast_update(
    stock_cfg,
    recent_df,
    historical_folder="historical_data",
    *,
    require_all_tf_triplets: bool = True,  # <— NEW: require entry/target/stoploss too
    verbose: bool = True,  # <— NEW: console hints about gating
):
    """
    Builds a *minimal, modern* cfg consisting of ONLY the multi-timeframe SR/levels
    (S0/R0 .. S8/R8 with entry/target/stoploss + respected_S/ respected_R per TF),
    plus a small context payload (ohlcv snapshot, signal/reason, preds, volume_threshold).

    If any timeframe is incomplete (per `require_all_tf_triplets`), this function will
    return the original `stock_cfg` UNCHANGED, so upstream code won’t publish/log
    partial rows.
    """
    stock_cfg = copy.deepcopy(stock_cfg)
    stock_code = stock_cfg.get("stock_code")

    # ---------------- 1) Load DAILY historical (for pivots / long windows) ----------------
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
            cols = {c.lower(): c for c in df_hist.columns}
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

    # ---------------- 2) Normalize recent (Redis) ----------------
    df_recent = (
        recent_df.copy()
        if (recent_df is not None and not recent_df.empty)
        else pd.DataFrame()
    )
    if not df_recent.empty:
        df_recent = df_recent.reset_index(drop=True)
        if "Timestamp" in df_recent.columns:
            df_recent["Timestamp"] = to_ist(df_recent["Timestamp"], keep_tz=True)
            df_recent = df_recent.drop_duplicates(subset=["Timestamp"])

    # Combined (used for preds / signal fallbacks)
    df_all = (
        pd.concat([df_hist, df_recent], ignore_index=True)
        if not df_hist.empty
        else df_recent.copy()
    )
    df_all = (
        df_all.sort_values("Timestamp").reset_index(drop=True)
        if not df_all.empty
        else df_all
    )

    # ---------------- 3) Context: volume threshold, signal, predictions ----------------
    vol_series = _get(df_recent, "Volume", "volume", default=0)
    volume_threshold = (
        float(vol_series.tail(50).mean() * 1.5) if len(vol_series) else 0.0
    )

    # Fallback SR from combined df to compute signal if needed
    if df_all is not None and not df_all.empty:
        tmp_sup, tmp_res, _, _ = compute_support_resistance(df_all)
    else:
        tmp_sup, tmp_res = None, None

    latest = (
        df_recent.iloc[-1]
        if (df_recent is not None and not df_recent.empty)
        else pd.Series(dtype="float64")
    )
    if not df_recent.empty and tmp_sup is not None and tmp_res is not None:
        signal, reasons = determine_signal(latest, tmp_sup, tmp_res, volume_threshold)
    else:
        signal, reasons = "No Action", ["Insufficient recent data"]

    # Predictions for UI/logic
    preds = (
        predict_forward_returns(df_all, horizons=("1d", "1w", "3m"))
        if (df_all is not None and not df_all.empty)
        else {
            "1d": {"ret": 0.0, "price": None},
            "1w": {"ret": 0.0, "price": None},
            "3m": {"ret": 0.0, "price": None},
        }
    )
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

    # ---------------- 4) Current OHLVC snapshot (from Redis minute candles) ----------------
    last_candles = fetch_candles(stock_code, n=1)  # chronological df or None
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
        current_price = (
            float(latest.get("Close"))
            if "Close" in latest
            else (float(latest.get("close")) if "close" in latest else None)
        )

    # ---------------- 5) Build the multi-timeframe SR+levels ONLY ----------------
    multi_tf = build_all_timeframe_levels(
        stock_code=stock_code,
        df_hist_daily=df_hist,
        df_recent_base=df_recent,
        current_px_hint=current_price,
        signal=signal,
    )

    # Early gate on completeness (before assembling new_cfg)
    # We check on a temp dict that only contains TF stuff.
    # If incomplete, we return the original config untouched (skip publish/log this cycle).
    tf_only = dict(multi_tf)  # shallow copy
    ok, missing = _all_tf_complete(tf_only, require_triplets=require_all_tf_triplets)
    if not ok:
        if verbose:
            print(
                f"[⏳] {stock_code}: TFs incomplete → {', '.join(missing)} (skipping update)"
            )
        return stock_cfg  # unchanged => upstream equality check prevents publish

    # ---------------- 6) Final cfg (ONLY new, multi-TF world) ----------------
    new_cfg = {
        "stock_code": stock_cfg.get("stock_code"),
        "instrument_token": stock_cfg.get("instrument_token"),
        "ohlcv": ohlcv_block,
        "signal": signal,
        "reason": reasons,
        "forecast": "basic_algo",
        "predicted_targets": predicted_targets,
        "volume_threshold": int(safe_number(volume_threshold)),
    }
    new_cfg.update(multi_tf)  # attach the full multi-timeframe bundle

    # Optional: compute legacy width for base S/R (handy for sheets)
    try:
        if new_cfg.get("support") is not None and new_cfg.get("resistance") is not None:
            new_cfg["sr_range_pct"] = round(
                (
                    (float(new_cfg["resistance"]) - float(new_cfg["support"]))
                    / float(new_cfg["support"])
                )
                * 100.0,
                4,
            )
        else:
            new_cfg["sr_range_pct"] = None
    except Exception:
        new_cfg["sr_range_pct"] = None

    # ---------------- 7) Change-detection vs incoming stock_cfg ----------------
    compare_keys = list(new_cfg.keys())
    changed = any(new_cfg.get(k) != stock_cfg.get(k) for k in compare_keys)

    if not changed:
        # keep original untouched so caller won't publish unnecessarily
        if verbose:
            print(f"[ℹ️] {stock_code}: no material changes")
        return stock_cfg
    else:
        return {**new_cfg, "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
