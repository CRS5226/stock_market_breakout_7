# llm_forecast.py

import os
import json
import csv
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from redis_utils import get_redis, get_recent_indicators, get_recent_candles

from typing import Optional, Tuple

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# === Model routing config ===
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4.1-mini")  # stronger / pricier
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "gpt-4.1-nano")  # cheaper scout

# === S/R lookback choices (tweak as you like) ===
REALTIME_MINUTE_LOOKBACK = int(
    os.getenv("REALTIME_MINUTE_LOOKBACK", "60")
)  # last 60 mins
WEEKLY_DAILY_LOOKBACK = 5  # ~1 week
QUARTER_DAILY_LOOKBACK = 60  # ~3 months


def route_model(stock_code: str, index: int) -> str:
    return PRIMARY_MODEL if (index % 2 == 0) else SECONDARY_MODEL


# -------------------- helpers --------------------


def _safe_number(x, default=0):
    try:
        if x is None:
            return default
        xv = float(x)
        if pd.isna(xv) or xv == float("inf") or xv == float("-inf"):
            return default
        return xv
    except Exception:
        return default


def _safe_str(x):
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return str(x)


def _coerce_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _coerce_ret(x):
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def fetch_candles(stock_code: str, n: int = 2):
    """Fetch last N minute candles from Redis (chronological)."""
    r = get_redis()
    items = get_recent_candles(r, stock_code, n)
    if not items:
        print(f"[⚠️] No candles found in Redis for {stock_code}")
        return None
    df = pd.DataFrame(items)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "minute" in df.columns:
        df["minute"] = pd.to_datetime(df["minute"])
        return df.sort_values("minute").reset_index(drop=True)
    return df.reset_index(drop=True)


def _last_close_from(df_recent, df_hist, cfg):
    try:
        if df_recent is not None and not df_recent.empty:
            col = (
                "Close"
                if "Close" in df_recent.columns
                else ("close" if "close" in df_recent.columns else None)
            )
            if col:
                v = df_recent[col].iloc[-1]
                if pd.notna(v):
                    return float(v)
    except Exception:
        pass
    try:
        oc = cfg.get("ohlcv", {})
        v = oc.get("close", None)
        if v is not None:
            return float(v)
    except Exception:
        pass
    try:
        if df_hist is not None and not df_hist.empty and "Close" in df_hist.columns:
            v = df_hist["Close"].iloc[-1]
            if pd.notna(v):
                return float(v)
    except Exception:
        pass
    return None


# ---------- NEW: SR helpers ----------


def _donchian_sr_from_minutes(min_df: pd.DataFrame) -> tuple | tuple[None, None]:
    """
    Compute S/R from minute candles using Donchian High/Low over the given frame.
    Expects columns: high/low (lowercase) or High/Low.
    """
    if min_df is None or min_df.empty:
        return None, None
    # be tolerant to schemas
    hcol = (
        "high"
        if "high" in min_df.columns
        else ("High" if "High" in min_df.columns else None)
    )
    lcol = (
        "low"
        if "low" in min_df.columns
        else ("Low" if "Low" in min_df.columns else None)
    )
    if not hcol or not lcol:
        return None, None
    try:
        return float(min_df[hcol].max()), float(min_df[lcol].min())
    except Exception:
        return None, None


def _donchian_sr_from_daily(
    df_hist: pd.DataFrame, lookback: int
) -> tuple | tuple[None, None]:
    """
    Compute S/R from daily historical candles over a lookback (in rows).
    Expects columns: 'High','Low'.
    """
    if (
        df_hist is None
        or df_hist.empty
        or "High" not in df_hist.columns
        or "Low" not in df_hist.columns
    ):
        return None, None
    tail = df_hist.tail(max(1, lookback))
    try:
        res = float(tail["High"].max())
        sup = float(tail["Low"].min())
        return res, sup
    except Exception:
        return None, None


# ---------- MERGE + NORMALIZE HELPERS (add once near other helpers) ----------


def _is_dict(x):
    return isinstance(x, dict)


def _is_list(x):
    return isinstance(x, list)


def _as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def _merge_into(base: dict, extra: dict) -> dict:
    """
    Deep-merge 'extra' into 'base', preferring non-empty values from extra.
    Special-cases:
      - reason: concatenates into a list (dedup preserves order)
      - dicts: recurse
    """
    if not _is_dict(base):
        base = {}
    if not _is_dict(extra):
        return base

    for k, v in extra.items():
        if k == "reason":
            # normalize both sides to lists of strings
            base_list = _as_list(base.get("reason"))
            extra_list = _as_list(v)
            # flatten any nested / non-strings
            flat = []
            for item in base_list + extra_list:
                if item is None:
                    continue
                if isinstance(item, (list, tuple)):
                    flat.extend([str(t) for t in item if t is not None])
                else:
                    flat.append(str(item))
            # dedup preserve order
            seen = set()
            out = []
            for it in flat:
                if it not in seen:
                    seen.add(it)
                    out.append(it)
            base["reason"] = out
        elif _is_dict(v):
            base[k] = _merge_into(base.get(k, {}), v)
        else:
            # prefer non-empty/None overrides
            if v is not None and v != "":
                base[k] = v
            else:
                base.setdefault(k, v)
    return base


def _pick_base_from_list(items, desired_stock_code: str | None):
    """
    Choose a 'base' dict from a list:
      1) one with stock_code == desired_stock_code
      2) else first having any stock_code / instrument_token
      3) else a new empty dict
    """
    idx = None
    if desired_stock_code:
        for i, it in enumerate(items):
            if (
                _is_dict(it)
                and str(it.get("stock_code", "")).upper()
                == str(desired_stock_code).upper()
            ):
                idx = i
                break
    if idx is None:
        for i, it in enumerate(items):
            if _is_dict(it) and ("stock_code" in it or "instrument_token" in it):
                idx = i
                break
    if idx is None:
        return {}, list(range(len(items)))  # nothing suitable, merge all into empty
    rest = [i for i in range(len(items)) if i != idx]
    return items[idx].copy(), rest


def _normalize_llm_cfg(raw_cfg, stock_block: dict):
    """
    Ensure we always return a SINGLE dict for the given stock_code.
    Handles shapes like:
      - dict (good)
      - list of dicts (merge all into the one with stock_code)
      - dict with "stocks": [ ... ] (merge inside)
    Also injects missing stock_code / instrument_token from stock_block.
    """
    stock_code = stock_block.get("stock_code")
    instr = stock_block.get("instrument_token")

    # a) list of dicts
    if _is_list(raw_cfg):
        base, rest_idx = _pick_base_from_list(raw_cfg, stock_code)
        for i in rest_idx:
            if _is_dict(raw_cfg[i]):
                base = _merge_into(base, raw_cfg[i])
        raw_cfg = base

    # b) dict with "stocks" list inside
    if _is_dict(raw_cfg) and "stocks" in raw_cfg and _is_list(raw_cfg["stocks"]):
        items = raw_cfg["stocks"]
        base, rest_idx = _pick_base_from_list(items, stock_code)
        for i in rest_idx:
            if _is_dict(items[i]):
                base = _merge_into(base, items[i])
        raw_cfg = base

    # c) ensure dict
    if not _is_dict(raw_cfg):
        raw_cfg = {}

    # Inject identity if missing
    raw_cfg.setdefault("stock_code", stock_code)
    if raw_cfg.get("instrument_token") in (None, "", 0) and instr is not None:
        raw_cfg["instrument_token"] = instr

    # Normalize reason
    if "reason" in raw_cfg:
        raw_cfg = _merge_into(raw_cfg, {"reason": raw_cfg.get("reason")})
    else:
        raw_cfg["reason"] = []

    return raw_cfg


def _sr_width_pct(
    sup: float | None, res: float | None, base_close: float | None
) -> float | None:
    try:
        if sup is None or res is None or base_close is None or base_close <= 0:
            return None
        return round(((res - sup) / base_close) * 100.0, 4)
    except Exception:
        return None


def _to_float(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return default
        return float(x)
    except Exception:
        return default


def _ensure_sr_order(s, r):
    s = _to_float(s, None)
    r = _to_float(r, None)
    if s is None or r is None:
        return s, r
    if s > r:
        s, r = r, s
    return s, r


def _pick_bias(cfg, last_close):
    # Prefer explicit signal first
    sig = (cfg.get("signal") or "").lower()
    if "short" in sig:
        return "short"
    if "long" in sig or "buy" in sig:
        return "long"
    # Infer from realtime SR vs close
    s_rt = _to_float(cfg.get("support_realtime"))
    r_rt = _to_float(cfg.get("resistance_realtime"))
    c = _to_float(last_close)
    if c and s_rt and r_rt:
        mid = (s_rt + r_rt) / 2.0
        return "long" if c >= mid else "short"
    return "long"  # safe default


def _synthesize_levels_from_sr(s, r, bias, last_close):
    """
    Creates (entry, target, stop) from S/R with simple, deterministic rules:
    - entry anchored at 'close' if inside band, else 40% into band toward bias
    - risk = 25% of band, RR ≈ 1.5
    - allow small overshoot (±5% of band)
    """
    s, r = _ensure_sr_order(s, r)
    if s is None or r is None or s == r:
        return None, None, None

    band = r - s
    if band <= 0:
        return None, None, None

    rr = 1.5
    c = _to_float(last_close)

    if bias == "long":
        entry = c if (c is not None and s < c < r) else s + 0.40 * band
        risk = 0.25 * band
        stop = entry - risk
        target = entry + rr * risk
        target = min(target, r + 0.05 * band)
        stop = max(stop, s - 0.05 * band)
        if not (stop < entry < target):
            return None, None, None
        return float(entry), float(target), float(stop)

    else:
        entry = c if (c is not None and s < c < r) else r - 0.40 * band
        risk = 0.25 * band
        stop = entry + risk
        target = entry - rr * risk
        target = max(target, s - 0.05 * band)
        stop = min(stop, r + 0.05 * band)
        if not (target < entry < stop):
            return None, None, None
        return float(entry), float(target), float(stop)


def _fill_one_triplet(cfg, prefix, s_key, r_key, bias, last_close, reasons):
    """
    Ensures entry/target/stoploss for horizon prefix ("", "1", "2", "3").
    """
    e_key = f"entry{prefix}"
    t_key = f"target{prefix}"
    sl_key = f"stoploss{prefix}"

    e = _to_float(cfg.get(e_key))
    t = _to_float(cfg.get(t_key))
    sl = _to_float(cfg.get(sl_key))

    # Keep if present and ordered
    ok = False
    if e is not None and t is not None and sl is not None:
        if bias == "long" and (sl < e < t):
            ok = True
        if bias == "short" and (t < e < sl):
            ok = True
    if ok:
        return

    # Try to synthesize from S/R
    s = cfg.get(s_key)
    r = cfg.get(r_key)
    se, st, ssl = _synthesize_levels_from_sr(s, r, bias, last_close)

    # If S/R absent, synthesize around close with a tiny synthetic band
    if se is None:
        c = _to_float(last_close)
        if c is not None:
            s_syn = c * 0.994
            r_syn = c * 1.006
            se, st, ssl = _synthesize_levels_from_sr(s_syn, r_syn, bias, last_close)

    if se is not None:
        cfg[e_key] = se
        cfg[t_key] = st
        cfg[sl_key] = ssl
        if isinstance(reasons, list):
            reasons.append(
                f"Auto-filled {e_key}/{t_key}/{sl_key} from {s_key}/{r_key} with bias={bias}."
            )
    else:
        if isinstance(reasons, list):
            reasons.append(
                f"Could not synthesize {e_key}/{t_key}/{sl_key} (missing/invalid S/R)."
            )


# -------------------- end helpers --------------------

# Example schema shape for LLM (expanded with new SR keys).
EXAMPLE_CONFIG = {
    "stock_code": "RELIANCE",
    "instrument_token": 128083204,
    "ohlcv": {  # system may overwrite from Redis
        "time": None,
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
    },
    # Legacy "general" SR (we will map these to daily S/R below to keep backward compatibility)
    "support": 1366.5,
    "resistance": 1373.05,
    # NEW: explicit SR sets
    "support_realtime": None,
    "resistance_realtime": None,
    "support_1d": None,
    "resistance_1d": None,
    "support_1w": None,
    "resistance_1w": None,
    "support_3m": None,
    "resistance_3m": None,
    # NEW: SR width percentages relative to last_close
    "sr_range_pct_realtime": None,
    "sr_range_pct_1d": None,
    "sr_range_pct_1w": None,
    "sr_range_pct_3m": None,
    "volume_threshold": 178607,
    "bollinger": {"mid_price": 1370.0, "upper_band": 1373.0, "lower_band": 1365.0},
    "adx": {"period": 14, "threshold": 20},
    "moving_averages": {"ma_fast": 9, "ma_slow": 20},
    "inside_bar": {"lookback": 1},
    "candle": {"min_body_percent": 0.7},
    "reason": [],
    "signal": "No Action",
    # NEW: trade levels
    "entry": None,
    "target": None,
    "stoploss": None,
    "entry1": None,
    "target1": None,
    "stoploss1": None,
    "entry2": None,
    "target2": None,
    "stoploss2": None,
    "entry3": None,
    "target3": None,
    "stoploss3": None,
    "predicted_targets": {
        "1d": {"ret": 0.0, "price": None},
        "1w": {"ret": 0.0, "price": None},
        "3m": {"ret": 0.0, "price": None},
    },
}

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
    Uses last row's atr_pct if available, else default_pct.
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


def _count_respects(df, level, side, tol_frac=0.0015, min_sep=2):
    """
    Count 'respect' events on DAILY candles.
    side: 'support' or 'resistance'
    Logic:
      SUPPORT respected if:
        (near) |Low - L|/L <= tol and Close >= L  OR  (wick) Low < L and Close > L
      RESISTANCE respected if:
        (near) |High - L|/L <= tol and Close <= L OR  (wick) High > L and Close < L
    """
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


# ---------- main function (drop-in replacement) ----------


def compute_respect_counts_from_cfg(
    df_hist: pd.DataFrame,
    cfg: dict,
    lookback_hist_bars_for_SR: int = 180,  # base S/R → daily window
    lookback_hist_bars_for_pairs: int = 180,  # 1d/1w/3m → daily window
    base_sr_fixed_tol_pct: float = 0.50,  # wide fixed band for base S/R (e.g., 0.50%)
    pair_auto_scale: float = 0.25,  # pairs use ATR% * 0.25 (clamped)
    pair_min_bps: int = 8,
    pair_max_bps: int = 60,
    min_sep_bars: int = 2,
) -> dict:
    """
    Counts 'respect' events for:
      - Base S/R: cfg['support'], cfg['resistance'] with a wide fixed tolerance on DAILY data.
      - 1d/1w/3m pairs: cfg['support_1d'], cfg['resistance_1d'], etc., with ATR-based tolerance on DAILY data.
    Returns a dict of 8 counts keyed like respected_S, respected_R, respected_S1, ...
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

    if df_hist is None or df_hist.empty:
        return out

    dly_sr = df_hist.tail(max(1, lookback_hist_bars_for_SR))
    dly_pair = df_hist.tail(max(1, lookback_hist_bars_for_pairs))

    # --- Base S/R with wide fixed tolerance ---
    base_tol_frac = max(
        0.0005, float(base_sr_fixed_tol_pct) / 100.0
    )  # e.g., 0.50% => 0.005
    S = cfg.get("support")
    R = cfg.get("resistance")
    if S is not None:
        out["respected_S"] = _count_respects(
            dly_sr, S, "support", base_tol_frac, min_sep_bars
        )
    if R is not None:
        out["respected_R"] = _count_respects(
            dly_sr, R, "resistance", base_tol_frac, min_sep_bars
        )

    # --- 1d/1w/3m with ATR-based tighter tolerance ---
    pair_tol_frac = _auto_tol_frac(
        dly_pair,
        atr_col="atr_pct",
        default_pct=0.35,
        scale=pair_auto_scale,
        min_bps=pair_min_bps,
        max_bps=pair_max_bps,
    )

    S1, R1 = cfg.get("support_1d"), cfg.get("resistance_1d")
    S2, R2 = cfg.get("support_1w"), cfg.get("resistance_1w")
    S3, R3 = cfg.get("support_3m"), cfg.get("resistance_3m")

    if S1 is not None:
        out["respected_S1"] = _count_respects(
            dly_pair, S1, "support", pair_tol_frac, min_sep_bars
        )
    if R1 is not None:
        out["respected_R1"] = _count_respects(
            dly_pair, R1, "resistance", pair_tol_frac, min_sep_bars
        )
    if S2 is not None:
        out["respected_S2"] = _count_respects(
            dly_pair, S2, "support", pair_tol_frac, min_sep_bars
        )
    if R2 is not None:
        out["respected_R2"] = _count_respects(
            dly_pair, R2, "resistance", pair_tol_frac, min_sep_bars
        )
    if S3 is not None:
        out["respected_S3"] = _count_respects(
            dly_pair, S3, "support", pair_tol_frac, min_sep_bars
        )
    if R3 is not None:
        out["respected_R3"] = _count_respects(
            dly_pair, R3, "resistance", pair_tol_frac, min_sep_bars
        )

    return out


# If you already have _last_close_from() elsewhere, you can skip this fallback.
def _last_close_from(df_recent, df_hist, cfg):
    v = None
    # Try cfg ohlcv
    try:
        v = _coerce_float((cfg or {}).get("ohlcv", {}).get("close"))
    except Exception:
        pass
    # Try recent indicators
    if v is None:
        try:
            if "Close" in df_recent:
                v = _coerce_float(df_recent["Close"].dropna().iloc[-1])
            elif "close" in df_recent:
                v = _coerce_float(df_recent["close"].dropna().iloc[-1])
        except Exception:
            pass
    # Try historical daily
    if v is None:
        try:
            if "Close" in df_hist:
                v = _coerce_float(df_hist["Close"].dropna().iloc[-1])
            elif "close" in df_hist:
                v = _coerce_float(df_hist["close"].dropna().iloc[-1])
        except Exception:
            pass
    return v


def _long_only_triplet(
    cfg,
    s_key,
    r_key,
    e_key,
    t_key,
    sl_key,
    last_close=None,
    atr_pct_fallback=0.35,
    use_last_close_anchor=True,
    risk_pct_by_band=0.25,
    rr=1.35,
):
    """
    Build/sanitize ONE long-only triplet for an S/R pair:
      stoploss < support < entry < target < resistance

    Parameters
    ----------
    use_last_close_anchor : if True, bias entry near last_close (clamped inside band).
                            if False, place entry inside the band using risk_pct_by_band.
    risk_pct_by_band      : when not anchoring, entry = band_lo + p*(band_hi - band_lo)
                            (0 => very near support, 1 => near resistance)
    rr                    : risk-reward multiple used for target if target not provided.
    """
    # --- S/R ---
    sup = _coerce_float(cfg.get(s_key))
    res = _coerce_float(cfg.get(r_key))
    if sup is None or res is None:
        cfg[e_key] = cfg[t_key] = cfg[sl_key] = None
        return

    # Ensure S < R
    sup, res = _ensure_sr_order(sup, res)

    # --- Buffers from ATR% ---
    atr_pct = _coerce_float(cfg.get("atr_pct")) or atr_pct_fallback  # percent
    atr_frac = max(0.001, atr_pct / 100.0)  # ≥0.10%
    buf_bps = 0.75 * atr_frac * 10000.0
    buf_frac = max(12 / 10000.0, min(60 / 10000.0, buf_bps))  # 12..60 bps
    tiny = max(atr_frac, 0.0008)  # ≥8 bps

    # --- Existing values (if any) ---
    entry = _coerce_float(cfg.get(e_key))
    target = _coerce_float(cfg.get(t_key))
    stoploss = _coerce_float(cfg.get(sl_key))

    # --- Inner tradable band ---
    band_lo = sup * (1.0 + buf_frac)  # just above support
    band_hi = res * (1.0 - buf_frac)  # just below resistance
    if band_hi <= band_lo:  # degenerate band, bail out safely
        cfg[e_key] = cfg[t_key] = cfg[sl_key] = None
        return

    # --- Entry placement ---
    if entry is None:
        if use_last_close_anchor and last_close is not None:
            # pull toward last_close but clamp to inner band
            base_ref = max(band_lo, min(float(last_close), band_hi))
        else:
            # horizon-driven: place between band_lo and band_hi
            p = max(0.0, min(1.0, float(risk_pct_by_band)))
            base_ref = band_lo + p * (band_hi - band_lo)
        entry = base_ref
    # clamp any provided entry to band
    entry = max(band_lo, min(entry, band_hi))

    # --- Stop below support, ATR-bounded ---
    lo_stop = sup * (1.0 - 1.25 * atr_frac)  # deepest allowed
    hi_stop = sup * (1.0 - tiny)  # just below support
    if stoploss is None:
        stoploss = lo_stop
    stoploss = max(lo_stop, min(stoploss, hi_stop))
    if stoploss >= entry:
        stoploss = entry * (1.0 - tiny)

    # --- Target via RR, capped by resistance ---
    risk = max(0.01, entry - stoploss)
    target_cap = band_hi
    if target is None:
        target = min(entry + rr * risk, target_cap)
    # keep > entry and < resistance cap
    target = min(max(target, entry * (1.0 + tiny)), target_cap)

    # --- Final write-back ---
    cfg[e_key] = round(float(entry), 3)
    cfg[t_key] = round(float(target), 3)
    cfg[sl_key] = round(float(stoploss), 3)


# ---------- local helpers (patches) ----------


def horizon_sr_from_daily(
    df_daily: pd.DataFrame, lookback: int, use_quantile: bool = False, q: float = 0.05
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute composite S/R from *daily* data over the last `lookback` rows.
    Returns (resistance, support).
    Prefers swing extremes; blends last-day floor pivots; optional BB if present.
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
    if not (H and L):
        return None, None

    # swing extremes (optionally robust via quantiles)
    if use_quantile:
        swing_high = float(tail[H].quantile(1.0 - q))
        swing_low = float(tail[L].quantile(q))
    else:
        swing_high = float(tail[H].max())
        swing_low = float(tail[L].min())

    # floor pivot from the last day if Close is available
    r1 = swing_high
    s1 = swing_low
    if C is not None and tail[C].notna().any():
        h = float(tail[H].iloc[-1])
        l = float(tail[L].iloc[-1])
        c = float(tail[C].iloc[-1])
        P = (h + l + c) / 3.0
        r1 = 2 * P - l
        s1 = 2 * P - h

    # optional Bollinger bounds across window
    bb_u = float(tail[U].max()) if U and tail[U].notna().any() else None
    bb_l = float(tail[D].min()) if D and tail[D].notna().any() else None

    candidates_res = [v for v in (r1, swing_high, bb_u) if v is not None]
    candidates_sup = [v for v in (s1, swing_low, bb_l) if v is not None]
    if not candidates_res or not candidates_sup:
        return None, None

    return max(candidates_res), min(candidates_sup)


def _fallback_horizon_sr_if_duplicated(cfg, df_hist):
    """If LLM duplicates base SR to all horizons or leaves them missing, recompute from daily."""
    for sk, rk, lb in [
        ("support_1d", "resistance_1d", 1),
        ("support_1w", "resistance_1w", 5),
        ("support_3m", "resistance_3m", 60),
    ]:
        s = _coerce_float(cfg.get(sk))
        r = _coerce_float(cfg.get(rk))
        base_s = _coerce_float(cfg.get("support"))
        base_r = _coerce_float(cfg.get("resistance"))
        if (s is None or r is None) or (s == base_s and r == base_r):
            r_h, s_h = horizon_sr_from_daily(df_hist, lb)
            if s_h is not None and r_h is not None:
                s_h, r_h = _ensure_sr_order(round(s_h, 2), round(r_h, 2))
                cfg[sk], cfg[rk] = s_h, r_h
                cfg.setdefault("reason", []).append(
                    f"LLM duplicated/missed {sk}/{rk}; recomputed via horizon_sr({lb})."
                )


def _push_apart_if_equal_long(cfg, e_key, t_key, r_key, min_eps_bps=12):
    e = _coerce_float(cfg.get(e_key))
    t = _coerce_float(cfg.get(t_key))
    r = _coerce_float(cfg.get(r_key))
    if e is None or t is None or r is None:
        return
    if t <= e:
        eps = max(min_eps_bps / 10000.0, 0.0008)  # >= 8–12 bps
        cap = r * (1.0 - 12 / 10000.0)  # keep inside R by 12 bps
        new_t = min(e * (1.0 + eps), cap)
        if new_t <= e:
            new_t = e + max(0.01, e * eps)  # tick bump if rounding collapses
        cfg[t_key] = round(new_t, 2)


def _audit_long_invariants(cfg, s, r, e, t, sl, tag):
    s = _coerce_float(cfg.get(s))
    r = _coerce_float(cfg.get(r))
    e = _coerce_float(cfg.get(e))
    t = _coerce_float(cfg.get(t))
    sl = _coerce_float(cfg.get(sl))
    ok = all(v is not None for v in (s, r, e, t, sl)) and (sl < s < e < t < r)
    if not ok:
        cfg.setdefault("reason", []).append(
            f"WARNING: invariant violated after post-process for {tag}."
        )
    return ok


def save_forecast_csv(stock_code, raw):
    os.makedirs("forecast", exist_ok=True)
    file_path = f"forecast/LLM_{stock_code}_response.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "raw_response"])
        writer.writerow([now, raw])


def update_token_monitor(
    stock_code: str,
    input_tokens: int,
    output_tokens: int,
    monitor_file="stock_llm_monitor.json",
    model_used: str = "",
):
    if os.path.exists(monitor_file):
        try:
            with open(monitor_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    if stock_code not in data:
        data[stock_code] = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_model": {},
        }

    data[stock_code]["total_input_tokens"] += input_tokens
    data[stock_code]["total_output_tokens"] += output_tokens

    if "by_model" not in data[stock_code]:
        data[stock_code]["by_model"] = {}
    if model_used not in data[stock_code]["by_model"]:
        data[stock_code]["by_model"][model_used] = {"input": 0, "output": 0}

    data[stock_code]["by_model"][model_used]["input"] += input_tokens
    data[stock_code]["by_model"][model_used]["output"] += output_tokens

    with open(monitor_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def forecast_config_update(
    stock_block: dict,
    historical_folder: str = "historical_data",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    escalate_on_signal: bool = True,
):
    import os, json
    import pandas as pd
    from datetime import datetime

    stock_code = stock_block.get("stock_code")

    # --- Step 1: Historical (for LLM context) ---
    hist_csv = None
    if os.path.isdir(historical_folder):
        for file in os.listdir(historical_folder):
            if file.startswith(f"{stock_code}_historical_") and file.endswith(".csv"):
                hist_csv = os.path.join(historical_folder, file)
                break

    df_hist = pd.DataFrame()
    if hist_csv and os.path.exists(hist_csv):
        try:
            df_hist = pd.read_csv(hist_csv)
            historical_excerpt = df_hist.tail(5).to_csv(index=False)
        except Exception as e:
            return None, None, f"Failed to read historical CSV {hist_csv}: {e}"
    else:
        historical_excerpt = "(No historical file found)"

    # --- Step 2: Recent indicators from Redis ---
    try:
        r = get_redis()
        recent_data = get_recent_indicators(r, stock_code, n=50)
        if not recent_data:
            return None, None, f"No recent indicators for {stock_code}"
        df_recent = pd.DataFrame(recent_data)
        try:
            df_recent.columns = [str(c).strip() for c in df_recent.columns]
        except Exception:
            pass
        recent_excerpt = df_recent.tail(10).to_csv(index=False)
    except Exception as e:
        return None, None, f"Failed to fetch recent indicators from Redis: {e}"

    # --- Step 3: Prompt (LLM generates SR + trade levels + targets) ---
    schema_str = json.dumps(EXAMPLE_CONFIG, indent=2)
    provided_cfg_str = json.dumps(stock_block, indent=2)
    prompt = f"""
You are a STRICT JSON generator for a stock trading config.

OBJECTIVE
- Read CONFIG, HISTORICAL DATA (~1 year daily candles), and RECENT DATA (~50 1-minute indicator rows).
- Update fields ONLY when justified by the data:
  • Core levels: support/resistance (default, realtime, 1d, 1w, 3m)
  • volume_threshold, bollinger, adx/moving_averages/inside_bar/candle params, and signal
  • Trade levels: entry/target/stoploss for each S/R pair
  • Forward predicted_targets (1d, 1w, 3m)
- Provide a SHORT list of reasons explaining what changed and why.

LONG-ONLY POLICY (NO SHORTS)
- Never propose or imply short trades.
- For EVERY S/R pair, the final levels MUST satisfy this strict ordering:
  stoploss < support < entry < target < resistance
- Keep entry and target INSIDE the S–R band; keep stoploss STRICTLY below support.
- If the market context is bearish, set a conservative signal (e.g., "No Action") but keep outputs long-only and ordered.

SUPPORT/RESISTANCE
- Infer support/resistance from historical + recent data.
- Always output numeric floats for:
  "support", "resistance",
  "support_realtime", "resistance_realtime",
  "support_1d", "resistance_1d",
  "support_1w", "resistance_1w",
  "support_3m", "resistance_3m".
- Ensure S < R for each horizon (swap if needed).

TRADE LEVELS (LONG-ONLY)
- For each S/R pair, output:
  Default SR → "entry","target","stoploss"
  S1/R1 (1d) → "entry1","target1","stoploss1"
  S2/R2 (1w) → "entry2","target2","stoploss2"
  S3/R3 (3m) → "entry3","target3","stoploss3"
- Check ordering BEFORE returning JSON; if any triplet would violate the ordering or leave the band, adjust minimally so the ordering holds.
- Risk–reward should be reasonable (typically ~1.1–2.5).

PREDICTED TARGETS
- Return "predicted_targets" with keys "1d","1w","3m".
- Each must include numeric "ret" (fractional, e.g., 0.012 = +1.2%) AND "price" (Close_t * (1 + ret)).
- These are forward estimates and independent from trade triplets.

RULES
- Preserve "stock_code" and "instrument_token".
- You MAY include/refresh "ohlcv" of the latest bar if you can infer it; if uncertain, leave it None (system may overwrite from Redis).
- Output ONE valid JSON object matching the SCHEMA keys exactly. No commentary, no markdown.
- Required top-level fields: "reason", "signal", "predicted_targets", all SR levels, and all entry/target/stoploss keys.
- ALL price-like fields must be numeric floats (no nulls / empty strings).

SCHEMA (shape example):
{schema_str}

CONFIG (current):
{provided_cfg_str}

HISTORICAL DATA (recent daily rows):
{historical_excerpt}

RECENT DATA (last indicator rows):
{recent_excerpt}
""".strip()

    # ---------- LLM call ----------
    def _call_llm(model_name: str):
        resp = client.responses.create(
            model=model_name,
            temperature=temperature,
            max_output_tokens=2048,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Output strict JSON matching the SCHEMA keys exactly. "
                        "Every required key present; no nulls or empty strings. "
                        "All price-like fields must be numeric floats. No markdown, no comments. "
                        "Long-only: never output short-side levels; enforce stoploss < support < entry < target < resistance."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.output_text.strip()

        # Clean possible code fences
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        save_forecast_csv(stock_code, cleaned)

        in_toks = getattr(getattr(resp, "usage", None), "input_tokens", 0)
        out_toks = getattr(getattr(resp, "usage", None), "output_tokens", 0)
        update_token_monitor(stock_code, in_toks, out_toks, model_used=model_name)

        # parse JSON
        try:
            cfg = json.loads(cleaned)
        except Exception as e:
            safe_excerpt = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
            return None, None, f"JSON parse error: {e}. Output excerpt: {safe_excerpt}"

        cfg = _normalize_llm_cfg(cfg, stock_block)

        # minimal safety defaults
        cfg.setdefault("reason", [])
        cfg.setdefault("signal", "No Action")
        cfg["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cfg["forecast"] = "AI"  # mark as LLM-produced

        # --- Overwrite OHLCV from Redis (latest minute candle) ---
        try:
            last_candles = fetch_candles(stock_code, n=max(REALTIME_MINUTE_LOOKBACK, 1))
            if last_candles is not None and not last_candles.empty:
                row = last_candles.iloc[-1]
                cfg["ohlcv"] = {
                    "time": _safe_str(row.get("minute")),
                    "open": _safe_number(row.get("open"), None),
                    "high": _safe_number(row.get("high"), None),
                    "low": _safe_number(row.get("low"), None),
                    "close": _safe_number(row.get("close"), None),
                    "volume": int(_safe_number(row.get("volume"), 0)),
                }
        except Exception:
            cfg.setdefault(
                "ohlcv",
                {
                    "time": None,
                    "open": None,
                    "high": None,
                    "low": None,
                    "close": None,
                    "volume": None,
                },
            )

        # --- Sanitize predicted_targets ---
        pt = cfg.get("predicted_targets", {}) or {}
        if not isinstance(pt, dict):
            pt = {}

        last_close = _last_close_from(df_recent, df_hist, cfg)
        CAPS = {"1d": 0.06, "1w": 0.12, "3m": 0.30}
        final_pt = {}
        for h in ("1d", "1w", "3m"):
            slot = pt.get(h, {}) or {}
            rhat = _coerce_ret(slot.get("ret", 0.0))
            cap = CAPS[h]
            rhat = max(-cap, min(cap, rhat))
            phat = slot.get("price", None)
            if (
                phat is None or (isinstance(phat, str) and phat.strip() == "")
            ) and last_close is not None:
                phat = float(last_close * (1.0 + rhat))
            else:
                try:
                    phat = float(phat) if phat is not None else None
                except Exception:
                    phat = (
                        float(last_close * (1.0 + rhat))
                        if last_close is not None
                        else None
                    )
            final_pt[h] = {"ret": float(rhat), "price": phat, "method": "llm_v1"}
        cfg["predicted_targets"] = final_pt

        # --- Ensure SR order across horizons ---
        cfg["support"], cfg["resistance"] = _ensure_sr_order(
            cfg.get("support"), cfg.get("resistance")
        )
        cfg["support_realtime"], cfg["resistance_realtime"] = _ensure_sr_order(
            cfg.get("support_realtime"), cfg.get("resistance_realtime")
        )
        cfg["support_1d"], cfg["resistance_1d"] = _ensure_sr_order(
            cfg.get("support_1d"), cfg.get("resistance_1d")
        )
        cfg["support_1w"], cfg["resistance_1w"] = _ensure_sr_order(
            cfg.get("support_1w"), cfg.get("resistance_1w")
        )
        cfg["support_3m"], cfg["resistance_3m"] = _ensure_sr_order(
            cfg.get("support_3m"), cfg.get("resistance_3m")
        )

        # --- PATCH A: recompute horizons if LLM duplicated/missed them ---
        _fallback_horizon_sr_if_duplicated(cfg, df_hist)

        # === LONG-ONLY synthesis/sanitization for every triplet ===
        _last_close = last_close  # reuse computed last_close above

        # Base / near-term: anchor to last_close
        _long_only_triplet(
            cfg,
            "support",
            "resistance",
            "entry",
            "target",
            "stoploss",
            last_close=_last_close,
            use_last_close_anchor=True,
        )

        # 1d: usually ok to anchor
        _long_only_triplet(
            cfg,
            "support_1d",
            "resistance_1d",
            "entry1",
            "target1",
            "stoploss1",
            last_close=_last_close,
            use_last_close_anchor=True,
        )

        # 1w: DO NOT anchor; use the band to create distinct levels
        _long_only_triplet(
            cfg,
            "support_1w",
            "resistance_1w",
            "entry2",
            "target2",
            "stoploss2",
            last_close=_last_close,
            use_last_close_anchor=False,
            risk_pct_by_band=0.30,
        )

        # 3m: DO NOT anchor; even deeper toward support by default
        _long_only_triplet(
            cfg,
            "support_3m",
            "resistance_3m",
            "entry3",
            "target3",
            "stoploss3",
            last_close=_last_close,
            use_last_close_anchor=False,
            risk_pct_by_band=0.35,
        )

        # --- PATCH B: push-apart guard (handles equal/rounded entry/target) ---
        _push_apart_if_equal_long(cfg, "entry", "target", "resistance")
        _push_apart_if_equal_long(cfg, "entry1", "target1", "resistance_1d")
        _push_apart_if_equal_long(cfg, "entry2", "target2", "resistance_1w")
        _push_apart_if_equal_long(cfg, "entry3", "target3", "resistance_3m")

        # --- Optional: audit invariants and log warnings ---
        _audit_long_invariants(
            cfg, "support", "resistance", "entry", "target", "stoploss", "base"
        )
        _audit_long_invariants(
            cfg, "support_1d", "resistance_1d", "entry1", "target1", "stoploss1", "1d"
        )
        _audit_long_invariants(
            cfg, "support_1w", "resistance_1w", "entry2", "target2", "stoploss2", "1w"
        )
        _audit_long_invariants(
            cfg, "support_3m", "resistance_3m", "entry3", "target3", "stoploss3", "3m"
        )

        # --- Compute SR width percentages (use last_close as base_close for clarity) ---
        try:
            base_close = last_close
            cfg["sr_range_pct_realtime"] = _sr_width_pct(
                cfg.get("support_realtime"), cfg.get("resistance_realtime"), base_close
            )
            cfg["sr_range_pct_1d"] = _sr_width_pct(
                cfg.get("support_1d"), cfg.get("resistance_1d"), base_close
            )
            cfg["sr_range_pct_1w"] = _sr_width_pct(
                cfg.get("support_1w"), cfg.get("resistance_1w"), base_close
            )
            cfg["sr_range_pct_3m"] = _sr_width_pct(
                cfg.get("support_3m"), cfg.get("resistance_3m"), base_close
            )

            if (
                cfg.get("support") is not None
                and cfg.get("resistance") is not None
                and base_close is not None
                and base_close > 0
            ):
                cfg["sr_range_pct"] = round(
                    (
                        (float(cfg["resistance"]) - float(cfg["support"]))
                        / float(cfg["support"])
                    )
                    * 100.0,
                    4,
                )
            else:
                cfg["sr_range_pct"] = None
        except Exception:
            cfg["sr_range_pct"] = None

        # --- Final guard: warn if any triplet still missing ---
        must_have = [
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
        if any(cfg.get(k) is None for k in must_have):
            cfg["reason"].append(
                "WARNING: Some trade levels still missing after synthesis."
            )

        # --- Respect counts derived from cfg (no pivots needed) ---
        try:
            respect_counts = compute_respect_counts_from_cfg(
                df_hist=df_hist,
                cfg=cfg,
                lookback_hist_bars_for_SR=180,
                lookback_hist_bars_for_pairs=180,
                base_sr_fixed_tol_pct=0.50,  # 0.50% wide band for base S/R
                pair_auto_scale=0.25,  # pairs: ATR% * 0.25 (clamped 8–60 bps)
                pair_min_bps=8,
                pair_max_bps=60,
                min_sep_bars=2,
            )
            cfg["respected_S"] = respect_counts.get("respected_S")
            cfg["respected_R"] = respect_counts.get("respected_R")
            cfg["respected_S1"] = respect_counts.get("respected_S1")
            cfg["respected_R1"] = respect_counts.get("respected_R1")
            cfg["respected_S2"] = respect_counts.get("respected_S2")
            cfg["respected_R2"] = respect_counts.get("respected_R2")
            cfg["respected_S3"] = respect_counts.get("respected_S3")
            cfg["respected_R3"] = respect_counts.get("respected_R3")
        except Exception:
            cfg.setdefault("respected_S", None)
            cfg.setdefault("respected_R", None)
            cfg.setdefault("respected_S1", None)
            cfg.setdefault("respected_R1", None)
            cfg.setdefault("respected_S2", None)
            cfg.setdefault("respected_R2", None)
            cfg.setdefault("respected_S3", None)
            cfg.setdefault("respected_R3", None)

        return cfg, cfg.get("reason", []), None

    # First pass with provided model
    updated_cfg, reasons, err = _call_llm(model)
    if err:
        return None, None, err

    return updated_cfg, reasons, None
