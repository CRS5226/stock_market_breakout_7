# llm_forecast.py

import os
import json
import csv
from datetime import datetime
import glob

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from redis_utils import (
    get_redis,
    get_recent_indicators,
    get_recent_candles,
    get_recent_indicators_tf,
    get_recent_candles_tf,
)

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


# ---------- Multi-timeframe plan (S0..S8) ----------
# idx=0 is "realtime/base" (1min) and uses unsuffixed fields
TF_PLAN = [
    ("1min", 0, "1d"),
    ("5min", 1, "1d"),
    ("15min", 2, "1d"),
    ("30min", 3, "1d"),
    ("45min", 4, "1d"),
    ("1hour", 5, "1w"),
    ("4hour", 6, "1w"),
    ("1day", 7, "1w"),
    ("1month", 8, "3m"),
]


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
                rows = (
                    legacy  # legacy assumed already chronological; if not, sort below
                )
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

    for src, dst in [
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
    ]:
        _copy_col(src, dst)

    # type coercion (safe)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        )

    return df.tail(n).reset_index(drop=True)


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


# -------------------- end helpers --------------------

# Example schema shape for LLM (expanded with new SR keys).
EXAMPLE_CONFIG = {
    "stock_code": "RELIANCE",
    "instrument_token": 128083204,
    "ohlcv": {
        "time": None,
        "open": None,
        "high": None,
        "low": None,
        "close": None,
        "volume": None,
    },
    # Base (S0) generic SR kept for compatibility
    "support": None,
    "resistance": None,
    # NEW multi-TF SR sets
    "support1": None,
    "resistance1": None,
    "support2": None,
    "resistance2": None,
    "support3": None,
    "resistance3": None,
    "support4": None,
    "resistance4": None,
    "support5": None,
    "resistance5": None,
    "support6": None,
    "resistance6": None,
    "support7": None,
    "resistance7": None,
    "support8": None,
    "resistance8": None,
    # Trade triplets (base + 1..8)
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
    "entry4": None,
    "target4": None,
    "stoploss4": None,
    "entry5": None,
    "target5": None,
    "stoploss5": None,
    "entry6": None,
    "target6": None,
    "stoploss6": None,
    "entry7": None,
    "target7": None,
    "stoploss7": None,
    "entry8": None,
    "target8": None,
    "stoploss8": None,
    # Respect counts per TF
    "respected_S": None,
    "respected_R": None,
    "respected_S1": None,
    "respected_R1": None,
    "respected_S2": None,
    "respected_R2": None,
    "respected_S3": None,
    "respected_R3": None,
    "respected_S4": None,
    "respected_R4": None,
    "respected_S5": None,
    "respected_R5": None,
    "respected_S6": None,
    "respected_R6": None,
    "respected_S7": None,
    "respected_R7": None,
    "respected_S8": None,
    "respected_R8": None,
    # Optional widths (we still compute some of these later)
    "sr_range_pct": None,
    "volume_threshold": None,
    "bollinger": {"mid_price": None, "upper_band": None, "lower_band": None},
    "adx": {"period": 14, "threshold": 20},
    "moving_averages": {"ma_fast": 9, "ma_slow": 20},
    "inside_bar": {"lookback": 1},
    "candle": {"min_body_percent": 0.7},
    "reason": [],
    "signal": "No Action",
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


# ----------- TF helpers (data + SR + respects) ---------
def _resample_daily_to_monthly(df_daily: pd.DataFrame) -> pd.DataFrame:
    if df_daily is None or df_daily.empty or "Timestamp" not in df_daily.columns:
        return pd.DataFrame()
    df = df_daily.copy()
    idx = pd.to_datetime(df["Timestamp"])
    df = df.set_index(idx)
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
        "atr_pct": "last",
        "BB_Upper": "last",
        "BB_Lower": "last",
        "HH20": "last",
        "LL20": "last",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = (
        df.resample("M")
        .agg(agg)
        .dropna(how="all")
        .reset_index()
        .rename(columns={"index": "Timestamp"})
    )
    return out


def _respect_counts_for_df(
    df_tf: pd.DataFrame, support: float, resistance: float
) -> dict:
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


def _load_tf_window(
    stock_code: str, tf: str, base_dir="historical_data_candles", bars: int = 400
) -> pd.DataFrame:
    """Load a recent window of candles for a TF from disk (latest file)."""
    tf_dir = os.path.join(base_dir, tf)
    if not os.path.isdir(tf_dir):
        return pd.DataFrame()
    patt = os.path.join(tf_dir, f"{stock_code}_{tf}_*.csv")
    files = glob.glob(patt)
    if not files:
        return pd.DataFrame()
    path = max(files, key=os.path.getmtime)
    try:
        df = pd.read_csv(path)
        # Normalize
        if "Timestamp" not in df.columns:
            for c in ("timestamp", "minute", "datetime"):
                if c in df.columns:
                    df = df.rename(columns={c: "Timestamp"})
                    break
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        for a, b in [
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
            ("volume", "Volume"),
        ]:
            if a in df.columns and b not in df.columns:
                df[b] = df[a]
        df = (
            df.dropna(subset=["Timestamp"])
            .sort_values("Timestamp")
            .tail(bars)
            .reset_index(drop=True)
        )
        return df
    except Exception:
        return pd.DataFrame()


def _fill_all_tf_respects(cfg: dict, stock_code: str) -> None:
    """
    Fills respected_S / respected_R for base (idx=0) and respected_S{idx}/R{idx} for idx=1..8
    using per-TF CSV windows. If a TF file is unavailable, counts remain 0.
    """
    # Base (idx=0) uses unsuffixed keys
    tf_map = [
        ("1min", 0),
        ("5min", 1),
        ("15min", 2),
        ("30min", 3),
        ("45min", 4),
        ("1hour", 5),
        ("4hour", 6),
        ("1day", 7),
        ("1month", 8),
    ]
    for tf, idx in tf_map:
        df_tf = _load_tf_window(stock_code, tf, bars=400)
        # pick correct S/R keys
        if idx == 0:
            s_key, r_key = "support", "resistance"
            rs_key_s, rs_key_r = "respected_S", "respected_R"
        else:
            s_key, r_key = f"support{idx}", f"resistance{idx}"
            rs_key_s, rs_key_r = f"respected_S{idx}", f"respected_R{idx}"

        sup = _coerce_float(cfg.get(s_key))
        res = _coerce_float(cfg.get(r_key))
        if sup is None or res is None or df_tf.empty:
            cfg[rs_key_s] = cfg.get(rs_key_s, 0) or 0
            cfg[rs_key_r] = cfg.get(rs_key_r, 0) or 0
            continue

        rc = _respect_counts_for_df(df_tf, sup, res)  # already ATR-tolerant
        cfg[rs_key_s] = int(rc.get("S", 0))
        cfg[rs_key_r] = int(rc.get("R", 0))


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
    stock_code = stock_block.get("stock_code")

    # ---------- 1) Build ONE latest candle per timeframe (Redis → Disk fallback) ----------
    base_dir = os.path.join(os.getcwd(), "historical_data_candles")
    tf_latest_records = []  # for prompt
    tf_source_rows = {}  # idx -> single-row DataFrame (for respect counts if needed)
    df_hist_daily = pd.DataFrame()  # full daily for respect counts & excerpts

    for tf, idx, _ in TF_PLAN:
        # try Redis (n=1 latest)
        df_r = pd.DataFrame()
        try:
            rows = get_recent_indicators_tf(get_redis(), stock_code, tf, n=1)
            if rows:
                df_r = pd.DataFrame(rows)
        except Exception:
            df_r = pd.DataFrame()

        # normalize redis row -> 'Timestamp' + OHLCV
        if not df_r.empty:
            if "Timestamp" not in df_r.columns:
                if "minute" in df_r.columns:
                    df_r["Timestamp"] = pd.to_datetime(df_r["minute"], errors="coerce")
                elif "timestamp" in df_r.columns:
                    df_r["Timestamp"] = pd.to_datetime(
                        df_r["timestamp"], errors="coerce"
                    )
                elif "datetime" in df_r.columns:
                    df_r["Timestamp"] = pd.to_datetime(
                        df_r["datetime"], errors="coerce"
                    )
            for a, b in [
                ("open", "Open"),
                ("high", "High"),
                ("low", "Low"),
                ("close", "Close"),
                ("volume", "Volume"),
            ]:
                if a in df_r.columns and b not in df_r.columns:
                    df_r[b] = df_r[a]
            df_r = df_r.dropna(subset=["Timestamp"]).sort_values("Timestamp").tail(1)

        # if Redis missing, fallback to disk: newest CSV in historical_data_candles/<tf>/
        df_d = pd.DataFrame()
        if df_r.empty:
            tf_dir = os.path.join(base_dir, tf)
            if os.path.isdir(tf_dir):
                patt = os.path.join(tf_dir, f"{stock_code}_{tf}_*.csv")
                files = glob.glob(patt)
                if files:
                    latest_path = max(files, key=os.path.getmtime)
                    try:
                        df_d = pd.read_csv(latest_path)
                        # timestamp normalization
                        if "Timestamp" not in df_d.columns:
                            for c in ("timestamp", "minute", "datetime"):
                                if c in df_d.columns:
                                    df_d = df_d.rename(columns={c: "Timestamp"})
                                    break
                        df_d["Timestamp"] = pd.to_datetime(
                            df_d["Timestamp"], errors="coerce"
                        )
                        for a, b in [
                            ("open", "Open"),
                            ("high", "High"),
                            ("low", "Low"),
                            ("close", "Close"),
                            ("volume", "Volume"),
                        ]:
                            if a in df_d.columns and b not in df_d.columns:
                                df_d[b] = df_d[a]
                        df_d = (
                            df_d.dropna(subset=["Timestamp"])
                            .sort_values("Timestamp")
                            .tail(1)
                        )
                    except Exception:
                        df_d = pd.DataFrame()

        # choose whichever is available (prefer Redis if present)
        df_one = df_r if not df_r.empty else df_d
        if not df_one.empty:
            row = df_one.iloc[-1]
            tf_latest_records.append(
                {
                    "tf": tf,
                    "time": str(row.get("Timestamp", row.get("minute", ""))),
                    "open": (
                        float(row.get("Open")) if pd.notna(row.get("Open")) else None
                    ),
                    "high": (
                        float(row.get("High")) if pd.notna(row.get("High")) else None
                    ),
                    "low": float(row.get("Low")) if pd.notna(row.get("Low")) else None,
                    "close": (
                        float(row.get("Close")) if pd.notna(row.get("Close")) else None
                    ),
                    "volume": (
                        int(row.get("Volume")) if pd.notna(row.get("Volume")) else None
                    ),
                }
            )
            tf_source_rows[idx] = df_one.copy()

            # keep full daily history for counts/excerpt if we have a daily CSV
            if tf == "1day" and df_hist_daily.empty:
                # try to load full 1day file for better context / counts
                tf_dir = os.path.join(base_dir, "1day")
                if os.path.isdir(tf_dir):
                    patt = os.path.join(tf_dir, f"{stock_code}_1day_*.csv")
                    files = glob.glob(patt)
                    if files:
                        try:
                            best = max(files, key=os.path.getmtime)
                            df_hist_daily = pd.read_csv(best)
                            if "Timestamp" not in df_hist_daily.columns:
                                for c in ("timestamp", "minute", "datetime"):
                                    if c in df_hist_daily.columns:
                                        df_hist_daily = df_hist_daily.rename(
                                            columns={c: "Timestamp"}
                                        )
                                        break
                            df_hist_daily["Timestamp"] = pd.to_datetime(
                                df_hist_daily["Timestamp"], errors="coerce"
                            )
                            for a, b in [
                                ("open", "Open"),
                                ("high", "High"),
                                ("low", "Low"),
                                ("close", "Close"),
                                ("volume", "Volume"),
                            ]:
                                if (
                                    a in df_hist_daily.columns
                                    and b not in df_hist_daily.columns
                                ):
                                    df_hist_daily[b] = df_hist_daily[a]
                            df_hist_daily = (
                                df_hist_daily.dropna(subset=["Timestamp"])
                                .sort_values("Timestamp")
                                .reset_index(drop=True)
                            )
                        except Exception:
                            df_hist_daily = pd.DataFrame()

    # Excerpts for prompt
    historical_excerpt = (
        df_hist_daily.tail(5).to_csv(index=False)
        if not df_hist_daily.empty
        else "(No 1day data found)"
    )
    try:
        tf_latest_json = json.dumps(tf_latest_records, indent=2, default=str)
    except Exception:
        tf_latest_json = "[]"

    # ---------- 2) Recent indicators (context block) ----------
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
        recent_excerpt = df_recent.tail(1).to_csv(index=False)
    except Exception as e:
        return None, None, f"Failed to fetch recent indicators from Redis: {e}"

    # ---------- 3) Build multi-TF prompt (strict JSON) ----------
    schema_str = json.dumps(EXAMPLE_CONFIG, indent=2)
    provided_cfg_str = json.dumps(stock_block, indent=2)
    prompt = f"""
You are a STRICT JSON generator for a stock trading config.

OBJECTIVE
- Read CONFIG, DAILY HISTORY (last rows), RECENT INTRADAY INDICATORS, and the LATEST PER-TIMEFRAME CANDLES (1min..1month; exactly one candle per TF).
- Update fields ONLY when justified by the data:
  • Core levels for base (S0) and S1..S8
  • volume_threshold, bollinger, adx/moving_averages/inside_bar/candle params, and signal
  • Trade levels: entry/target/stoploss for base and for each S/R pair k=1..8
  • Forward predicted_targets (1d, 1w, 3m)
- Provide a SHORT list of reasons explaining what changed and why.

LONG-ONLY POLICY (NO SHORTS)
- Never propose or imply short trades.
- For EVERY S/R pair (base and k=1..8), enforce:
  stoploss < support < entry < target < resistance
- Keep entry & target INSIDE the S–R band; keep stoploss STRICTLY below support.

SUPPORT/RESISTANCE (MULTI-TF)
- Always output numeric floats for base S/R: "support","resistance"
- And for k = 1..8 (TFs: 1min=0,5m=1,15m=2,30m=3,45m=4,1h=5,4h=6,1d=7,1mth=8):
  "support k","resistance k"
- Ensure S < R for base and each k (swap if needed).

TRADE LEVELS (LONG-ONLY)
- Base → "entry","target","stoploss"
- For k=1..8 → "entry k","target k","stoploss k"
- Check ordering BEFORE returning JSON; adjust minimally if needed (RR ~1.1–2.5).

PREDICTED TARGETS
- Return "predicted_targets" with keys "1d","1w","3m" (numeric "ret" fraction + "price").

RULES
- Preserve "stock_code" and "instrument_token".
- You MAY fill "ohlcv" of the latest bar if you can infer it; if uncertain, leave None (system may overwrite).
- Output ONE valid JSON object matching the SCHEMA keys exactly. No commentary, no markdown.
- ALL price-like fields must be numeric floats (no nulls / empty strings).

SCHEMA (example):
{schema_str}

CONFIG (current):
{provided_cfg_str}

LATEST CANDLES PER TIMEFRAME (S0..S8, one per TF):
{tf_latest_json}

DAILY HISTORY (recent rows):
{historical_excerpt}

RECENT INTRADAY INDICATORS (last rows):
{recent_excerpt}
""".strip()

    # ---------- 4) LLM call (inline — no nested funcs) ----------
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=2048,
        input=[
            {
                "role": "system",
                "content": (
                    "Output strict JSON matching the SCHEMA keys exactly. "
                    "Every required key present; no nulls or empty strings. "
                    "All price-like fields must be numeric floats. No markdown, no comments. "
                    "Long-only: enforce stoploss < support < entry < target < resistance for base and k=1..8."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.output_text.strip()
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    save_forecast_csv(stock_code, cleaned)
    in_toks = getattr(getattr(resp, "usage", None), "input_tokens", 0)
    out_toks = getattr(getattr(resp, "usage", None), "output_tokens", 0)
    update_token_monitor(stock_code, in_toks, out_toks, model_used=model)

    try:
        cfg = json.loads(cleaned)
    except Exception as e:
        safe_excerpt = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
        return None, None, f"JSON parse error: {e}. Output excerpt: {safe_excerpt}"

    # ---------- 5) Post-LLM long-only sanitization & enrich ----------
    cfg = _normalize_llm_cfg(cfg, stock_block)
    cfg.setdefault("reason", [])
    cfg.setdefault("signal", "No Action")
    cfg["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg["forecast"] = "AI"

    print("[*]llm repose : ", json.dumps(cfg, indent=2))

    # latest 1m OHLCV snapshot (Redis legacy)
    try:
        last_candles = fetch_candles(stock_code, n=max(REALTIME_MINUTE_LOOKBACK, 1))
        if last_candles is not None and not last_candles.empty:
            row = last_candles.iloc[-1]

            # time → iso string if Timestamp, else str
            minute_val = row.get("minute")
            if isinstance(minute_val, pd.Timestamp):
                minute_str = minute_val.isoformat()
            else:
                minute_str = str(minute_val) if minute_val is not None else None

            # numeric fields
            open_val = (
                float(row.get("open"))
                if row.get("open") is not None and pd.notna(row.get("open"))
                else None
            )
            high_val = (
                float(row.get("high"))
                if row.get("high") is not None and pd.notna(row.get("high"))
                else None
            )
            low_val = (
                float(row.get("low"))
                if row.get("low") is not None and pd.notna(row.get("low"))
                else None
            )
            close_val = (
                float(row.get("close"))
                if row.get("close") is not None and pd.notna(row.get("close"))
                else None
            )

            vol_raw = row.get("volume")
            if vol_raw is not None and pd.notna(vol_raw):
                try:
                    volume_val = int(float(vol_raw))
                except Exception:
                    volume_val = None
            else:
                volume_val = None

            cfg["ohlcv"] = {
                "time": minute_str,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close_val,
                "volume": volume_val,
            }
        else:
            cfg["ohlcv"] = {
                "time": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None,
            }
    except Exception:
        cfg["ohlcv"] = {
            "time": None,
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
        }

    # predicted targets
    pt = cfg.get("predicted_targets", {}) or {}
    if not isinstance(pt, dict):
        pt = {}
    last_close = _last_close_from(df_recent, df_hist_daily, cfg)
    for h, cap in (("1d", 0.06), ("1w", 0.12), ("3m", 0.30)):
        slot = pt.get(h, {}) or {}
        rhat = _coerce_ret(slot.get("ret", 0.0))
        rhat = max(-cap, min(cap, rhat))
        phat = slot.get("price", None)
        try:
            phat = float(phat) if phat is not None else None
        except Exception:
            phat = float(last_close * (1.0 + rhat)) if last_close is not None else None
        if phat is None and last_close is not None:
            phat = float(last_close * (1.0 + rhat))
        pt[h] = {"ret": float(rhat), "price": phat, "method": "llm_v1"}
    cfg["predicted_targets"] = pt

    # ensure SR order base + k=1..8
    cfg["support"], cfg["resistance"] = _ensure_sr_order(
        cfg.get("support"), cfg.get("resistance")
    )
    for k in range(1, 9):
        sk, rk = _ensure_sr_order(cfg.get(f"support{k}"), cfg.get(f"resistance{k}"))
        cfg[f"support{k}"], cfg[f"resistance{k}"] = sk, rk

    # synthesize long-only triplets
    _long_only_triplet(
        cfg,
        "support",
        "resistance",
        "entry",
        "target",
        "stoploss",
        last_close=last_close,
        use_last_close_anchor=True,
    )
    # mild anchor for 1/2; band-only for others
    for k, anch, risk_p in [
        (1, True, 0.30),
        (2, True, 0.30),
        (3, False, 0.30),
        (4, False, 0.30),
        (5, False, 0.30),
        (6, False, 0.30),
        (7, False, 0.30),
        (8, False, 0.35),
    ]:
        _long_only_triplet(
            cfg,
            f"support{k}",
            f"resistance{k}",
            f"entry{k}",
            f"target{k}",
            f"stoploss{k}",
            last_close=last_close,
            use_last_close_anchor=anch,
            risk_pct_by_band=risk_p,
        )

    _push_apart_if_equal_long(cfg, "entry", "target", "resistance")
    for k in range(1, 9):
        _push_apart_if_equal_long(cfg, f"entry{k}", f"target{k}", f"resistance{k}")

    _audit_long_invariants(
        cfg, "support", "resistance", "entry", "target", "stoploss", "base"
    )
    for k in range(1, 9):
        _audit_long_invariants(
            cfg,
            f"support{k}",
            f"resistance{k}",
            f"entry{k}",
            f"target{k}",
            f"stoploss{k}",
            f"S{k}/R{k}",
        )

    try:
        _fill_all_tf_respects(cfg, stock_code)
    except Exception:
        # ensure keys exist even if counting failed
        cfg.setdefault("respected_S", 0)
        cfg.setdefault("respected_R", 0)
        for k in range(1, 9):
            cfg.setdefault(f"respected_S{k}", 0)
            cfg.setdefault(f"respected_R{k}", 0)

    # legacy width
    try:
        if cfg.get("support") is not None and cfg.get("resistance") is not None:
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

    # final missing guards
    must_have = ["entry", "target", "stoploss"] + [
        k
        for trio in [(f"entry{i}", f"target{i}", f"stoploss{i}") for i in range(1, 9)]
        for k in trio
    ]
    if any(cfg.get(k) is None for k in must_have):
        cfg["reason"].append(
            "WARNING: Some trade levels still missing after synthesis."
        )

    return cfg, cfg.get("reason", []), None
