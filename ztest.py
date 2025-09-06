# llm_forecast.py

import os
import json
import csv
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from redis_utils import get_redis, get_recent_indicators, get_recent_candles

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
- Read CONFIG, HISTORICAL DATA (~1 year daily candles), and RECENT DATA (last ~50 1-minute indicator rows).
- Update all relevant fields ONLY if justified:
  - Core levels: support/resistance (default, realtime, 1d, 1w, 3m)
  - volume_threshold, bollinger, adx/moving_averages/inside_bar/candle params, and signal
  - Trade levels: entry/target/stoploss for each S/R pair
  - Forward predicted_targets (1d, 1w, 3m)
- Produce a SHORT list of reasons (what changed and why).

SUPPORT/RESISTANCE RULES
- Infer support/resistance directly from historical + recent data.
- Always provide numeric floats for:
  - "support", "resistance"
  - "support_realtime", "resistance_realtime"
  - "support_1d", "resistance_1d"
  - "support_1w", "resistance_1w"
  - "support_3m", "resistance_3m"

TRADE LEVEL RULES
- For each S/R pair, compute and output:
  - Default SR → "entry", "target", "stoploss"
  - S1/R1 (1d) → "entry1", "target1", "stoploss1"
  - S2/R2 (1w) → "entry2", "target2", "stoploss2"
  - S3/R3 (3m) → "entry3", "target3", "stoploss3"
- If long: stoploss < entry < target
- If short: target < entry < stoploss
- Risk-reward typically between 1.1 and 2.5 unless strong trend conviction.

PREDICTED TARGETS RULES
- Return "predicted_targets" with keys "1d", "1w", "3m".
- Each must include numeric "ret" (fractional return, e.g., 0.012 = +1.2%) AND "price" (Close_t * (1 + ret)).

RULES
- Preserve "stock_code" and "instrument_token".
- You MAY include/refresh the "ohlcv" of the latest bar if you can infer it; if not sure, leave None (system may overwrite from Redis).
- Output ONE valid JSON object matching the SCHEMA shape. No commentary, no markdown.
- Required top-level fields: "reason", "signal", "predicted_targets", all SR levels, and all entry/target/stoploss keys.

SCHEMA (shape example):
{schema_str}

CONFIG (current):
{provided_cfg_str}

HISTORICAL DATA (recent daily rows):
{historical_excerpt}

RECENT DATA (last indicator rows):
{recent_excerpt}
""".strip()

    def _call_llm(model_name: str):
        resp = client.responses.create(
            model=model_name,
            temperature=temperature,
            max_output_tokens=2048,
            input=[
                {
                    "role": "system",
                    "content": "Output: strict JSON config only, matching schema shape. No markdown, no comments.",
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

        # --- Compute SR width percentages (if SR + close available) ---
        try:
            base_close = last_close
            cfg["sr_range_pct_realtime"] = _sr_width_pct(
                cfg.get("support_realtime"),
                cfg.get("resistance_realtime"),
                cfg.get("support_realtime"),
            )
            cfg["sr_range_pct_1d"] = _sr_width_pct(
                cfg.get("support_1d"), cfg.get("resistance_1d"), cfg.get("support_1d")
            )
            cfg["sr_range_pct_1w"] = _sr_width_pct(
                cfg.get("support_1w"), cfg.get("resistance_1w"), cfg.get("support_1w")
            )
            cfg["sr_range_pct_3m"] = _sr_width_pct(
                cfg.get("support_3m"), cfg.get("resistance_3m"), cfg.get("support_3m")
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


# gsheet_logger.py

import gspread
from datetime import datetime


def get_gsheet_client(sheet_name: str):
    """Authorize and return a specific Google Sheet client by name."""
    gc = gspread.service_account(filename="cred.json")
    sh = gc.open(sheet_name)
    return sh


def _get_target_fields(stock_cfg: dict):
    """
    Extract target fields from stock_cfg.get("targets", {}).

    Expected structure:
    stock_cfg["targets"] = {
        "1d": {"time": "...", "price": 123.4, "ret": 0.0123},
        "1w": {"time": "...", "price": 125.0, "ret": 0.0211},
        "3m": {"time": "...", "price": 150.5, "ret": 0.0742},
    }
    """
    tgt = stock_cfg.get("targets", {}) or {}

    def pick(key, field):
        try:
            v = (tgt.get(key) or {}).get(field)
            return v if v is not None else ""
        except Exception:
            return ""

    return {
        # 1 day
        "target_time_1d": pick("1d", "time"),
        "target_price_1d": pick("1d", "price"),
        "target_ret_1d": pick("1d", "ret"),
        # 1 week
        "target_time_1w": pick("1w", "time"),
        "target_price_1w": pick("1w", "price"),
        "target_ret_1w": pick("1w", "ret"),
        # 3 months
        "target_time_3m": pick("3m", "time"),
        "target_price_3m": pick("3m", "price"),
        "target_ret_3m": pick("3m", "ret"),
    }


def _get_pred_fields(stock_cfg: dict):
    """
    Extract prediction fields from stock_cfg.get("predicted_targets", {}).

    Expected structure:
    stock_cfg["predicted_targets"] = {
        "1d": {"price": 123.4, "ret": 0.0123, "method": "heuristic_v1"},
        "1w": {"price": 125.0, "ret": 0.0211, "method": "heuristic_v1"},
        "3m": {"price": 150.5, "ret": 0.0742, "method": "heuristic_v1"},
    }
    """
    pred = stock_cfg.get("predicted_targets", {}) or {}

    def pick(key, field):
        try:
            v = (pred.get(key) or {}).get(field)
            # Keep blanks for missing instead of 0
            return "" if v is None else v
        except Exception:
            return ""

    return {
        # 1 day
        "pred_price_1d": pick("1d", "price"),
        "pred_ret_1d": pick("1d", "ret"),
        "pred_method_1d": pick("1d", "method"),
        # 1 week
        "pred_price_1w": pick("1w", "price"),
        "pred_ret_1w": pick("1w", "ret"),
        "pred_method_1w": pick("1w", "method"),
        # 3 months
        "pred_price_3m": pick("3m", "price"),
        "pred_ret_3m": pick("3m", "ret"),
        "pred_method_3m": pick("3m", "method"),
    }


def _get_ohlcv_fields(stock_cfg: dict):
    """
    Pulls latest OHLCV + current price from config.
    Expects:
      stock_cfg["current_price"] = float | None
      stock_cfg["ohlcv"] = {"time","open","high","low","close","volume"} | None
    Case-insensitive safe reads, blanks when missing.
    """

    def pick(d, key):
        if not isinstance(d, dict) or d is None:
            return ""
        # case-insensitive
        for k, v in d.items():
            if str(k).lower() == str(key).lower():
                return "" if v is None else v
        return ""

    ohlcv = stock_cfg.get("ohlcv") or {}
    return {
        # "price_time": pick(ohlcv, "time"),
        "open": pick(ohlcv, "open"),
        "high": pick(ohlcv, "high"),
        "low": pick(ohlcv, "low"),
        "close": pick(ohlcv, "close"),
        "volume": pick(ohlcv, "volume"),
        # "current_price": (
        #     ""
        #     if stock_cfg.get("current_price") is None
        #     else stock_cfg.get("current_price")
        # ),
    }


def _blank(x):
    return "" if x is None else x


def flatten_config(stock_cfg: dict) -> dict:
    """
    Flatten nested stock config dict for Google Sheet logging.

    Includes:
    - Legacy SR + % width
    - Realtime / 1d / 1w / 3m SR + their % widths
    - Entry/Target/Stoploss for default, S1/R1, S2/R2, S3/R3
    - Respect counts for each S/R pair
    - OHLCV close & volume
    - Signal
    """

    # ---- Core SR values ----
    S = stock_cfg.get("support")
    R = stock_cfg.get("resistance")
    # SRT = stock_cfg.get("support_realtime")
    # RRT = stock_cfg.get("resistance_realtime")
    S1 = stock_cfg.get("support_1d")
    R1 = stock_cfg.get("resistance_1d")
    S2 = stock_cfg.get("support_1w")
    R2 = stock_cfg.get("resistance_1w")
    S3 = stock_cfg.get("support_3m")
    R3 = stock_cfg.get("resistance_3m")

    # ---- SR widths ----
    SR_pct = stock_cfg.get("sr_range_pct")
    # SR_rt_pct = stock_cfg.get("sr_range_pct_realtime")
    SR1_pct = stock_cfg.get("sr_range_pct_1d")
    SR2_pct = stock_cfg.get("sr_range_pct_1w")
    SR3_pct = stock_cfg.get("sr_range_pct_3m")

    # ---- Trade levels (entries/targets/stops) ----
    entry = stock_cfg.get("entry")
    target = stock_cfg.get("target")
    stoploss = stock_cfg.get("stoploss")

    entry1 = stock_cfg.get("entry1")
    target1 = stock_cfg.get("target1")
    stoploss1 = stock_cfg.get("stoploss1")

    entry2 = stock_cfg.get("entry2")
    target2 = stock_cfg.get("target2")
    stoploss2 = stock_cfg.get("stoploss2")

    entry3 = stock_cfg.get("entry3")
    target3 = stock_cfg.get("target3")
    stoploss3 = stock_cfg.get("stoploss3")

    # ---- Respect counts (new) ----
    respected_S = stock_cfg.get("respected_S")
    respected_R = stock_cfg.get("respected_R")
    respected_S1 = stock_cfg.get("respected_S1")
    respected_R1 = stock_cfg.get("respected_R1")
    respected_S2 = stock_cfg.get("respected_S2")
    respected_R2 = stock_cfg.get("respected_R2")
    respected_S3 = stock_cfg.get("respected_S3")
    respected_R3 = stock_cfg.get("respected_R3")

    flat = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock_code": stock_cfg.get("stock_code"),
        # OHLCV
        "close": stock_cfg.get("ohlcv", {}).get("close"),
        "volume": stock_cfg.get("ohlcv", {}).get("volume"),
        # Legacy/default SR (mapped to 1d in your pipeline)
        "S": _blank(S),
        "R": _blank(R),
        "SR_pct": _blank(SR_pct),
        # Realtime SR
        # "S_RT": _blank(SRT),
        # "R_RT": _blank(RRT),
        # "SR_RT_pct": _blank(SR_rt_pct),
        # 1d SR
        "S1": _blank(S1),
        "R1": _blank(R1),
        "SR1_pct": _blank(SR1_pct),
        # 1w SR
        "S2": _blank(S2),
        "R2": _blank(R2),
        "SR2_pct": _blank(SR2_pct),
        # 3m SR
        "S3": _blank(S3),
        "R3": _blank(R3),
        "SR3_pct": _blank(SR3_pct),
        # Trade levels
        "entry": _blank(entry),
        "target": _blank(target),
        "stoploss": _blank(stoploss),
        "entry1": _blank(entry1),
        "target1": _blank(target1),
        "stoploss1": _blank(stoploss1),
        "entry2": _blank(entry2),
        "target2": _blank(target2),
        "stoploss2": _blank(stoploss2),
        "entry3": _blank(entry3),
        "target3": _blank(target3),
        "stoploss3": _blank(stoploss3),
        # Respect counts (new)
        "respected_S": _blank(respected_S),
        "respected_R": _blank(respected_R),
        "respected_S1": _blank(respected_S1),
        "respected_R1": _blank(respected_R1),
        "respected_S2": _blank(respected_S2),
        "respected_R2": _blank(respected_R2),
        "respected_S3": _blank(respected_S3),
        "respected_R3": _blank(respected_R3),
        # Meta
        "signal": stock_cfg.get("signal"),
    }

    # flat = {
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "stock_code": stock_cfg.get("stock_code"),
    #     # ----- Legacy SR (kept for backward compatibility) -----
    #     "support": stock_cfg.get("support"),
    #     "resistance": stock_cfg.get("resistance"),
    #     "sr_range_pct": stock_cfg.get("sr_range_pct"),
    #     # ----- NEW SR sets -----
    #     # Realtime (1-min window)
    #     # "support_realtime": stock_cfg.get("support_realtime"),
    #     # "resistance_realtime": stock_cfg.get("resistance_realtime"),
    #     # "sr_range_pct_realtime": stock_cfg.get("sr_range_pct_realtime"),
    #     # 1 Day window
    #     "support_1d": stock_cfg.get("support_1d"),
    #     "resistance_1d": stock_cfg.get("resistance_1d"),
    #     "sr_range_pct_1d": stock_cfg.get("sr_range_pct_1d"),
    #     # 1 Week window
    #     "support_1w": stock_cfg.get("support_1w"),
    #     "resistance_1w": stock_cfg.get("resistance_1w"),
    #     "sr_range_pct_1w": stock_cfg.get("sr_range_pct_1w"),
    #     # 3 Months window
    #     "support_3m": stock_cfg.get("support_3m"),
    #     "resistance_3m": stock_cfg.get("resistance_3m"),
    #     "sr_range_pct_3m": stock_cfg.get("sr_range_pct_3m"),
    #     # Misc (existing fields)
    #     "volume_threshold": stock_cfg.get("volume_threshold"),
    #     "signal": stock_cfg.get("signal"),
    #     # "forecast": stock_cfg.get("forecast"),
    #     # "last_updated": stock_cfg.get("last_updated"),
    # }

    # Latest OHLCV
    # flat.update(_get_ohlcv_fields(stock_cfg))

    # Predicted (forecasts)
    # flat.update(_get_pred_fields(stock_cfg))

    # If you also want realized target fields, uncomment:
    # flat.update(_get_target_fields(stock_cfg))

    return flat


def ensure_headers(worksheet, desired_headers: list[str]) -> list[str]:
    """
    Ensure the sheet's first row contains all desired headers.
    - If the sheet is empty, write desired_headers.
    - If headers exist, append any missing new headers to the end (no clearing).
    Returns the final header order present on the sheet.
    """
    first_row = worksheet.row_values(1)

    if not first_row:
        # Empty sheet → write fresh headers
        worksheet.update("A1", [desired_headers])
        return desired_headers

    # Extend with any missing columns (preserve existing order)
    missing = [h for h in desired_headers if h not in first_row]
    if missing:
        new_headers = first_row + missing
        # Overwrite only the header row with the extended header list
        worksheet.update("A1", [new_headers])
        return new_headers

    return first_row


# ============== NEW: small utilities for A1 ranges & header lookups ==============


def _col_index(headers: list[str], col_name: str) -> int | None:
    """Return 0-based index of a header name; None if missing."""
    try:
        return [h.strip() for h in headers].index(col_name)
    except ValueError:
        return None


def _a1_from_row(headers_len: int, row_number: int) -> str:
    """
    Make an A1 range for an entire row given header length.
    Example: headers_len=10, row_number=5 -> 'A5:J5'
    """

    def _col_letter(n: int) -> str:
        # 1-based to letters
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    start_col_letter = "A"
    end_col_letter = _col_letter(headers_len)
    return f"{start_col_letter}{row_number}:{end_col_letter}{row_number}"


# ===================== UPDATED: upsert instead of append =========================


def log_config_upsert(
    stock_cfg: dict, sheet_name: str, tab_name: str, key_col: str = "stock_code"
):
    """
    Upsert a single row by `key_col` (default: 'stock_code'):
      - Ensures headers (adds missing columns if needed).
      - If a row with the same stock_code exists, updates that row only.
      - Otherwise appends a new row.

    :param stock_cfg: dict of stock configuration
    :param sheet_name: Google Sheet file name
    :param tab_name: Worksheet/tab name within the sheet
    :param key_col: Column used as unique key (must exist in headers)
    """
    sh = get_gsheet_client(sheet_name)
    try:
        worksheet = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=tab_name, rows="20000", cols="100")

    # 1) Flatten and ensure headers
    flat_cfg = flatten_config(stock_cfg)
    desired_headers = list(flat_cfg.keys())
    final_headers = ensure_headers(worksheet, desired_headers)

    # 2) Ensure key column exists in headers
    key_idx0 = _col_index(final_headers, key_col)
    if key_idx0 is None:
        # If somehow missing, extend headers and recompute
        final_headers = ensure_headers(worksheet, final_headers + [key_col])
        key_idx0 = _col_index(final_headers, key_col)
        if key_idx0 is None:
            raise RuntimeError(
                f"Cannot find or create key column '{key_col}' in sheet headers."
            )

    # 3) Build row aligned to final_headers
    #    (Any header not in flat_cfg becomes "")
    row_values = [flat_cfg.get(h, "") for h in final_headers]

    # 4) Upsert by key
    key_value = flat_cfg.get(key_col, "")
    if key_value is None or str(key_value).strip() == "":
        raise ValueError(
            f"Upsert requires non-empty '{key_col}' in data. Got: {key_value!r}"
        )

    # Only scan the single key column to find matching row
    # (faster and avoids false positives in other columns)
    key_col_1based = key_idx0 + 1
    # Read only that column (from row 2 downwards; row 1 is header)
    key_column_cells = worksheet.col_values(key_col_1based)[1:]  # skip header row

    target_row_number = None
    for i, cell_val in enumerate(
        key_column_cells, start=2
    ):  # sheet rows start at 1; row 1 is header
        if str(cell_val).strip() == str(key_value).strip():
            target_row_number = i
            break

    if target_row_number is None:
        # 5a) No match -> append as new row
        worksheet.append_row(row_values, value_input_option="USER_ENTERED")
        print(f"[INFO] UPSERT (append) for {key_value} → {sheet_name}/{tab_name}")
    else:
        # 5b) Match found -> update that exact row range (A1 style)
        rng = _a1_from_row(len(final_headers), target_row_number)
        worksheet.update(rng, [row_values], value_input_option="USER_ENTERED")
        print(
            f"[INFO] UPSERT (update row {target_row_number}) for {key_value} → {sheet_name}/{tab_name}"
        )


# def log_config_update(stock_cfg: dict, sheet_name: str, tab_name: str):
#     """
#     Append config update as new row in a tab of a specific Google Sheet.

#     :param stock_cfg: dict of stock configuration
#     :param sheet_name: Google Sheet file name (e.g. 'TradingStocksLogs')
#     :param tab_name: Worksheet/tab name within the sheet (e.g. 'stocks_30_gpt')
#     """
#     sh = get_gsheet_client(sheet_name)

#     try:
#         worksheet = sh.worksheet(tab_name)
#     except gspread.exceptions.WorksheetNotFound:
#         # Pre-allocate more columns to reduce future reflows
#         worksheet = sh.add_worksheet(title=tab_name, rows="20000", cols="50")

#     # Flatten config → dict
#     flat_cfg = flatten_config(stock_cfg)
#     desired_headers = list(flat_cfg.keys())

#     # Ensure headers exist/extend without clearing past data
#     final_headers = ensure_headers(worksheet, desired_headers)

#     # Build the row aligned to final_headers (some columns might not be in flat_cfg yet)
#     row = [flat_cfg.get(col, "") for col in final_headers]

#     worksheet.append_row(row, value_input_option="USER_ENTERED")

#     print(
#         f"[INFO] Config update logged for {stock_cfg.get('stock_code')} → {sheet_name}/{tab_name}"
#     )


def log_config_update(stock_cfg: dict, sheet_name: str, tab_name: str):
    """
    Backwards-compatible wrapper: now performs UPSERT by 'stock_code'
    instead of always appending a new row.
    """
    return log_config_upsert(stock_cfg, sheet_name, tab_name, key_col="stock_code")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# if __name__ == "__main__":
#     # Example manual test…
#     pass


# # gsheet_logger.py

# import gspread
# from datetime import datetime


# def get_gsheet_client(sheet_name: str):
#     """Authorize and return a specific Google Sheet client by name."""
#     gc = gspread.service_account(filename="cred.json")
#     sh = gc.open(sheet_name)
#     return sh


# def _get_target_fields(stock_cfg: dict):
#     """
#     Extract target fields from stock_cfg.get("targets", {}).

#     Expected structure:
#     stock_cfg["targets"] = {
#         "1d": {"time": "...", "price": 123.4, "ret": 0.0123},
#         "1w": {"time": "...", "price": 125.0, "ret": 0.0211},
#         "3m": {"time": "...", "price": 150.5, "ret": 0.0742},
#     }
#     """
#     tgt = stock_cfg.get("targets", {}) or {}

#     def pick(key, field):
#         try:
#             v = (tgt.get(key) or {}).get(field)
#             return v if v is not None else ""
#         except Exception:
#             return ""

#     return {
#         # 1 day
#         "target_time_1d": pick("1d", "time"),
#         "target_price_1d": pick("1d", "price"),
#         "target_ret_1d": pick("1d", "ret"),
#         # 1 week
#         "target_time_1w": pick("1w", "time"),
#         "target_price_1w": pick("1w", "price"),
#         "target_ret_1w": pick("1w", "ret"),
#         # 3 months
#         "target_time_3m": pick("3m", "time"),
#         "target_price_3m": pick("3m", "price"),
#         "target_ret_3m": pick("3m", "ret"),
#     }


# def _get_pred_fields(stock_cfg: dict):
#     """
#     Extract prediction fields from stock_cfg.get("predicted_targets", {}).

#     Expected structure:
#     stock_cfg["predicted_targets"] = {
#         "1d": {"price": 123.4, "ret": 0.0123, "method": "heuristic_v1"},
#         "1w": {"price": 125.0, "ret": 0.0211, "method": "heuristic_v1"},
#         "3m": {"price": 150.5, "ret": 0.0742, "method": "heuristic_v1"},
#     }
#     """
#     pred = stock_cfg.get("predicted_targets", {}) or {}

#     def pick(key, field):
#         try:
#             v = (pred.get(key) or {}).get(field)
#             # Keep blanks for missing instead of 0
#             return "" if v is None else v
#         except Exception:
#             return ""

#     return {
#         # 1 day
#         "pred_price_1d": pick("1d", "price"),
#         "pred_ret_1d": pick("1d", "ret"),
#         "pred_method_1d": pick("1d", "method"),
#         # 1 week
#         "pred_price_1w": pick("1w", "price"),
#         "pred_ret_1w": pick("1w", "ret"),
#         "pred_method_1w": pick("1w", "method"),
#         # 3 months
#         "pred_price_3m": pick("3m", "price"),
#         "pred_ret_3m": pick("3m", "ret"),
#         "pred_method_3m": pick("3m", "method"),
#     }


# def _get_ohlcv_fields(stock_cfg: dict):
#     """
#     Pulls latest OHLCV + current price from config.
#     Expects:
#       stock_cfg["current_price"] = float | None
#       stock_cfg["ohlcv"] = {"time","open","high","low","close","volume"} | None
#     Case-insensitive safe reads, blanks when missing.
#     """

#     def pick(d, key):
#         if not isinstance(d, dict) or d is None:
#             return ""
#         # case-insensitive
#         for k, v in d.items():
#             if str(k).lower() == str(key).lower():
#                 return "" if v is None else v
#         return ""

#     ohlcv = stock_cfg.get("ohlcv") or {}
#     return {
#         # "price_time": pick(ohlcv, "time"),
#         "open": pick(ohlcv, "open"),
#         "high": pick(ohlcv, "high"),
#         "low": pick(ohlcv, "low"),
#         "close": pick(ohlcv, "close"),
#         "volume": pick(ohlcv, "volume"),
#         # "current_price": (
#         #     ""
#         #     if stock_cfg.get("current_price") is None
#         #     else stock_cfg.get("current_price")
#         # ),
#     }


# def flatten_config(stock_cfg: dict) -> dict:
#     """Flatten nested stock config dict for Google Sheet logging."""
#     flat = {
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "stock_code": stock_cfg.get("stock_code"),
#         "support": stock_cfg.get("support"),
#         "resistance": stock_cfg.get("resistance"),
#         "volume_threshold": stock_cfg.get("volume_threshold"),
#         "signal": stock_cfg.get("signal"),
#         "sr_range_pct": stock_cfg.get("sr_range_pct"),
#         # "forecast": stock_cfg.get("forecast"),
#         # "last_updated": stock_cfg.get("last_updated"),
#     }

#     flat.update(_get_ohlcv_fields(stock_cfg))

#     # realized (labels)
#     # flat.update(_get_target_fields(stock_cfg))

#     # predicted (forecasts)  ← NEW
#     flat.update(_get_pred_fields(stock_cfg))

#     # # Bollinger
#     # boll = stock_cfg.get("bollinger", {})
#     # flat.update(
#     #     {
#     #         "boll_mid_price": boll.get("mid_price"),
#     #         "boll_upper_band": boll.get("upper_band"),
#     #         "boll_lower_band": boll.get("lower_band"),
#     #     }
#     # )

#     # # MACD
#     # macd = stock_cfg.get("macd", {})
#     # flat.update(
#     #     {
#     #         "macd_signal_line": macd.get("signal_line"),
#     #         "macd_histogram": macd.get("histogram"),
#     #         "macd_ma_fast": macd.get("ma_fast"),
#     #         "macd_ma_slow": macd.get("ma_slow"),
#     #         "macd_ma_signal": macd.get("ma_signal"),
#     #     }
#     # )

#     # # ADX
#     # adx = stock_cfg.get("adx", {})
#     # flat.update(
#     #     {
#     #         "adx_period": adx.get("period"),
#     #         "adx_threshold": adx.get("threshold"),
#     #     }
#     # )

#     # # Moving Averages
#     # ma = stock_cfg.get("moving_averages", {})
#     # flat.update(
#     #     {
#     #         "ma_fast": ma.get("ma_fast"),
#     #         "ma_slow": ma.get("ma_slow"),
#     #     }
#     # )

#     # # Inside Bar
#     # inside = stock_cfg.get("inside_bar", {})
#     # flat.update(
#     #     {
#     #         "inside_lookback": inside.get("lookback"),
#     #     }
#     # )

#     # # Candle
#     # candle = stock_cfg.get("candle", {})
#     # flat.update(
#     #     {
#     #         "candle_min_body_percent": candle.get("min_body_percent"),
#     #     }
#     # )

#     return flat


# def ensure_headers(worksheet, desired_headers: list[str]) -> list[str]:
#     """
#     Ensure the sheet's first row contains all desired headers.
#     - If the sheet is empty, write desired_headers.
#     - If headers exist, append any missing new headers to the end (no clearing).
#     Returns the final header order present on the sheet.
#     """
#     first_row = worksheet.row_values(1)

#     if not first_row:
#         # Empty sheet → write fresh headers
#         worksheet.update("A1", [desired_headers])
#         return desired_headers

#     # Extend with any missing columns (preserve existing order)
#     missing = [h for h in desired_headers if h not in first_row]
#     if missing:
#         new_headers = first_row + missing
#         # Overwrite only the header row with the extended header list
#         worksheet.update("A1", [new_headers])
#         return new_headers

#     return first_row


# def log_config_update(stock_cfg: dict, sheet_name: str, tab_name: str):
#     """
#     Append config update as new row in a tab of a specific Google Sheet.

#     :param stock_cfg: dict of stock configuration
#     :param sheet_name: Google Sheet file name (e.g. 'TradingStocksLogs')
#     :param tab_name: Worksheet/tab name within the sheet (e.g. 'stocks_30_gpt')
#     """
#     sh = get_gsheet_client(sheet_name)

#     try:
#         worksheet = sh.worksheet(tab_name)
#     except gspread.exceptions.WorksheetNotFound:
#         # Pre-allocate more columns to reduce future reflows
#         worksheet = sh.add_worksheet(title=tab_name, rows="20000", cols="50")

#     # Flatten config → dict
#     flat_cfg = flatten_config(stock_cfg)
#     desired_headers = list(flat_cfg.keys())

#     # Ensure headers exist/extend without clearing past data
#     final_headers = ensure_headers(worksheet, desired_headers)

#     # Build the row aligned to final_headers (some columns might not be in flat_cfg yet)
#     row = [flat_cfg.get(col, "") for col in final_headers]

#     worksheet.append_row(row, value_input_option="USER_ENTERED")

#     print(
#         f"[INFO] Config update logged for {stock_cfg.get('stock_code')} → {sheet_name}/{tab_name}"
#     )


# # if __name__ == "__main__":
# #     from datetime import datetime

# #     # --- Mock config payload (matches flattener fields) ---
# #     test_cfg = {
# #         "stock_code": "CDSL",
# #         "support": 1565.94,
# #         "resistance": 1571.10,
# #         "volume_threshold": 7478,
# #         "signal": "Hold",
# #         "forecast": "basic_algo",
# #         "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

# #         # NEW: latest OHLCV (from Redis candle) + current price
# #         "ohlcv": {
# #             "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #             "open": 1568.00,
# #             "high": 1572.50,
# #             "low": 1561.25,
# #             "close": 1569.40,
# #             "volume": 123456,
# #         },
# #         "current_price": 1569.40,

# #         # Realized targets (labels) — sample values
# #         "targets": {
# #             "1d": {"time": "2025-08-29 15:30:00", "price": 1580.0, "ret": 0.0068, "status": "resolved"},
# #             "1w": {"time": "2025-09-05 15:30:00", "price": 1605.5, "ret": 0.0230, "status": "resolved"},
# #             "3m": {"time": "2025-11-28 15:30:00", "price": 1710.0, "ret": 0.0895, "status": "pending"},  # example pending
# #         },

# #         # Predicted targets (forecasts) — sample values
# #         "predicted_targets": {
# #             "1d": {"price": 1574.2, "ret": 0.0031, "method": "heuristic_v1"},
# #             "1w": {"price": 1601.8, "ret": 0.0207, "method": "heuristic_v1"},
# #             "3m": {"price": 1690.0, "ret": 0.0769, "method": "heuristic_v1"},
# #         },

# #         # (Optional) keep your other nested blocks; flattener ignores them unless used
# #         "bollinger": {"mid_price": 1568.52, "upper_band": 1571.10, "lower_band": 1565.94},
# #         "macd": {"signal_line": -0.35, "histogram": 0.01, "ma_fast": 1580.9, "ma_slow": 1581.1, "ma_signal": 1580.9},
# #         "adx": {"period": 14, "threshold": 20},
# #         "moving_averages": {"ma_fast": 9, "ma_slow": 20},
# #         "inside_bar": {"lookback": 1},
# #         "candle": {"min_body_percent": 0.7},
# #         "reason": ["test reason"],
# #     }

# #     SHEET_NAME = "TradingStocksLogs"  # your sheet name
# #     TAB_NAME   = "test_tab"           # temporary tab for testing

# #     try:
# #         log_config_update(test_cfg, SHEET_NAME, TAB_NAME)
# #         print("[✅ SUCCESS] Test log written. Check your Google Sheet.")
# #     except Exception as e:
# #         print("[❌ ERROR] Google Sheets test failed:", e)
