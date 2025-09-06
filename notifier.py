# notifier.py
# -*- coding: utf-8 -*-

"""
SR/S1R1/S2R2/S3R3 level notifier for Telegram.
Maps your config fields to:
  SR   -> entry,  target,  stoploss
  S1R1 -> entry1, target1, stoploss1
  S2R2 -> entry2, target2, stoploss2
  S3R3 -> entry3, target3, stoploss3

Exports:
  notify_sr_levels(code, df, stock_cfg, *, redis_client=None, source_tag=None, ttl_sec=86400)

Usage in monitor_shard (after is_breakout):
    from notifier import notify_sr_levels
    notify_sr_levels(code, df, stock, redis_client=r, source_tag=stock.get("forecast"))

Relies on your existing telegram_alert30b functions.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import time

import pandas as pd

from telegram_alert30b import (
    send_trade_alert,
    send_error_alert,
)

# -----------------------
# State (Redis or memory)
# -----------------------

_MEM_STATE: Dict[str, Dict[str, Any]] = {}  # key -> dict


def _k(code: str, pair: str) -> str:
    # one state per (stock, pair)
    return f"sr_notif:{code}:{pair}"


def _get_state(code: str, pair: str, redis_client=None) -> Dict[str, Any]:
    """
    State fields:
      in_trade: bool
      entry_sent, s1_sent, s2_sent: bool
      t: int (epoch)
    """
    key = _k(code, pair)
    now = int(time.time())
    if redis_client:
        raw = redis_client.get(key)
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        st = {
            "in_trade": False,
            "entry_sent": False,
            "s1_sent": False,
            "s2_sent": False,
            "t": now,
        }
        redis_client.set(key, json.dumps(st))
        return st
    st = _MEM_STATE.get(key)
    if st is None:
        st = {
            "in_trade": False,
            "entry_sent": False,
            "s1_sent": False,
            "s2_sent": False,
            "t": now,
        }
        _MEM_STATE[key] = st
    return st


def _set_state(
    code: str, pair: str, st: Dict[str, Any], redis_client=None, ttl_sec: int = 86400
) -> None:
    st["t"] = int(time.time())
    key = _k(code, pair)
    if redis_client:
        redis_client.set(key, json.dumps(st), ex=ttl_sec)
    else:
        _MEM_STATE[key] = st


# -------------
# Core helpers
# -------------


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.2f}" if abs(x) >= 100 else f"{x:.3f}"


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _detect_side(entry: float, target: float, stop: float) -> str:
    # Long if target > entry > stop; Short if target < entry < stop; else infer by distances.
    if (target > entry) and (entry > stop):
        return "long"
    if (target < entry) and (entry < stop):
        return "short"
    return "long" if abs(target - entry) >= abs(stop - entry) else "short"


def _crossed_up(prev_: float, now_: float, level: float) -> bool:
    return (prev_ < level) and (now_ >= level)


def _crossed_dn(prev_: float, now_: float, level: float) -> bool:
    return (prev_ > level) and (now_ <= level)


def _pairs_from_config(cfg: Dict[str, Any]) -> List[Tuple[str, float, float, float]]:
    """
    Returns list of (pair_name, entry, target(S1), stoploss(S2))
    Only includes sets that are all finite numbers.
    """
    out: List[Tuple[str, float, float, float]] = []
    mapping = [
        ("SR", "entry", "target", "stoploss"),
        ("S1R1", "entry1", "target1", "stoploss1"),
        ("S2R2", "entry2", "target2", "stoploss2"),
        ("S3R3", "entry3", "target3", "stoploss3"),
    ]
    for pair, e_key, t_key, s_key in mapping:
        e, t, s = cfg.get(e_key), cfg.get(t_key), cfg.get(s_key)
        try:
            if e is None or t is None or s is None:
                continue
            e, t, s = float(e), float(t), float(s)
            if not (math.isfinite(e) and math.isfinite(t) and math.isfinite(s)):
                continue
            out.append((pair, e, t, s))
        except Exception:
            continue
    return out


# --------------------------
# Public: notify SR levels
# --------------------------


def notify_sr_levels(
    code: str,
    df: pd.DataFrame,
    stock_cfg: Dict[str, Any],
    *,
    redis_client=None,
    source_tag: Optional[str] = None,
    ttl_sec: int = 86400,
) -> None:
    """
    Sends Telegram messages for each SR pair:
      - 🚀 ENTRY   once when price crosses the entry level (direction-aware)
      - 🎯 S1      once when price hits the target
      - 🛑 S2      once when price hits the stoploss

    Args:
      code: stock code (e.g., "CDSL")
      df: DataFrame with columns ["Timestamp","Open","High","Low","Close","Volume"]
      stock_cfg: full config dict containing entry/target/stoploss fields
      redis_client: optional redis for persistent flags
      source_tag: tag shown in header; defaults to stock_cfg["forecast"] or "ALGO"
      ttl_sec: TTL for state keys (default 1 day)
    """
    try:
        ts = df["Timestamp"].iloc[-1]
        price_now = float(df["Close"].iloc[-1])
        price_prev = float(df["Close"].iloc[-2]) if len(df) > 1 else price_now
        tag = (source_tag or stock_cfg.get("forecast") or "ALGO").upper()

        for pair, entry, target, stop in _pairs_from_config(stock_cfg):
            side = _detect_side(entry, target, stop)
            st = _get_state(code, pair, redis_client)

            # --- ENTRY detection
            if not st["entry_sent"]:
                hit = (
                    _crossed_up(price_prev, price_now, entry)
                    if side == "long"
                    else _crossed_dn(price_prev, price_now, entry)
                )
                if hit:
                    rr = abs(target - entry) / max(1e-9, abs(entry - stop))
                    text = (
                        f"🚀 *ENTRY* [{pair}] ({side.upper()})\n"
                        f"Entry: *{_fmt(entry)}* | S1(Target): *{_fmt(target)}* | S2(SL): *{_fmt(stop)}*\n"
                        f"Live: *{_fmt(price_now)}* | RR≈ *{rr:.2f}*\n"
                        f"_S1 is target alert, S2 is stoploss alert._"
                    )
                    send_trade_alert(code, text, price_now, ts, source_tag=tag)
                    st["entry_sent"], st["in_trade"] = True, True
                    _set_state(code, pair, st, redis_client, ttl_sec)

            # --- In-trade: S1 or S2
            if st["in_trade"]:
                if side == "long":
                    s1_hit = _crossed_up(price_prev, price_now, target)
                    s2_hit = _crossed_dn(price_prev, price_now, stop)
                    gain_pct = _pct(target, entry)
                    loss_pct = _pct(stop, entry)
                else:
                    s1_hit = _crossed_dn(price_prev, price_now, target)
                    s2_hit = _crossed_up(price_prev, price_now, stop)
                    gain_pct = _pct(entry, target)  # for short, entry -> lower target
                    loss_pct = _pct(entry, stop)

                if (not st["s1_sent"]) and s1_hit:
                    text = (
                        f"🎯 *S1 TARGET HIT* [{pair}] ({side.upper()})\n"
                        f"Target: *{_fmt(target)}* | Entry: *{_fmt(entry)}*\n"
                        f"P/L: *{gain_pct:.2f}%* from entry\n"
                        f"Live: *{_fmt(price_now)}*"
                    )
                    send_trade_alert(code, text, price_now, ts, source_tag=tag)
                    st["s1_sent"] = True
                    st["in_trade"] = (
                        False  # assume flat after target; change if you trail
                    )
                    _set_state(code, pair, st, redis_client, ttl_sec)

                if (not st["s2_sent"]) and s2_hit:
                    text = (
                        f"🛑 *S2 STOPLOSS HIT* [{pair}] ({side.UPPER() if hasattr(side,'UPPER') else side.upper()})\n"
                        f"Stop: *{_fmt(stop)}* | Entry: *{_fmt(entry)}*\n"
                        f"P/L: *{loss_pct:.2f}%* from entry\n"
                        f"Live: *{_fmt(price_now)}*"
                    )
                    send_trade_alert(code, text, price_now, ts, source_tag=tag)
                    st["s2_sent"] = True
                    st["in_trade"] = False
                    _set_state(code, pair, st, redis_client, ttl_sec)

    except Exception as e:
        try:
            send_error_alert(f"[{code}] notifier error: {type(e).__name__}: {e}")
        finally:
            # swallow to keep shard loop alive
            pass


def is_breakout(df, resistance, support, config):
    if len(df) < 5:
        return None, None, None, "Insufficient data"

    current = df.iloc[-1]

    close = current["Close"]
    volume = current["Volume"]
    vol_thresh = config.get("volume_threshold", 0)
    action = config.get("signal", "No Action")
    rsi = current.get("RSI", None)

    # Breakout logic
    if close > resistance:
        reason = f"📈 Breakout: Close ({close:.2f}) > Resistance ({resistance:.2f})"

        # Volume filter
        if volume < vol_thresh:
            reason += " | Low Volume"
        else:
            reason += " | Volume OK"

        # Bollinger confirmation
        if close > current["BB_Upper"]:
            reason += " | BB confirms"
        else:
            reason += " | BB no confirm"

        # ADX strength
        if current["ADX"] < config["adx"]["threshold"]:
            reason += f" | ADX={round(current['ADX'], 1)} < threshold"
        else:
            reason += f" | ADX={round(current['ADX'], 1)} confirms"

        # RSI confirmation
        if rsi is not None:
            if rsi > 50 and rsi < 80:
                reason += f" | RSI={round(rsi,1)} confirms"
                return "breakout", close, resistance, reason
            else:
                reason += f" | RSI={round(rsi,1)} invalid"
                return None, None, None, reason
        else:
            reason += " | RSI missing"
            return "breakout", close, resistance, reason

    # Breakdown logic
    elif close < support:
        reason = f"📉 Breakdown: Close ({close:.2f}) < Support ({support:.2f})"

        # Volume filter
        if volume < vol_thresh:
            reason += " | Low Volume"
        else:
            reason += " | Volume OK"

        # Bollinger confirmation
        if close < current["BB_Lower"]:
            reason += " | BB confirms"
        else:
            reason += " | BB no confirm"

        # ADX strength
        if current["ADX"] < config["adx"]["threshold"]:
            reason += f" | ADX={round(current['ADX'], 1)} < threshold"
        else:
            reason += f" | ADX={round(current['ADX'], 1)} confirms"

        # RSI confirmation
        if rsi is not None:
            if rsi < 50 and rsi > 20:
                reason += f" | RSI={round(rsi,1)} confirms"
                return "breakdown", close, support, reason
            else:
                reason += f" | RSI={round(rsi,1)} invalid"
                return None, None, None, reason
        else:
            reason += " | RSI missing"
            return "breakdown", close, support, reason

    return None, None, None, "No breakout/breakdown"
