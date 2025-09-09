# collector.py (tick collector + multi-timeframe candle builder)
import os
import json
import logging
from multiprocessing import Process
from datetime import datetime
from kiteconnect import KiteTicker
from dotenv import load_dotenv
from redis_utils import (
    get_redis,
    # backward-compat 1m helpers (kept)
    floor_minute,
    set_current_candle,
    finalize_and_roll_new_candle,
    # new generic TF helpers
    floor_period,
    set_current_candle_tf,
    finalize_and_roll_new_candle_tf,
)

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
CONFIG_FILE = "config30a.json"

r = get_redis()

# ==== CONFIG: shards ====
SHARDS = 2
SHARD_MODE = "roundrobin"  # ("contiguous" or "roundrobin")

# Timeframe definitions: tf label -> bucket size in minutes
TF_DEFS = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "45min": 45,
    "1hour": 60,
    "4hour": 240,
}


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return data.get("stocks", [])
    except Exception as e:
        print(f"[Config Error] Failed to load {CONFIG_FILE}: {e}")
        return []


def split_contiguous_equally(stocks, shard_index, shard_count):
    n = len(stocks)
    base = n // shard_count
    rem = n % shard_count
    start = shard_index * base + min(shard_index, rem)
    end = start + base + (1 if shard_index < rem else 0)
    return stocks[start:end]


def assign_stocks(stocks, shard_index, shard_count, mode="contiguous"):
    if shard_count <= 1:
        return stocks
    if mode == "roundrobin":
        return [s for i, s in enumerate(stocks) if i % shard_count == shard_index]
    return split_contiguous_equally(stocks, shard_index, shard_count)


def start_collector(
    stock_configs, shard_index=0, shard_count=1, shard_mode="contiguous"
):
    logging.basicConfig(level=logging.INFO, force=True)

    assigned = assign_stocks(stock_configs, shard_index, shard_count, shard_mode)
    if not assigned:
        print(
            f"⚠️ Shard {shard_index + 1}/{shard_count} has no assigned stocks. Exiting."
        )
        return

    instrument_map = {s["instrument_token"]: s["stock_code"].upper() for s in assigned}

    # Per-symbol, per-timeframe state
    # Structure: states[CODE][TF] = {"bucket": str, "current": {...}}
    # We also track last cumulative daily volume to compute delta
    states = {
        code: {
            "_last_cum_vol": None,
            **{tf: {"bucket": None, "current": None} for tf in TF_DEFS.keys()},
        }
        for code in instrument_map.values()
    }

    kws = KiteTicker(API_KEY, ACCESS_TOKEN)

    def _update_timeframe(
        code: str, tf: str, bucket_key: str, price: float, delta_vol: int
    ):
        st = states[code][tf]

        # New bucket? finalize previous and start a new one.
        if st["bucket"] is None or bucket_key != st["bucket"]:
            if st["current"]:
                finalize_and_roll_new_candle_tf(
                    r, code, tf, st["current"], max_candles=300
                )

            st["current"] = {
                "bucket": bucket_key,  # ISO string start of bucket (UTC minutes)
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": max(0, delta_vol),  # add this tick's delta
            }
            st["bucket"] = bucket_key
        else:
            cur = st["current"]
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price
            cur["volume"] += max(0, delta_vol)

        # Persist snapshot for this TF
        cur = st["current"]
        set_current_candle_tf(
            r,
            code,
            tf,
            st["bucket"],
            cur["open"],
            cur["high"],
            cur["low"],
            cur["close"],
            cur["volume"],
        )

    def on_ticks(ws, ticks):
        for tick in ticks:
            token = tick.get("instrument_token")
            code = instrument_map.get(token)
            if not code:
                continue

            ts = tick.get("exchange_timestamp")  # tz-aware datetime
            if not ts:
                continue

            price = float(tick.get("last_price"))
            cum_vol = int(
                tick.get("volume_traded")
            )  # cumulative daily volume from exchange

            # Compute delta volume once per symbol
            prev_cum = states[code]["_last_cum_vol"]
            if prev_cum is None:
                delta_vol = 0
            else:
                delta_vol = max(0, cum_vol - prev_cum)
            states[code]["_last_cum_vol"] = cum_vol

            # ---- Update all TFs ----
            # 1m path (backward-compat keys as well)
            minute_key = floor_minute(ts)
            # Keep legacy "current minute" hash & list
            # Finalize previous minute candle & start new happens via TF engine too,
            # but we retain the legacy write so any existing consumers keep working.
            # We synthesize the 1m candle state off the generic TF state.
            bucket_1m = floor_period(ts, TF_DEFS["1min"])
            _update_timeframe(code, "1min", bucket_1m, price, delta_vol)

            # Mirror current 1m candle into legacy hash for compatibility
            cur_1m = states[code]["1min"]["current"]
            set_current_candle(
                r,
                code,
                minute_key,
                cur_1m["open"],
                cur_1m["high"],
                cur_1m["low"],
                cur_1m["close"],
                cur_1m["volume"],
            )
            # NOTE: legacy finalize (finalize_and_roll_new_candle) happens only when TF engine rolls,
            # because we store finalized 1m into candles:{code}:1min now.
            # If you need *also* candles:{code} legacy list, uncomment below:
            # finalize_and_roll_new_candle(r, code, {...}) on 1m bucket flip.

            # Other TFs
            for tf, mins in TF_DEFS.items():
                if tf == "1min":
                    continue
                bucket_key = floor_period(ts, mins)
                _update_timeframe(code, tf, bucket_key, price, delta_vol)

            print(f"[{code}] tick ok")

    def on_connect(ws, response):
        tokens = list(instrument_map.keys())
        print(
            f"✅ Shard {shard_index + 1}/{shard_count}: Connected — Subscribing to {len(tokens)} tokens"
        )
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_close(ws, code_, reason):
        print(f"❌ Shard {shard_index + 1}: Disconnected {code_} - {reason}")

    def on_error(ws, code_, reason):
        print(f"⚠️ Shard {shard_index + 1}: Error {code_} - {reason}")

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error

    kws.connect(threaded=False)


if __name__ == "__main__":
    stocks = load_config()
    if not stocks:
        print("❌ No stocks loaded. Exiting.")
        raise SystemExit(1)

    print(
        f"🚀 Launching {SHARDS} shard(s) for {len(stocks)} stock(s) — mode={SHARD_MODE}"
    )
    procs = []
    for idx in range(SHARDS):
        p = Process(
            target=start_collector, args=(stocks, idx, SHARDS, SHARD_MODE), daemon=False
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
