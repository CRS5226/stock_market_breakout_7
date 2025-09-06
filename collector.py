# collector.py (tick collector + 1min candle builder with per-minute volume)
import os
import json
import math
import logging
from multiprocessing import Process
from kiteconnect import KiteTicker
from dotenv import load_dotenv
from redis_utils import (
    get_redis,
    floor_minute,
    set_current_candle,
    finalize_and_roll_new_candle,
)

load_dotenv()

API_KEY = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

CONFIG_FILE = "config30a.json"  # 🔑 All stocks here
r = get_redis()

# ==== CONFIG: set only this ====
SHARDS = 2  # 🔧 Number of shards to run in parallel (0..SHARDS-1 will be started automatically)
SHARD_MODE = "roundrobin"  # ("contiguous" or "roundrobin")


def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return data.get("stocks", [])
    except Exception as e:
        print(f"[Config Error] Failed to load {CONFIG_FILE}: {e}")
        return []


def split_contiguous_equally(stocks, shard_index, shard_count):
    """
    Split into nearly equal contiguous chunks.
    Example: n=30, k=2 -> [0..14], [15..29]
             n=5,  k=2 -> [0..2],  [3..4]
    """
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
    # default: contiguous equal split
    return split_contiguous_equally(stocks, shard_index, shard_count)


def start_collector(
    stock_configs, shard_index=0, shard_count=1, shard_mode="contiguous"
):
    logging.basicConfig(level=logging.INFO, force=True)

    # split work by shard
    assigned = assign_stocks(stock_configs, shard_index, shard_count, shard_mode)
    if not assigned:
        print(
            f"⚠️ Shard {shard_index + 1}/{shard_count} has no assigned stocks. Exiting."
        )
        return

    instrument_map = {s["instrument_token"]: s["stock_code"].upper() for s in assigned}

    # print(
    #     f"🧩 Shard {shard_index + 1}/{shard_count} mode={shard_mode} assigned {len(assigned)} stocks:"
    # )
    for s in assigned:
        pass
        # print(f"   • {s['stock_code']} ({s['instrument_token']})")

    # in-memory candle state per stock
    candles_state = {
        code: {"prev_minute": None, "current": None, "last_tick_volume": None}
        for code in instrument_map.values()
    }

    kws = KiteTicker(API_KEY, ACCESS_TOKEN)

    def on_ticks(ws, ticks):
        for tick in ticks:
            token = tick.get("instrument_token")
            code = instrument_map.get(token)
            if not code:
                continue

            ts = tick.get("exchange_timestamp")
            if not ts:
                continue

            price = float(tick.get("last_price"))
            cum_vol = int(tick.get("volume_traded"))  # cumulative daily volume
            minute_key = floor_minute(ts)

            state = candles_state[code]

            # delta volume (per-tick)
            if state["last_tick_volume"] is None:
                delta_vol = 0
            else:
                delta_vol = max(0, cum_vol - state["last_tick_volume"])
            state["last_tick_volume"] = cum_vol

            # new minute -> finalize previous candle, start new
            if state["prev_minute"] is None or minute_key != state["prev_minute"]:
                if state["current"]:
                    finalize_and_roll_new_candle(
                        r, code, state["current"], max_candles=100
                    )

                state["current"] = {
                    "minute": str(minute_key),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": delta_vol,
                }
                state["prev_minute"] = minute_key
            else:
                # update current candle
                cur = state["current"]
                cur["high"] = max(cur["high"], price)
                cur["low"] = min(cur["low"], price)
                cur["close"] = price
                cur["volume"] += delta_vol

            # save snapshot in Redis
            set_current_candle(
                r,
                code,
                minute_key,
                state["current"]["open"],
                state["current"]["high"],
                state["current"]["low"],
                state["current"]["close"],
                state["current"]["volume"],
            )

            print(f"[{code}] Tick saved")

    def on_connect(ws, response):
        tokens = list(instrument_map.keys())
        print(
            f"✅ Shard {shard_index + 1}/{shard_count}: Connected — Subscribing to {len(tokens)} tokens"
        )
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)

    def on_close(ws, code, reason):
        print(f"❌ Shard {shard_index + 1}: Disconnected {code} - {reason}")

    def on_error(ws, code, reason):
        print(f"⚠️ Shard {shard_index + 1}: Error {code} - {reason}")

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close
    kws.on_error = on_error

    # each shard runs its own websocket connection
    kws.connect(threaded=False)


if __name__ == "__main__":
    stocks = load_config()
    if not stocks:
        print("❌ No stocks loaded. Exiting.")
        raise SystemExit(1)

    # Spawn one process per shard automatically (indices 0..SHARDS-1)
    print(
        f"🚀 Launching {SHARDS} shard(s) for {len(stocks)} stock(s) — mode={SHARD_MODE}"
    )
    procs = []
    for idx in range(SHARDS):
        p = Process(
            target=start_collector,
            args=(stocks, idx, SHARDS, SHARD_MODE),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Wait for all shards (each runs its own KiteTicker loop)
    for p in procs:
        p.join()
