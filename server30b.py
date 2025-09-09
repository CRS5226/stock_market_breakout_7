# server30b.py

import os
import time
import json
import pandas as pd
from multiprocessing import Process, Manager
from indicator import add_indicators
from telegram_alert30b import (
    send_trade_alert,
    send_pipeline_status,
    send_error_alert,
    send_config_update,
    send_server_feedback,
)
from llm_forecast import forecast_config_update, route_model

# from basic_algo_forecaster import basic_forecast_update
from basic_algo3 import basic_forecast_update
from throughput_monitor import ThroughputMonitor

# from redis_utils import get_redis, get_recent_candles, get_recent_indicators
from redis_utils import (
    get_redis,
    get_recent_candles_tf,
    get_recent_indicators_tf,
    push_indicator_row_tf,
    get_last_indicator_bucket_tf,
    set_last_indicator_bucket_tf,
)

# from redis_utils import (
#         get_redis,
#         get_recent_candles_tf,
#         push_indicator_row_tf,
#         get_last_indicator_bucket_tf,
#         set_last_indicator_bucket_tf,
#     )

from gsheet_logger import log_config_update
from gforecast_logger import write_pretty_to_sheet_from_sheets
from notifier import notify_sr_levels, is_breakout


r = get_redis()

CONFIG_PATH = "config30b.json"
DATA_FOLDER = "data"
STATS_FILE = "monitor_stats.json"
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")  # always TradingStocksLogs
TAB_NAME = os.getenv("TAB_NAME_30B")  # unique per server

# TAB_NAME = os.getenv("TAB_NAME_TEST")  # unique per server


USE_LLM = False  # 🔀 Toggle here: True = LLM forecast, False = custom algo

os.makedirs(DATA_FOLDER, exist_ok=True)

BREAKOUT_STATE = {}
LAST_CONFIG = {}
last_forecast_time = {}


# Example TF set (match what collector produces)
MONITOR_TFS = ["1min", "5min", "15min", "30min", "45min", "1hour", "4hour"]

# Keep separate breakout state per code & timeframe
BREAKOUT_STATE_TF = (
    {}
)  # shape: { code: { tf: {"above_resistance": bool, "below_support": bool} } }

TS_CANDIDATES = ("minute", "bucket", "timestamp", "datetime", "time")


def _find_ts_field(row: dict) -> str | None:
    for k in TS_CANDIDATES:
        if k in row and row[k] is not None:
            return k
    return None


def _to_iso_bucket(val) -> str:
    """
    Normalize any time-like value to ISO string so you can safely compare & store in Redis.
    Handles pandas.Timestamp, datetime, epoch-like str/int, and plain ISO strings.
    """
    try:
        ts = pd.to_datetime(val, utc=True, errors="coerce")
        if pd.isna(ts):
            # try to rescue strings like "Timestamp('2025-09-08 14:22:00+0000', tz='UTC')"
            if isinstance(val, str) and val.startswith("Timestamp("):
                # extract the inside
                import re

                m = re.search(r"Timestamp\('([^']+)'", val)
                if m:
                    ts = pd.to_datetime(m.group(1), utc=True, errors="coerce")
        if pd.isna(ts):
            return str(val)  # last resort: stable string compare
        return ts.isoformat()
    except Exception:
        return str(val)


def _ensure_state(code, tf):
    BREAKOUT_STATE_TF.setdefault(code, {})
    BREAKOUT_STATE_TF[code].setdefault(
        tf, {"above_resistance": False, "below_support": False}
    )
    return BREAKOUT_STATE_TF[code][tf]


def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Config Error] Failed to read config: {e}")
        return {"stocks": []}


def fetch_latest_config_for_stock(stock_code):
    config = load_config()
    for stock in config.get("stocks", []):
        if stock["stock_code"] == stock_code:
            return stock
    return None


def print_config_changes(stock_code, new_config):
    last_config = LAST_CONFIG.get(stock_code, {})
    messages = []
    support_changed = False
    resistance_changed = False

    for key in new_config:
        new_val = new_config.get(key)
        old_val = last_config.get(key)

        if isinstance(new_val, dict):
            for sub_key in new_val:
                old_sub = old_val.get(sub_key) if old_val else None
                new_sub = new_val[sub_key]
                # 🚫 Skip "None → X" changes
                if old_sub is None and new_sub is not None:
                    continue
                if old_sub != new_sub:
                    messages.append(f"{key}.{sub_key}: {old_sub} → {new_sub}")
        else:
            # 🚫 Skip "None → X" changes
            if old_val is None and new_val is not None:
                continue
            if old_val != new_val:
                messages.append(f"{key}: {old_val} → {new_val}")
                if key == "support":
                    support_changed = True
                if key == "resistance":
                    resistance_changed = True

    if messages:
        print(f"[🔁 CONFIG CHANGE] {stock_code} → " + ", ".join(messages))
        send_config_update(
            f"⚙️ Config Updated: {stock_code}\n" + "\n".join(messages), stock_code
        )

    # ✅ Update LAST_CONFIG immediately so future diffs are accurate
    LAST_CONFIG[stock_code] = new_config.copy()

    return support_changed or resistance_changed


def init_last_config():
    config = load_config()
    for stock in config.get("stocks", []):
        stock_code = stock["stock_code"]
        LAST_CONFIG[stock_code] = stock.copy()


def detect_removed_stocks(existing_codes):
    removed = []
    for stock_code in list(LAST_CONFIG.keys()):
        if stock_code not in existing_codes:
            removed.append(stock_code)
            send_pipeline_status(f"❌ Stock Removed: {stock_code}", stock_code)
            del LAST_CONFIG[stock_code]
            del BREAKOUT_STATE[stock_code]
    return removed


def forecast_manager():
    print("🧠 Forecast Manager started.")

    monitor = ThroughputMonitor(window_sec=60, csv_path="forecast/throughput.csv")

    while True:
        try:
            config_data = load_config()
            stocks = config_data.get("stocks", [])

            for i, stock_cfg in enumerate(stocks):
                stock_code = stock_cfg["stock_code"]

                updated_cfg = None
                reasons = None
                err = None

                try:
                    if USE_LLM:
                        try:
                            # 🔮 LLM Forecast
                            model_choice = route_model(stock_code, i)
                            print(
                                f"[🔮 Forecast] {stock_code} via {model_choice} (LLM)"
                            )

                            t0 = time.time()
                            updated_cfg, reasons, err = forecast_config_update(
                                stock_cfg,
                                historical_folder="historical_data",
                                model=model_choice,
                                temperature=0.2,
                                escalate_on_signal=True,
                            )
                            latency_ms = (time.time() - t0) * 1000.0
                            monitor.record(stock_code, model_choice, latency_ms)

                        except Exception as llm_e:
                            # 🚨 LLM failed → fallback to basic algo
                            print(f"[⚠️ LLM Error for {stock_code}] {llm_e}")
                            print(f"[📊 Fallback] Using basic algo for {stock_code}...")

                            rows = get_recent_indicators_tf(
                                r, stock_code, "1min", n=200
                            )
                            if rows:
                                df = pd.DataFrame(reversed(rows))
                            else:
                                print(
                                    f"[⚠️] No recent indicators found for {stock_code}"
                                )
                                df = pd.DataFrame()

                            updated_cfg = basic_forecast_update(
                                stock_cfg,
                                recent_df=df,
                                historical_folder="historical_data",
                            )
                            reasons = "Fallback to basic algo due to LLM error"

                    else:
                        # 📊 Custom Algorithm Forecast only
                        print(f"[📊 Forecast] {stock_code} via basic algo...")

                        rows = get_recent_indicators_tf(r, stock_code, "1min", n=200)
                        if rows:
                            df = pd.DataFrame(reversed(rows))  # chronological
                            # normalize column names that basic_forecast_update expects
                            cols = {c.lower(): c for c in df.columns}
                            if "timestamp" not in cols:
                                # rows may have 'minute' or lowercase 'timestamp'
                                if "minute" in cols:
                                    df["Timestamp"] = pd.to_datetime(
                                        df[cols["minute"]], errors="coerce"
                                    )
                                elif "timestamp" in cols:
                                    df["Timestamp"] = pd.to_datetime(
                                        df[cols["timestamp"]], errors="coerce"
                                    )
                                else:
                                    # no time column present; allow function to proceed with empty df
                                    df["Timestamp"] = pd.NaT
                            else:
                                df["Timestamp"] = pd.to_datetime(
                                    df[cols["timestamp"]], errors="coerce"
                                )

                            # standardize OHLCV names if only lowercase exist
                            for a, b in [
                                ("open", "Open"),
                                ("high", "High"),
                                ("low", "Low"),
                                ("close", "Close"),
                                ("volume", "Volume"),
                            ]:
                                if a in cols and b not in df.columns:
                                    df[b] = df[cols[a]]

                            # drop rows without usable time (keeps prompt lean)
                            df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)
                        else:
                            print(
                                f"[⚠️] No recent indicators found for {stock_code} (1min)"
                            )
                            df = pd.DataFrame()

                        updated_cfg = basic_forecast_update(
                            stock_cfg,
                            recent_df=df,
                            historical_folder="historical_data",
                        )

                except Exception as inner_e:
                    err = str(inner_e)

                # --- Apply updates ---
                if err:
                    print(f"[❌ Forecast Error - {stock_code}] {err}")
                else:
                    if updated_cfg != stock_cfg:
                        stocks[i] = updated_cfg
                        with open(CONFIG_PATH, "w") as f:
                            json.dump(config_data, f, indent=2)
                        print(f"[✅ Forecast Updated - {stock_code}]")
                        if reasons:
                            print(f"[ℹ️ Reasons] {reasons}")

                        # 🚀 Log to Google Sheets
                        try:
                            log_config_update(
                                updated_cfg, GOOGLE_SHEET_NAME, tab_name=TAB_NAME
                            )

                            write_pretty_to_sheet_from_sheets(
                                spreadsheet_name=GOOGLE_SHEET_NAME,  # "TradingStocksLogs"
                                gpt_tab=os.getenv(
                                    "TAB_NAME_30A", "stocks_30_gpt"
                                ),  # GPT tab
                                algo_tab=os.getenv(
                                    "TAB_NAME_30B", "stocks_30_algo"
                                ),  # ALGO tab
                                pretty_tab=os.getenv(
                                    "TAB_NAME_30_ALL", "stocks_30_all"
                                ),
                                spacer_rows=3,
                                service_account_json="cred.json",
                                # service_account_json="/path/to/service_account.json"  # optional if env var is set
                            )

                        except Exception as gsheet_e:
                            print(
                                f"[⚠️ Google Sheets Logging Failed - {stock_code}] {gsheet_e}"
                            )
                    else:
                        print(f"[🧠 Forecast - {stock_code}] No changes detected")

                time.sleep(10)  # pacing delay

        except Exception as e:
            print(f"[Forecast Manager Error] {type(e).__name__}: {e}")
            time.sleep(30)


def monitor_shard(stock_list, stats, shard_id, total_shards, lookback_candles=200):
    # single redis handle for the loop

    r = get_redis()

    # shard routing (unchanged)
    stocks_for_this_shard = [
        s for i, s in enumerate(stock_list) if i % total_shards == shard_id
    ]
    print(f"🟢 Monitoring Shard {shard_id} with {len(stocks_for_this_shard)} stocks")

    while True:
        start_time = time.time()

        for stock in stocks_for_this_shard:
            code = stock["stock_code"]

            try:
                # --- 1) Pull latest cfg and update state (your existing helpers) ---
                updated_config = fetch_latest_config_for_stock(code)
                if updated_config:
                    stock = updated_config
                    config_changed = print_config_changes(code, updated_config)
                    if config_changed:
                        BREAKOUT_STATE_TF[code] = {
                            tf: {"above_resistance": False, "below_support": False}
                            for tf in MONITOR_TFS
                        }

                # --- 2) Loop over timeframes ---
                for tf in MONITOR_TFS:
                    rows = get_recent_candles_tf(r, code, tf, n=lookback_candles)

                    if not rows:
                        # nothing at all for this TF
                        print(f"[ℹ️] {code} {tf}: no candles in Redis.")
                        continue
                    # if len(rows) < 20:
                    #     # warn if warm-up may be insufficient for BB/RSI/ADX
                    #     print(
                    #         f"[⏳] {code} {tf}: only {len(rows)} candles; indicators may be partial."
                    #     )

                    # detect timestamp field from the newest row
                    ts_field = _find_ts_field(rows[0])
                    if not ts_field:
                        print(
                            f"[⚠️] {code} {tf}: no timestamp field among {list(rows[0].keys())}"
                        )
                        continue

                    # dedup guard on last *closed* candle (newest)
                    last_bucket_raw = rows[0].get(ts_field)
                    last_bucket_iso = _to_iso_bucket(last_bucket_raw)

                    prev_done = get_last_indicator_bucket_tf(r, code, tf)
                    if isinstance(prev_done, bytes):
                        prev_done = prev_done.decode("utf-8", errors="ignore")
                    if prev_done == last_bucket_iso:
                        # already processed this candle
                        continue

                    # chronological order for indicator calc
                    df = pd.DataFrame(reversed(rows)).copy()
                    # normalize Timestamp
                    df["Timestamp"] = pd.to_datetime(
                        df[ts_field], errors="coerce", utc=True
                    )

                    # standardize OHLCV
                    df.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        },
                        inplace=True,
                    )

                    # sanity check after normalization
                    if df["Timestamp"].isna().all():
                        print(
                            f"[⚠️] {code} {tf}: all Timestamps NA after parsing {ts_field}."
                        )
                        continue

                    # --- 3) Compute indicators (your function) ---
                    df = add_indicators(df, stock)

                    # --- 4) Write latest row to timeframe indicators stream ---
                    latest = df.iloc[-1].to_dict()
                    # store timestamp as ISO string for consistency
                    latest["Timestamp"] = pd.to_datetime(
                        latest["Timestamp"], utc=True
                    ).isoformat()
                    latest["timeframe"] = tf
                    push_indicator_row_tf(r, code, tf, latest, maxlen=200)
                    stats["redis_writes"] = stats.get("redis_writes", 0) + 1

                    # mark computed bucket (store ISO)
                    set_last_indicator_bucket_tf(r, code, tf, last_bucket_iso)

                    # --- 5) Breakout / breakdown per timeframe (unchanged logic) ---
                    signal, price, levels, reason = is_breakout(
                        df,
                        stock.get("resistance", 0),
                        stock.get("support", 0),
                        stock,
                    )

                    tf_state = _ensure_state(code, tf)
                    bucket_dt = df["Timestamp"].iloc[-1]

                    if signal == "breakout" and not tf_state["above_resistance"]:
                        send_trade_alert(
                            code,
                            f"[{tf}] 📈 Breakout Above {stock.get('resistance', 0)}\n🧠 {reason}",
                            price,
                            bucket_dt,
                        )
                        tf_state["above_resistance"], tf_state["below_support"] = (
                            True,
                            False,
                        )

                    elif signal == "breakdown" and not tf_state["below_support"]:
                        send_trade_alert(
                            code,
                            f"[{tf}] 📉 Breakdown Below {stock.get('support', 0)}\n🧠 {reason}",
                            price,
                            bucket_dt,
                        )
                        tf_state["below_support"], tf_state["above_resistance"] = (
                            True,
                            False,
                        )

                    # Optional: tag SR notifier with TF
                    notify_sr_levels(
                        code=code,
                        df=df,
                        stock_cfg=stock,
                        redis_client=r,
                        source_tag=(
                            f"{stock.get('forecast')}_{tf}".upper()
                            if stock.get("forecast")
                            else tf
                        ),
                    )

            except Exception as e:
                print(
                    f"[❌ Monitor Error] {code} {tf if 'tf' in locals() else ''}: {type(e).__name__}: {e}"
                )

        stats["monitor_cycles"] = stats.get("monitor_cycles", 0) + 1
        stats["monitor_time"] = stats.get("monitor_time", 0) + (
            time.time() - start_time
        )
        time.sleep(2)


def stats_writer(stats):
    start_time = time.time()
    while True:
        try:
            elapsed = time.time() - start_time
            per_sec = {
                "csv_writes_per_sec": stats["csv_writes"] / elapsed if elapsed else 0,
                "monitor_cycles_per_sec": (
                    stats["monitor_cycles"] / elapsed if elapsed else 0
                ),
                "ticks_per_sec": stats["tick_count"] / elapsed if elapsed else 0,
            }
            full_stats = dict(stats)
            full_stats.update(per_sec)
            with open(STATS_FILE, "w") as f:
                json.dump(full_stats, f, indent=2)
        except Exception as e:
            print(f"[Stats Error] {e}")
        time.sleep(15)  # 15 seconds interval


def run():
    processes = {}
    manager = Manager()
    stats = manager.dict(
        {"csv_writes": 0, "monitor_cycles": 0, "monitor_time": 0.0, "tick_count": 0}
    )

    stat_proc = Process(target=stats_writer, args=(stats,))
    stat_proc.start()

    print("🚀 Real-Time Stock Monitor started. Watching for changes...")
    send_server_feedback()

    MONITOR_SHARDS = 2  # monitors

    while True:
        try:
            config = load_config()
            stock_list = config.get("stocks", [])

            # Forecaster
            if "forecaster" not in processes:
                forecast_proc = Process(target=forecast_manager)
                forecast_proc.start()
                processes["forecaster"] = forecast_proc

            # Monitors (sharded)
            for i in range(MONITOR_SHARDS):
                proc_name = f"monitor_{i}"
                if proc_name not in processes:
                    monitor_proc = Process(
                        target=monitor_shard,
                        args=(stock_list, stats, i, MONITOR_SHARDS),
                    )
                    monitor_proc.start()
                    processes[proc_name] = monitor_proc
                    print(f"✅ Started Monitor Shard {i}")

            time.sleep(5)

        except KeyboardInterrupt:
            print("⛔️ Stopped by user.")
            break
        except Exception as e:
            send_error_alert(f"[Server Error] {type(e).__name__}: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run()
