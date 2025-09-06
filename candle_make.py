# candle_make.py
import os
import pandas as pd
from datetime import datetime, timezone
from redis_utils import (
    get_redis,
    floor_minute,
    set_current_candle,
    finalize_and_roll_new_candle,
)


def start_candle_builder(
    stock_code: str, timeframe: str = "1min", output_folder: str = "candles"
):
    """
    Build candles from Redis ticks and store in Redis + CSV.
    """
    os.makedirs(output_folder, exist_ok=True)
    r = get_redis()

    stream = f"ticks:{stock_code}"
    last_id = "0-0"
    candle_df = pd.DataFrame()

    prev_minute = None
    current_candle = None

    print(f"🟣 Candle builder started for {stock_code}")

    while True:
        resp = r.xread({stream: last_id}, block=1000, count=500)
        if not resp:
            continue

        _, messages = resp[0]
        for msg_id, fields in messages:
            last_id = msg_id
            ts = pd.to_datetime(fields[b"ts"])
            price = float(fields[b"price"])
            vol = int(fields[b"vol"])

            minute_key = floor_minute(ts)

            # New candle
            if prev_minute is None or minute_key != prev_minute:
                if current_candle:
                    # finalize last candle into Redis + CSV
                    finalize_and_roll_new_candle(
                        r, stock_code, current_candle, max_candles=300
                    )

                    # also update CSV
                    candle_df.loc[prev_minute] = [
                        current_candle["open"],
                        current_candle["high"],
                        current_candle["low"],
                        current_candle["close"],
                        current_candle["volume"],
                    ]
                    filepath = os.path.join(output_folder, f"{stock_code}_candle.csv")
                    candle_df.sort_index(inplace=True)
                    candle_df.to_csv(filepath)

                # start new candle
                current_candle = {
                    "minute": minute_key,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": vol,
                }
                prev_minute = minute_key
            else:
                # update existing candle
                current_candle["high"] = max(current_candle["high"], price)
                current_candle["low"] = min(current_candle["low"], price)
                current_candle["close"] = price
                current_candle["volume"] = vol

            # update live candle in Redis
            set_current_candle(
                r,
                stock_code,
                minute_key,
                current_candle["open"],
                current_candle["high"],
                current_candle["low"],
                current_candle["close"],
                current_candle["volume"],
            )
