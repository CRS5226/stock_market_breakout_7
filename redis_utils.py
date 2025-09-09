# redis_utils.py
import os, json
import redis
from datetime import datetime, timezone, timedelta


def get_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def floor_minute(ts: datetime) -> str:
    """(Backward-compat) Floor to 1-minute UTC, ISO string (timespec=minutes)."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    return ts.replace(second=0, microsecond=0).isoformat(timespec="minutes")


def floor_period(ts: datetime, minutes: int) -> str:
    """
    Floor timestamp to N-minute bucket (UTC) and return ISO string
    marking the bucket *start* (timespec=minutes).
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    # Compute minutes from day start and floor
    day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    minutes_since_day_start = int((ts - day_start).total_seconds() // 60)
    floored = (minutes_since_day_start // minutes) * minutes
    bucket_start = day_start + timedelta(minutes=floored)
    return bucket_start.isoformat(timespec="minutes")


def lpush_cap(r: redis.Redis, key: str, item: dict, maxlen: int):
    p = r.pipeline()
    p.lpush(key, json.dumps(item, default=str))
    p.ltrim(key, 0, maxlen - 1)
    p.execute()


def hgetall_json(r: redis.Redis, key: str):
    d = r.hgetall(key)
    return {
        k: (float(v) if isinstance(v, str) and v.replace(".", "", 1).isdigit() else v)
        for k, v in d.items()
    }


# ---------- Single-TF (backward compat: minute) ----------
def set_current_candle(r: redis.Redis, code: str, minute_key: str, o, h, l, c, v):
    key = f"candle:cur:{code}"
    r.hset(
        key,
        mapping={
            "minute": minute_key,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        },
    )


def finalize_and_roll_new_candle(
    r: redis.Redis, code: str, prev_candle: dict, max_candles: int = 300
):
    if not prev_candle:
        return
    prev_candle["stock"] = code
    lpush_cap(r, f"candles:{code}", prev_candle, max_candles)


# ---------- Multi-timeframe helpers ----------
def set_current_candle_tf(
    r: redis.Redis, code: str, tf: str, bucket_key: str, o, h, l, c, v
):
    key = f"candle:cur:{code}:{tf}"
    r.hset(
        key,
        mapping={
            "tf": tf,
            "bucket": bucket_key,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        },
    )


def finalize_and_roll_new_candle_tf(
    r: redis.Redis, code: str, tf: str, prev_candle: dict, max_candles: int = 300
):
    if not prev_candle:
        return
    prev_candle["stock"] = code
    prev_candle["tf"] = tf
    lpush_cap(r, f"candles:{code}:{tf}", prev_candle, max_candles)


def get_recent_candles(r: redis.Redis, code: str, n: int = 5):
    """Backward-compat: last N 1-minute candles (newest first)."""
    items = r.lrange(f"candles:{code}", 0, n - 1)
    return [json.loads(x) for x in items]


def get_recent_candles_tf(r: redis.Redis, code: str, tf: str, n: int = 5):
    """Last N candles for a timeframe (newest first)."""
    items = r.lrange(f"candles:{code}:{tf}", 0, n - 1)
    return [json.loads(x) for x in items]


def get_recent_indicators(r, code: str, n: int = 5):
    items = r.lrange(f"indicators:{code}", 0, n - 1)
    return [json.loads(x) for x in items]


# --- add to redis_utils.py ---


def push_indicator_row_tf(
    r: redis.Redis, code: str, tf: str, row: dict, maxlen: int = 200
):
    """Append newest indicator row for timeframe to a capped list."""
    key = f"indicators:{code}:{tf}"
    lpush_cap(r, key, row, maxlen)


def get_recent_indicators_tf(r: redis.Redis, code: str, tf: str, n: int = 5):
    """Read most recent indicator rows for timeframe (newest first)."""
    items = r.lrange(f"indicators:{code}:{tf}", 0, n - 1)
    return [json.loads(x) for x in items]


def get_last_indicator_bucket_tf(r: redis.Redis, code: str, tf: str) -> str | None:
    """Return last bucket for which indicators were written (simple de-dup guard)."""
    return r.get(f"indicators:last_bucket:{code}:{tf}")


def set_last_indicator_bucket_tf(r: redis.Redis, code: str, tf: str, bucket: str):
    r.set(f"indicators:last_bucket:{code}:{tf}", bucket)
