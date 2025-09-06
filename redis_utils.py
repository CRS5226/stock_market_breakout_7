# redis_utils.py
import os, json
import redis
from datetime import datetime, timezone


def get_redis():
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.Redis.from_url(url, decode_responses=True)


def floor_minute(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.astimezone(timezone.utc)
    return ts.replace(second=0, microsecond=0).isoformat(timespec="minutes")


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


def get_recent_candles(r: redis.Redis, code: str, n: int = 5):
    """Return last N candles as list of dicts (newest first)."""
    items = r.lrange(f"candles:{code}", 0, n - 1)
    return [json.loads(x) for x in items]


def get_recent_indicators(r, code: str, n: int = 5):
    """Return last N candles+indicators as list of dicts (newest first)."""
    items = r.lrange(f"indicators:{code}", 0, n - 1)
    return [json.loads(x) for x in items]
