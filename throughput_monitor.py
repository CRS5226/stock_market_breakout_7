# throughput_monitor.py
import os
import csv
import time
from collections import deque, defaultdict


class ThroughputMonitor:
    """
    Tracks how many unique stocks were processed in the last N seconds,
    with per-model breakdown and simple latency stats.
    Logs periodic snapshots to CSV.
    """

    def __init__(self, window_sec=60, csv_path="forecast/throughput.csv"):
        self.window_sec = window_sec
        self.events = deque()  # (ts, symbol, model, latency_ms)
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "timestamp",
                        "window_sec",
                        "unique_symbols_last_window",
                        "events_last_window",
                        "pct_of_400",
                        "avg_latency_ms",
                        "p50_latency_ms",
                        "p90_latency_ms",
                        "p99_latency_ms",
                        "per_model_counts_json",
                    ]
                )

    def _prune(self, now=None):
        if now is None:
            now = time.time()
        cutoff = now - self.window_sec
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def record(self, symbol: str, model: str, latency_ms: float):
        """
        Call this once per completed GPT call.
        """
        ts = time.time()
        self.events.append((ts, symbol, model, latency_ms))
        self._prune(ts)

    def _percentile(self, values, p):
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values) - 1) * p
        f = int(k)
        c = min(f + 1, len(values) - 1)
        if f == c:
            return float(values[int(k)])
        d0 = values[f] * (c - k)
        d1 = values[c] * (k - f)
        return float(d0 + d1)

    def snapshot(self):
        """
        Returns a dict of current stats for the last window.
        """
        self._prune()
        now_iso = time.strftime("%Y-%m-%d %H:%M:%S")
        # Unique symbols in window (how many distinct stocks processed in last minute)
        symbols = {sym for (_, sym, _, _) in self.events}
        unique_symbols = len(symbols)
        # Events (total completions) in window
        events = len(self.events)
        # Per-model counts
        per_model = defaultdict(int)
        latencies = []
        for _, _, model, lat_ms in self.events:
            per_model[model] += 1
            latencies.append(lat_ms)

        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        p50 = self._percentile(latencies, 0.50)
        p90 = self._percentile(latencies, 0.90)
        p99 = self._percentile(latencies, 0.99)

        pct_of_400 = round(100.0 * unique_symbols / 400.0, 2)

        return {
            "timestamp": now_iso,
            "window_sec": self.window_sec,
            "unique_symbols_last_window": unique_symbols,
            "events_last_window": events,
            "pct_of_400": pct_of_400,
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p90_latency_ms": round(p90, 2),
            "p99_latency_ms": round(p99, 2),
            "per_model_counts": dict(per_model),
        }

    def log_snapshot_csv(self):
        snap = self.snapshot()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    snap["timestamp"],
                    snap["window_sec"],
                    snap["unique_symbols_last_window"],
                    snap["events_last_window"],
                    snap["pct_of_400"],
                    snap["avg_latency_ms"],
                    snap["p50_latency_ms"],
                    snap["p90_latency_ms"],
                    snap["p99_latency_ms"],
                    json_dumps_safe(snap["per_model_counts"]),
                ]
            )


def json_dumps_safe(d):
    try:
        import json

        return json.dumps(d, separators=(",", ":"))
    except Exception:
        # Fallback to simple str if json fails
        return str(d)
