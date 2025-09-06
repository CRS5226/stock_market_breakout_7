# llm_forecast.py

import os
import json
import csv

# import pandas as pd
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from redis_utils import get_redis, get_recent_indicators

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# === Model routing config ===
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4.1-mini")  # stronger / pricier
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "gpt-4.1-nano")  # cheaper scout


def route_model(stock_code: str, index: int) -> str:
    """
    Simple round-robin: even-index stocks -> PRIMARY_MODEL, odd-index -> SECONDARY_MODEL.
    You can swap this to any rule (by symbol, volatility, volume, etc.).
    """
    return PRIMARY_MODEL if (index % 2 == 0) else SECONDARY_MODEL


EXAMPLE_CONFIG = {
    "stock_code": "RELIANCE",
    "instrument_token": 128083204,
    "support": 1366.5,
    "resistance": 1373.05,
    "volume_threshold": 178607,
    "adx": {"period": 14, "threshold": 20},
    "reason": [],  # LLM will fill with reasons for update
    "last_updated": "2025-08-12 00:00:00",  # placeholder, replaced dynamically
    "signal": "No Action",  # LLM will fill with signal
}


def save_forecast_csv(stock_code, raw):
    """Append raw LLM responses to a CSV log."""
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
    escalate_on_signal: bool = True,  # <— new
):
    stock_code = stock_block.get("stock_code")

    # --- Step 1: Historical ---
    hist_csv = None
    if os.path.isdir(historical_folder):
        for file in os.listdir(historical_folder):
            if file.startswith(f"{stock_code}_historical_") and file.endswith(".csv"):
                hist_csv = os.path.join(historical_folder, file)
                break

    if hist_csv and os.path.exists(hist_csv):
        try:
            df_hist = pd.read_csv(hist_csv)
            historical_excerpt = df_hist.tail(5).to_csv(index=False)
        except Exception as e:
            return None, None, f"Failed to read historical CSV {hist_csv}: {e}"
    else:
        historical_excerpt = "(No historical file found)"

    # --- Step 2: Recent candles+indicators from Redis ---
    try:
        r = get_redis()
        recent_data = get_recent_indicators(r, stock_code, n=5)
        if not recent_data:
            return (
                None,
                None,
                f"No recent indicators for {stock_code}",
            )

        df_recent = pd.DataFrame(recent_data)
        recent_excerpt = df_recent.to_csv(index=False)
    except Exception as e:
        return None, None, f"Failed to fetch recent indicators from Redis: {e}"

    # --- Step 3: Prompt ---
    schema_str = json.dumps(EXAMPLE_CONFIG, indent=2)
    provided_cfg_str = json.dumps(stock_block, indent=2)
    prompt = f"""
You are a strict JSON generator for stock configs.

TASK:
- Read CONFIG, HISTORICAL DATA (~1 year daily candles of size 1 hour), and RECENT DATA (last few candles of 1 minuite).
- Update support/resistance/volume_threshold/indicator params ONLY if justified by trends, breakouts, or major changes in recent days.
- Use historical for context but prioritize recent for short-term.
- Preserve 'stock_code' and 'instrument_token'.
- Always include:
  - "reason": list of changes
  - "last_updated": YYYY-MM-DD HH:MM:SS
  - "signal": "Buy" | "Sell" | "Hold" | "No Action"
Return ONE JSON object only. No markdown. No prose.

SCHEMA (example):
{schema_str}

CONFIG (current):
{provided_cfg_str}

HISTORICAL DATA (recent daily candles):
{historical_excerpt}

RECENT DATA (last candles):
{recent_excerpt}
"""

    def _call_llm(model_name: str):
        resp = client.responses.create(
            model=model_name,
            temperature=temperature,
            max_output_tokens=2048,
            input=[
                {
                    "role": "system",
                    "content": "Output: strict JSON config only, matching schema exactly. Do NOT include triple backticks or Markdown.",
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

        # tokens
        in_toks = getattr(getattr(resp, "usage", None), "input_tokens", 0)
        out_toks = getattr(getattr(resp, "usage", None), "output_tokens", 0)
        update_token_monitor(stock_code, in_toks, out_toks, model_used=model_name)

        # parse JSON
        try:
            cfg = json.loads(cleaned)
        except Exception as e:
            safe_excerpt = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
            return None, None, f"JSON parse error: {e}. Output excerpt: {safe_excerpt}"
        cfg.setdefault("reason", [])
        cfg.setdefault("signal", "No Action")
        cfg.setdefault("last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cfg["forecast"] = "AI"
        return cfg, cfg.get("reason", []), None

    # First pass with provided model
    updated_cfg, reasons, err = _call_llm(model)

    if err:
        return None, None, err

    # Optional escalation: if cheaper model triggers a trade, confirm with PRIMARY_MODEL
    # if escalate_on_signal and (model == SECONDARY_MODEL):
    #     sig = (updated_cfg or {}).get("signal", "No Action")
    #     if sig in ("Buy", "Sell"):
    #         updated_cfg2, reasons2, err2 = _call_llm(PRIMARY_MODEL)
    #         if err2 is None:
    #             # Keep PRIMARY decision if it matches or is stricter
    #             updated_cfg = updated_cfg2
    #             reasons = reasons2

    return updated_cfg, reasons, None
