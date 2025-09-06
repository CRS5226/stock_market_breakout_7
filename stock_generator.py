import pandas as pd
import json
from datetime import datetime

# ✅ Parameters
N_STOCKS = 400  # Change this to any number you want
INSTRUMENT_CSV = "filtered_data.csv"
OUTPUT_JSON = "config30.json"
OUTPUT_CSV = "stocks_reference.csv"

# ✅ Load top NSE stock list
with open("sir_stocks.txt", "r") as f:
    top_stocks = [line.strip() for line in f.readlines()][:N_STOCKS]

# ✅ Load instrument CSV
df = pd.read_csv(INSTRUMENT_CSV)

# ✅ Filter to NSE equity symbols
df_eq = df[(df["exchange"] == "NSE") & (df["instrument_type"] == "EQ")]

# ✅ Prepare outputs
config_stocks = []
csv_rows = []
missing = []

# ---------- Default values (BACKWARD-COMPATIBLE) ----------
default_support = 1000.0
default_resistance = 2000.0
default_volume_threshold = 1_000_000

# Keep your existing shape…
default_bollinger = {
    "mid_price": 0.0,
    "upper_band": 0.0,
    "lower_band": 0.0,
    # …and add tunables your code can use (safe defaults)
    "period": 20,
    "std_dev": 2,
}

# Keep MACD container and add periods used by indicator code
default_macd = {
    "signal_line": 0.0,
    "histogram": 0.0,
    "ma_fast": 0.0,
    "ma_slow": 0.0,
    "ma_signal": 0.0,
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9,
}

default_adx = {"period": 14, "threshold": 20}
default_ma = {"ma_fast": 9, "ma_slow": 20}
default_inside_bar = {"lookback": 1}
default_candle = {"min_body_percent": 0.7}
default_reason = ["Initial configuration with default values."]
default_signal = "Hold"

# New: feature params for advanced indicators / rules (optional; safe to ignore)
default_feature_params = {
    "hhll_lookback": 20,  # HH20/LL20
    "vwap_window": 60,  # rolling VWAP bars if session VWAP not available
    "vol_z_window": 20,  # volume z-score window
    "atr_period": 14,
    "min_atr_pct": 0.25,  # used by basic forecaster thresholds
    "near_barrier_bps": 15,  # "near" S/R definition
    "ema_slope_bps_ok": 2.0,  # slope threshold for trend bias
    "squeeze_multiplier": 1.10,  # bb_squeeze = width < min_width*mult
}

for stock in top_stocks:
    row = df_eq[df_eq["tradingsymbol"] == stock]
    if not row.empty:
        token = int(row["instrument_token"].values[0])
        stock_name = row["name"].values[0] if "name" in row.columns else stock

        # ✅ JSON entry
        config_stocks.append(
            {
                "stock_code": stock,
                "instrument_token": token,
                # Core levels
                "support": default_support,
                "resistance": default_resistance,
                "volume_threshold": default_volume_threshold,
                # Indicators + tunables
                "bollinger": default_bollinger.copy(),
                "macd": default_macd.copy(),
                "adx": default_adx.copy(),
                "moving_averages": default_ma.copy(),
                # Patterns / candles
                "inside_bar": default_inside_bar.copy(),
                "candle": default_candle.copy(),
                # Extra parameters for advanced features (optional consumer)
                "feature_params": default_feature_params.copy(),
                # Meta
                "reason": default_reason.copy(),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "signal": default_signal,
            }
        )

        # ✅ CSV row
        csv_rows.append(
            {"instrument_token": token, "stock_code": stock, "stock_name": stock_name}
        )
    else:
        missing.append(stock)

# ✅ Save JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump({"stocks": config_stocks}, f, indent=2)

# ✅ Save CSV
pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

# ✅ Summary
print(f"\n🎯 Requested: {len(top_stocks)} stocks")
print(f"✅ Added to config: {len(config_stocks)}")
print(f"❌ Not Found in instrument.csv: {len(missing)}")

if missing:
    print("\n🚫 Missing stocks:")
    for stock in missing:
        print(" -", stock)
else:
    print("🎉 All stocks found successfully!")

print(f"\n✅ JSON saved to {OUTPUT_JSON}")
print(f"✅ CSV saved to {OUTPUT_CSV}")
