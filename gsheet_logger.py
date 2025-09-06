# gsheet_logger.py

import gspread
from datetime import datetime


def get_gsheet_client(sheet_name: str):
    """Authorize and return a specific Google Sheet client by name."""
    gc = gspread.service_account(filename="cred.json")
    sh = gc.open(sheet_name)
    return sh


def _get_ohlcv_fields(stock_cfg: dict):
    """
    Pulls latest OHLCV + current price from config.
    Expects:
      stock_cfg["current_price"] = float | None
      stock_cfg["ohlcv"] = {"time","open","high","low","close","volume"} | None
    Case-insensitive safe reads, blanks when missing.
    """

    def pick(d, key):
        if not isinstance(d, dict) or d is None:
            return ""
        # case-insensitive
        for k, v in d.items():
            if str(k).lower() == str(key).lower():
                return "" if v is None else v
        return ""

    ohlcv = stock_cfg.get("ohlcv") or {}
    return {
        # "price_time": pick(ohlcv, "time"),
        "open": pick(ohlcv, "open"),
        "high": pick(ohlcv, "high"),
        "low": pick(ohlcv, "low"),
        "close": pick(ohlcv, "close"),
        "volume": pick(ohlcv, "volume"),
        # "current_price": (
        #     ""
        #     if stock_cfg.get("current_price") is None
        #     else stock_cfg.get("current_price")
        # ),
    }


def _blank(x):
    return "" if x is None else x


def flatten_config(stock_cfg: dict) -> dict:
    """
    Flatten nested stock config dict for Google Sheet logging.

    Includes:
    - Legacy SR + % width
    - Realtime / 1d / 1w / 3m SR + their % widths
    - Entry/Target/Stoploss for default, S1/R1, S2/R2, S3/R3
    - Respect counts for each S/R pair
    - OHLCV close & volume
    - Signal
    """

    # ---- Core SR values ----
    S = stock_cfg.get("support")
    R = stock_cfg.get("resistance")
    # SRT = stock_cfg.get("support_realtime")
    # RRT = stock_cfg.get("resistance_realtime")
    S1 = stock_cfg.get("support_1d")
    R1 = stock_cfg.get("resistance_1d")
    S2 = stock_cfg.get("support_1w")
    R2 = stock_cfg.get("resistance_1w")
    S3 = stock_cfg.get("support_3m")
    R3 = stock_cfg.get("resistance_3m")

    # ---- SR widths ----
    SR_pct = stock_cfg.get("sr_range_pct")
    # SR_rt_pct = stock_cfg.get("sr_range_pct_realtime")
    SR1_pct = stock_cfg.get("sr_range_pct_1d")
    SR2_pct = stock_cfg.get("sr_range_pct_1w")
    SR3_pct = stock_cfg.get("sr_range_pct_3m")

    # ---- Trade levels (entries/targets/stops) ----
    entry = stock_cfg.get("entry")
    target = stock_cfg.get("target")
    stoploss = stock_cfg.get("stoploss")

    entry1 = stock_cfg.get("entry1")
    target1 = stock_cfg.get("target1")
    stoploss1 = stock_cfg.get("stoploss1")

    entry2 = stock_cfg.get("entry2")
    target2 = stock_cfg.get("target2")
    stoploss2 = stock_cfg.get("stoploss2")

    entry3 = stock_cfg.get("entry3")
    target3 = stock_cfg.get("target3")
    stoploss3 = stock_cfg.get("stoploss3")

    # ---- Respect counts (new) ----
    respected_S = stock_cfg.get("respected_S")
    respected_R = stock_cfg.get("respected_R")
    respected_S1 = stock_cfg.get("respected_S1")
    respected_R1 = stock_cfg.get("respected_R1")
    respected_S2 = stock_cfg.get("respected_S2")
    respected_R2 = stock_cfg.get("respected_R2")
    respected_S3 = stock_cfg.get("respected_S3")
    respected_R3 = stock_cfg.get("respected_R3")

    flat = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock_code": stock_cfg.get("stock_code"),
        # OHLCV
        "close": stock_cfg.get("ohlcv", {}).get("close"),
        "volume": stock_cfg.get("ohlcv", {}).get("volume"),
        # Legacy/default SR (mapped to 1d in your pipeline)
        "S": _blank(S),
        "R": _blank(R),
        "SR_pct": _blank(SR_pct),
        # Realtime SR
        # "S_RT": _blank(SRT),
        # "R_RT": _blank(RRT),
        # "SR_RT_pct": _blank(SR_rt_pct),
        # 1d SR
        "S1": _blank(S1),
        "R1": _blank(R1),
        "SR1_pct": _blank(SR1_pct),
        # 1w SR
        "S2": _blank(S2),
        "R2": _blank(R2),
        "SR2_pct": _blank(SR2_pct),
        # 3m SR
        "S3": _blank(S3),
        "R3": _blank(R3),
        "SR3_pct": _blank(SR3_pct),
        # Trade levels
        "entry": _blank(entry),
        "target": _blank(target),
        "stoploss": _blank(stoploss),
        "entry1": _blank(entry1),
        "target1": _blank(target1),
        "stoploss1": _blank(stoploss1),
        "entry2": _blank(entry2),
        "target2": _blank(target2),
        "stoploss2": _blank(stoploss2),
        "entry3": _blank(entry3),
        "target3": _blank(target3),
        "stoploss3": _blank(stoploss3),
        # Respect counts (new)
        "respected_S": _blank(respected_S),
        "respected_R": _blank(respected_R),
        "respected_S1": _blank(respected_S1),
        "respected_R1": _blank(respected_R1),
        "respected_S2": _blank(respected_S2),
        "respected_R2": _blank(respected_R2),
        "respected_S3": _blank(respected_S3),
        "respected_R3": _blank(respected_R3),
        # Meta
        "signal": stock_cfg.get("signal"),
    }

    # Latest OHLCV
    # flat.update(_get_ohlcv_fields(stock_cfg))

    # Predicted (forecasts)
    # flat.update(_get_pred_fields(stock_cfg))

    # If you also want realized target fields, uncomment:
    # flat.update(_get_target_fields(stock_cfg))

    return flat


def ensure_headers(worksheet, desired_headers: list[str]) -> list[str]:
    """
    Ensure the sheet's first row contains all desired headers.
    - If the sheet is empty, write desired_headers.
    - If headers exist, append any missing new headers to the end (no clearing).
    Returns the final header order present on the sheet.
    """
    first_row = worksheet.row_values(1)

    if not first_row:
        # Empty sheet → write fresh headers
        worksheet.update("A1", [desired_headers])
        return desired_headers

    # Extend with any missing columns (preserve existing order)
    missing = [h for h in desired_headers if h not in first_row]
    if missing:
        new_headers = first_row + missing
        # Overwrite only the header row with the extended header list
        worksheet.update("A1", [new_headers])
        return new_headers

    return first_row


# ============== NEW: small utilities for A1 ranges & header lookups ==============


def _col_index(headers: list[str], col_name: str) -> int | None:
    """Return 0-based index of a header name; None if missing."""
    try:
        return [h.strip() for h in headers].index(col_name)
    except ValueError:
        return None


def _a1_from_row(headers_len: int, row_number: int) -> str:
    """
    Make an A1 range for an entire row given header length.
    Example: headers_len=10, row_number=5 -> 'A5:J5'
    """

    def _col_letter(n: int) -> str:
        # 1-based to letters
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(65 + r) + s
        return s

    start_col_letter = "A"
    end_col_letter = _col_letter(headers_len)
    return f"{start_col_letter}{row_number}:{end_col_letter}{row_number}"


# ===================== UPDATED: upsert instead of append =========================


def log_config_upsert(
    stock_cfg: dict, sheet_name: str, tab_name: str, key_col: str = "stock_code"
):
    """
    Upsert a single row by `key_col` (default: 'stock_code'):
      - Ensures headers (adds missing columns if needed).
      - If a row with the same stock_code exists, updates that row only.
      - Otherwise appends a new row.

    :param stock_cfg: dict of stock configuration
    :param sheet_name: Google Sheet file name
    :param tab_name: Worksheet/tab name within the sheet
    :param key_col: Column used as unique key (must exist in headers)
    """
    sh = get_gsheet_client(sheet_name)
    try:
        worksheet = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = sh.add_worksheet(title=tab_name, rows="20000", cols="100")

    # 1) Flatten and ensure headers
    flat_cfg = flatten_config(stock_cfg)
    desired_headers = list(flat_cfg.keys())
    final_headers = ensure_headers(worksheet, desired_headers)

    # 2) Ensure key column exists in headers
    key_idx0 = _col_index(final_headers, key_col)
    if key_idx0 is None:
        # If somehow missing, extend headers and recompute
        final_headers = ensure_headers(worksheet, final_headers + [key_col])
        key_idx0 = _col_index(final_headers, key_col)
        if key_idx0 is None:
            raise RuntimeError(
                f"Cannot find or create key column '{key_col}' in sheet headers."
            )

    # 3) Build row aligned to final_headers
    #    (Any header not in flat_cfg becomes "")
    row_values = [flat_cfg.get(h, "") for h in final_headers]

    # 4) Upsert by key
    key_value = flat_cfg.get(key_col, "")
    if key_value is None or str(key_value).strip() == "":
        raise ValueError(
            f"Upsert requires non-empty '{key_col}' in data. Got: {key_value!r}"
        )

    # Only scan the single key column to find matching row
    # (faster and avoids false positives in other columns)
    key_col_1based = key_idx0 + 1
    # Read only that column (from row 2 downwards; row 1 is header)
    key_column_cells = worksheet.col_values(key_col_1based)[1:]  # skip header row

    target_row_number = None
    for i, cell_val in enumerate(
        key_column_cells, start=2
    ):  # sheet rows start at 1; row 1 is header
        if str(cell_val).strip() == str(key_value).strip():
            target_row_number = i
            break

    if target_row_number is None:
        # 5a) No match -> append as new row
        worksheet.append_row(row_values, value_input_option="USER_ENTERED")
        print(f"[INFO] UPSERT (append) for {key_value} → {sheet_name}/{tab_name}")
    else:
        # 5b) Match found -> update that exact row range (A1 style)
        rng = _a1_from_row(len(final_headers), target_row_number)
        worksheet.update(rng, [row_values], value_input_option="USER_ENTERED")
        print(
            f"[INFO] UPSERT (update row {target_row_number}) for {key_value} → {sheet_name}/{tab_name}"
        )


def log_config_update(stock_cfg: dict, sheet_name: str, tab_name: str):
    """
    Backwards-compatible wrapper: now performs UPSERT by 'stock_code'
    instead of always appending a new row.
    """
    return log_config_upsert(stock_cfg, sheet_name, tab_name, key_col="stock_code")
