# gsheet_logger.py

import gspread
from datetime import datetime


def get_gsheet_client(sheet_name: str):
    """Authorize and return a specific Google Sheet client by name."""
    gc = gspread.service_account(filename="cred.json")
    sh = gc.open(sheet_name)
    return sh


def _blank(x):
    return "" if x is None else x


def _get_num(d, *keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] not in (None, ""):
            try:
                return float(d[k])
            except Exception:
                pass
    return None


def _sr_pct_from(s, r):
    try:
        if s is None or r is None or float(s) == 0.0:
            return None
        return ((float(r) - float(s)) / float(s)) * 100.0
    except Exception:
        return None


def flatten_config(stock_cfg: dict) -> dict:
    """
    Flatten the (new) multi-TF config into a single row for Google Sheets.
    Outputs base (S0) + S1..S8:
      - S, R, SR*_pct
      - entry*, target*, stoploss*
      - respected_S*, respected_R*
    Also includes timestamp, stock_code, close, volume, signal, model.
    """
    out = {
        # meta
        "timestamp": stock_cfg.get("last_updated")
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stock_code": stock_cfg.get("stock_code"),
        "signal": stock_cfg.get("signal"),
        "model": (
            "GPT"
            if str(stock_cfg.get("forecast", "")).lower() in ("AI", "LLM")
            else "ALGO"
        ),
        # snapshot
        "close": _get_num(stock_cfg.get("ohlcv", {}) or {}, "close")
        or _get_num(stock_cfg, "close"),
        "volume": _get_num(stock_cfg.get("ohlcv", {}) or {}, "volume"),
    }

    # ----- base (idx=0) uses unsuffixed keys -----
    base_s = _get_num(stock_cfg, "support")
    base_r = _get_num(stock_cfg, "resistance")
    out["S"] = _blank(base_s)
    out["R"] = _blank(base_r)
    # prefer precomputed base width if present
    base_pct = _get_num(stock_cfg, "sr_range_pct")
    if base_pct is None:
        base_pct = _sr_pct_from(base_s, base_r)
    out["SR_pct"] = _blank(base_pct)

    out["entry"] = _blank(_get_num(stock_cfg, "entry"))
    out["target"] = _blank(_get_num(stock_cfg, "target"))
    out["stoploss"] = _blank(_get_num(stock_cfg, "stoploss"))

    out["respected_S"] = _blank(_get_num(stock_cfg, "respected_S"))
    out["respected_R"] = _blank(_get_num(stock_cfg, "respected_R"))

    # ----- S1..S8 -----
    for idx in range(1, 9):
        s_key = f"support{idx}"
        r_key = f"resistance{idx}"
        e_key = f"entry{idx}"
        t_key = f"target{idx}"
        sl_key = f"stoploss{idx}"
        rs_key = f"respected_S{idx}"
        rr_key = f"respected_R{idx}"
        pct_key = f"SR{idx}_pct"

        s = _get_num(stock_cfg, s_key)
        r = _get_num(stock_cfg, r_key)

        # prefer precomputed SR% if you ever store it; else compute on the fly
        sr_pct = _get_num(stock_cfg, pct_key)
        if sr_pct is None:
            sr_pct = _sr_pct_from(s, r)

        out[f"S{idx}"] = _blank(s)
        out[f"R{idx}"] = _blank(r)
        out[pct_key] = _blank(sr_pct)

        out[e_key] = _blank(_get_num(stock_cfg, e_key))
        out[t_key] = _blank(_get_num(stock_cfg, t_key))
        out[sl_key] = _blank(_get_num(stock_cfg, sl_key))

        out[rs_key] = _blank(_get_num(stock_cfg, rs_key))
        out[rr_key] = _blank(_get_num(stock_cfg, rr_key))

    return out


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
