# # pretty_forecasts.py

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional
import math
import pandas as pd

try:
    import gspread  # optional
except Exception:
    gspread = None


# ============================= helpers =============================


def _safe(d: Dict, key: str, default=""):
    v = d.get(key, default)
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


# def _latest_per_stock(df: pd.DataFrame) -> pd.DataFrame:
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     if "stock_code" in df.columns:
#         df["stock_code"] = df["stock_code"].astype(str).str.upper()
#     if "model" not in df.columns:
#         df["model"] = "ALGO"
#     df = df.sort_values(["stock_code", "model", "timestamp"])
#     idx = (
#         df.groupby(["stock_code", "model"], dropna=False)["timestamp"].transform("max")
#         == df["timestamp"]
#     )
#     return df[idx].copy()


def _latest_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    # accept either 'timestamp' or 'last_updated'
    ts_col = None
    for cand in ("timestamp", "last_updated", "last_updated_at"):
        if cand in df.columns:
            ts_col = cand
            break

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        # create a uniform helper column for grouping
        df["_ts"] = df[ts_col]
    else:
        # no timestamp-like column—treat all as same time so last row wins within each group
        df["_ts"] = pd.NaT

    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].astype(str).str.upper()
    if "model" not in df.columns:
        df["model"] = "ALGO"

    df = df.sort_values(["stock_code", "model", "_ts"])
    idx = (
        df.groupby(["stock_code", "model"], dropna=False)["_ts"].transform("max")
        == df["_ts"]
    )
    out = df[idx].copy()
    out.drop(columns=["_ts"], inplace=True, errors="ignore")
    return out


# ---------- NEW: TF labels + schema-agnostic getters ----------

TF_LABELS = {
    0: "1min",
    1: "5min",
    2: "15min",
    3: "30min",
    4: "45min",
    5: "1hour",
    6: "4hour",
    7: "1day",
    8: "1month",
}


def _get_num(row: Optional[Dict], *keys):
    if not row:
        return None
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                v = row[k]
                # normalize common string sentinels
                if isinstance(v, str) and v.strip().lower() in (
                    "nan",
                    "na",
                    "null",
                    "none",
                    "inf",
                    "+inf",
                    "-inf",
                ):
                    continue
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
                return fv
            except Exception:
                pass
    return None


def _get_str(row: Optional[Dict], *keys, default=""):
    if not row:
        return default
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return default


def _sr_pair(row: Optional[Dict], idx: int):
    """Return (support, resistance) for base (idx=0) or idx>0. Accepts new and legacy names."""
    if idx == 0:
        s = _get_num(row, "support", "S")
        r = _get_num(row, "resistance", "R")
    else:
        s = _get_num(row, f"support{idx}", f"S{idx}")
        r = _get_num(row, f"resistance{idx}", f"R{idx}")
    return s, r


def _triplet(row: Optional[Dict], idx: int):
    if idx == 0:
        e = _get_num(row, "entry")
        t = _get_num(row, "target")
        sl = _get_num(row, "stoploss")
    else:
        e = _get_num(row, f"entry{idx}")
        t = _get_num(row, f"target{idx}")
        sl = _get_num(row, f"stoploss{idx}")
    return e, t, sl


def _sr_pct(row: Optional[Dict], idx: int):
    """SR% from row if present; else compute ((R-S)/S)*100."""
    if idx == 0:
        v = _get_num(row, "sr_range_pct", "SR_pct")
    else:
        v = _get_num(row, f"SR{idx}_pct")  # legacy spare key if present
    if v is not None:
        return v
    s, r = _sr_pair(row, idx)
    if s is None or r is None or s == 0:
        return None
    try:
        return ((r - s) / s) * 100.0
    except Exception:
        return None


def _respect_pair(row: Optional[Dict], idx: int):
    if idx == 0:
        rs = _get_num(row, "respected_S")
        rr = _get_num(row, "respected_R")
    else:
        rs = _get_num(row, f"respected_S{idx}")
        rr = _get_num(row, f"respected_R{idx}")

    def safe_int(v, default=0):
        try:
            if v is None:
                return default
            v = float(v)
            if math.isnan(v) or math.isinf(v):
                return default
            return int(v)
        except Exception:
            return default

    return safe_int(rs), safe_int(rr)


# ============================= block builder =============================


def _make_block(
    stock: str,
    ts_display_unused: str,
    gpt_row: Optional[Dict],
    algo_row: Optional[Dict],
) -> List[List]:
    """
    Output per-model sections (ALGO then GPT):
    <STOCK> [MODEL] (last updated : ts) (close price : <close>) (signal : <sig>)
    tf | S | R | SR_pct | entry | target | stoploss | respected_S | respected_R
    one row per TF (1min..1month)
    """

    def ts_of(row: Optional[Dict]) -> str:
        v = _get_str(row, "timestamp", "last_updated", "last_updated_at")
        if not v:
            return ""
        try:
            return pd.to_datetime(v).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(v)

    def header(model_tag: str, row: Optional[Dict]) -> List[str]:
        ts = ts_of(row)
        close = _get_num(row, "close")
        if close is None and isinstance(row, dict):
            close = _get_num(row.get("ohlcv", {}) if row else {}, "close")
        close_txt = f"{close:.6g}" if close is not None else ""
        sig = _get_str(row, "signal")
        return [
            f"{stock} [{model_tag}] (last updated : {ts}) (close price : {close_txt}) (signal : {sig})"
        ]

    def one_row(row: Optional[Dict], idx: int) -> List:
        tf = TF_LABELS.get(idx, f"tf{idx}")
        s, r = _sr_pair(row, idx)
        srp = _sr_pct(row, idx)
        e, t, sl = _triplet(row, idx)
        rs, rr = _respect_pair(row, idx)

        def fmt(x):
            return (
                ""
                if x is None
                else (f"{x:.6g}" if isinstance(x, (int, float)) else str(x))
            )

        return [
            tf,
            fmt(s),
            fmt(r),
            fmt(srp),
            fmt(e),
            fmt(t),
            fmt(sl),
            str(rs),
            str(rr),
        ]

    block: List[List] = []
    # ALGO section
    block.append(header("ALGO", algo_row))
    block.append(
        [
            "tf",
            "S",
            "R",
            "SR_pct",
            "entry",
            "target",
            "stoploss",
            "respected_S",
            "respected_R",
        ]
    )
    for idx in range(0, 9):
        block.append(one_row(algo_row, idx))
    block.append([])  # spacer

    # GPT section
    block.append(header("GPT", gpt_row))
    block.append(
        [
            "tf",
            "S",
            "R",
            "SR_pct",
            "entry",
            "target",
            "stoploss",
            "respected_S",
            "respected_R",
        ]
    )
    for idx in range(0, 9):
        block.append(one_row(gpt_row, idx))

    return block


# def _blocks_from_combined_df(
#     combined_df: pd.DataFrame, spacer_rows: int = 3
# ) -> List[List[List]]:
#     df = _latest_per_stock(combined_df)
#     blocks: List[List[List]] = []
#     for stock, g in df.groupby("stock_code", dropna=False):
#         gpt_row = g[g["model"] == "GPT"].sort_values("timestamp").tail(1)
#         algo_row = g[g["model"] == "ALGO"].sort_values("timestamp").tail(1)
#         gpt_row = gpt_row.iloc[0].to_dict() if not gpt_row.empty else None
#         algo_row = algo_row.iloc[0].to_dict() if not algo_row.empty else None

#         # combined last-updated if you ever want it
#         ts_vals = []
#         if gpt_row and gpt_row.get("timestamp"):
#             ts_vals.append(pd.to_datetime(gpt_row["timestamp"]))
#         if algo_row and algo_row.get("timestamp"):
#             ts_vals.append(pd.to_datetime(algo_row["timestamp"]))
#         ts_display = max(ts_vals).strftime("%Y-%m-%d %H:%M:%S") if ts_vals else ""

#         blocks.append(_make_block(stock, ts_display, gpt_row, algo_row))

#     spaced_blocks: List[List[List]] = []
#     for i, blk in enumerate(blocks):
#         spaced_blocks.append(blk)
#         if i < len(blocks) - 1 and spacer_rows > 0:
#             spaced_blocks.append([[] for _ in range(spacer_rows)])
#     return spaced_blocks


def _blocks_from_combined_df(
    combined_df: pd.DataFrame, spacer_rows: int = 3
) -> List[List[List]]:
    df = _latest_per_stock(combined_df)
    blocks: List[List[List]] = []

    # --- DUMMY ALGO FILTER block ---
    dummy_algo = [
        [
            "DUMMY_ALGO [ALGO FILTER] (last updated : - ) (close price : - ) (signal : - )"
        ],
        [
            "tf",
            "S",
            "R",
            "SR_pct",
            "entry",
            "target",
            "stoploss",
            "respected_S",
            "respected_R",
        ],
    ]
    for idx in range(0, 9):
        dummy_algo.append([TF_LABELS.get(idx, f"tf{idx}")] + [""] * 8)
    blocks.append(dummy_algo)

    blocks.append([[]])  # spacer row between dummy blocks

    # --- DUMMY GPT FILTER block ---
    dummy_gpt = [
        ["DUMMY_GPT [GPT FILTER] (last updated : - ) (close price : - ) (signal : - )"],
        [
            "tf",
            "S",
            "R",
            "SR_pct",
            "entry",
            "target",
            "stoploss",
            "respected_S",
            "respected_R",
        ],
    ]
    for idx in range(0, 9):
        dummy_gpt.append([TF_LABELS.get(idx, f"tf{idx}")] + [""] * 8)
    blocks.append(dummy_gpt)

    blocks.append([[] for _ in range(spacer_rows)])  # spacer rows after dummy blocks

    # --- Real stock blocks ---
    for stock, g in df.groupby("stock_code", dropna=False):
        gpt_row = g[g["model"] == "GPT"].sort_values("timestamp").tail(1)
        algo_row = g[g["model"] == "ALGO"].sort_values("timestamp").tail(1)
        gpt_row = gpt_row.iloc[0].to_dict() if not gpt_row.empty else None
        algo_row = algo_row.iloc[0].to_dict() if not algo_row.empty else None

        ts_vals = []
        if gpt_row and gpt_row.get("timestamp"):
            ts_vals.append(pd.to_datetime(gpt_row["timestamp"]))
        if algo_row and algo_row.get("timestamp"):
            ts_vals.append(pd.to_datetime(algo_row["timestamp"]))
        ts_display = max(ts_vals).strftime("%Y-%m-%d %H:%M:%S") if ts_vals else ""

        blocks.append(_make_block(stock, ts_display, gpt_row, algo_row))
        blocks.append([[] for _ in range(spacer_rows)])  # spacing between stocks

    return blocks


# ---------- NO-FLICKER SHEET WRITER (no ws.clear, skip if unchanged, single range update) ----------


def _col_letter(n: int) -> str:
    # 1 -> A, 2 -> B, ...
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _pad_grid(rows: List[List]) -> List[List]:
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [list(r) + [""] * (width - len(r)) for r in rows]


def _get_current_values(ws) -> List[List]:
    # get_all_values returns only the used range
    return ws.get_all_values()


def _grids_equal(a: List[List], b: List[List]) -> bool:
    # Compare after padding to same rectangular shape
    pa = _pad_grid(a)
    pb = _pad_grid(b)
    if len(pa) != len(pb):
        return False
    if pa and pb and len(pa[0]) != len(pb[0]):
        # different widths → re-pad to the max width
        maxw = max(len(pa[0]), len(pb[0]))
        pa = [r + [""] * (maxw - len(r)) for r in pa]
        pb = [r + [""] * (maxw - len(r)) for r in pb]
    return pa == pb


def _update_sheet_values_no_flicker(ws, new_rows: List[List]):
    """
    Overwrite the visible grid without ws.clear():
    - Skip if unchanged
    - Update a single explicit A1 range sized to the max(current,new)
    - Pad blanks to wipe leftover cells (no separate clear)
    """
    cur = _get_current_values(ws)
    new_grid = _pad_grid(new_rows)
    if _grids_equal(cur, new_grid):
        return  # no change -> no flicker

    # Determine target rectangle = max of current vs new
    tgt_rows = max(len(cur), len(new_grid))
    tgt_cols = max(len(cur[0]) if cur else 0, len(new_grid[0]) if new_grid else 0)

    # Pad new_grid up to target size (so it overwrites everything)
    padded = [row + [""] * (tgt_cols - len(row)) for row in new_grid]
    padded += [[""] * tgt_cols for _ in range(tgt_rows - len(padded))]

    end_col = _col_letter(tgt_cols or 1)
    end_row = tgt_rows if tgt_rows > 0 else 1
    a1 = f"A1:{end_col}{end_row}"
    ws.update(a1, padded, value_input_option="RAW")


# ============== PUBLIC ENTRYPOINT (WRITE DIRECTLY TO SHEET TAB) ==============


def write_pretty_to_sheet_from_sheets(
    spreadsheet_name: str,
    gpt_tab: str,
    algo_tab: str,
    pretty_tab: str = "stocks_30_pretty",
    spacer_rows: int = 3,
    service_account_json: Optional[str] = None,
):
    if gspread is None:
        raise RuntimeError("gspread is not installed. pip install gspread oauth2client")

    gc = (
        gspread.service_account(filename=service_account_json)
        if service_account_json
        else gspread.service_account()
    )
    sh = gc.open(spreadsheet_name)

    ws_gpt = sh.worksheet(gpt_tab)
    ws_algo = sh.worksheet(algo_tab)

    df_gpt = pd.DataFrame(ws_gpt.get_all_records())
    df_algo = pd.DataFrame(ws_algo.get_all_records())
    df_gpt["model"] = "GPT"
    df_algo["model"] = "ALGO"

    blocks = _blocks_from_combined_df(
        pd.concat([df_gpt, df_algo], ignore_index=True), spacer_rows=spacer_rows
    )
    flat_rows: List[List] = []
    for blk in blocks:
        flat_rows.extend(blk)

    # ensure target tab
    try:
        ws_pretty = sh.worksheet(pretty_tab)
    except Exception:
        ws_pretty = sh.add_worksheet(title=pretty_tab, rows=100, cols=50)

    # no clear; single A1 range update; skip when unchanged
    _update_sheet_values_no_flicker(ws_pretty, flat_rows)
