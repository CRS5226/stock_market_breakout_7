# pretty_forecasts.py
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional

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


def _latest_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].astype(str).str.upper()
    if "model" not in df.columns:
        df["model"] = "ALGO"
    df = df.sort_values(["stock_code", "model", "timestamp"])
    idx = (
        df.groupby(["stock_code", "model"], dropna=False)["timestamp"].transform("max")
        == df["timestamp"]
    )
    return df[idx].copy()


# def _make_block(
#     stock: str, ts_display: str, gpt_row: Optional[Dict], algo_row: Optional[Dict]
# ) -> List[List]:
#     """
#     Layout:

#     <STOCK> (last updated : <timestamp>)
#     close price : <val>, (signal : <val>)
#     [ "", "GPT", "ALGO", "", "GPT", "ALGO", "", "GPT", "ALGO" ]  (model header for 3 triplets)
#     S   [GPT] [ALGO]   R   [GPT] [ALGO]   SR_pct [GPT] [ALGO]
#     S1  [GPT] [ALGO]   R1  [GPT] [ALGO]   SR1_pct [GPT] [ALGO]
#     S2  [GPT] [ALGO]   R2  [GPT] [ALGO]   SR2_pct [GPT] [ALGO]
#     S3  [GPT] [ALGO]   R3  [GPT] [ALGO]   SR3_pct [GPT] [ALGO]
#     <blank>
#     entry  [GPT] [ALGO]  target  [GPT] [ALGO]  stoploss  [GPT] [ALGO]
#     entry1 [GPT] [ALGO]  target1 [GPT] [ALGO]  stoploss1 [GPT] [ALGO]
#     entry2 [GPT] [ALGO]  target2 [GPT] [ALGO]  stoploss2 [GPT] [ALGO]
#     entry3 [GPT] [ALGO]  target3 [GPT] [ALGO]  stoploss3 [GPT] [ALGO]
#     <blank>
#     respected_S  [GPT] [ALGO]  respected_R  [GPT] [ALGO]
#     respected_S1 [GPT] [ALGO]  respected_R1 [GPT] [ALGO]
#     respected_S2 [GPT] [ALGO]  respected_R2 [GPT] [ALGO]
#     respected_S3 [GPT] [ALGO]  respected_R3 [GPT] [ALGO]
#     """
#     # ------- headers -------
#     header1 = [f"{stock} (last updated : {ts_display})"]

#     close_g = _safe(gpt_row or {}, "close", "")
#     close_a = _safe(algo_row or {}, "close", "")
#     if close_g and close_a and str(close_g) != str(close_a):
#         close_txt = f"close price : GPT={close_g}, ALGO={close_a}"
#     else:
#         close_txt = f"close price : {close_g or close_a}"

#     sig_g = _safe(gpt_row or {}, "signal", "")
#     sig_a = _safe(algo_row or {}, "signal", "")
#     if sig_g and sig_a and str(sig_g) != str(sig_a):
#         sig_txt = f"(signal : GPT={sig_g}, ALGO={sig_a})"
#     else:
#         sig_txt = f"(signal : {sig_g or sig_a})"

#     header2 = [f"{close_txt}, {sig_txt}"]

#     # helper: one metric triplet
#     def tri(label: str, key: str) -> List:
#         return [label, _safe(gpt_row or {}, key, ""), _safe(algo_row or {}, key, "")]

#     # one-time model header for 3 triplets (S/R/SR_pct width)
#     model_header = ["", "GPT", "ALGO", "", "GPT", "ALGO", "", "GPT", "ALGO"]

#     # ------- level rows -------
#     base_row = tri("S", "S") + tri("R", "R") + tri("SR_pct", "SR_pct")
#     row_S1 = tri("S1", "S1") + tri("R1", "R1") + tri("SR1_pct", "SR1_pct")
#     row_S2 = tri("S2", "S2") + tri("R2", "R2") + tri("SR2_pct", "SR2_pct")
#     row_S3 = tri("S3", "S3") + tri("R3", "R3") + tri("SR3_pct", "SR3_pct")

#     # ------- entries section (NO blank between entry and entry1) -------
#     entry_row = (
#         tri("entry", "entry") + tri("target", "target") + tri("stoploss", "stoploss")
#     )
#     entry1_row = (
#         tri("entry1", "entry1")
#         + tri("target1", "target1")
#         + tri("stoploss1", "stoploss1")
#     )
#     entry2_row = (
#         tri("entry2", "entry2")
#         + tri("target2", "target2")
#         + tri("stoploss2", "stoploss2")
#     )
#     entry3_row = (
#         tri("entry3", "entry3")
#         + tri("target3", "target3")
#         + tri("stoploss3", "stoploss3")
#     )

#     # ------- respected section -------
#     res_row0 = tri("respected_S", "respected_S") + tri("respected_R", "respected_R")
#     res_row1 = tri("respected_S1", "respected_S1") + tri("respected_R1", "respected_R1")
#     res_row2 = tri("respected_S2", "respected_S2") + tri("respected_R2", "respected_R2")
#     res_row3 = tri("respected_S3", "respected_S3") + tri("respected_R3", "respected_R3")

#     # assemble with blanks: one before entries, none between entry and entry1, one before respected
#     block: List[List] = [
#         header1,
#         header2,
#         model_header,
#         base_row,
#         row_S1,
#         row_S2,
#         row_S3,
#         [],  # blank before base entry row
#         entry_row,
#         entry1_row,  # <-- no blank here
#         entry2_row,
#         entry3_row,
#         [],  # blank before respected rows
#         res_row0,
#         res_row1,
#         res_row2,
#         res_row3,
#     ]
#     return block


def _make_block(
    stock: str,
    ts_display_unused: str,
    gpt_row: Optional[Dict],
    algo_row: Optional[Dict],
) -> List[List]:
    """
    Per-model layout (ALGO block, then GPT) with close price in the header.

    CDSL [ALGO] (last updated : ts_algo) (close price : <close_algo>) (signal : <sig_algo>)
    S, R, SR_pct, entry, target, stoploss, respected_S, respected_R
    S1, R1, SR1_pct, entry1, target1, stoploss1, respected_S1, respected_R1
    S2, R2, SR2_pct, entry2, target2, stoploss2, respected_S2, respected_R2
    S3, R3, SR3_pct, entry3, target3, stoploss3, respected_S3, respected_R3

    <blank>

    CDSL [GPT] (last updated : ts_gpt) (close price : <close_gpt>) (signal : <sig_gpt>)
    (same rows)
    """
    import pandas as pd

    def ts_of(row: Optional[Dict]) -> str:
        if row and row.get("timestamp"):
            try:
                return pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(row["timestamp"])
        return ""

    def one_model_header(model_tag: str, row: Optional[Dict]) -> List[str]:
        ts = ts_of(row)
        close = _safe(row or {}, "close", "")
        sig = _safe(row or {}, "signal", "")
        # e.g., "CDSL [ALGO] (last updated : 2025-09-04 11:39:05) (close price : 1508.8) (signal : No Action)"
        return [
            f"{stock} [{model_tag}] (last updated : {ts}) (close price : {close}) (signal : {sig})"
        ]

    def pair(row: Optional[Dict], label: str, key: str) -> List:
        return [label, _safe(row or {}, key, "")]

    def s_row(row: Optional[Dict]) -> List:
        # S line with base metrics + respected_S/R
        r: List = []
        r += pair(row, "S", "S")
        r += pair(row, "R", "R")
        r += pair(row, "SR_pct", "SR_pct")
        r += pair(row, "entry", "entry")
        r += pair(row, "target", "target")
        r += pair(row, "stoploss", "stoploss")
        r += pair(row, "respected_S", "respected_S")
        r += pair(row, "respected_R", "respected_R")
        return r

    def sN_row(row: Optional[Dict], n: int) -> List:
        # S1/S2/S3 rows with R*, SR*%, entry*, target*, stoploss*, respected_S*, respected_R*
        r: List = []
        r += pair(row, f"S{n}", f"S{n}")
        r += pair(row, f"R{n}", f"R{n}")
        r += pair(row, f"SR{n}_pct", f"SR{n}_pct")
        r += pair(row, f"entry{n}", f"entry{n}")
        r += pair(row, f"target{n}", f"target{n}")
        r += pair(row, f"stoploss{n}", f"stoploss{n}")
        r += pair(row, f"respected_S{n}", f"respected_S{n}")
        r += pair(row, f"respected_R{n}", f"respected_R{n}")
        return r

    # ----- ALGO block -----
    block: List[List] = [
        one_model_header("ALGO", algo_row),
        s_row(algo_row),
        sN_row(algo_row, 1),
        sN_row(algo_row, 2),
        sN_row(algo_row, 3),
        [],  # spacer between ALGO and GPT blocks (remove if you don't want a blank line)
    ]

    # ----- GPT block -----
    block += [
        one_model_header("GPT", gpt_row),
        s_row(gpt_row),
        sN_row(gpt_row, 1),
        sN_row(gpt_row, 2),
        sN_row(gpt_row, 3),
    ]

    return block


def _blocks_from_combined_df(
    combined_df: pd.DataFrame, spacer_rows: int = 3
) -> List[List[List]]:
    df = _latest_per_stock(combined_df)
    blocks: List[List[List]] = []
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

    spaced_blocks: List[List[List]] = []
    for i, blk in enumerate(blocks):
        spaced_blocks.append(blk)
        if i < len(blocks) - 1 and spacer_rows > 0:
            spaced_blocks.append([[] for _ in range(spacer_rows)])
    return spaced_blocks


# ---------- NO-FLICKER WRITER (no ws.clear, skip if unchanged, single range update) ----------


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
        # different widths
        # re-pad to the max width
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
    # flatten blocks -> list of rows; ensure rectangular grid
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


# ====================== PUBLIC ENTRYPOINTS (CSV) ======================


def write_pretty_csv_from_two_csvs(
    gpt_csv_path: str,
    algo_csv_path: str,
    out_csv_path: str = "forecasts_pretty_view.csv",
    spacer_rows: int = 3,
):
    def _load_tag(p, tag):
        df = pd.read_csv(p)
        df["model"] = tag
        return df

    gpt = _load_tag(gpt_csv_path, "GPT")
    algo = _load_tag(algo_csv_path, "ALGO")
    blocks = _blocks_from_combined_df(
        pd.concat([gpt, algo], ignore_index=True), spacer_rows=spacer_rows
    )
    flat_rows: List[List] = []
    for blk in blocks:
        flat_rows.extend(blk)
    out = Path(out_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(flat_rows)


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

    # <-- key change: no clear, update a fixed A1 range; skip if unchanged
    _update_sheet_values_no_flicker(ws_pretty, flat_rows)
