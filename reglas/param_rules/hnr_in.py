# hnr_in.py
from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Dict, Any, Iterable, Union

WINDOW_DAYS    = 30
BASE_MIN_CLP   = 1000.0
PCTS           = (95, 97, 99)

def _as_list(x: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(x, str):
        return [x]
    return list(map(str, x))

def _max_count_window30(ts: np.ndarray, days: int) -> int:
    if ts.size == 0:
        return 0
    ts = np.sort(ts)
    best = 0
    j = 0
    delta = np.timedelta64(days, "D")
    for i in range(ts.size):
        end = ts[i] + delta
        while j < ts.size and ts[j] <= end:
            j += 1
        best = max(best, j - i)
    return best

def run_parameters_hnr_in(path: str, subsubsegments: Union[str, Iterable[str]], *, verbose: bool=False) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    req = {"customer_id","tx_date_time","tx_amount","tx_base_amount","tx_direction","tx_type","customer_sub_sub_type"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Faltan columnas para HNR-IN: {miss}")

    df["tx_date_time"]   = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_amount"]      = pd.to_numeric(df["tx_amount"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    df["tx_direction"]   = df["tx_direction"].astype(str).str.title()
    df["tx_type"]        = df["tx_type"].astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    is_round = np.isfinite(df["tx_amount"]) & np.isclose(df["tx_amount"] % 1000.0, 0.0, atol=1e-9)
    m = (
        df["tx_direction"].eq("Inbound") &
        df["tx_type"].eq("Cash") &
        is_round &
        (df["tx_base_amount"] > BASE_MIN_CLP) &
        df["tx_date_time"].notna() &
        df["customer_id"].notna()
    )
    g = df.loc[m, ["customer_id", "tx_date_time"]].sort_values(["customer_id","tx_date_time"])

    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{p}" for p in PCTS], "Number_max30d":[np.nan]*len(PCTS)})
        return {"meta":{"clients":0}, "percentiles": tbl}

    rows = []
    for cid, sub in g.groupby("customer_id", sort=False):
        rows.append({"customer_id": cid, "max_30d": _max_count_window30(sub["tx_date_time"].values, WINDOW_DAYS)})
    res = pd.DataFrame(rows)

    s = pd.to_numeric(res["max_30d"], errors="coerce").dropna()
    pct_vals = {f"p{p}": (float(np.percentile(s, p)) if len(s) else np.nan) for p in PCTS}
    tbl = pd.DataFrame({"percentil":[f"p{p}" for p in PCTS],
                        "Number_max30d":[pct_vals[f"p{p}"] for p in PCTS]})
    suggested = int(math.ceil(pct_vals["p95"])) if np.isfinite(pct_vals.get("p95", np.nan)) else np.nan
    return {"meta":{"clients":res.shape[0], "suggested_number": suggested}, "percentiles": tbl}
