from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (95, 97, 99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_lbal(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[int] = DEFAULT_PCTS,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    df["tx_date_time"]   = pd.to_datetime(df.get("tx_date_time"), errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")
    df["tx_direction"]   = df.get("tx_direction","").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type","").astype(str).str.title()

    mask = df["tx_direction"].eq("Inbound") & df["tx_type"].eq("Cash") & df["tx_base_amount"].notna()
    g = df.loc[mask, ["customer_id","tx_date_time","customer_account_balance","tx_base_amount"]].copy()
    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles], "Balance_after_tx":[np.nan]*len(percentiles)})
        return {"meta":{"n":0, "suggested_balance_p95": np.nan}, "percentiles": tbl}

    g["customer_account_balance"] = pd.to_numeric(g["customer_account_balance"], errors="coerce").fillna(0.0)
    g["tx_base_amount"] = g["tx_base_amount"].abs()
    s = (g["customer_account_balance"] + g["tx_base_amount"]).astype(float).dropna()

    stats = {p: (float(np.percentile(s, p)) if len(s) else np.nan) for p in percentiles}
    tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles], "Balance_after_tx":[stats[p] for p in percentiles]})
    rec = int(round(stats.get(95, np.nan))) if np.isfinite(stats.get(95, np.nan)) else np.nan
    return {"meta":{"n": int(len(s)), "suggested_balance_p95": rec}, "percentiles": tbl}
