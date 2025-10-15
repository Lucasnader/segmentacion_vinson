from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (90, 95, 97, 99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_p_pct_bal(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[int] = DEFAULT_PCTS,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    df["customer_account_balance"] = pd.to_numeric(df["customer_account_balance"], errors="coerce")
    has_time = "tx_date_time" in df.columns
    if has_time: df["tx_date_time"] = pd.to_datetime(df["tx_date_time"], errors="coerce")

    cols = ["customer_id","customer_account_balance"] + (["tx_date_time"] if has_time else [])
    g = df[cols].dropna(subset=["customer_id","customer_account_balance"]).copy()
    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles], "Balance":[np.nan]*len(percentiles)})
        return {"meta":{"clients":0,"suggested_balance_p95":np.nan}, "percentiles": tbl}

    if has_time:
        g = g.sort_values(["customer_id","tx_date_time"]).groupby("customer_id", as_index=False).tail(1)
    else:
        g = g.groupby("customer_id", as_index=False).tail(1)

    s = g.loc[g["customer_account_balance"]>0, "customer_account_balance"].astype(float)
    if s.empty:
        tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles], "Balance":[np.nan]*len(percentiles)})
        return {"meta":{"clients":0,"suggested_balance_p95":np.nan}, "percentiles": tbl}

    stats = {p: float(np.percentile(s, p)) for p in percentiles}
    tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles], "Balance":[stats[p] for p in percentiles]})
    suggested = int(round(stats.get(95, np.nan))) if pd.notna(stats.get(95, np.nan)) else np.nan
    return {"meta":{"clients":int(s.shape[0]), "suggested_balance_p95": suggested}, "percentiles": tbl}
