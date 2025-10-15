from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (0.85, 0.90, 0.95, 0.97, 0.98, 0.99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_in_gt_out(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    window_days: int = 30,
    percentiles: Iterable[float] = DEFAULT_PCTS,
    filter_to_cash: bool = True,
    use_abs: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"]   = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    df["tx_direction"]   = df["tx_direction"].astype(str).str.title()
    if filter_to_cash and "tx_type" in df.columns:
        df["tx_type"] = df["tx_type"].astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    mask = df["tx_direction"].eq("Inbound") & df["tx_date_time"].notna() & df["tx_base_amount"].notna()
    if filter_to_cash and "tx_type" in df.columns:
        mask &= df["tx_type"].eq("Cash")
    g = df.loc[mask, ["customer_id","tx_date_time","tx_base_amount"]].copy()

    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles], "Amount_IN_30d":[np.nan]*len(percentiles)})
        return {"meta":{"clients":0,"windows":0}, "percentiles": tbl}

    g["amt"] = g["tx_base_amount"].abs() if use_abs else g["tx_base_amount"]
    parts = []
    for cid, sub in g.groupby("customer_id", sort=False):
        daily = sub.set_index("tx_date_time")["amt"].resample("D").sum()
        parts.append(daily.rolling(f"{window_days}D").sum().rename(cid))

    s = pd.concat([ser.dropna().astype(float) for ser in parts], axis=0) if parts else pd.Series(dtype=float)
    q = s.quantile(list(percentiles)) if len(s) else pd.Series(index=list(percentiles), dtype=float)

    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "Amount_IN_30d":[q.get(p, np.nan) for p in percentiles]})
    return {"meta":{"clients":int(g["customer_id"].nunique()), "windows":int(len(s))}, "percentiles": tbl}
