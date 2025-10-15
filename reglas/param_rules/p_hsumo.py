from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (0.85, 0.90, 0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_hsumo(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[float] = DEFAULT_PCTS,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"]   = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    df["tx_direction"]   = df.get("tx_direction","").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type","").astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    g = df[df["tx_direction"].eq("Outbound") & df["tx_type"].eq("Cash")
           & df["tx_date_time"].notna() & df["tx_base_amount"].notna()][["customer_id","tx_date_time","tx_base_amount"]]

    max_rows = []
    for cid, sub in g.groupby("customer_id", sort=False):
        daily = sub.set_index("tx_date_time")["tx_base_amount"].abs().resample("D").sum()
        if daily.empty: continue
        S30 = daily.rolling("30D").sum()
        max_rows.append({"customer_id": cid, "S30_max": float(S30.max())})

    R = pd.DataFrame(max_rows)
    s = R["S30_max"].astype(float) if not R.empty else pd.Series(dtype=float)
    q = s.quantile(list(percentiles)) if len(s) else pd.Series(index=list(percentiles), dtype=float)
    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "Amount_30d_max_per_customer_CLP":[q.get(p, np.nan) for p in percentiles]})
    return {"meta":{"clients": int(s.shape[0])}, "percentiles": tbl}
