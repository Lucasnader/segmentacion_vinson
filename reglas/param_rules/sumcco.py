from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

PCTS_DEF = (0.90, 0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_sumcco(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    tx_type: str = "Cash",
    percentiles: Iterable[float] = PCTS_DEF,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string","counterparty_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"]   = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    df["tx_direction"]   = df["tx_direction"].astype(str).str.title()
    df["tx_type"]        = df["tx_type"].astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    m = (df["tx_direction"].eq("Outbound") & df["tx_type"].eq(tx_type.title()) &
         df["customer_id"].notna() & df["counterparty_id"].notna() &
         df["tx_date_time"].notna() & df["tx_base_amount"].notna())
    g = df.loc[m, ["customer_id","counterparty_id","tx_date_time","tx_base_amount"]].copy()
    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles], "Amount_CLP":[np.nan]*len(percentiles)})
        return {"meta":{"pairs":0}, "percentiles": tbl}

    g["amt"] = g["tx_base_amount"].abs().astype(float)
    out_max = []
    for (cid, cpid), sub in g.groupby(["customer_id","counterparty_id"], sort=False):
        sub = sub.sort_values("tx_date_time")
        ts = sub["tx_date_time"].values; am = sub["amt"].values
        j = 0; pref = np.concatenate([[0.0], np.cumsum(am)]); best_s = 0.0
        delta = np.timedelta64(14, "D")
        for i in range(ts.size):
            end = ts[i] + delta
            while j < ts.size and ts[j] <= end: j += 1
            s = pref[j] - pref[i]
            if s > best_s: best_s = s
        out_max.append(best_s)

    s = pd.Series(out_max, dtype=float)
    q = s.quantile(percentiles) if len(s) else pd.Series(index=percentiles, dtype=float)
    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "Amount_CLP":[q.get(p, np.nan) for p in percentiles]})
    return {"meta":{"pairs": int(len(out_max))}, "percentiles": tbl}
