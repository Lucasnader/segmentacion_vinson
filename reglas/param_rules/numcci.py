from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

DEFAULT_NUM_QS = (0.50, 0.75, 0.90, 0.95, 0.97, 0.98, 0.99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_numcci(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    tx_type: str = "Cash",
    window_days: int = 14,
    percentiles: Iterable[float] = DEFAULT_NUM_QS,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string","counterparty_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"] = pd.to_datetime(df.get("tx_date_time"), errors="coerce")
    df["tx_direction"] = df.get("tx_direction","").astype(str).str.title()
    df["tx_type"]      = df.get("tx_type","").astype(str).str.title()
    df["counterparty_id"] = df.get("counterparty_id","").astype(str).str.strip()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    m = (df["tx_direction"].eq("Inbound") & df["tx_type"].eq(tx_type) &
         df["customer_id"].notna() & df["counterparty_id"].notna() &
         df["counterparty_id"].ne("NA") & df["tx_date_time"].notna())
    g = df.loc[m, ["customer_id","counterparty_id","tx_date_time"]].copy()

    parts, pairs = [], 0
    for (cid, cpid), sub in g.groupby(["customer_id","counterparty_id"], sort=False):
        daily_cnt = sub.set_index("tx_date_time").assign(x=1)["x"].resample("D").sum().astype(float)
        if daily_cnt.empty: continue
        C14 = daily_cnt.rolling(f"{window_days}D", min_periods=1).sum()
        parts.append(C14)
        pairs += 1

    if not parts:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                            "Number_raw":[np.nan]*len(percentiles),
                            "Number_ceil":[np.nan]*len(percentiles)})
        return {"meta":{"pairs":0,"windows":0}, "percentiles": tbl}

    s = pd.concat([ser.dropna().astype(float) for ser in parts], axis=0)
    q = s.quantile(list(percentiles)) if len(s) else pd.Series(index=list(percentiles), dtype=float)
    tbl = pd.DataFrame({
        "percentil":   [f"p{int(p*100)}" for p in percentiles],
        "Number_raw":  [q.get(p, np.nan) for p in percentiles],
        "Number_ceil": [int(math.ceil(q.get(p))) if pd.notna(q.get(p, np.nan)) else np.nan for p in percentiles],
    })
    return {"meta":{"pairs":pairs, "windows":int(len(s))}, "percentiles": tbl}
