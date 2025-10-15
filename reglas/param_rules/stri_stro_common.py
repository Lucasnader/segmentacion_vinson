from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Dict, Any, Optional

PCTS_DEF = (0.85, 0.90, 0.95, 0.97, 0.99)

def _counts_7d(dates: pd.Series) -> list[int]:
    arr = np.array(dates.values, dtype="datetime64[ns]")
    n, j, out = len(arr), 0, []
    for i in range(n):
        end = arr[i] + np.timedelta64(7, "D")
        while j < n and arr[j] <= end: j += 1
        out.append(j - i)
    return out

def _run_str(
    path: str,
    *,
    direction: str,      # "Inbound" | "Outbound"
    currency: str,       # "CLP" | "EUR" | "USD"
    subsubsegments: Optional[Iterable[str]] = None,
    percentiles: Iterable[float] = PCTS_DEF,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"] = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_amount"]    = pd.to_numeric(df["tx_amount"], errors="coerce")

    if subsubsegments is not None and "customer_sub_sub_type" in df.columns:
        targets = set([subsubsegments] if isinstance(subsubsegments, str) else map(str, subsubsegments))
        df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    mask = (
        df["tx_direction"].astype(str).str.title().eq(direction) &
        df["tx_type"].astype(str).str.title().eq("Cash") &
        df["tx_currency"].astype(str).str.upper().eq(currency) &
        df["tx_date_time"].notna() & df["tx_amount"].notna() &
        df["tx_amount"].abs().between(9950, 10000)
    )
    g = df.loc[mask, ["customer_id","tx_date_time"]].sort_values(["customer_id","tx_date_time"])
    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                            "X_candidatos":[np.nan]*len(percentiles)})
        return {"meta":{"windows":0, "clients":0}, "percentiles": tbl}

    counts = []
    for _, sub in g.groupby("customer_id", sort=False):
        if len(sub): counts.extend(_counts_7d(sub["tx_date_time"]))
    s = pd.Series(counts, dtype=float)
    q = s.quantile(percentiles) if len(s) else pd.Series(index=percentiles, dtype=float)
    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "X_candidatos":[q.get(p, np.nan) for p in percentiles]})
    return {"meta":{"windows": int(len(s)), "clients": int(g["customer_id"].nunique())}, "percentiles": tbl}
