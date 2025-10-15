from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (90, 95, 97, 99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_hvi(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    window_days: int = 30,
    percentiles: Iterable[int] = DEFAULT_PCTS,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()
    df["tx_date_time"] = pd.to_datetime(df["tx_date_time"], errors="coerce")

    mask = ((df["tx_direction"].astype(str).str.title() == "Inbound") &
            (df["tx_type"].astype(str).str.title() == "Cash") &
            df["tx_date_time"].notna() & df["customer_id"].notna())
    g = df.loc[mask, ["customer_id","tx_date_time"]].copy()

    def max_count_30d(group: pd.DataFrame) -> int:
        dates = np.sort(group["tx_date_time"].values); n = len(dates)
        j = 0; best = 0
        for i in range(n):
            end = dates[i] + np.timedelta64(window_days, "D")
            while j < n and dates[j] <= end: j += 1
            best = max(best, j - i)
        return best

    m = (g.sort_values(["customer_id","tx_date_time"])
           .groupby("customer_id", as_index=False)
           .apply(lambda sub: pd.Series({"max_30d": max_count_30d(sub)}))
           .reset_index(drop=True))
    s = pd.to_numeric(m["max_30d"], errors="coerce").dropna()

    stats = {p: (float(np.percentile(s, p)) if len(s) else np.nan) for p in percentiles}
    tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles],
                        "Number_max30d":[stats[p] for p in percentiles]})
    rec = int(math.ceil(stats.get(95, np.nan))) if np.isfinite(stats.get(95, np.nan)) else np.nan
    return {"meta":{"clients": int(m.shape[0]), "suggested_whole_number": rec}, "percentiles": tbl}
