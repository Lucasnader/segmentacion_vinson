from __future__ import annotations
import pandas as pd, numpy as np, math
from collections import Counter, deque
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (0.50, 0.75, 0.90, 0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_ocmc_1(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    window_days: int = 30,
    percentiles: Iterable[float] = DEFAULT_PCTS,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string","counterparty_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"]  = pd.to_datetime(df["tx_date_time"], errors="coerce")

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    mask = (df["tx_date_time"].notna() & df["customer_id"].notna() &
            df["counterparty_id"].notna() & (df["counterparty_id"].astype(str).str.upper().str.strip() != "NA"))
    g = df.loc[mask, ["customer_id","tx_date_time","counterparty_id"]].copy()

    counts = []
    for _, sub in g.groupby("customer_id", sort=False):
        sub = sub.sort_values("tx_date_time")
        times = sub["tx_date_time"].to_numpy()
        cps   = sub["counterparty_id"].astype(str).to_numpy()

        win = deque()
        freq = Counter()
        distinct = 0
        for t, cp in zip(times, cps):
            win.append((t, cp))
            prev = freq[cp]; freq[cp] += 1
            if prev == 0: distinct += 1

            cutoff = t - np.timedelta64(window_days, "D")
            while win and win[0][0] < cutoff:
                t0, cp0 = win.popleft()
                freq[cp0] -= 1
                if freq[cp0] == 0: distinct -= 1

            counts.append(distinct)

    s = pd.Series(counts, dtype=float)
    if s.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                            "Counterparties_30d":[np.nan]*len(percentiles),
                            "Ceil":[np.nan]*len(percentiles)})
        return {"meta":{"windows":0}, "percentiles": tbl}

    q = s.quantile(list(percentiles))
    tbl = pd.DataFrame({
        "percentil": [f"p{int(p*100)}" for p in percentiles],
        "Counterparties_30d": [q.get(p, np.nan) for p in percentiles],
        "Ceil": [int(math.ceil(q.get(p))) if pd.notna(q.get(p, np.nan)) else np.nan for p in percentiles]
    })
    return {"meta":{"windows":int(len(s))}, "percentiles": tbl}
