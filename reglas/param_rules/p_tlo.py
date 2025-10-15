from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (90, 95, 97, 99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_tlo(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[int] = DEFAULT_PCTS,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    mask = ((df["tx_direction"].astype(str).str.title() == "Outbound") &
            (df["tx_type"].astype(str).str.title() == "Cash") &
            (df["tx_base_amount"] > 0))
    s = df.loc[mask, "tx_base_amount"].astype(float).dropna()

    stats = {p: (float(np.percentile(s, p)) if len(s) else np.nan) for p in percentiles}
    tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles],
                        "Amount_CLP":[stats[p] for p in percentiles]})
    rec = int(round(stats.get(95, np.nan))) if np.isfinite(stats.get(95, np.nan)) else np.nan
    return {"meta":{"n": int(len(s)), "suggested_amount_p95": rec}, "percentiles": tbl}
