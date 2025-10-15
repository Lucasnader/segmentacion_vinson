from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (0.85, 0.90, 0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_in_out_1_amount(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[float] = DEFAULT_PCTS,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    m = (df["tx_direction"].astype(str).str.title().eq("Outbound") &
         df["tx_type"].astype(str).str.title().eq("Cash") &
         df["tx_base_amount"].notna() &
         (df["tx_base_amount"] > 0))
    s = df.loc[m, "tx_base_amount"].astype(float)

    q = s.quantile(list(percentiles)) if len(s) else pd.Series(index=list(percentiles), dtype=float)
    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "Amount_CLP":[q.get(p, np.nan) for p in percentiles]})
    # p90 sugerido (igual que tu celda)
    suggested = int(round(q.get(0.90))) if pd.notna(q.get(0.90, np.nan)) else np.nan
    return {"meta":{"n":int(len(s)), "suggested_amount_p90": suggested}, "percentiles": tbl}
