# out_avg.py
from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_QS       = (0.85, 0.90, 0.95, 0.97, 0.99)
DEFAULT_MIN_PREV = 1
DEFAULT_MIN_AMT  = 0.0

def _as_list(x: Union[str, Iterable[str]]) -> list[str]:
    if isinstance(x, str):
        return [x]
    return list(map(str, x))

def run_parameters_out_avg(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[float] = DEFAULT_QS,
    min_prev_tx: int = DEFAULT_MIN_PREV,
    min_amount: float = DEFAULT_MIN_AMT,
    verbose: bool = False,
) -> Dict[str, Any]:
    tx = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    req = {"customer_id","tx_date_time","tx_base_amount","tx_direction","tx_type","customer_sub_sub_type"}
    miss = [c for c in req if c not in tx.columns]
    if miss:
        raise KeyError(f"Faltan columnas para OUT>AVG: {miss}")

    tx["tx_date_time"]   = pd.to_datetime(tx["tx_date_time"], errors="coerce")
    tx["tx_base_amount"] = pd.to_numeric(tx["tx_base_amount"], errors="coerce")

    targets = set(_as_list(subsubsegments))
    tx = tx[tx["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    mask_out = (
        tx["tx_direction"].astype(str).str.upper().str.startswith("OUT") &
        tx["tx_type"].astype(str).str.upper().str.startswith("CASH") &
        tx["tx_date_time"].notna() &
        tx["tx_base_amount"].notna()
    )
    g = tx.loc[mask_out, ["customer_id","tx_date_time","tx_base_amount"]].copy()
    g = g[g["tx_base_amount"] > min_amount]
    g = g.sort_values(["customer_id","tx_date_time"])

    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(q*100)}" for q in percentiles],
                            "Amount":[np.nan]*len(percentiles),
                            "Factor":[np.nan]*len(percentiles)})
        return {"meta":{"n_amount":0,"n_factor":0,"min_prev_tx":min_prev_tx,"min_amount":min_amount},
                "percentiles": tbl}

    g["prev_avg"] = g.groupby("customer_id")["tx_base_amount"].transform(lambda s: s.shift().expanding().mean())
    g["prev_cnt"] = g.groupby("customer_id").cumcount()
    elig = (g["prev_cnt"] >= min_prev_tx) & (g["prev_avg"] > 0)
    g["factor"] = np.where(elig, g["tx_base_amount"] / g["prev_avg"], np.nan)

    Q = list(percentiles)
    amount_s = g["tx_base_amount"].astype(float).dropna()
    factor_s = pd.to_numeric(g["factor"], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()

    amount_q = amount_s.quantile(Q) if len(amount_s) else pd.Series(index=Q, dtype=float)
    factor_q = factor_s.quantile(Q) if len(factor_s) else pd.Series(index=Q, dtype=float)

    tbl = pd.DataFrame({
        "percentil": [f"p{int(q*100)}" for q in Q],
        "Amount":    [amount_q.get(q, np.nan) for q in Q],
        "Factor":    [factor_q.get(q, np.nan) for q in Q],
    })

    return {"meta":{"n_amount":int(len(amount_s)), "n_factor":int(len(factor_s)),
                    "min_prev_tx":min_prev_tx, "min_amount":min_amount},
            "percentiles": tbl}
