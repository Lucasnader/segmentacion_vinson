# HASUMO — OUT: Amount (S3) y Factor (S3/AVG177)
from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

DEFAULT_AMOUNT_QS = (0.85, 0.90, 0.95, 0.97, 0.99)
DEFAULT_FACTOR_QS = (0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_hasumo(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    amount_qs: Iterable[float] = DEFAULT_AMOUNT_QS,
    factor_qs: Iterable[float] = DEFAULT_FACTOR_QS,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    req = {"customer_id","tx_date_time","tx_base_amount","tx_direction","tx_type","customer_sub_sub_type"}
    miss = [c for c in req if c not in df.columns]
    if miss: raise KeyError(f"Faltan columnas HASUMO: {miss}")

    df["tx_date_time"]   = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    df["tx_direction"]   = df["tx_direction"].astype(str).str.title()
    df["tx_type"]        = df["tx_type"].astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    mask = df["tx_direction"].eq("Outbound") & df["tx_type"].eq("Cash") & df["tx_date_time"].notna() & df["tx_base_amount"].notna()
    g = df.loc[mask, ["customer_id","tx_date_time","tx_base_amount"]].copy()

    amt_points, fac_points = [], []
    for _, sub in g.groupby("customer_id", sort=False):
        sub = sub.sort_values("tx_date_time")
        daily = sub.set_index("tx_date_time")["tx_base_amount"].abs().resample("D").sum()
        if daily.empty: continue
        S3 = daily.rolling("3D").sum()
        AVG177 = S3.shift(3).rolling("177D", min_periods=1).mean()

        ok_amt = (S3 > 0) & S3.notna()
        if ok_amt.any(): amt_points.append(S3.loc[ok_amt])

        ok_fac = ok_amt & (AVG177 > 0) & AVG177.notna()
        if ok_fac.any():
            fac_points.append((S3.loc[ok_fac] / AVG177.loc[ok_fac]).replace([np.inf,-np.inf], np.nan).dropna())

    S_amt = pd.concat(amt_points) if amt_points else pd.Series(dtype=float)
    S_fac = pd.concat(fac_points) if fac_points else pd.Series(dtype=float)

    amount_q = S_amt.quantile(list(amount_qs)) if len(S_amt) else pd.Series(index=list(amount_qs), dtype=float)
    factor_q = S_fac.quantile(list(factor_qs)) if len(S_fac) else pd.Series(index=list(factor_qs), dtype=float)

    idx = sorted(set(list(amount_qs)) | set(list(factor_qs)))
    out = pd.DataFrame({
        "percentil":   [f"p{int(q*100)}" for q in idx],
        "Amount_S3":   [amount_q.get(q, np.nan) for q in idx],
        "Factor_raw":  [factor_q.get(q, np.nan) for q in idx],
        "Factor_rec":  [int(math.ceil(factor_q.get(q))) if pd.notna(factor_q.get(q, np.nan)) and np.isfinite(factor_q.get(q)) else np.nan for q in idx],
    })
    return {"meta":{"n_amount":int(len(S_amt)), "n_factor":int(len(S_fac))}, "percentiles": out}
