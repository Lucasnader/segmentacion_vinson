# HANUMO — OUT: Number (S3N) y Factor (S3N/AVG177N)
from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

DEFAULT_NUMBER_QS = (0.85, 0.90, 0.95, 0.97, 0.99)
DEFAULT_FACTOR_QS = (0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x,str) else list(map(str,x))

def run_parameters_hanumo(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    number_qs: Iterable[float] = DEFAULT_NUMBER_QS,
    factor_qs: Iterable[float] = DEFAULT_FACTOR_QS,
    verbose: bool = False,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    req = {"customer_id","tx_date_time","tx_direction","tx_type","customer_sub_sub_type"}
    miss = [c for c in req if c not in df.columns]
    if miss: raise KeyError(f"Faltan columnas HANUMO: {miss}")

    df["tx_date_time"] = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_direction"] = df["tx_direction"].astype(str).str.title()
    df["tx_type"]      = df["tx_type"].astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    mask = df["tx_direction"].eq("Outbound") & df["tx_type"].eq("Cash") & df["tx_date_time"].notna()
    g = df.loc[mask, ["customer_id","tx_date_time"]].copy()

    num_points, fac_points = [], []
    for _, sub in g.groupby("customer_id", sort=False):
        sub = sub.sort_values("tx_date_time")
        daily_cnt = sub.set_index("tx_date_time").assign(x=1)["x"].resample("D").sum().fillna(0)
        if daily_cnt.empty: continue
        S3N = daily_cnt.rolling("3D").sum()
        AVG177N = S3N.shift(3).rolling("177D", min_periods=1).mean()

        ok_num = S3N.notna() & (S3N > 0)
        if ok_num.any(): num_points.append(S3N.loc[ok_num])

        ok_fac = ok_num & AVG177N.notna() & (AVG177N > 0)
        if ok_fac.any():
            fac_points.append((S3N.loc[ok_fac] / AVG177N.loc[ok_fac]).replace([np.inf,-np.inf], np.nan).dropna())

    S_num = pd.concat(num_points) if num_points else pd.Series(dtype=float)
    S_fac = pd.concat(fac_points) if fac_points else pd.Series(dtype=float)

    num_q = S_num.quantile(list(number_qs)) if len(S_num) else pd.Series(index=list(number_qs), dtype=float)
    fac_q = S_fac.quantile(list(factor_qs)) if len(S_fac) else pd.Series(index=list(factor_qs), dtype=float)

    idx = sorted(set(list(number_qs)) | set(list(factor_qs)))
    out = pd.DataFrame({
        "percentil":        [f"p{int(q*100)}" for q in idx],
        "Number_raw_S3N":   [num_q.get(q, np.nan) for q in idx],
        "Number_ceiled":    [int(math.ceil(num_q.get(q))) if pd.notna(num_q.get(q, np.nan)) else np.nan for q in idx],
        "Factor_raw":       [fac_q.get(q, np.nan) for q in idx],
        "Factor_ceiled":    [int(math.ceil(fac_q.get(q))) if pd.notna(fac_q.get(q, np.nan)) and np.isfinite(fac_q.get(q)) else np.nan for q in idx],
    })
    return {"meta":{"n_number":int(len(S_num)), "n_factor":int(len(S_fac))}, "percentiles": out}
