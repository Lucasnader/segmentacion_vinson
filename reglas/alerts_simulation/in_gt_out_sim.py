from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np
from utils import load_tx_base, filter_subsubs, restrict_counts_after

# Por petición: Low/High como variables fijas en el archivo (aplican a TODOS los escenarios)
FIXED_IN_GT_OUT_LOW_PCT: float = 80.0
FIXED_IN_GT_OUT_HIGH_PCT: float = 100.0

def simulate_in_gt_out(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],   # sólo se usará Amount_IN_30d del bundle
    count_from: str = "2025-03-01",
) -> pd.DataFrame:
    """
    IN>%OUT: ventanas cliente–día
      IN30 > Amount_IN_30d  AND  IN30 ∈ [Low% , High%] de OUT30
    Low/High se definen aquí como constantes fijas para toda la simulación.
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    base_mask = df["tx_date_time"].notna() & df["tx_base_amount"].notna() & df["customer_id"].notna()
    IN_  = df[base_mask & df["tx_direction"].eq("Inbound")][["customer_id","tx_date_time","tx_base_amount"]].copy()
    OUT_ = df[base_mask & df["tx_direction"].eq("Outbound")][["customer_id","tx_date_time","tx_base_amount"]].copy()

    if IN_.empty and OUT_.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts=[]
    # construir series IN/OUT diarias por cliente y sus rolling 30d
    for cid, sub_in in IN_.groupby("customer_id", sort=False):
        in_daily = (sub_in.set_index("tx_date_time")["tx_base_amount"].abs().resample("D").sum())
        sub_out = OUT_[OUT_["customer_id"].eq(cid)]
        if sub_out.empty:
            out_daily = pd.Series(0.0, index=in_daily.index)
        else:
            out_daily = (sub_out.set_index("tx_date_time")["tx_base_amount"].abs().resample("D").sum())

        start = min(in_daily.index.min(), out_daily.index.min())
        end   = max(in_daily.index.max(), out_daily.index.max())
        idx   = pd.date_range(start, end, freq="D")

        in_daily  = in_daily.reindex(idx,  fill_value=0.0)
        out_daily = out_daily.reindex(idx, fill_value=0.0)

        IN30  = in_daily.rolling("30D", min_periods=1).sum()
        OUT30 = out_daily.rolling("30D", min_periods=1).sum()
        parts.append(pd.DataFrame({"customer_id": cid, "date": idx, "IN30": IN30.values, "OUT30": OUT30.values}))

    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","date","IN30","OUT30"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    countable = restrict_counts_after(M, "date", count_from)
    L = FIXED_IN_GT_OUT_LOW_PCT / 100.0
    H = FIXED_IN_GT_OUT_HIGH_PCT / 100.0

    rows=[]
    for name, pars in scenarios.items():
        A = float(pars.get("Amount_IN_30d", np.inf))
        m = (
            (M["IN30"] > A) &
            (M["OUT30"] > 0) &
            (M["IN30"] >= M["OUT30"] * L) &
            (M["IN30"] <= M["OUT30"] * H)
        )
        cnt = int(M.loc[m & countable, ["customer_id","date"]].drop_duplicates().shape[0])
        rows.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(rows)
