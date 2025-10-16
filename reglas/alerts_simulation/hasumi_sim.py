from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np
from utils import load_tx_base, filter_subsubs, restrict_counts_after

def simulate_hasumi(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
) -> pd.DataFrame:
    """
    HASUMI (Inbound Cash): ventanas cliente–día
      S3 > Amount  AND  avg3_hist = (S180 - S3)/59 > 0  AND  S3 > Factor * avg3_hist
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    base = df[
        (df["tx_direction"].eq("Inbound")) &
        (df["tx_type"].eq("Cash")) &
        (df["customer_id"].notna()) &
        (df["tx_date_time"].notna()) &
        (df["tx_base_amount"].notna())
    ][["customer_id","tx_date_time","tx_base_amount"]].copy()

    if base.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts = []
    for cid, sub in base.groupby("customer_id", sort=False):
        daily = (sub.set_index("tx_date_time")["tx_base_amount"].abs().resample("D").sum())
        if daily.empty:
            continue
        s3 = daily.rolling("3D").sum()
        s180 = daily.rolling("180D").sum()
        parts.append(pd.DataFrame({"customer_id": cid, "date": daily.index, "S3": s3.values, "S180": s180.values}))
    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","date","S3","S180"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    M["avg3_hist"] = (M["S180"] - M["S3"]) / 59.0
    countable = restrict_counts_after(M, "date", count_from)

    rows = []
    for name, pars in scenarios.items():
        A = float(pars.get("Amount", np.inf))
        F = float(pars.get("Factor", np.inf))
        m = (M["S3"] > A) & (M["avg3_hist"] > 0) & (M["S3"] > F * M["avg3_hist"])
        cnt = int(M.loc[m & countable, ["customer_id","date"]].drop_duplicates().shape[0])
        rows.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(rows)
