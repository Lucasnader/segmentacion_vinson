from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np
from utils import load_tx_base, filter_subsubs, restrict_counts_after

FIXED_HNR_OUT_NUMBER: float | None = None

def simulate_hnr_out(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
) -> pd.DataFrame:
    """
    HNR-OUT: ventanas cliente–día
      Outbound Cash, tx_base_amount > 1000, amount original “redondo”
      y CNT30 (rolling 30d) > Number
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    amt_orig = df["tx_amount"].fillna(0.0001)
    is_round = np.isfinite(amt_orig) & np.isclose(amt_orig % 1000, 0, atol=1e-9)

    base = df[
        (df["tx_direction"].eq("Outbound")) &
        (df["tx_type"].eq("Cash")) &
        (df["tx_base_amount"] > 1000) &
        (df["tx_date_time"].notna()) &
        (df["customer_id"].notna()) &
        is_round
    ][["customer_id","tx_date_time"]].copy()

    if base.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts=[]
    for cid, sub in base.groupby("customer_id", sort=False):
        daily_cnt = (sub.set_index("tx_date_time").assign(x=1)["x"].resample("D").sum().fillna(0.0))
        cnt30 = daily_cnt.rolling("30D").sum()
        parts.append(pd.DataFrame({"customer_id": cid, "date": cnt30.index, "CNT30": cnt30.values}))
    M = pd.concat(parts, ignore_index=True)

    countable = restrict_counts_after(M, "date", count_from)
    rows=[]
    for name, pars in scenarios.items():
        N = FIXED_HNR_OUT_NUMBER if FIXED_HNR_OUT_NUMBER is not None else float(pars.get("Number", np.inf))
        m = (M["CNT30"] > N)
        cnt = int(M.loc[m & countable, ["customer_id","date"]].drop_duplicates().shape[0])
        rows.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(rows)
