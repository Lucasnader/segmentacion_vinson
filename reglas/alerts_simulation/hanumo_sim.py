from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np
from utils import load_tx_base, filter_subsubs, restrict_counts_after

FIXED_HANUMO_NUMBER: float | None = None
FIXED_HANUMO_FACTOR: float | None = None

def simulate_hanumo(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    HANUMO: ventanas cliente–día (Outbound Cash)
      S3N >= Number  AND  AVG177N > 0  AND  (S3N / AVG177N) > Factor
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    base = df[
        (df["tx_direction"].eq("Outbound")) &
        (df["tx_type"].eq("Cash")) &
        (df["customer_id"].notna()) &
        (df["tx_date_time"].notna())
    ][["customer_id","tx_date_time"]].copy()

    if base.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts = []
    for cid, sub in base.groupby("customer_id", sort=False):
        daily_n = (sub.set_index("tx_date_time")
                      .assign(x=1)["x"]
                      .resample("D").sum()
                      .astype(float))
        if daily_n.empty:
            continue
        s3n = daily_n.rolling("3D").sum()
        avg177n = s3n.shift(3).rolling("177D", min_periods=1).mean()
        parts.append(pd.DataFrame({"customer_id": cid, "date": daily_n.index, "S3N": s3n.values, "AVG177N": avg177n.values}))
    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","date","S3N","AVG177N"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    M["Factor"] = np.where(M["AVG177N"] > 0, M["S3N"] / M["AVG177N"], np.nan)
    countable = restrict_counts_after(M, "date", count_from)

    rows = []
    for name, pars in scenarios.items():
        Number = FIXED_HANUMO_NUMBER if FIXED_HANUMO_NUMBER is not None else float(pars.get("Number", np.nan))
        Factor = FIXED_HANUMO_FACTOR if FIXED_HANUMO_FACTOR is not None else float(pars.get("Factor", np.nan))
        m = (
            (M["S3N"] >= Number) &
            (M["AVG177N"] > 0) &
            (M["Factor"] > Factor)
        ).fillna(False)
        cnt = int(M.loc[m & countable, ["customer_id","date"]].drop_duplicates().shape[0])
        rows.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(rows)
