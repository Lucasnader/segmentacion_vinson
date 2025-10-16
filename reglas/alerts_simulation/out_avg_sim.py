from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

# =================== Variables fijas editables ===================
OUT_AVG_NUMBER_FIXED: float = 9.0
# ================================================================

def simulate_out_avg(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    OUT>AVG — cuenta transacciones que cumplen:
      - Outbound & Cash
      - tx_base_amount >= Amount
      - prev_cnt > Number
      - factor >= Factor  (factor = monto / promedio previo excl.)
    Solo se cuentan tx con fecha >= count_from (el histórico previo sí se usa).
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    g = df[
        df["tx_direction"].eq("Outbound")
        & df["tx_type"].eq("Cash")
        & df["tx_date_time"].notna()
        & df["tx_base_amount"].notna()
        & df["customer_id"].notna()
    ][["customer_id","tx_date_time","tx_base_amount"]].copy()

    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    g = g.sort_values(["customer_id","tx_date_time"]).reset_index(drop=True)
    g["prev_avg"] = (
        g.groupby("customer_id")["tx_base_amount"]
         .transform(lambda s: s.shift().expanding().mean())
    )
    g["prev_cnt"] = g.groupby("customer_id").cumcount()
    g["factor"]   = np.where((g["prev_cnt"]>=1) & (g["prev_avg"]>0),
                             g["tx_base_amount"] / g["prev_avg"], np.nan)

    countable = restrict_counts_after(g, "tx_date_time", count_from)

    rows = []
    for name, pars in scenarios.items():
        A = float(pars.get("Amount", np.inf))
        F = float(pars.get("Factor", np.inf))
        N = float(pars.get("Number", OUT_AVG_NUMBER_FIXED))

        elig = (g["tx_base_amount"] >= A) & (g["prev_cnt"] > N) & np.isfinite(g["factor"])
        m_ok = elig & (g["factor"] >= F)
        rows.append({"escenario": name, "alertas": int((m_ok & countable).sum())})

    return pd.DataFrame(rows)
