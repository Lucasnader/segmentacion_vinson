from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

# =================== Variables fijas editables ===================
NUMCCI_TYPE_FIXED: str = "Cash"
WINDOW_DAYS: int = 14
# ================================================================

def simulate_numcci(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    NUMCCI — ventanas (customer_id, counterparty_id, día):
      - Inbound & tx_type == TYPE
      - count 14d por (cid, cpid, direction, type) > Number
    Cuenta solo ventanas con fecha >= count_from (rolling usa histórico).
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    df["counterparty_id"] = df["counterparty_id"].astype(str).str.strip()

    m = (
        df["tx_direction"].eq("Inbound")
        & df["tx_type"].eq(NUMCCI_TYPE_FIXED)
        & df["customer_id"].notna()
        & df["counterparty_id"].notna()
        & df["counterparty_id"].ne("NA")
        & df["tx_date_time"].notna()
    )
    g = df.loc[m, ["customer_id","counterparty_id","tx_date_time"]].copy()
    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts=[]
    for (cid, cpid), sub in g.groupby(["customer_id","counterparty_id"], sort=False):
        daily_cnt = (sub.set_index("tx_date_time")
                        .assign(x=1)["x"]
                        .resample("D").sum()
                        .astype(float))
        if daily_cnt.empty:
            continue
        C14 = daily_cnt.rolling(f"{WINDOW_DAYS}D", min_periods=1).sum()
        parts.append(pd.DataFrame({"customer_id": cid, "counterparty_id": cpid, "date": C14.index, "C14": C14.values}))
    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","counterparty_id","date","C14"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    countable = restrict_counts_after(M, "date", count_from)

    rows=[]
    for name, pars in scenarios.items():
        # JSON trae Number_ceil/Number_raw — preferimos "Number_ceil" y lo mapeamos a Number
        N = float(pars.get("Number", pars.get("Number_ceil", pars.get("Number_raw", np.inf))))
        m_ok = (M["C14"] > N)
        rows.append({"escenario": name, "alertas": int((m_ok & countable).sum())})

    return pd.DataFrame(rows)
