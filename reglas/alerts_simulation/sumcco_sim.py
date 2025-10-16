from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd

from utils import load_tx_base, filter_subsubs

def simulate_sumcco(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
    tx_type_fixed: str = "Cash",
) -> pd.DataFrame:
    """
    SUMCCO:
      OUT & {type}, por (customer, counterparty) ventana 14d:
      sum(tx_base_amount) > Amount.
      Unidad = (customer_id, counterparty_id, día). Cuenta >= count_from.
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    df["tx_date_time"]   = pd.to_datetime(df.get("tx_date_time"), errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")
    df["tx_direction"]   = df.get("tx_direction", "").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type", "").astype(str).str.title()
    df["counterparty_id"] = df.get("counterparty_id", "").astype(str).str.strip()

    m = (
        df["tx_direction"].eq("Outbound")
        & df["tx_type"].eq(tx_type_fixed)
        & df["customer_id"].notna()
        & df["counterparty_id"].notna()
        & df["counterparty_id"].ne("NA")
        & df["tx_date_time"].notna()
        & df["tx_base_amount"].notna()
    )
    g = df.loc[m, ["customer_id", "counterparty_id", "tx_date_time", "tx_base_amount"]].copy()
    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    g["amt"] = g["tx_base_amount"].abs().astype(float)

    parts = []
    for (cid, cpid), sub in g.groupby(["customer_id", "counterparty_id"], sort=False):
        sub = sub.sort_values("tx_date_time")
        daily_sum = sub.set_index("tx_date_time")["amt"].resample("D").sum().astype(float)
        if daily_sum.empty:
            continue
        S14 = daily_sum.rolling("14D").sum()
        parts.append(pd.DataFrame({"customer_id": cid, "counterparty_id": cpid, "date": S14.index, "S14": S14.values}))

    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","counterparty_id","date","S14"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    cutoff = pd.to_datetime(count_from)
    M_after = M[M["date"] >= cutoff]

    out = []
    for name, v in scenarios.items():
        A = float(v.get("Amount", float("inf")))
        m_ok = (M_after["S14"] > A)
        cnt = int(M_after.loc[m_ok, ["customer_id", "counterparty_id", "date"]].drop_duplicates().shape[0])
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
