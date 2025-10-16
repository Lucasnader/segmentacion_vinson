from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

from utils import load_tx_base, filter_subsubs

def simulate_rvt_in(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
) -> pd.DataFrame:
    """
    RVT-IN:
      IN & Cash, montos redondos (mod(tx_amount,1000)=0 con default 0.0001)
      Ventana 30d: count > Number AND sum(tx_base_amount) > Amount
      Unidad = ventanas (cliente, día).
      Se cuentan SOLO ventanas con fecha >= count_from, usando historia previa para el rolling.
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    # Casts defensivos
    df["tx_date_time"]   = pd.to_datetime(df.get("tx_date_time"), errors="coerce")
    df["tx_amount"]      = pd.to_numeric(df.get("tx_amount"), errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")
    df["tx_direction"]   = df.get("tx_direction", "").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type", "").astype(str).str.title()

    base = (
        df["tx_direction"].eq("Inbound")
        & df["tx_type"].eq("Cash")
        & df["customer_id"].notna()
        & df["tx_date_time"].notna()
        & df["tx_base_amount"].notna()
    )

    # Redonda en moneda original
    amt_orig = df["tx_amount"].fillna(0.0001)
    is_round = np.isfinite(amt_orig) & np.isclose(amt_orig % 1000, 0, atol=1e-9)

    g = df.loc[base & is_round, ["customer_id", "tx_date_time", "tx_base_amount"]].copy()
    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    g["amt"] = g["tx_base_amount"].abs().astype(float)

    parts = []
    for cid, sub in g.groupby("customer_id", sort=False):
        if sub.empty:
            continue
        sub = sub.sort_values("tx_date_time")
        daily_cnt = sub.set_index("tx_date_time").assign(x=1)["x"].resample("D").sum().fillna(0.0)
        daily_sum = sub.set_index("tx_date_time")["amt"].resample("D").sum().fillna(0.0)
        N30 = daily_cnt.rolling("30D").sum()
        S30 = daily_sum.rolling("30D").sum()
        parts.append(pd.DataFrame({"customer_id": cid, "date": N30.index, "N30": N30.values, "S30": S30.values}))

    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id", "date", "N30", "S30"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    cutoff = pd.to_datetime(count_from)
    M_after = M[M["date"] >= cutoff]

    out = []
    for name, v in scenarios.items():
        N = float(v.get("Number", np.inf))  # si falta en pct, no gatilla
        A = float(v.get("Amount", np.inf))
        m_ok = (M_after["N30"] > N) & (M_after["S30"] > A)
        cnt = int(M_after.loc[m_ok, ["customer_id", "date"]].drop_duplicates().shape[0])
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
