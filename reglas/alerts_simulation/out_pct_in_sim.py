from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np

from utils import load_tx_base, filter_subsubs

# parámetros “globales” fijos para la regla (los puedes editar aquí)
OUT_PCT_IN_LOW_DEFAULT  = 90.0
OUT_PCT_IN_HIGH_DEFAULT = 110.0

def simulate_out_pct_in(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    OUT>%IN — Ventanas cliente–día:
      OUT30 > Amount_OUT_30d
      y OUT30 ∈ [Low%, High%] de IN30
    Se cuentan solo ventanas con date >= count_from, pero se usa todo el histórico
    para los rollings de 30 días.
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    # base
    base_mask = (
        df["tx_date_time"].notna() &
        df["tx_base_amount"].notna() &
        df["customer_id"].notna()
    )
    OUT_ = df[base_mask & df["tx_direction"].astype(str).str.title().eq("Outbound")][
        ["customer_id", "tx_date_time", "tx_base_amount"]
    ].copy()
    IN_  = df[base_mask & df["tx_direction"].astype(str).str.title().eq("Inbound")][
        ["customer_id", "tx_date_time", "tx_base_amount"]
    ].copy()

    rows = []

    if OUT_.empty:
        # si no hay OUT, todos los escenarios dan 0
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts = []
    for cid, sub_out in OUT_.groupby("customer_id", sort=False):
        out_daily = (
            sub_out.set_index("tx_date_time")["tx_base_amount"]
                   .abs()
                   .resample("D").sum()
        )

        sub_in = IN_[IN_["customer_id"].eq(cid)]
        if sub_in.empty:
            in_daily = pd.Series(dtype=float)  # vacío; lo rellenamos luego
        else:
            in_daily = (
                sub_in.set_index("tx_date_time")["tx_base_amount"]
                      .abs()
                      .resample("D").sum()
            )

        # Rango de fechas robusto a vacíos
        mins, maxs = [], []
        if not out_daily.index.empty:
            mins.append(out_daily.index.min())
            maxs.append(out_daily.index.max())
        if not in_daily.index.empty:
            mins.append(in_daily.index.min())
            maxs.append(in_daily.index.max())

        if not mins:
            # cliente sin series válidas
            continue

        idx = pd.date_range(min(mins), max(maxs), freq="D")

        out_daily = out_daily.reindex(idx, fill_value=0.0)
        # si in_daily estaba vacío, quedará todo NaN; rellenamos a 0
        in_daily  = in_daily.reindex(idx).fillna(0.0)

        OUT30 = out_daily.rolling("30D", min_periods=1).sum()
        IN30  = in_daily.rolling("30D", min_periods=1).sum()

        parts.append(pd.DataFrame({
            "customer_id": cid,
            "date": idx,
            "OUT30": OUT30.values,
            "IN30": IN30.values
        }))

    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","date","OUT30","IN30"])

    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    # máscara de conteo (desde marzo inclusive)
    count_from_ts = pd.Timestamp(count_from).normalize()
    countable = (M["date"] >= count_from_ts)

    # valores fijos (si no vienen en el escenario)
    def low_high_for(s: Dict[str, Any]) -> tuple[float, float]:
        L = float(s.get("Low", OUT_PCT_IN_LOW_DEFAULT))   # %
        H = float(s.get("High", OUT_PCT_IN_HIGH_DEFAULT)) # %
        return L, H

    out_rows = []
    for name, pars in scenarios.items():
        A = float(pars.get("Amount_OUT_30d", 0.0))
        L, H = low_high_for(pars)

        m = (
            (M["OUT30"] > A) &
            (M["IN30"] > 0) &
            (M["OUT30"] >= M["IN30"] * (L/100.0)) &
            (M["OUT30"] <= M["IN30"] * (H/100.0))
        )

        # contar solo ventanas desde count_from
        count = int(M.loc[m & countable, ["customer_id", "date"]].drop_duplicates().shape[0])
        out_rows.append({"escenario": name, "alertas": count})

    return pd.DataFrame(out_rows)
