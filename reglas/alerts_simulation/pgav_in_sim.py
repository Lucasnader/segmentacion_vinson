from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np

from utils import load_tx_base, filter_subsubs, restrict_counts_after

# ============================================================================
# 👉 EDITA AQUÍ: valor fijo para Number en PGAV-IN
#    - Si dejas un entero (ej: 800), se usará SIEMPRE ese valor para todas las
#      corridas/escenarios, sin importar lo que venga en el JSON.
#    - Si lo dejas como None, se intentará usar Number desde el escenario.
# ============================================================================
FIXED_PGAV_IN_NUMBER: int | None = 139
# ============================================================================


def _roll_7d_sum_count(sub: pd.DataFrame) -> pd.DataFrame:
    out = (
        sub.set_index("tx_date_time")["tx_base_amount"]
           .rolling("7D")
           .agg(["sum", "count"])
    )
    out.index = sub.index  # alinear con 'g'
    return out


def simulate_pgav_in(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
) -> pd.DataFrame:
    """
    PGAV-IN: cuenta transacciones que cumplen los umbrales del escenario.
    Condiciones:
      - Amount: tx_base_amount >= Amount   (si el escenario lo define)
      - Factor: factor_excl > Factor       (si el escenario lo define)
      - Number: prev_cnt7 >= Number        (si el escenario lo define)
        * Si FIXED_PGAV_IN_NUMBER no es None, se usa SIEMPRE ese valor para Number.

    Se cuentan solo las transacciones con tx_date_time >= count_from,
    pero el rolling de 7d usa todo el histórico previo para contexto.
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    GROUP_COL = "customer_sub_sub_type" if "customer_sub_sub_type" in df.columns else "customer_type"

    g = df[
        (df["tx_direction"].eq("Inbound"))
        & (df["tx_type"].eq("Cash"))
        & (df["tx_base_amount"].notna())
        & (df["tx_date_time"].notna())
    ].copy()

    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    g = g.sort_values([GROUP_COL, "tx_date_time"]).reset_index(drop=True)
    tmp = (
        g[[GROUP_COL, "tx_date_time", "tx_base_amount"]]
        .groupby(GROUP_COL, group_keys=False)
        .apply(_roll_7d_sum_count)
    )

    g["prev_sum7"] = tmp["sum"] - g["tx_base_amount"]
    g["prev_cnt7"] = tmp["count"] - 1
    g["peer_avg7_excl"] = np.where(g["prev_cnt7"] > 0, g["prev_sum7"] / g["prev_cnt7"], np.nan)
    g["factor"] = np.where(g["peer_avg7_excl"] > 0, g["tx_base_amount"] / g["peer_avg7_excl"], np.nan)

    # máscara de conteo por fecha (solo contamos desde count_from en adelante)
    countable = restrict_counts_after(g, "tx_date_time", count_from)

    def _mask_for_scenario(dfm: pd.DataFrame, pars: Dict[str, Any]) -> pd.Series:
        m = pd.Series(True, index=dfm.index)

        # Amount
        if "Amount" in pars and pd.notna(pars["Amount"]):
            m &= dfm["tx_base_amount"] >= float(pars["Amount"])

        # Factor
        if "Factor" in pars and pd.notna(pars["Factor"]):
            m &= dfm["factor"] > float(pars["Factor"])

        # Number (usar fijo si está definido; si no, usar del escenario si existe)
        if FIXED_PGAV_IN_NUMBER is not None:
            m &= dfm["prev_cnt7"] >= float(FIXED_PGAV_IN_NUMBER)
        else:
            if "Number" in pars and pd.notna(pars["Number"]):
                m &= dfm["prev_cnt7"] >= float(pars["Number"])

        return m.fillna(False)

    rows = []
    for name, pars in scenarios.items():
        m_alert = _mask_for_scenario(g, pars)
        # Unidad = transacciones que cumplen
        count = int(g.loc[m_alert & countable].shape[0])
        rows.append({"escenario": name, "alertas": count})

    return pd.DataFrame(rows)
