from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

# =================== Variables fijas editables ===================
P_PCTBAL_PERCENTAGE_FIXED: float = 95.0  # %
# ================================================================

def simulate_p_pctbal(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    P-%BAL — por transacción OUT:
      - balance_prev >= Balance
      - y (balance_prev - monto <= 0)  OR  (monto > balance_prev * Percentage%)
    Cuenta solo tx fecha >= count_from (balance leído de columna base).
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    g = df[df["tx_direction"].eq("Outbound") & df["tx_base_amount"].notna()].copy()
    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    bal = pd.to_numeric(g.get("customer_account_balance"), errors="coerce").fillna(-1e16)
    g["_bal_prev"] = bal

    countable = restrict_counts_after(g, "tx_date_time", count_from)

    rows = []
    for name, pars in scenarios.items():
        B = float(pars.get("Balance", np.inf))
        P = float(pars.get("Percentage", P_PCTBAL_PERCENTAGE_FIXED))

        pre = g["_bal_prev"] >= B
        cond1 = (g["_bal_prev"] - g["tx_base_amount"]) <= 0
        cond2 = ((g["_bal_prev"] - g["tx_base_amount"]) > 0) & (g["tx_base_amount"] > g["_bal_prev"] * (P/100.0))
        rows.append({"escenario": name, "alertas": int(((pre & (cond1 | cond2)) & countable).sum())})

    return pd.DataFrame(rows)
