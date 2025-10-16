from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np

from utils import load_tx_base, filter_subsubs, restrict_counts_after

def simulate_p_lbal(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    bal = pd.to_numeric(df.get("customer_account_balance"), errors="coerce").fillna(0.0)
    df["_bal_prev"] = bal

    g = df[
        df["tx_direction"].astype(str).str.title().eq("Inbound")
        & df["tx_base_amount"].notna()
        & df["tx_date_time"].notna()
    ][["_bal_prev","tx_base_amount","tx_date_time"]].copy()

    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    countable = restrict_counts_after(g, "tx_date_time", count_from)

    out=[]
    for name, pars in scenarios.items():
        B = float(pars.get("Balance", 0.0))
        m = (g["_bal_prev"] + g["tx_base_amount"] > B)
        cnt = int((m & countable).sum())
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
