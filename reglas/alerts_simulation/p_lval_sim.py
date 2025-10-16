from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

def simulate_p_lval(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    df["_exp"] = pd.to_numeric(df.get("customer_expected_amount"), errors="coerce").fillna(0.0)

    g = df[df["tx_base_amount"].notna() & df["tx_date_time"].notna()][["_exp","tx_base_amount","tx_date_time"]].copy()
    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    base_elig = g["_exp"] != 0
    countable = restrict_counts_after(g, "tx_date_time", count_from)

    out=[]
    for name, pars in scenarios.items():
        F = float(pars.get("Factor", 0.0))
        m = base_elig & (g["tx_base_amount"] > g["_exp"] * F)
        cnt = int((m & countable).sum())
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
