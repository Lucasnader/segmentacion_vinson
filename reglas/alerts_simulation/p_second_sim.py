from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

P_SECOND_DAYS_DEFAULT = 7  # fijo

def simulate_p_second(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
    days_fixed: int | None = None,
) -> pd.DataFrame:
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    df["tx_date_time"] = pd.to_datetime(df.get("tx_date_time"), errors="coerce")
    df["customer_account_creation_date"] = pd.to_datetime(df.get("customer_account_creation_date"), errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")

    g = df[
        df["tx_date_time"].notna()
        & df["customer_account_creation_date"].notna()
        & df["customer_id"].notna()
        & df["tx_base_amount"].notna()
    ][["customer_id","tx_date_time","customer_account_creation_date","tx_base_amount"]].copy()

    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    g = g.sort_values(["customer_id","tx_date_time"])
    g["tx_order"] = g.groupby("customer_id").cumcount() + 1

    td = g["tx_date_time"] - g["customer_account_creation_date"]
    g["days_from_open"] = td.dt.total_seconds() / 86400.0

    countable = restrict_counts_after(g, "tx_date_time", count_from)

    out = []
    for name, pars in scenarios.items():
        A = float(pars.get("Amount", 0.0))
        D = int(pars.get("Days", days_fixed or P_SECOND_DAYS_DEFAULT))
        m = (
            g["tx_order"].eq(2)
            & (g["days_from_open"] >= 0)
            & (g["days_from_open"] <= D)
            & (g["tx_base_amount"] > A)
        )
        cnt = int((m & countable).sum())
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
