from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd

from utils import load_tx_base, filter_subsubs

def simulate_p_hsumo(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
    collapse_runs: bool = False,
) -> pd.DataFrame:
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    g = df[
        df["tx_direction"].astype(str).str.title().eq("Outbound")
        & df["tx_type"].astype(str).str.title().eq("Cash")
        & df["customer_id"].notna()
        & df["tx_date_time"].notna()
        & df["tx_base_amount"].notna()
    ][["customer_id","tx_date_time","tx_base_amount"]].copy()

    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts=[]
    for cid, sub in g.groupby("customer_id", sort=False):
        daily = sub.set_index("tx_date_time")["tx_base_amount"].abs().resample("D").sum()
        if daily.empty:
            continue
        S30 = daily.rolling("30D", min_periods=1).sum()
        parts.append(pd.DataFrame({"customer_id": cid, "date": S30.index, "S30": S30.values}))

    M = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["customer_id","date","S30"])
    if M.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    count_from_ts = pd.Timestamp(count_from).normalize()
    M = M.loc[M["date"] >= count_from_ts].copy()

    def count_alerts(dfm, amount, collapse=False):
        m = dfm["S30"] > amount
        if not collapse:
            return int(dfm.loc[m, ["customer_id","date"]].drop_duplicates().shape[0])
        df2 = dfm.loc[m, ["customer_id","date"]].sort_values(["customer_id","date"])
        df2["prev"] = df2.groupby("customer_id")["date"].shift(1)
        df2["is_new"] = df2["prev"].isna() | ((df2["date"] - df2["prev"]).dt.days > 1)
        return int(df2.loc[df2["is_new"]].shape[0])

    out = []
    for name, pars in scenarios.items():
        A = float(pars.get("Amount", 0.0))
        cnt = count_alerts(M, A, collapse_runs)
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
