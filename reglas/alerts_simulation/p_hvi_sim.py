from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd

from utils import load_tx_base, filter_subsubs

def simulate_p_hvi(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    g = df[
        df["tx_direction"].astype(str).str.title().eq("Inbound")
        & df["tx_type"].astype(str).str.title().eq("Cash")
        & df["customer_id"].notna()
        & df["tx_date_time"].notna()
    ][["customer_id","tx_date_time"]].copy()

    if g.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    parts=[]
    for cid, sub in g.groupby("customer_id", sort=False):
        daily = sub.set_index("tx_date_time").assign(x=1)["x"].resample("D").sum().fillna(0.0)
        C30 = daily.rolling("30D", min_periods=1).sum()
        parts.append(pd.DataFrame({"customer_id": cid, "date": C30.index, "C30": C30.values}))

    M = pd.concat(parts, ignore_index=True)
    count_from_ts = pd.Timestamp(count_from).normalize()
    M = M.loc[M["date"] >= count_from_ts].copy()

    out=[]
    for name, pars in scenarios.items():
        N = float(pars.get("Number", 0))
        cnt = int(M.loc[M["C30"] > N, ["customer_id","date"]].drop_duplicates().shape[0])
        out.append({"escenario": name, "alertas": cnt})
    return pd.DataFrame(out)
