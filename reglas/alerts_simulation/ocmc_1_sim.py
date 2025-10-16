from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

# =================== Variables fijas editables ===================
# Solo hay Number; el JSON trae "Counterparties_30d" por percentil,
# lo mapeamos a Number cuando venga del bundle, o puedes fijarlo acá
OCMC1_NUMBER_FALLBACK: float = 2.0
# ================================================================

def simulate_ocmc_1(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    OCMC_1 — transacciones (primerizas):
      - unique count(counterparty_id) por cliente en 30d > Number
      - y la tx es la primera con esa contraparte en 30d (cid, cpid)
      - Excluye cpid 'NA'
    Solo se cuentan tx con fecha >= count_from (ventanas 30d usan histórico).
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    df = df[
        df["tx_date_time"].notna()
        & df["customer_id"].notna()
        & df["counterparty_id"].astype(str).ne("NA")
    ][["customer_id","counterparty_id","tx_date_time"]].copy()

    if df.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    df = df.sort_values(["customer_id","tx_date_time"]).reset_index(drop=True)

    parts = []
    for cid, sub in df.groupby("customer_id", sort=False):
        day_idx = pd.date_range(sub["tx_date_time"].min().normalize(),
                                sub["tx_date_time"].max().normalize(), freq="D")
        cp_daily = (
            sub.groupby([pd.Grouper(key="tx_date_time", freq="D"), "counterparty_id"])
               .size()
               .unstack("counterparty_id", fill_value=0)
               .reindex(day_idx, fill_value=0)
        )
        uniq30 = (cp_daily.gt(0).rolling(window=30, min_periods=1).sum().gt(0).sum(axis=1))
        # marcar si es primera en 30d para ese par (cid, cpid)
        first_flags = []
        for _, row in sub.iterrows():
            d = row["tx_date_time"].normalize()
            mask_30 = (sub["tx_date_time"] >= d - pd.Timedelta(days=29)) & (sub["tx_date_time"] <= d)
            cnt_pair_30 = (sub.loc[mask_30, "counterparty_id"] == row["counterparty_id"]).sum()
            first_flags.append(int(cnt_pair_30 == 1))
        sub = sub.copy()
        sub["_is_first30"] = first_flags
        sub["_uniq30_at_day"] = sub["tx_date_time"].dt.normalize().map(uniq30).astype(float)
        parts.append(sub)

    G = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns.tolist()+["_is_first30","_uniq30_at_day"])
    if G.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    countable = restrict_counts_after(G, "tx_date_time", count_from)

    rows = []
    for name, pars in scenarios.items():
        N = float(pars.get("Number", pars.get("Counterparties_30d", OCMC1_NUMBER_FALLBACK)))
        m = (G["_is_first30"].eq(1) & (G["_uniq30_at_day"] > N))
        rows.append({"escenario": name, "alertas": int((m & countable).sum())})

    return pd.DataFrame(rows)
