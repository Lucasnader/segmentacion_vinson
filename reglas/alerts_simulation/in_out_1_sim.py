from __future__ import annotations
from typing import Dict, Any, Iterable
import numpy as np
import pandas as pd

from utils import load_tx_base, filter_subsubs, restrict_counts_after

# =================== Variables fijas editables ===================
IN_OUT_1_NUMBER_FIXED: float = 2.0     # IN_cnt_14d > Number
IN_OUT_1_PERCENTAGE_FIXED: float = 80  # OUT >= (P%)*IN_sum_14d
WINDOW_DAYS: int = 14
# ================================================================

def simulate_in_out_1(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-03-01",
) -> pd.DataFrame:
    """
    IN-OUT-1 — por transacción OUT:
      - Outbound & Cash
      - amount > Amount
      - IN_cnt_14d > Number
      - amount >= (Percentage/100)*IN_sum_14d
    Cuenta solo OUT con fecha >= count_from (IN 14d usa histórico).
    """
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    base = df[df["tx_date_time"].notna() & df["tx_base_amount"].notna() & df["customer_id"].notna()].copy()
    IN_  = base[base["tx_direction"].eq("Inbound")  & base["tx_type"].eq("Cash")][["customer_id","tx_date_time","tx_base_amount"]]
    OUT_ = base[base["tx_direction"].eq("Outbound") & base["tx_type"].eq("Cash")][["customer_id","tx_date_time","tx_base_amount"]]

    if OUT_.empty:
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    # Precalcular IN diarias por cliente
    in_daily = {}
    for cid, sub in IN_.groupby("customer_id", sort=False):
        daily = (sub.set_index("tx_date_time")["tx_base_amount"].abs()
                    .resample("D").agg(["sum","count"])
                    .rename(columns={"sum":"IN_sum","count":"IN_cnt"}))
        in_daily[cid] = daily

    rows = []
    # máscara de conteo por fecha para OUT
    OUT_ = OUT_.sort_values(["customer_id","tx_date_time"]).reset_index(drop=True)
    countable = restrict_counts_after(OUT_, "tx_date_time", count_from)

    for name, pars in scenarios.items():
        A = float(pars.get("Amount", np.inf))
        N = float(pars.get("Number", IN_OUT_1_NUMBER_FIXED))
        P = float(pars.get("Percentage", IN_OUT_1_PERCENTAGE_FIXED))

        ok_flags = []
        for cid, sub_out in OUT_.groupby("customer_id", sort=False):
            ind = in_daily.get(cid)
            if ind is None or ind.empty:
                continue
            IN14_sum = ind["IN_sum"].rolling(f"{WINDOW_DAYS}D").sum()
            IN14_cnt = ind["IN_cnt"].rolling(f"{WINDOW_DAYS}D").sum()

            for t, amt in zip(sub_out["tx_date_time"], sub_out["tx_base_amount"].abs()):
                d = pd.Timestamp(t.normalize())
                if d in IN14_sum.index:
                    # valores hasta ese día
                    val_sum = float(IN14_sum.loc[:d].iloc[-1]) if not IN14_sum.loc[:d].empty else 0.0
                    val_cnt = float(IN14_cnt.loc[:d].iloc[-1]) if not IN14_cnt.loc[:d].empty else 0.0
                else:
                    val_sum = 0.0; val_cnt = 0.0

                cond = (amt > A) & (val_cnt > N) & (amt >= (P/100.0) * val_sum)
                ok_flags.append(cond)

        ok_flags = pd.Series(ok_flags, index=OUT_.index, dtype=bool) if len(ok_flags)==len(OUT_) else pd.Series(False, index=OUT_.index)
        rows.append({"escenario": name, "alertas": int((ok_flags & countable).sum())})

    return pd.DataFrame(rows)
