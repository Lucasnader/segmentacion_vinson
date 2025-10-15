from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (0.85, 0.90, 0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_second(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[float] = DEFAULT_PCTS,
    window_days: int = 7,
    filter_to_cash: bool = True,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"] = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["customer_account_creation_date"] = pd.to_datetime(df["customer_account_creation_date"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    if filter_to_cash and "tx_type" in df.columns:
        df["tx_type"] = df["tx_type"].astype(str).str.title()

    m = df["tx_date_time"].notna() & df["customer_account_creation_date"].notna() & df["tx_base_amount"].notna()
    if filter_to_cash and "tx_type" in df.columns:
        m &= df["tx_type"].eq("Cash")

    g = df.loc[m, ["customer_id","tx_date_time","customer_account_creation_date","tx_base_amount"]].copy()
    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles], "Amount_CLP":[np.nan]*len(percentiles)})
        return {"meta":{"n_second":0}, "percentiles": tbl}

    g = g.sort_values(["customer_id","tx_date_time"])
    g["tx_order"] = g.groupby("customer_id").cumcount() + 1
    within = (g["tx_date_time"] - g["customer_account_creation_date"]).dt.days.between(0, window_days)
    second_tx = g[(g["tx_order"] == 2) & within & (g["tx_base_amount"] > 0)]["tx_base_amount"].astype(float)

    if second_tx.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles], "Amount_CLP":[np.nan]*len(percentiles)})
        return {"meta":{"n_second":0}, "percentiles": tbl}

    q = second_tx.quantile(list(percentiles))
    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "Amount_CLP":[q.get(p, np.nan) for p in percentiles]})
    return {"meta":{"n_second": int(second_tx.shape[0])}, "percentiles": tbl}
