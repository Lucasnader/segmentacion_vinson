from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (0.85, 0.90, 0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_first(
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

    g = df.loc[m, ["customer_name","tx_date_time","customer_account_creation_date","tx_base_amount"]].copy()
    if g.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles], "Amount_CLP":[np.nan]*len(percentiles)})
        return {"meta":{"n_clients_window":0}, "percentiles": tbl}

    g["tx_date"]   = g["tx_date_time"].dt.normalize()
    g["open_date"] = g["customer_account_creation_date"].dt.normalize()
    idx_first = g.sort_values(["customer_name","tx_date_time"]).groupby("customer_name").head(1).index
    first = g.loc[idx_first].copy()
    first["days_since_open"] = (first["tx_date"] - first["open_date"]).dt.days
    within = first["days_since_open"].between(0, window_days, inclusive="both") & (first["tx_base_amount"] > 0)
    first_in_window = first.loc[within, "tx_base_amount"].astype(float)

    if first_in_window.empty:
        tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles], "Amount_CLP":[np.nan]*len(percentiles)})
        return {"meta":{"n_clients_window":0}, "percentiles": tbl}

    q = first_in_window.quantile(list(percentiles))
    tbl = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in percentiles],
                        "Amount_CLP":[q.get(p, np.nan) for p in percentiles]})
    return {"meta":{"n_clients_window": int(first_in_window.shape[0])}, "percentiles": tbl}
