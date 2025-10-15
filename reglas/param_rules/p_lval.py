from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

DEFAULT_PCTS = (90, 95, 97, 99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def run_parameters_p_lval(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    percentiles: Iterable[int] = DEFAULT_PCTS,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    df["tx_base_amount"] = pd.to_numeric(df["tx_base_amount"], errors="coerce")
    df["customer_expected_amount"] = pd.to_numeric(df["customer_expected_amount"], errors="coerce")
    is_cash = (df["tx_type"].astype(str).str.title() == "Cash")
    m = df.loc[is_cash, ["customer_id","tx_base_amount","customer_expected_amount"]].dropna()

    if m.empty:
        tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles],
                            "Factor_raw":[np.nan]*len(percentiles),
                            "Factor_int":[np.nan]*len(percentiles)})
        return {"meta":{"n":0, "mean_int": np.nan, "suggested_factor": np.nan}, "percentiles": tbl}

    exp_by_cust = (m.groupby("customer_id", as_index=False)["customer_expected_amount"].max()
                     .rename(columns={"customer_expected_amount":"expected_max"}))
    m = m.merge(exp_by_cust, on="customer_id", how="left")
    m = m[(m["expected_max"] > 0) & (m["tx_base_amount"] > 0)].copy()
    if m.empty:
        tbl = pd.DataFrame({"percentil":[f"p{p}" for p in percentiles],
                            "Factor_raw":[np.nan]*len(percentiles),
                            "Factor_int":[np.nan]*len(percentiles)})
        return {"meta":{"n":0, "mean_int": np.nan, "suggested_factor": np.nan}, "percentiles": tbl}

    m["factor_raw"] = m["tx_base_amount"] / m["expected_max"]
    m["factor_int"] = np.rint(m["factor_raw"])

    s_raw = m["factor_raw"].astype(float).replace([np.inf,-np.inf], np.nan).dropna()
    s_int = m["factor_int"].astype(float).replace([np.inf,-np.inf], np.nan).dropna()

    stats_raw = {p: float(np.percentile(s_raw, p)) for p in percentiles} if len(s_raw) else {}
    stats_int = {p: float(np.percentile(s_int, p)) for p in percentiles} if len(s_int) else {}

    tbl = pd.DataFrame({
        "percentil":[f"p{p}" for p in percentiles],
        "Factor_raw":[stats_raw.get(p, np.nan) for p in percentiles],
        "Factor_int":[stats_int.get(p, np.nan) for p in percentiles],
    })
    mean_int = float(s_int.mean()) if len(s_int) else np.nan
    suggested = int(np.floor(mean_int)) + 1 if np.isfinite(mean_int) else np.nan
    return {"meta":{"n": int(m.shape[0]), "mean_int": mean_int, "suggested_factor": suggested}, "percentiles": tbl}
