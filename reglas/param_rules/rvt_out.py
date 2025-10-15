from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

NUM_QS_DEF = (0.95, 0.97, 0.99)
AMT_QS_DEF = (0.95, 0.97, 0.99)

def _as_list(x): return [x] if isinstance(x, str) else list(map(str, x))

def _max_count_sum_30d(ts: np.ndarray, amts: np.ndarray, days=30):
    if ts.size == 0: return (0, 0.0)
    idx = np.argsort(ts); ts = ts[idx]; amts = amts[idx]
    j = 0; best_c = 0; best_s = 0.0
    prefix = np.concatenate([[0.0], np.cumsum(amts)])
    delta = np.timedelta64(days, "D")
    for i in range(ts.size):
        end = ts[i] + delta
        while j < ts.size and ts[j] <= end: j += 1
        c = j - i; s = prefix[j] - prefix[i]
        if c > best_c: best_c = c
        if s > best_s: best_s = s
    return (best_c, best_s)

def run_parameters_rvt_out(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    window_days: int = 30,
    number_qs: Iterable[float] = NUM_QS_DEF,
    amount_qs: Iterable[float] = AMT_QS_DEF,
) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    df["tx_date_time"]   = pd.to_datetime(df.get("tx_date_time"), errors="coerce")
    df["tx_amount"]      = pd.to_numeric(df.get("tx_amount"), errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")
    df["tx_direction"]   = df.get("tx_direction","").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type","").astype(str).str.title()

    targets = set(_as_list(subsubsegments))
    df = df[df["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    is_round = np.isfinite(df["tx_amount"]) & np.isclose(df["tx_amount"] % 1000.0, 0.0, atol=1e-9)
    m = (df["tx_direction"].eq("Outbound") & df["tx_type"].eq("Cash") & is_round &
         df["tx_date_time"].notna() & df["tx_base_amount"].notna() & df["customer_id"].notna())
    g = df.loc[m, ["customer_id","tx_date_time","tx_base_amount"]].sort_values(["customer_id","tx_date_time"]).copy()
    if g.empty:
        tblN = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in number_qs],
                             "Number_raw":[np.nan]*len(number_qs),
                             "Number_ceil":[np.nan]*len(number_qs)})
        tblA = pd.DataFrame({"percentil":[f"p{int(p*100)}" for p in amount_qs],
                             "Amount_CLP":[np.nan]*len(amount_qs)})
        return {"meta":{"clients":0}, "percentiles":{"number": tblN, "amount": tblA}}

    g["amt"] = g["tx_base_amount"].abs().astype(float)
    out_rows=[]
    for cid, sub in g.groupby("customer_id", sort=False):
        mc, ms = _max_count_sum_30d(sub["tx_date_time"].values, sub["amt"].values, window_days)
        out_rows.append({"customer_id": cid, "max_count_30d": mc, "max_sum_30d": ms})

    res = pd.DataFrame(out_rows)
    sN = res["max_count_30d"].astype(float); sA = res["max_sum_30d"].astype(float)
    qN = {p: (float(np.percentile(sN, int(p*100))) if len(sN) else np.nan) for p in number_qs}
    qA = {p: (float(np.percentile(sA, int(p*100))) if len(sA) else np.nan) for p in amount_qs}

    df_number = pd.DataFrame({
        "percentil":[f"p{int(p*100)}" for p in number_qs],
        "Number_raw":[qN[p] for p in number_qs],
        "Number_ceil":[int(math.ceil(qN[p])) if np.isfinite(qN[p]) else np.nan for p in number_qs],
    })
    df_amount = pd.DataFrame({
        "percentil":[f"p{int(p*100)}" for p in amount_qs],
        "Amount_CLP":[qA[p] for p in amount_qs],
    })
    return {"meta":{"clients": res.shape[0]}, "percentiles":{"number": df_number, "amount": df_amount}}
