from __future__ import annotations
import pandas as pd, numpy as np, math
from typing import Iterable, Union, Dict, Any

AMOUNT_QS_DEF = (0.85, 0.90, 0.95, 0.97, 0.99)
FACTOR_QS_DEF = (0.90, 0.95, 0.97, 0.99)
NUMBER_QS_DEF = (0.50, 0.75, 0.90)

def _as_list(x):
    return [x] if isinstance(x, str) else list(map(str, x))

def _qdict(series, qs):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {q: np.nan for q in qs}
    q = s.quantile(qs)
    return {float(k): float(v) for k, v in q.items()}

def run_parameters_pgav_out(
    path: str,
    *,
    subsubsegments: Union[str, Iterable[str]],
    amount_qs: Iterable[float] = AMOUNT_QS_DEF,
    factor_qs: Iterable[float] = FACTOR_QS_DEF,
    number_qs: Iterable[float] = NUMBER_QS_DEF,
) -> Dict[str, Any]:
    tx = pd.read_csv(path, dtype={"customer_id":"string"}, encoding="utf-8-sig", low_memory=False)
    tx["tx_date_time"]   = pd.to_datetime(tx["tx_date_time"], errors="coerce")
    tx["tx_base_amount"] = pd.to_numeric(tx["tx_base_amount"], errors="coerce")
    tx["tx_direction"]   = tx["tx_direction"].astype(str).str.title()
    tx["tx_type"]        = tx["tx_type"].astype(str).str.title()

    # sub-subsegmento
    targets = set(_as_list(subsubsegments))
    if "customer_sub_sub_type" in tx.columns:
        tx = tx[tx["customer_sub_sub_type"].astype(str).isin(targets)].copy()

    GROUP_COL = "customer_sub_type" if "customer_sub_type" in tx.columns else "customer_type"
    if GROUP_COL not in tx.columns:
        raise KeyError("No encontré ni 'customer_sub_type' ni 'customer_type' en el CSV.")

    g = tx[
        (tx["tx_direction"].eq("Outbound")) &
        (tx["tx_type"].eq("Cash")) &
        (tx["tx_base_amount"].notna()) &
        (tx["tx_date_time"].notna())
    ].copy()

    if g.empty:
        cols = [GROUP_COL]
        cols += [f"Amount_p{int(q*100)}" for q in amount_qs]
        for q in factor_qs: cols += [f"Factor_p{int(q*100)}_raw", f"Factor_p{int(q*100)}"]
        for q in number_qs: cols += [f"Number_p{int(q*100)}_raw", f"Number_p{int(q*100)}"]
        out = pd.DataFrame(columns=cols)
        return {"meta": {"groups": 0, "rows": 0}, "percentiles": out}

    g = g.sort_values([GROUP_COL, "tx_date_time"]).reset_index(drop=True)

    # ---- Rolling 7D por grupo, alineado 1:1 con g ----
    def _roll_7d_sum_count(sub: pd.DataFrame) -> pd.DataFrame:
        out = (
            sub.set_index("tx_date_time")["tx_base_amount"]
               .rolling("7D")
               .agg(sum="sum", count="count")
        )
        out.index = sub.index  # alinear con 'g'
        return out

    tmp = g.groupby(GROUP_COL, group_keys=False).apply(_roll_7d_sum_count)

    g["prev_sum7"] = tmp["sum"] - g["tx_base_amount"]
    g["prev_cnt7"] = tmp["count"] - 1
    g["peer_avg7_excl"] = np.where(g["prev_cnt7"] > 0, g["prev_sum7"] / g["prev_cnt7"], np.nan)
    g["factor"] = np.where(g["peer_avg7_excl"] > 0, g["tx_base_amount"] / g["peer_avg7_excl"], np.nan)
    g["number_prev7"] = g["prev_cnt7"].clip(lower=0)

    rows = []
    for grp, sub in g.groupby(GROUP_COL):
        amt_q = _qdict(sub["tx_base_amount"], amount_qs)
        fac_q = _qdict(sub["factor"],        factor_qs)
        num_q = _qdict(sub["number_prev7"],  number_qs)
        row = {GROUP_COL: grp}
        for q,v in amt_q.items():
            row[f"Amount_p{int(q*100)}"] = v
        for q,v in fac_q.items():
            row[f"Factor_p{int(q*100)}_raw"] = v
            row[f"Factor_p{int(q*100)}"]     = int(math.ceil(v)) if np.isfinite(v) else np.nan
        for q,v in num_q.items():
            row[f"Number_p{int(q*100)}_raw"] = v
            row[f"Number_p{int(q*100)}"]     = int(math.floor(v)) if np.isfinite(v) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(GROUP_COL).reset_index(drop=True)
    return {"meta": {"groups": out.shape[0], "rows": int(g.shape[0])}, "percentiles": out}
