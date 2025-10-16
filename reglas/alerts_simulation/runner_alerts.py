from __future__ import annotations
from pathlib import Path
import json
import warnings
import pandas as pd

from pgav_in_sim import simulate_pgav_in
from pgav_out_sim import simulate_pgav_out
from hanumi_sim import simulate_hanumi
from hanumo_sim import simulate_hanumo
from hasumi_sim import simulate_hasumi
from hasumo_sim import simulate_hasumo
from hnr_in_sim import simulate_hnr_in
from hnr_out_sim import simulate_hnr_out
from in_gt_out_sim import simulate_in_gt_out
from in_avg_sim import simulate_in_avg          
from out_avg_sim import simulate_out_avg       
from in_out_1_sim import simulate_in_out_1     
from out_pct_in_sim import simulate_out_pct_in 
from numcci_sim import simulate_numcci         
from numcco_sim import simulate_numcco         
from ocmc_1_sim import simulate_ocmc_1          
from p_pctbal_sim import simulate_p_pctbal   
from p_first_sim import simulate_p_first
from p_second_sim import simulate_p_second
from p_hsumi_sim import simulate_p_hsumi
from p_hsumo_sim import simulate_p_hsumo
from p_hvi_sim import simulate_p_hvi
from p_hvo_sim import simulate_p_hvo
from p_lbal_sim import simulate_p_lbal
from p_lval_sim import simulate_p_lval
from p_tli_sim import simulate_p_tli
from p_tlo_sim import simulate_p_tlo
from rvt_in_sim import simulate_rvt_in
from rvt_out_sim import simulate_rvt_out
from sumcci_sim import simulate_sumcci
from sumcco_sim import simulate_sumcco

import re, unicodedata

def _slugify_segment(seg):
    """Convierte SUBSUBS_NUEVO (str o iterable) a un nombre de archivo seguro."""
    if isinstance(seg, (list, tuple, set)):
        seg = "__".join(map(str, seg))
    seg = str(seg)
    seg = unicodedata.normalize("NFKD", seg).encode("ascii", "ignore").decode("ascii")
    seg = seg.strip().lower()
    seg = re.sub(r"[^a-z0-9._-]+", "_", seg)
    return seg or "segmento"



# ============================================

# apaga DtypeWarning de pandas.read_csv y FutureWarning (groupby.apply, etc.)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
TX_PATH = ROOT / "data" / "tx" / "datos_trx__with_subsub_oficial.csv"
PARAMS_BUNDLE = ROOT / "outputs" / "params" / "R-Low" / "params_R-Low.json"

COUNT_FROM = pd.Timestamp("2025-02-21", tz="UTC")
SUBSUBS_ACTUAL = ["R-Low", "R-High"]
SUBSUBS_NUEVO  = ["R-High"]

# Parámetros “Actuales” de reglas (si quieres comparar contra situación vigente)
ACTUAL_PARAMS = {
    "PGAV-IN":  {"Amount": 20000000, "Factor": 5, "Number": 139},
    "PGAV-OUT": {"Amount": 17983025, "Factor": 4, "Number": 203},
    "HANUMI":   {"Number": 2, "Factor": 59},
    "HANUMO":   {"Number": 2, "Factor": 59},
    "HASUMI":   {"Amount": 45_700_000, "Factor": 12},
    "HASUMO":   {"Amount": 16_000_000, "Factor": 169},
    "HNR-IN":   {"Number": 6},
    "HNR-OUT":  {"Number": 4},
    "IN>%OUT":  {"Amount_IN_30d": 49084774},
    "IN>AVG":   {"Amount": 15446792, "Factor": 6, "Number": 10},
    "OUT>AVG":  {"Amount": 17000000,  "Factor": 7, "Number": 9},
    "IN-OUT-1": {"Amount": 100000000,  "Number": 2, "Percentage": 80},
    "OUT>%IN":  {"Amount_OUT_30d": 45000000, "Low": 90, "High": 110},
    "NUMCCI":   {"Number": 2},
    "NUMCCO":   {"Number": 2},
    "OCMC_1":   {"Number": 2},
    "P-%BAL":   {"Balance": 1500000000, "Percentage": 95},
    "P-1st": {"Days": 7, "Amount": 389142381},
    "P-2nd": {"Days": 7, "Amount": 386816508},
    "P-HSUMI": {"Amount": 299000000},
    "P-HSUMO": {"Amount": 373635900},
    "P-HVI": {"Number": 37},
    "P-HVO": {"Number": 12},
    "P-LBAL": {"Balance": 200000000},
    "P-LVAL": {"Factor": 1.67},
    "P-TLI": {"Amount": 200_000_000},
    "P-TLO": {"Amount": 182_633_523},
    "RVT-IN":  {"Number": 6, "Amount": 99800000},
    "RVT-OUT": {"Number": 4, "Amount": 54000000},
    "SUMCCI":  {"Amount": 100000000},
    "SUMCCO":  {"Amount": 100000000},
}

OUT_DIR = ROOT / "outputs" / "alerts_sim"


# ------------------------------------------------------------
# Helpers de bundle
# ------------------------------------------------------------
def _load_bundle(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _rows(bundle: dict, rule_key: str, table_key: str) -> list[dict]:
    return (
        bundle.get("rules", {})
              .get(rule_key, {})
              .get(table_key, {})
              .get("percentiles", [])
        or []
    )

def _first_float(row: dict, *keys: str) -> float | None:
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except Exception:
                pass
    return None


# ------------------------------------------------------------
# Builders de escenarios por regla (desde bundle, sin pcts hardcode)
# ------------------------------------------------------------
def build_rvt_scenarios(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    # amount
    for r in _rows(bundle, rule_key, "amount"):
        pct = str(r.get("percentil","")).strip()
        a = _first_float(r, "Amount_CLP", "Amount")
        if pct and a is not None:
            scen.setdefault(pct, {})["Amount"] = a
    # number
    for r in _rows(bundle, rule_key, "number"):
        pct = str(r.get("percentil","")).strip()
        n = _first_float(r, "Number", "Number_ceil", "Number_ceiled", "Number_raw")
        if pct and n is not None:
            scen.setdefault(pct, {})["Number"] = n
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_sumcc_scenarios(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        a = _first_float(r, "Amount_CLP", "Amount")
        if pct and a is not None:
            scen.setdefault(pct, {})["Amount"] = a
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_amount_only(bundle: dict, rule_key: str, json_field: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        a = _first_float(r, json_field, "Amount", "Amount_CLP", "Amount_S3")
        if pct and a is not None:
            scen.setdefault(pct, {})["Amount"] = a
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_number_only(bundle: dict, rule_key: str, *json_fields: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        n = _first_float(r, *json_fields, "Number", "Number_ceil", "Number_ceiled", "Number_raw")
        if pct and n is not None:
            scen.setdefault(pct, {})["Number"] = n
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_factor_only(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        f = _first_float(r, "Factor", "Factor_int", "Factor_rec", "Factor_raw")
        if pct and f is not None:
            scen.setdefault(pct, {})["Factor"] = f
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_p_first(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-1st: Amount_CLP
    return build_amount_only(bundle, "P-1st", "Amount_CLP", include_actual)

def build_p_second(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-2nd: Amount_CLP
    return build_amount_only(bundle, "P-2nd", "Amount_CLP", include_actual)

def build_p_hsumi(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-HSUMI: Amount_30d_max_per_customer_CLP
    return build_amount_only(bundle, "P-HSUMI", "Amount_30d_max_per_customer_CLP", include_actual)

def build_p_hsumo(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-HSUMO: Amount_30d_max_per_customer_CLP
    return build_amount_only(bundle, "P-HSUMO", "Amount_30d_max_per_customer_CLP", include_actual)

def build_p_hvi(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-HVI: Number_max30d
    return build_number_only(bundle, "P-HVI", "Number_max30d", include_actual=include_actual)

def build_p_hvo(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-HVO: Number_max30d
    return build_number_only(bundle, "P-HVO", "Number_max30d", include_actual=include_actual)

def build_p_lbal(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-LBAL: Balance_after_tx
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("P-LBAL", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        b  = _first_float(r, "Balance_after_tx", "Balance")
        if pct and b is not None:
            scen.setdefault(pct, {})["Balance"] = b
    if include_actual and "P-LBAL" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["P-LBAL"])
    return scen

def build_p_lval(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-LVAL: Factor_(int|raw)
    return build_factor_only(bundle, "P-LVAL", include_actual)

def build_p_tli(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-TLI: Amount_CLP
    return build_amount_only(bundle, "P-TLI", "Amount_CLP", include_actual)

def build_p_tlo(bundle: dict, include_actual: bool) -> dict[str, dict]:
    # P-TLO: Amount_CLP
    return build_amount_only(bundle, "P-TLO", "Amount_CLP", include_actual)

def build_pgav_scenarios(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}

    for r in _rows(bundle, rule_key, "amount"):
        pct = str(r.get("percentil","")).strip()
        val = _first_float(r, "Amount_CLP", "Amount")
        if pct and val is not None:
            scen.setdefault(pct, {})["Amount"] = val
    for r in _rows(bundle, rule_key, "factor"):
        pct = str(r.get("percentil","")).strip()
        val = _first_float(r, "Factor", "Factor_int", "Factor_raw")
        if pct and val is not None:
            scen.setdefault(pct, {})["Factor"] = val
    for r in _rows(bundle, rule_key, "number"):
        pct = str(r.get("percentil","")).strip()
        val = _first_float(r, "Number", "Number_ceil", "Number_ceiled", "Number_raw")
        if pct and val is not None:
            scen.setdefault(pct, {})["Number"] = val

    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_hanum_xy_scenarios(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        if not pct:
            continue
        n = _first_float(r, "Number", "Number_ceil", "Number_ceiled", "Number_raw")
        f = _first_float(r, "Factor", "Factor_int", "Factor_ceiled", "Factor_raw")
        if n is not None or f is not None:
            scen.setdefault(pct, {})
            if n is not None: scen[pct]["Number"] = n
            if f is not None: scen[pct]["Factor"] = f
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_hasum_xy_scenarios(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        if not pct:
            continue
        a = _first_float(r, "Amount_S3", "Amount")
        f = _first_float(r, "Factor", "Factor_int", "Factor_raw")
        if a is not None or f is not None:
            scen.setdefault(pct, {})
            if a is not None: scen[pct]["Amount"] = a
            if f is not None: scen[pct]["Factor"] = f
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

def build_in_gt_out_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("IN>%OUT", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        a = _first_float(r, "Amount_IN_30d")
        if pct and a is not None:
            scen.setdefault(pct, {})["Amount_IN_30d"] = a
    if include_actual and "IN>%OUT" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["IN>%OUT"])
    return scen

def build_hnr_scenarios(bundle: dict, rule_key: str, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get(rule_key, {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        n = _first_float(r, "Number", "Number_ceil", "Number_ceiled", "Number_raw", "Number_max30d")
        if pct and n is not None:
            scen.setdefault(pct, {})["Number"] = n
    if include_actual and rule_key in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS[rule_key])
    return scen

# ====== NUEVOS builders ======
def build_in_avg_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("IN>AVG", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        A = _first_float(r, "Amount")
        F = _first_float(r, "Factor", "Factor_raw")
        if not pct:
            continue
        if A is not None or F is not None:
            scen.setdefault(pct, {})
            if A is not None: scen[pct]["Amount"] = A
            if F is not None: scen[pct]["Factor"] = F
    if include_actual and "IN>AVG" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["IN>AVG"])
    return scen

def build_out_avg_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("OUT>AVG", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        A = _first_float(r, "Amount")
        F = _first_float(r, "Factor", "Factor_raw")
        if not pct:
            continue
        if A is not None or F is not None:
            scen.setdefault(pct, {})
            if A is not None: scen[pct]["Amount"] = A
            if F is not None: scen[pct]["Factor"] = F
    if include_actual and "OUT>AVG" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["OUT>AVG"])
    return scen

def build_in_out_1_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("IN-OUT-1 Amount", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        A = _first_float(r, "Amount_CLP", "Amount")
        if pct and A is not None:
            scen.setdefault(pct, {})["Amount"] = A
    if include_actual and "IN-OUT-1" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["IN-OUT-1"])
    return scen

def build_out_pct_in_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("OUT>%IN", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        A = _first_float(r, "Amount_OUT_30d")
        if pct and A is not None:
            scen.setdefault(pct, {})["Amount_OUT_30d"] = A
    if include_actual and "OUT>%IN" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["OUT>%IN"])
    return scen

def build_numcci_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("NUMCCI", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        n = _first_float(r, "Number", "Number_ceil", "Number_raw")
        if pct and n is not None:
            scen.setdefault(pct, {})["Number"] = n
    if include_actual and "NUMCCI" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["NUMCCI"])
    return scen

def build_numcco_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("NUMCCO", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        n = _first_float(r, "Number", "Number_ceil", "Number_raw")
        if pct and n is not None:
            scen.setdefault(pct, {})["Number"] = n
    if include_actual and "NUMCCO" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["NUMCCO"])
    return scen

def build_ocmc_1_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("OCMC_1", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        n = _first_float(r, "Counterparties_30d", "Ceil")
        if pct and n is not None:
            scen.setdefault(pct, {})["Counterparties_30d"] = n
    if include_actual and "OCMC_1" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["OCMC_1"])
    return scen

def build_p_pctbal_scenarios(bundle: dict, include_actual: bool) -> dict[str, dict]:
    scen: dict[str, dict] = {}
    rows = bundle.get("rules", {}).get("P-%BAL", {}).get("percentiles", [])
    for r in rows:
        pct = str(r.get("percentil","")).strip()
        b = _first_float(r, "Balance")
        if pct and b is not None:
            scen.setdefault(pct, {})["Balance"] = b
    if include_actual and "P-%BAL" in ACTUAL_PARAMS:
        scen["Actual"] = dict(ACTUAL_PARAMS["P-%BAL"])
    return scen
# ========================================

# ------------------------------------------------------------
# Runner principal
# ------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(PARAMS_BUNDLE)

    # ===================== Simulación ACTUAL (segmento completo) =====================
    res_actual = []

    # PGAV-IN / PGAV-OUT (solo "Actual")
    # sc_in_act  = build_pgav_scenarios(bundle, "PGAV-IN", include_actual=True)
    # sc_out_act = build_pgav_scenarios(bundle, "PGAV-OUT", include_actual=True)
    # df = simulate_pgav_in(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_in_act["Actual"]}, count_from=COUNT_FROM).assign(regla="PGAV-IN")
    # res_actual.append(df)
    # df = simulate_pgav_out(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_out_act["Actual"]}, count_from=COUNT_FROM).assign(regla="PGAV-OUT")
    # res_actual.append(df)
    # print("Simulated PGAV-IN and PGAV-OUT for Actual.")

    # # HANUMI / HANUMO
    # sc_hmi = build_hanum_xy_scenarios(bundle, "HANUMI", include_actual=True)
    # if "Actual" in sc_hmi:
    #     df = simulate_hanumi(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_hmi["Actual"]}, count_from=COUNT_FROM).assign(regla="HANUMI")
    #     res_actual.append(df)
    # sc_hmo = build_hanum_xy_scenarios(bundle, "HANUMO", include_actual=True)
    # if "Actual" in sc_hmo:
    #     df = simulate_hanumo(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_hmo["Actual"]}, count_from=COUNT_FROM).assign(regla="HANUMO")
    #     res_actual.append(df)
    # print("Simulated HANUMI and HANUMO for Actual.")

    # # HASUMI / HASUMO
    # sc_hsi = build_hasum_xy_scenarios(bundle, "HASUMI", include_actual=True)
    # if "Actual" in sc_hsi:
    #     df = simulate_hasumi(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_hsi["Actual"]}, count_from=COUNT_FROM).assign(regla="HASUMI")
    #     res_actual.append(df)
    # sc_hso = build_hasum_xy_scenarios(bundle, "HASUMO", include_actual=True)
    # if "Actual" in sc_hso:
    #     df = simulate_hasumo(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_hso["Actual"]}, count_from=COUNT_FROM).assign(regla="HASUMO")
    #     res_actual.append(df)
    # print("Simulated HASUMI and HASUMO for Actual.")

    # # HNR-IN / HNR-OUT
    # sc_hnri = build_hnr_scenarios(bundle, "HNR-IN", include_actual=True)
    # if "Actual" in sc_hnri:
    #     df = simulate_hnr_in(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_hnri["Actual"]}, count_from=COUNT_FROM).assign(regla="HNR-IN")
    #     res_actual.append(df)
    # sc_hnro = build_hnr_scenarios(bundle, "HNR-OUT", include_actual=True)
    # if "Actual" in sc_hnro:
    #     df = simulate_hnr_out(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_hnro["Actual"]}, count_from=COUNT_FROM).assign(regla="HNR-OUT")
    #     res_actual.append(df)
    # print("Simulated HNR-IN and HNR-OUT for Actual.")

    # # IN>%OUT
    # sc_in_gt_out = build_in_gt_out_scenarios(bundle, include_actual=True)
    # if "Actual" in sc_in_gt_out:
    #     df = simulate_in_gt_out(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc_in_gt_out["Actual"]}, count_from=COUNT_FROM).assign(regla="IN>%OUT")
    #     res_actual.append(df)
    # print("Simulated IN>%OUT for Actual.")

    # # IN>AVG
    # sc = build_in_avg_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_in_avg(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="IN>AVG")
    #     res_actual.append(df)
    # print("Simulated IN>AVG for Actual.")

    # # OUT>AVG
    # sc = build_out_avg_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_out_avg(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="OUT>AVG")
    #     res_actual.append(df)
    # print("Simulated OUT>AVG for Actual.")

    # # IN-OUT-1
    # sc = build_in_out_1_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_in_out_1(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="IN-OUT-1")
    #     res_actual.append(df)
    # print("Simulated IN-OUT-1 for Actual.")

    # # OUT>%IN
    # sc = build_out_pct_in_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_out_pct_in(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="OUT>%IN")
    #     res_actual.append(df)
    # print("Simulated OUT>%IN for Actual.")

    # # NUMCCI
    # sc = build_numcci_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_numcci(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="NUMCCI")
    #     res_actual.append(df)
    # print("Simulated NUMCCI for Actual.")

    # # NUMCCO
    # sc = build_numcco_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_numcco(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="NUMCCO")
    #     res_actual.append(df)
    # print("Simulated NUMCCO for Actual.")

    # # OCMC_1
    # sc = build_ocmc_1_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_ocmc_1(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="OCMC_1")
    #     res_actual.append(df)
    # print("Simulated OCMC_1 for Actual.")

    # # P-%BAL
    # sc = build_p_pctbal_scenarios(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_pctbal(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-%BAL")
    #     res_actual.append(df)
    # print("Simulated P-%BAL for Actual.")

    # # P-1st
    # sc = build_p_first(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_first(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-1st")
    #     res_actual.append(df)
    # print("Simulated P-1st for Actual.")

    # # P-2nd
    # sc = build_p_second(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_second(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-2nd")
    #     res_actual.append(df)
    # print("Simulated P-2nd for Actual.")

    # # P-HSUMI
    # sc = build_p_hsumi(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_hsumi(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-HSUMI")
    #     res_actual.append(df)
    # print("Simulated P-HSUMI for Actual.")

    # # P-HSUMO
    # sc = build_p_hsumo(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_hsumo(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-HSUMO")
    #     res_actual.append(df)
    # print("Simulated P-HSUMO for Actual.")

    # # P-HVI
    # sc = build_p_hvi(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_hvi(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-HVI")
    #     res_actual.append(df)
    # print("Simulated P-HVI for Actual.")

    # # P-HVO
    # sc = build_p_hvo(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_hvo(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-HVO")
    #     res_actual.append(df)
    # print("Simulated P-HVO for Actual.")

    # # P-LBAL
    # sc = build_p_lbal(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_lbal(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-LBAL")
    #     res_actual.append(df)
    # print("Simulated P-LBAL for Actual.")

    # # P-LVAL
    # sc = build_p_lval(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_lval(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-LVAL")
    #     res_actual.append(df)
    # print("Simulated P-LVAL for Actual.")

    # # P-TLI
    # sc = build_p_tli(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_tli(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-TLI")
    #     res_actual.append(df)
    # print("Simulated P-TLI for Actual.")

    # # P-TLO
    # sc = build_p_tlo(bundle, include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_p_tlo(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="P-TLO")
    #     res_actual.append(df)
    # print("Simulated P-TLO for Actual.")

    # # RVT-IN
    # sc = build_rvt_scenarios(bundle, "RVT-IN", include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_rvt_in(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="RVT-IN")
    #     res_actual.append(df)
    # print("Simulated RVT-IN for Actual.")

    # # RVT-OUT
    # sc = build_rvt_scenarios(bundle, "RVT-OUT", include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_rvt_out(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="RVT-OUT")
    #     res_actual.append(df)
    # print("Simulated RVT-OUT for Actual.")

    # # SUMCCI
    # sc = build_sumcc_scenarios(bundle, "SUMCCI", include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_sumcci(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="SUMCCI")
    #     res_actual.append(df)
    # print("Simulated SUMCCI for Actual.")

    # # SUMCCO
    # sc = build_sumcc_scenarios(bundle, "SUMCCO", include_actual=True)
    # if "Actual" in sc:
    #     df = simulate_sumcco(str(TX_PATH), subsubs=SUBSUBS_ACTUAL, scenarios={"Actual": sc["Actual"]}, count_from=COUNT_FROM).assign(regla="SUMCCO")
    #     res_actual.append(df)
    # print("Simulated SUMCCO for Actual.")


    df_actual = pd.concat(res_actual, ignore_index=True) if res_actual else pd.DataFrame(columns=["regla","escenario","alertas"])

    # ===================== Simulación NUEVA (subsub objetivo, todos pXX) =============
    res_new = []

    # PGAV
    sc = build_pgav_scenarios(bundle, "PGAV-IN", include_actual=False)
    if sc:
        df = simulate_pgav_in(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="PGAV-IN")
        res_new.append(df)
    sc = build_pgav_scenarios(bundle, "PGAV-OUT", include_actual=False)
    if sc:
        df = simulate_pgav_out(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="PGAV-OUT")
        res_new.append(df)
    print("Simulated PGAV-IN and PGAV-OUT for new scenarios.")

    # HANUMI/HANUMO
    sc = build_hanum_xy_scenarios(bundle, "HANUMI", include_actual=False)
    if sc:
        df = simulate_hanumi(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="HANUMI")
        res_new.append(df)
    sc = build_hanum_xy_scenarios(bundle, "HANUMO", include_actual=False)
    if sc:
        df = simulate_hanumo(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="HANUMO")
        res_new.append(df)
    print("Simulated HANUMI and HANUMO for new scenarios.")

    # HASUMI/HASUMO
    sc = build_hasum_xy_scenarios(bundle, "HASUMI", include_actual=False)
    if sc:
        df = simulate_hasumi(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="HASUMI")
        res_new.append(df)
    sc = build_hasum_xy_scenarios(bundle, "HASUMO", include_actual=False)
    if sc:
        df = simulate_hasumo(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="HASUMO")
        res_new.append(df)
    print("Simulated HASUMI and HASUMO for new scenarios.")

    # HNR-IN/OUT
    sc = build_hnr_scenarios(bundle, "HNR-IN", include_actual=False)
    if sc:
        df = simulate_hnr_in(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="HNR-IN")
        res_new.append(df)
    sc = build_hnr_scenarios(bundle, "HNR-OUT", include_actual=False)
    if sc:
        df = simulate_hnr_out(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="HNR-OUT")
        res_new.append(df)
    print("Simulated HNR-IN and HNR-OUT for new scenarios.")

    # IN>%OUT (bundle)
    sc = build_in_gt_out_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_in_gt_out(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="IN>%OUT")
        res_new.append(df)
    print("Simulated IN>%OUT for new scenarios.")

    # IN>AVG
    sc = build_in_avg_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_in_avg(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="IN>AVG")
        res_new.append(df)
    print("Simulated IN>AVG for new scenarios.")

    # OUT>AVG
    sc = build_out_avg_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_out_avg(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="OUT>AVG")
        res_new.append(df)
    print("Simulated OUT>AVG for new scenarios.")

    # IN-OUT-1
    sc = build_in_out_1_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_in_out_1(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="IN-OUT-1")
        res_new.append(df)
    print("Simulated IN-OUT-1 for new scenarios.")

    # OUT>%IN
    sc = build_out_pct_in_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_out_pct_in(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="OUT>%IN")
        res_new.append(df)
    print("Simulated OUT>%IN for new scenarios.")

    # NUMCCI
    sc = build_numcci_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_numcci(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="NUMCCI")
        res_new.append(df)
    print("Simulated NUMCCI for new scenarios.")

    # NUMCCO
    sc = build_numcco_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_numcco(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="NUMCCO")
        res_new.append(df)
    print("Simulated NUMCCO for new scenarios.")

    # OCMC_1
    sc = build_ocmc_1_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_ocmc_1(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="OCMC_1")
        res_new.append(df)
    print("Simulated OCMC_1 for new scenarios.")

    # P-%BAL
    sc = build_p_pctbal_scenarios(bundle, include_actual=False)
    if sc:
        df = simulate_p_pctbal(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-%BAL")
        res_new.append(df)
    print("Simulated P-%BAL for new scenarios.")

    # P-1st
    sc = build_p_first(bundle, include_actual=False)
    if sc:
        df = simulate_p_first(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-1st")
        res_new.append(df)
    print("Simulated P-1st for new scenarios.")

    # P-2nd
    sc = build_p_second(bundle, include_actual=False)
    if sc:
        df = simulate_p_second(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-2nd")
        res_new.append(df)
    print("Simulated P-2nd for new scenarios.")

    # P-HSUMI
    sc = build_p_hsumi(bundle, include_actual=False)
    if sc:
        df = simulate_p_hsumi(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-HSUMI")
        res_new.append(df)
    print("Simulated P-HSUMI for new scenarios.")

    # P-HSUMO
    sc = build_p_hsumo(bundle, include_actual=False)
    if sc:
        df = simulate_p_hsumo(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-HSUMO")
        res_new.append(df)
    print("Simulated P-HSUMO for new scenarios.")

    # P-HVI
    sc = build_p_hvi(bundle, include_actual=False)
    if sc:
        df = simulate_p_hvi(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-HVI")
        res_new.append(df)
    print("Simulated P-HVI for new scenarios.")

    # P-HVO
    sc = build_p_hvo(bundle, include_actual=False)
    if sc:
        df = simulate_p_hvo(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-HVO")
        res_new.append(df)
    print("Simulated P-HVO for new scenarios.")

    # P-LBAL
    sc = build_p_lbal(bundle, include_actual=False)
    if sc:
        df = simulate_p_lbal(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-LBAL")
        res_new.append(df)
    print("Simulated P-LBAL for new scenarios.")

    # P-LVAL
    sc = build_p_lval(bundle, include_actual=False)
    if sc:
        df = simulate_p_lval(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-LVAL")
        res_new.append(df)
    print("Simulated P-LVAL for new scenarios.")

    # P-TLI
    sc = build_p_tli(bundle, include_actual=False)
    if sc:
        df = simulate_p_tli(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-TLI")
        res_new.append(df)
    print("Simulated P-TLI for new scenarios.")

    # P-TLO
    sc = build_p_tlo(bundle, include_actual=False)
    if sc:
        df = simulate_p_tlo(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="P-TLO")
        res_new.append(df)
    print("Simulated P-TLO for new scenarios.")

    # RVT-IN
    sc = build_rvt_scenarios(bundle, "RVT-IN", include_actual=False)
    if sc:
        df = simulate_rvt_in(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="RVT-IN")
        res_new.append(df)
    print("Simulated RVT-IN for new scenarios.")

    # RVT-OUT
    sc = build_rvt_scenarios(bundle, "RVT-OUT", include_actual=False)
    if sc:
        df = simulate_rvt_out(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="RVT-OUT")
        res_new.append(df)
    print("Simulated RVT-OUT for new scenarios.")

    # SUMCCI
    sc = build_sumcc_scenarios(bundle, "SUMCCI", include_actual=False)
    if sc:
        df = simulate_sumcci(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="SUMCCI")
        res_new.append(df)
    print("Simulated SUMCCI for new scenarios.")

    # SUMCCO
    sc = build_sumcc_scenarios(bundle, "SUMCCO", include_actual=False)
    if sc:
        df = simulate_sumcco(str(TX_PATH), subsubs=SUBSUBS_NUEVO, scenarios=sc, count_from=COUNT_FROM).assign(regla="SUMCCO")
        res_new.append(df)
    print("Simulated SUMCCO for new scenarios.")



    df_new = pd.concat(res_new, ignore_index=True) if res_new else pd.DataFrame(columns=["regla","escenario","alertas"])

    # ===================== Resumen largo + compacto ===============================
    summary = pd.concat(
        [df_actual.assign(tipo="actual"), df_new.assign(tipo="nuevo")],
        ignore_index=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "alerts_summary_long.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    compact: dict[str, dict[str, int]] = {}
    for _, row in summary.iterrows():
        regla = str(row.get("regla", ""))
        escenario = str(row.get("escenario", ""))
        tipo = str(row.get("tipo", "nuevo")).lower()
        alertas = int(row.get("alertas", 0))
        if not regla or not escenario:
            continue
        compact.setdefault(regla, {})
        if tipo == "actual":
            compact[regla]["actual"] = alertas
        else:
            compact[regla][escenario.lower()] = alertas

    # >>> nombre de archivo dependiente del segmento
    seg_slug = _slugify_segment(SUBSUBS_NUEVO)
    out_compact_path = OUT_DIR / f"alerts_summary_compact__{seg_slug}.json"

    with open(out_compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    print("✔ Simulación de alertas terminada.")
    print(f"  - Resumen largo (JSON): {OUT_DIR/'alerts_summary_long.json'}")
    print(f"  - Resumen compacto:     {out_compact_path}")



if __name__ == "__main__":
    main()
