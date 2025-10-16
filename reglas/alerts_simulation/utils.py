from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, Tuple
import json
import pandas as pd
import numpy as np
from datetime import datetime

# --------- Lectura de bundle de parámetros ---------

def load_params_bundle(bundle_path: str | Path) -> Dict[str, Any]:
    with open(bundle_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --------- Helpers para extraer escenarios desde el bundle ---------

def _pick_percentiles(
    rows: Iterable[dict],
    wanted: Iterable[str]
) -> Dict[str, dict]:
    """
    Convierte una lista de rows con "percentil" a dict: {"p95": {...}, "p97": {...}}
    filtrando solo los 'wanted'.
    """
    out = {}
    wanted_set = set(wanted)
    for r in rows:
        p = str(r.get("percentil", "")).lower()
        if p in wanted_set:
            out[p] = r
    return out

def scenarios_from_bundle_rowwise(
    bundle: Dict[str, Any],
    rule_name: str,
    *,
    percentiles: Iterable[str],
    field_map: Dict[str, str],
    scenario_prefix: str = "",
    include_actual: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Para reglas cuyo JSON tiene estructura:
      "rules": {
        "X": { "percentiles": [ {"percentil":"p95", <campos>}, ... ] }
      }
    Devuelve dict {"p95": {"ParamA":..., "ParamB":...}, ...} + (opcional) "Actual": {...}

    field_map: cómo mapear {"ParamA": "FieldNameEnJSON", ...}
    """
    node = bundle["rules"].get(rule_name, {})
    rows = node.get("percentiles", [])
    byp = _pick_percentiles(rows, [p.lower() for p in percentiles])
    scenarios = {}
    # agrega percentiles
    for p, row in byp.items():
        scenario_name = f"{scenario_prefix}{p}"
        scenarios[scenario_name] = {k: row.get(v, np.nan) for k, v in field_map.items()}
    # agrega actual si corresponde
    if include_actual:
        scenarios["Actual"] = dict(include_actual)
    return scenarios

def scenarios_from_bundle_split_tables(
    bundle: Dict[str, Any],
    rule_name: str,
    *,
    percentiles: Iterable[str],
    amount_map: Optional[Dict[str, str]] = None,
    factor_map: Optional[Dict[str, str]] = None,
    number_map: Optional[Dict[str, str]] = None,
    include_actual: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Para reglas con estructura anidada (ej. PGAV-*):
      "rules": {
        "PGAV-IN": {
          "amount": {"percentiles":[...]},
          "factor": {"percentiles":[...]},
          "number": {"percentiles":[...]}
        }
      }
    Devuelve escenarios con combinación de columnas mapeadas.

    * Para PGAV la idea típica es construir escenarios con:
        - Amount_CLP de "amount" (p.ej p90/p95/p97/p99)
        - Factor (o Factor_ceiled) de "factor" (p90/p95/p97/p99)
        - Number (o Number_ceiled) de "number" (p50/p75/p90)  <- OJO: percentiles diferentes
      Por eso llamamos a esta función por partes (o con percentiles distintos por dimensión).
    """
    node = bundle["rules"].get(rule_name, {})
    scenarios = {}
    if include_actual:
        scenarios["Actual"] = dict(include_actual)

    # helper para rellenar dictEscenarios con un bloque (amount/factor/number)
    def add_block(dim_key: str, m: Dict[str, str] | None, pcts: Iterable[str]):
        if not m:
            return
        rows = node.get(dim_key, {}).get("percentiles", [])
        byp = _pick_percentiles(rows, [p.lower() for p in pcts])
        for p, row in byp.items():
            sc = scenarios.setdefault(p, {})
            for k, v in m.items():
                sc[k] = row.get(v, np.nan)

    # amount/factor usan la misma lista 'percentiles' que pasaste
    add_block("amount", amount_map, percentiles)
    add_block("factor", factor_map, percentiles)

    # number puede tener percentiles distintos (p50/p75/p90 normalmente).
    # Si quieres que number consuma otra lista, llama a esta función de nuevo con otros percentiles
    # o usa un segundo llamado (ver runner_alerts).
    add_block("number", number_map, percentiles)

    # homogeneiza nombres de escenarios (prefijo "p" por si vienen "p95")
    # ya vienen con 'p' del JSON
    return scenarios

# --------- Carga de transacciones y filtros comunes ---------

def load_tx_base(tx_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(tx_path, dtype={"customer_id": "string"}, encoding="utf-8-sig")
    df["tx_date_time"]   = pd.to_datetime(df["tx_date_time"], errors="coerce")
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")
    df["tx_amount"]      = pd.to_numeric(df.get("tx_amount"), errors="coerce")
    df["tx_direction"]   = df.get("tx_direction", "").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type", "").astype(str).str.title()
    df["tx_currency"]    = df.get("tx_currency", "").astype(str).str.upper()
    return df

def filter_subsubs(df: pd.DataFrame, subsubs: Iterable[str] | str) -> pd.DataFrame:
    if isinstance(subsubs, str):
        target = {subsubs}
    else:
        target = set(map(str, subsubs))
    if "customer_sub_sub_type" in df.columns:
        return df[df["customer_sub_sub_type"].astype(str).isin(target)].copy()
    return df.copy()

# --------- Utilidad: contar solo desde COUNT_FROM (con contexto completo) ---------

def restrict_counts_after(df, date_col, count_from):
    # Normaliza el umbral a UTC
    ts = pd.to_datetime(count_from, utc=True)

    # Normaliza la columna a UTC (segura aunque ya venga en UTC)
    col = pd.to_datetime(df[date_col], errors="coerce", utc=True)

    return col >= ts

