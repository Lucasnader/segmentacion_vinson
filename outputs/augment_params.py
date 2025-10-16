# augment_params.py
from __future__ import annotations
from pathlib import Path
import json
from copy import deepcopy

# === INPUTS ===================================================================

ROOT = Path(__file__).resolve().parents[2]  # .../LV_vsc
BUNDLE_IN  = ROOT / "outputs" / "params" / "R-Low" / "params_R-Low.json"
BUNDLE_OUT = ROOT / "outputs" / "params" / "R-Low" / "params_R-Low__augmented.json"

# Percentiles a usar cuando tengamos que CREAR una regla que no existe en el bundle
PCTS_ALL = ["p85","p90","p95","p97","p99"]

# ---------------------------------------------------------------------------
# MAPA de valores "actuales" (los que quieres inyectar cuando falten en el JSON).
# Claves:
#   - regla -> dict de:
#       - "flat":   {"Campo1": valor, "Campo2": valor, ...}   (para reglas con 1 sola tabla 'percentiles')
#       - "split":  {"amount": {...}, "factor": {...}, "number": {...}} (para PGAV, etc.)
#
# Los nombres de campo deben ser los que usa tu simulación. Procuro seguir los que tu JSON ya trae:
#   - Amount_CLP, Balance, Factor, Number, Number_max30d, Percentage, PctLow/PctHigh, etc.
#   - Para OUT>AVG / IN>AVG usas "Amount" y "Factor" en bundle; aquí los dejo igual.
# ---------------------------------------------------------------------------

CURRENT_DEFAULTS = {
    # === promedios por dirección ===
    "IN>AVG": {
        "flat": {
            "Amount": 134_712_184,
            "Factor": 4,
            "Number": 38,
        }
    },
    "OUT>AVG": {
        "flat": {
            "Amount": 45_402_580,
            "Factor": 4,
            "Number": 65,
        }
    },

    # === percentiles de balance/amount/factor simples ===
    "P-%BAL": { "flat": {"Balance": 1_048_744_284, "Percentage": 95} },
    "P-TLI":  { "flat": {"Amount_CLP": 1_206_067_841} },
    "P-TLO":  { "flat": {"Amount_CLP": 1_005_152_890} },
    "P-HVI":  { "flat": {"Number_max30d": 26} },
    "P-HVO":  { "flat": {"Number_max30d": 121} },
    "P-HSUMI":{ "flat": {"Amount_30d_max_per_customer_CLP": 9_662_835_000} },
    "P-HSUMO":{ "flat": {"Amount_30d_max_per_customer_CLP": 9_941_685_250} },
    "P-LBAL": { "flat": {"Balance_after_tx": 1_206_067_841} },
    # P-LVAL tiene en tu bundle Factor_raw/Factor_int; si quieres fijar "Factor" literal, márcalo aquí:
    "P-LVAL": { "flat": {"Factor": 4.36} },

    # === IN>%OUT / OUT>%IN con Low/High ===
    "OUT>%IN": { "flat": {"Amount_OUT_30d": 804_076_122, "PctLow": 125, "PctHigh": 200} },
    "IN>%OUT": { "flat": {"Amount_IN_30d": 1_090_365_821, "PctLow": 110, "PctHigh": 150} },

    # === RVT con amount + number ===
    "RVT-IN":  { "split": {
        "amount": {"Amount_CLP": 3_834_223_610},
        "number": {"Number_ceil": 13},  # en tu bundle usas Number_raw/Number_ceil; fijamos el ceiled
    }},
    "RVT-OUT": { "split": {
        "amount": {"Amount_CLP": 4_240_240_400},
        "number": {"Number_ceil": 17},
    }},

    # === PGAV (amount, factor, number) – si quieres fijar valores actuales, colócalos aquí:
    # Si no quieres fijarlos, deja el dict vacío o comenta la regla.
    "PGAV-IN": { "split": {
        # "amount": {"Amount_CLP": 20_000_000},
        # "factor": {"Factor": 5},
        # "number": {"Number": 800},
    }},
    "PGAV-OUT": { "split": {
        # "amount": {"Amount_CLP": 20_000_000},
        # "factor": {"Factor": 5},
        # "number": {"Number": 800},
    }},

    "OCMC_1": { "flat": {"Ceil": 35} },

    # === HANU* / HASU* ===
    "HANUMI": { "flat": {"Factor_ceiled": 51, "Number_ceiled": 2} },
    "HANUMO": { "flat": {"Factor_ceiled": 45, "Number_ceiled": 2} },
    "HASUMI": { "flat": {"Amount_S3": 509_895_324, "Factor_rec": 4} },
    "HASUMO": { "flat": {"Amount_S3": 188_002_425, "Factor_rec": 68} },

    "P-1st":  { "flat": {"Amount_CLP": 244_844_689} },
    "P-2nd":  { "flat": {"Amount_CLP": 244_844_689} },

    # STRI/STRO – contadores
    "STRINCLP": { "flat": {"X_candidatos": 3} },
    "STROTCLP": { "flat": {"X_candidatos": 3} },
    "STRINEUR": { "flat": {"X_candidatos": 3} },
    "STRINUSD": { "flat": {"X_candidatos": 3} },
    "STROTEUR": { "flat": {"X_candidatos": 3} },
    "STROTUSD": { "flat": {"X_candidatos": 3} },

    "IN-OUT-1 Amount": { "flat": {"Amount_CLP": 43_785_100, "Number": 3, "Percentage": 80} },

    "HNR-IN":  { "flat": {"Number_max30d": 18} },
    "HNR-OUT": { "flat": {"Number_max30d": 26} },

    "NUMCCI":  { "flat": {"Number_ceil": 2} },
    # "NUMCCO": no definido (NA) – si quieres fijarlo: {"flat": {"Number_ceil": <valor>}}
}

# === LOGICA ===================================================================

def _ensure_rule_exists(bundle: dict, rule: str):
    bundle.setdefault("rules", {})
    bundle["rules"].setdefault(rule, {})

def _ensure_flat_percentiles(rule_obj: dict, percentiles: list[str]):
    """Asegura la estructura 'percentiles': [ {percentil: ...}, ... ] para reglas 'flat'."""
    if "percentiles" not in rule_obj or not isinstance(rule_obj["percentiles"], list):
        rule_obj["percentiles"] = [{"percentil": p} for p in percentiles]
    else:
        # Asegura que cada entrada tenga 'percentil'; si no, las emparejamos por orden
        have = [r.get("percentil") for r in rule_obj["percentiles"]]
        if not all(isinstance(x, str) for x in have):
            # reindex por orden
            for i, p in enumerate(percentiles):
                if i < len(rule_obj["percentiles"]):
                    rule_obj["percentiles"][i]["percentil"] = rule_obj["percentiles"][i].get("percentil", p)

def _inject_flat(rule_obj: dict, fields: dict, percentiles_for_new=PCTS_ALL):
    """
    Inyecta campos fijos en una regla 'flat' (estructura: {"percentiles":[{...}]})
    Creará la estructura si no existe.
    """
    _ensure_flat_percentiles(rule_obj, percentiles_for_new)
    for row in rule_obj["percentiles"]:
        for k, v in fields.items():
            row.setdefault(k, v)

def _ensure_split_table(rule_obj: dict, table: str, percentiles: list[str]):
    """
    Asegura estructura split:
        rule_obj[table] = { "percentiles": [ { "percentil": ... }, ... ] }
    """
    rule_obj.setdefault(table, {})
    if "percentiles" not in rule_obj[table] or not isinstance(rule_obj[table]["percentiles"], list):
        rule_obj[table]["percentiles"] = [{"percentil": p} for p in percentiles]
    else:
        have = [r.get("percentil") for r in rule_obj[table]["percentiles"]]
        if not all(isinstance(x, str) for x in have):
            for i, p in enumerate(percentiles):
                if i < len(rule_obj[table]["percentiles"]):
                    rule_obj[table]["percentiles"][i]["percentil"] = rule_obj[table]["percentiles"][i].get("percentil", p)

def _inject_split(rule_obj: dict, split_fields: dict[str, dict], pcts_for_new=PCTS_ALL):
    """
    Inyecta campos fijos en tablas split (amount/factor/number).
    - Si la tabla existe, agrega los campos faltantes a CADA percentil ya presente.
    - Si no existe, la crea con todos los percentiles de pcts_for_new.
    """
    for table, fields in split_fields.items():
        _ensure_split_table(rule_obj, table, pcts_for_new)
        rows = rule_obj[table]["percentiles"]
        for row in rows:
            for k, v in fields.items():
                row.setdefault(k, v)

def _collect_existing_percentiles(rule_obj: dict) -> list[str]:
    """
    Si la regla ya tiene 'percentiles' (flat), toma esos percentiles.
    Si tiene split tables, intenta colectar la unión de percentiles por tabla.
    Si no encuentra, devuelve PCTS_ALL.
    """
    if "percentiles" in rule_obj and isinstance(rule_obj["percentiles"], list):
        p = [r.get("percentil") for r in rule_obj["percentiles"] if isinstance(r, dict)]
        p = [x for x in p if isinstance(x, str)]
        return p if p else PCTS_ALL

    # split
    pset = set()
    for t in ("amount","factor","number"):
        if t in rule_obj and isinstance(rule_obj[t], dict):
            rows = rule_obj[t].get("percentiles", [])
            for r in rows:
                p = r.get("percentil")
                if isinstance(p, str):
                    pset.add(p)
    return sorted(pset) if pset else PCTS_ALL

def augment_bundle(bundle: dict, defaults: dict) -> dict:
    out = deepcopy(bundle)

    for rule, cfg in defaults.items():
        _ensure_rule_exists(out, rule)
        rule_obj = out["rules"][rule]
        pcts_present = _collect_existing_percentiles(rule_obj)

        # Caso split
        if "split" in cfg and isinstance(cfg["split"], dict) and cfg["split"]:
            _inject_split(rule_obj, cfg["split"], pcts_present if pcts_present else PCTS_ALL)

        # Caso flat
        if "flat" in cfg and isinstance(cfg["flat"], dict) and cfg["flat"]:
            # Si la regla ya es split, y 'flat' viene, lo agregamos como 'percentiles' (sin romper split)
            if any(k in rule_obj for k in ("amount","factor","number")):
                # Agregamos/mezclamos una tabla adicional plana
                _ensure_flat_percentiles(rule_obj, pcts_present if pcts_present else PCTS_ALL)
                _inject_flat(rule_obj, cfg["flat"], pcts_present if pcts_present else PCTS_ALL)
            else:
                _inject_flat(rule_obj, cfg["flat"], pcts_present if pcts_present else PCTS_ALL)

        # Si no trae ni split ni flat (dict vacío), no hacemos nada: regla “documentada” pero sin defaults.

    return out

# === RUN ======================================================================

if __name__ == "__main__":
    with open(BUNDLE_IN, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    augmented = augment_bundle(bundle, CURRENT_DEFAULTS)

    BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(BUNDLE_OUT, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print("✔ Bundle augmentado.")
    print(f"  IN : {BUNDLE_IN}")
    print(f"  OUT: {BUNDLE_OUT}")
