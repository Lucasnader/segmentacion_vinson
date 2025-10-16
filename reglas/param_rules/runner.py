# runner.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from out_avg import run_parameters_out_avg
from in_avg  import run_parameters_in_avg
from hnr_in  import run_parameters_hnr_in
from hnr_out import run_parameters_hnr_out
from hanumi import run_parameters_hanumi
from hanumo import run_parameters_hanumo
from hasumi import run_parameters_hasumi
from hasumo import run_parameters_hasumo
from in_gt_out import run_parameters_in_gt_out
from out_gt_in import run_parameters_out_gt_in
from in_out_1 import run_parameters_in_out_1_amount
from numcci import run_parameters_numcci
from numcco import run_parameters_numcco
from ocmc_1 import run_parameters_ocmc_1
from p_pct_bal import run_parameters_p_pct_bal
from p_1st import run_parameters_p_first
from p_2nd import run_parameters_p_second
from p_hsumi import run_parameters_p_hsumi
from p_hsumo import run_parameters_p_hsumo
from p_hvi import run_parameters_p_hvi
from p_hvo import run_parameters_p_hvo
from p_lbal import run_parameters_p_lbal
from p_lval import run_parameters_p_lval
from p_tli import run_parameters_p_tli
from p_tlo import run_parameters_p_tlo
from pgav_in import run_parameters_pgav_in
from pgav_out import run_parameters_pgav_out
from rvt_in import run_parameters_rvt_in
from rvt_out import run_parameters_rvt_out
from strinclp import run_parameters_strinclp
from strineur import run_parameters_strineur
from strinusd import run_parameters_strinusd
from strotclp import run_parameters_strotclp
from stroteur import run_parameters_stroteur
from strotusd import run_parameters_strotusd
from sumcci import run_parameters_sumcci
from sumcco import run_parameters_sumcco

# --- Helpers de guardado -------------------------------------------------------
import json
from datetime import datetime

def _sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("%", "pct")
        .replace("/", "_")
        .replace("—", "-")
        .replace("–", "-")
    )

def _df_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DF igual a 'df' pero con columnas numéricas convertidas a float,
    dejando intactas las columnas de texto (como customer_sub_type).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c.lower() == "percentil":
            continue
        # intento de parseo numérico (quitando separador de miles)
        raw = out[c].astype(str).str.replace(",", "")
        as_num = pd.to_numeric(raw, errors="coerce")
        # Heurística: si al menos el 50% son números, tratamos la columna como numérica
        if as_num.notna().sum() >= (len(as_num) * 0.5):
            out[c] = as_num
        # si no, dejamos la columna como está (texto)
    return out


def _bundle_from_results(results: dict, subsub: str, tx_path: str) -> dict:
    """
    Convierte `results` (DFs ya formateados para display) a un JSON
    con valores NUMÉRICOS por percentil/regla. Soporta que un valor sea
    un DataFrame o un dict de DataFrames.
    """
    bundle = {
        "meta": {
            "subsubsegment": subsub,
            "tx_path": tx_path,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "rules": {}
    }

    def df_to_dict(df: pd.DataFrame) -> dict:
        """
        Convierte un DataFrame (formateado para display) a una lista de filas JSON.
        - Intenta parsear cada celda a número (quitando comas).
        - Si no puede, la deja como string.
        - 'percentil' siempre se serializa como string.
        """
        rows = []
        for _, r in df.iterrows():
            row = {}
            for k, v in r.items():
                if k.lower() == "percentil":
                    row["percentil"] = "" if pd.isna(v) else str(v)
                    continue

                # Intento de parseo numérico por celda
                s = "" if pd.isna(v) else str(v)
                if s != "":
                    # quita separadores de miles si los hubiera
                    num = pd.to_numeric(s.replace(",", ""), errors="coerce")
                else:
                    num = pd.NA

                if pd.notna(num):
                    row[k] = float(num)
                else:
                    # deja el valor como string (etiquetas de grupo, etc.)
                    row[k] = s
            rows.append(row)
        return {"percentiles": rows}


    for rule_name, obj in results.items():
        if isinstance(obj, dict):  # ej. reglas que devuelven {"number": DF, "amount": DF}
            bundle["rules"][rule_name] = {}
            for subname, df in obj.items():
                bundle["rules"][rule_name][_sanitize_name(subname)] = df_to_dict(df)
        else:  # DataFrame
            bundle["rules"][rule_name] = df_to_dict(obj)

    return bundle

def save_results_bundle(results: dict, out_dir: Path, subsub: str, tx_path: str) -> None:
    """
    Guarda:
      - Un CSV por regla (o subtabla) con valores numéricos
      - Un JSON maestro con todo centralizado
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CSVs numéricos
    # for rule_name, obj in results.items():
    #     base = _sanitize_name(rule_name)
    #     if isinstance(obj, dict):
    #         for subname, df in obj.items():
    #             p = out_dir / f"{base}__{_sanitize_name(subname)}.csv"
    #             _df_to_numeric(df).to_csv(p, index=False, encoding="utf-8-sig")
    #     else:
    #         p = out_dir / f"{base}.csv"
    #         _df_to_numeric(obj).to_csv(p, index=False, encoding="utf-8-sig")

    # 2) JSON maestro
    bundle = _bundle_from_results(results, subsub=subsub, tx_path=tx_path)
    json_path = out_dir / f"params_{_sanitize_name(subsub)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"\n✔ Parámetros guardados en: {out_dir}")
    print(f"   - JSON maestro: {json_path.name}")
    print(f"   - CSV por regla (n={len(list(out_dir.glob('*.csv')))} archivos)")

def _fmt_thousands(v, decimals=0):
    if pd.isna(v): return ""
    if decimals == 0:
        return f"{float(v):,.0f}"
    return f"{float(v):,.{decimals}f}"

def _format_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c.lower() == "percentil":
            continue

        # Parseo numérico suave (quitando comas). No toca el DF aún.
        parsed = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")
        mask = parsed.notna()

        if mask.any():
            # Para poder asignar strings sin warnings, forzamos dtype object
            out[c] = out[c].astype(object)

            if "Factor" in c:
                out.loc[mask, c] = parsed[mask].map(lambda x: _fmt_thousands(x, 2))
            else:
                out.loc[mask, c] = parsed[mask].map(lambda x: _fmt_thousands(x, 0))
        # Si no hay ningún numérico en la columna, la dejamos tal cual (p. ej., labels de grupo)

    return out


def _format_percentiles_any(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return _format_percentiles(x)
    if isinstance(x, dict):
        return {k: _format_percentiles(v) if isinstance(v, pd.DataFrame) else v for k, v in x.items()}
    return x

def _print_result_block(name: str, obj):
    print(f"\n— {name} —")
    if isinstance(obj, pd.DataFrame):
        print(obj.to_string(index=False))
    elif isinstance(obj, dict):
        # p. ej. {"number": df1, "amount": df2}
        for subname, subobj in obj.items():
            print(f"\n  [{subname}]")
            if isinstance(subobj, pd.DataFrame):
                print(subobj.to_string(index=False))
            else:
                print(str(subobj))
    else:
        print(str(obj))

def run_parametrization(tx_path: str, subsub: str):
    results = {}

    results["OUT>AVG"] = _format_percentiles(run_parameters_out_avg(tx_path, subsubsegments=subsub, verbose=False)["percentiles"])
    print("OUT>AVG done.")
    results["IN>AVG"]  = _format_percentiles(run_parameters_in_avg(tx_path,  subsubsegments=subsub, verbose=False)["percentiles"])
    print("IN>AVG done.")
    results["HNR-IN"]  = _format_percentiles(run_parameters_hnr_in(tx_path,  subsubsegments=subsub, verbose=False)["percentiles"])
    print("HNR-IN done.")
    results["HNR-OUT"] = _format_percentiles(run_parameters_hnr_out(tx_path, subsubsegments=subsub, verbose=False)["percentiles"])
    print("HNR-OUT done.")
    results["HANUMI"]  = _format_percentiles(run_parameters_hanumi(tx_path, subsubsegments=subsub)["percentiles"])
    print("HANUMI done.")
    results["HANUMO"]  = _format_percentiles(run_parameters_hanumo(tx_path, subsubsegments=subsub)["percentiles"])
    print("HANUMO done.")
    results["HASUMI"]  = _format_percentiles(run_parameters_hasumi(tx_path, subsubsegments=subsub)["percentiles"])
    print("HASUMI done.")
    results["HASUMO"]  = _format_percentiles(run_parameters_hasumo(tx_path, subsubsegments=subsub)["percentiles"])
    print("HASUMO done.")
    results["IN>%OUT"] = _format_percentiles(run_parameters_in_gt_out(tx_path, subsubsegments=subsub)["percentiles"])
    print("IN>%OUT done.")
    results["OUT>%IN"] = _format_percentiles(run_parameters_out_gt_in(tx_path, subsubsegments=subsub)["percentiles"])
    print("OUT>%IN done.")
    results["IN-OUT-1 Amount"] = _format_percentiles(run_parameters_in_out_1_amount(tx_path, subsubsegments=subsub)["percentiles"])
    print("IN-OUT-1 Amount done.")
    results["NUMCCI"]  = _format_percentiles(run_parameters_numcci(tx_path, subsubsegments=subsub)["percentiles"])
    print("NUMCCI done.")
    results["NUMCCO"]  = _format_percentiles(run_parameters_numcco(tx_path, subsubsegments=subsub)["percentiles"])
    print("NUMCCO done.")
    results["OCMC_1"]  = _format_percentiles(run_parameters_ocmc_1(tx_path, subsubsegments=subsub)["percentiles"])
    print("OCMC_1 done.")
    results["P-%BAL"]  = _format_percentiles(run_parameters_p_pct_bal(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-%BAL done.")
    results["P-1st"]  = _format_percentiles(run_parameters_p_first(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-1st done.")
    results["P-2nd"]  = _format_percentiles(run_parameters_p_second(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-2nd done.")
    results["P-HSUMI"]= _format_percentiles(run_parameters_p_hsumi(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-HSUMI done.")
    results["P-HSUMO"]= _format_percentiles(run_parameters_p_hsumo(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-HSUMO done.")
    results["P-HVI"]  = _format_percentiles(run_parameters_p_hvi(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-HVI done.")
    results["P-HVO"]  = _format_percentiles(run_parameters_p_hvo(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-HVO done.")
    results["P-LBAL"] = _format_percentiles(run_parameters_p_lbal(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-LBAL done.")
    results["P-LVAL"] = _format_percentiles(run_parameters_p_lval(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-LVAL done.")
    results["P-TLI"]  = _format_percentiles(run_parameters_p_tli(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-TLI done.")
    results["P-TLO"]  = _format_percentiles(run_parameters_p_tlo(tx_path, subsubsegments=subsub)["percentiles"])
    print("P-TLO done.")
    results["PGAV-IN"]= _format_percentiles_any(run_parameters_pgav_in(tx_path, subsubsegments=subsub)["percentiles"])
    print("PGAV-IN done.")
    results["PGAV-OUT"]= _format_percentiles_any(run_parameters_pgav_out(tx_path, subsubsegments=subsub)["percentiles"])
    print("PGAV-OUT done.")
    results["RVT-IN"] = _format_percentiles_any(run_parameters_rvt_in(tx_path, subsubsegments=subsub)["percentiles"])
    print("RVT-IN done.")
    results["RVT-OUT"]= _format_percentiles_any(run_parameters_rvt_out(tx_path, subsubsegments=subsub)["percentiles"])
    print("RVT-OUT done.")
    results["STRINCLP"]= _format_percentiles(run_parameters_strinclp(tx_path, subsubsegments=subsub)["percentiles"])
    print("STRINCLP done.")
    results["STRINEUR"]= _format_percentiles(run_parameters_strineur(tx_path, subsubsegments=subsub)["percentiles"])
    print("STRINEUR done.")
    results["STRINUSD"]= _format_percentiles(run_parameters_strinusd(tx_path, subsubsegments=subsub)["percentiles"])
    print("STRINUSD done.")
    results["STROTCLP"]= _format_percentiles(run_parameters_strotclp(tx_path, subsubsegments=subsub)["percentiles"])
    print("STROTCLP done.")
    results["STROTEUR"]= _format_percentiles(run_parameters_stroteur(tx_path, subsubsegments=subsub)["percentiles"])
    print("STROTEUR done.")
    results["STROTUSD"]= _format_percentiles(run_parameters_strotusd(tx_path, subsubsegments=subsub)["percentiles"])
    print("STROTUSD done.")
    results["SUMCCI"] = _format_percentiles(run_parameters_sumcci(tx_path, subsubsegments=subsub)["percentiles"])
    print("SUMCCI done.")
    results["SUMCCO"] = _format_percentiles(run_parameters_sumcco(tx_path, subsubsegments=subsub)["percentiles"])
    print("SUMCCO done.")


    # ---- Salida única, limpia ----
    print(f"\n=== Parametrización — sub-subsegmento: {subsub} ===")
    for name, obj in results.items():
        _print_result_block(name, obj)

    return results

if __name__ == "__main__": 
    # EDITA SOLO ESTAS DOS VARIABLES
    SUBSUB   = "I-2"
    THIS_DIR = Path(__file__).resolve().parent
    ROOT     = THIS_DIR.parents[1]
    TX_PATH  = ROOT / "data" / "tx" / "datos_trx__with_subsub.csv"

    if not TX_PATH.exists():
        raise FileNotFoundError(f"No encuentro el CSV en: {TX_PATH}")

    res = run_parametrization(str(TX_PATH), SUBSUB)

    # Guarda todo en carpeta de salida (puedes cambiar esta ruta si quieres)
    OUT_DIR = ROOT / "outputs" / "params" / SUBSUB
    save_results_bundle(res, OUT_DIR, subsub=SUBSUB, tx_path=str(TX_PATH))
