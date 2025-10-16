# p_hsumi_sim.py
from __future__ import annotations
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np

from utils import load_tx_base, filter_subsubs, restrict_counts_after

def _to_dt_utc(s):
    """Convierte a datetime UTC de forma segura."""
    return pd.to_datetime(s, errors="coerce", utc=True)

def simulate_p_hsumi(
    tx_path: str,
    *,
    subsubs: Iterable[str] | str,
    scenarios: Dict[str, Dict[str, Any]],
    count_from: str = "2025-02-21",
) -> pd.DataFrame:
    """
    P-HSUMI — Gatillo por transacción:
      Regla: tx_direction = Inbound AND tx_type = Cash AND S30_after(tx) > Amount
      Gatillo: se cuenta la transacción que hace cruzar el umbral (S30_before <= A < S30_after)

    Notas:
      - El rolling de 30 días usa TODO el historial previo.
      - El conteo final considera sólo transacciones con tx_date_time >= count_from.
      - Se trabaja en UTC para evitar choques tz-aware/naive.

    Retorna:
      DataFrame con columnas: ['escenario', 'alertas']
    """
    # -------------------- Carga base y normalización --------------------
    df = load_tx_base(tx_path)
    df = filter_subsubs(df, subsubs)

    # Normaliza columnas usadas
    df["tx_direction"]   = df.get("tx_direction", "").astype(str).str.title()
    df["tx_type"]        = df.get("tx_type", "").astype(str).str.title()
    df["tx_base_amount"] = pd.to_numeric(df.get("tx_base_amount"), errors="coerce")
    df["tx_date_time"]   = _to_dt_utc(df.get("tx_date_time"))
    df["customer_id"]    = df.get("customer_id")

    # Filtro mínimo de elegibilidad (Inbound + Cash, fechas/montos válidos)
    m = (
        df["tx_direction"].eq("Inbound")
        & df["tx_type"].eq("Cash")
        & df["customer_id"].notna()
        & df["tx_date_time"].notna()
        & df["tx_base_amount"].notna()
    )
    base = df.loc[m, ["customer_id", "tx_date_time", "tx_base_amount"]].copy()

    if base.empty:
        # estructura de salida consistente si no hay datos
        return pd.DataFrame([{"escenario": k, "alertas": 0} for k in scenarios])

    # -------------------- Preparación por cliente -----------------------
    # (No cortamos por fecha aquí: el rolling necesita historial completo)
    # Se calcularán triggers por transacción y luego se filtrará por count_from.
    count_from_mask_dummy = restrict_counts_after(
        base.rename(columns={"tx_date_time": "_tmp_dt"}).assign(_tmp_dt=_to_dt_utc(base["tx_date_time"])),
        "_tmp_dt",
        count_from,
    )
    # Solo usamos count_from_mask_dummy para forzar la conversión interna a UTC sin errores.
    # El filtrado real lo aplicaremos a los gatillos, no a la base (para no romper el rolling).

    # -------------------- Función interna: triggers para un Amount -------
    def _triggers_for_amount(amount: float) -> pd.DataFrame:
        """
        Devuelve las transacciones (filas) que gatillan la alerta para el umbral 'amount'.
        Implementa:
          - S30_after = suma(|monto|) en (t-30d, t]
          - S30_before = S30_after - monto_actual
          - Gatillo si: S30_before <= amount < S30_after
        """
        parts = []
        # Procesar por cliente mantiene la lógica de ventanas separadas.
        for cid, sub in base.groupby("customer_id", sort=False):
            sub = sub.sort_values("tx_date_time").copy()

            # Serie por transacción, indexada por el timestamp de esa transacción
            # (puede haber duplicados de timestamp sin problema)
            amt_abs = sub["tx_base_amount"].abs().astype(float).values
            s = pd.Series(amt_abs, index=sub["tx_date_time"].values)

            # Rolling de 30 días basado en tiempo; resultado alineado por posición
            S30_after = s.rolling("30D").sum().values
            S30_before = S30_after - amt_abs  # quita el aporte de la transacción actual

            sub["S30_after"]  = S30_after
            sub["S30_before"] = S30_before

            mask = (sub["S30_before"] <= amount) & (sub["S30_after"] > amount)
            if mask.any():
                parts.append(sub.loc[mask, ["customer_id", "tx_date_time", "tx_base_amount"]])

        if not parts:
            return pd.DataFrame(columns=["customer_id", "tx_date_time", "tx_base_amount"])

        return pd.concat(parts, ignore_index=True)

    # -------------------- Ejecutar escenarios ---------------------------
    # Filtro final por fecha de la transacción gatillo (no del historial)
    out_rows = []
    count_from_ts = pd.to_datetime(count_from, utc=True)

    for name, pars in scenarios.items():
        A = float(pars.get("Amount", 0.0))
        trig = _triggers_for_amount(A)

        if trig.empty:
            out_rows.append({"escenario": name, "alertas": 0})
            continue

        # Asegurar tz UTC y filtrar solo desde count_from en adelante
        trig["tx_date_time"] = _to_dt_utc(trig["tx_date_time"])
        trig = trig[trig["tx_date_time"] >= count_from_ts]

        out_rows.append({"escenario": name, "alertas": int(trig.shape[0])})

    return pd.DataFrame(out_rows)
