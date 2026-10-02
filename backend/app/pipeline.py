"""Pipeline de análisis: carga → filtrado → segmentación → modelo → resultado.

Orquesta los pasos del flujo del MVP:

1. **Carga/validación** (Pydantic en el router; aquí
   conversión de unidades).
2. **Filtrado** pasa-banda y **normalización** (``app/signal``).
3. **Segmentación** en latidos y features.
4. **Modelo** heurístico (``app/model``).
5. **Persistencia** en PostgreSQL (``app/repository``).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from app.model import MODEL_VERSION, classify
from app.repository import EcgRepository, StoredAnalysis
from app.signal import (
    bandpass_filter,
    detect_r_peaks,
    extract_features,
    to_millivolts,
    zscore,
)

#: Separadores aceptados en un CSV (coma, punto y coma, tab).
_CSV_SEPARATORS = re.compile(r"[,;\t]")


def parse_csv_values(text: str) -> list[float]:
    """Parsea un CSV de **una columna** de valores.

    Las filas no numéricas (p. ej. una cabecera) se
    ignoran; si ninguna fila es numérica la lista queda
    vacía y la validación de Pydantic la rechaza (422).
    """
    values: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        first = _CSV_SEPARATORS.split(line)[0].strip()
        if not first:
            continue
        try:
            values.append(float(first))
        except ValueError:
            continue
    return values


def _default_filename(units: str) -> str:
    """Nombre de archivo por defecto (la DB exige no nulo).

    Incluye la unidad de la señal (p. ej. ``..._mV.json``)
    para que el nombre sea informativo aunque la petición
    no lo haya proporcionado.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"api_{stamp}_{uuid.uuid4().hex[:8]}_{units}.json"


def run_analysis(
    *,
    signal: list[float],
    sample_rate_hz: int,
    units: str,
    filename: str | None,
    repository: EcgRepository,
) -> StoredAnalysis:
    """Ejecuta el pipeline completo y persiste el resultado."""
    # --- 1. Carga: conversión de unidades a mV -------------
    original_mv = to_millivolts(signal, units)

    # --- 2. Filtrado y normalización -----------------------
    filtered = bandpass_filter(original_mv, float(sample_rate_hz))
    normalized = zscore(filtered)

    # --- 3. Segmentación en latidos y features -------------
    r_peaks = detect_r_peaks(normalized, float(sample_rate_hz))
    features = extract_features(
        original_mv=original_mv,
        filtered=filtered,
        r_peaks=r_peaks,
        fs=float(sample_rate_hz),
        sample_rate_hz=sample_rate_hz,
    )

    # --- 4. Modelo -----------------------------------------
    detection = classify(features)

    # --- 5. Persistencia -----------------------------------
    return repository.save_analysis(
        filename=filename if filename else _default_filename(units),
        signal_metadata={
            "sample_rate_hz": sample_rate_hz,
            "n_channels": 1,
            "units": units,
        },
        anomalia=detection.anomalia,
        confianza=round(float(detection.confianza), 3),
        model_version=MODEL_VERSION,
        zonas_anomalas=[
            {"inicio": int(z["inicio"]), "fin": int(z["fin"]), "tipo": str(z["tipo"])}
            for z in detection.zonas
        ],
    )


def analyze_result(stored: StoredAnalysis) -> dict[str, Any]:
    """Serializa un ``StoredAnalysis`` para la respuesta."""
    return {
        "id": str(stored.id),
        "ecg_record_id": str(stored.ecg_record_id),
        "anomalia": stored.anomalia,
        "confianza": float(stored.confianza),
        "zonas_anomalas": stored.zonas_anomalas,
        "model_version": stored.model_version,
    }
