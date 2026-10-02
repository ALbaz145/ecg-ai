"""Esquemas Pydantic: el contrato de la API con el frontend.

Proyecto escolar, NO uso clínico.

Convenciones:

- ``snake_case`` en todo (lenguaje del proyecto: español).
- La señal es una lista plana de floats (un solo canal,
  ``n_channels = 1`` en el MVP).
- ``zonas_anomalas`` usa **índices de muestra** (enteros,
  ``inicio`` inclusivo, ``fin`` exclusivo) sobre la señal
  tal como fue subida.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --- Validación de la señal (los límites son parte del contrato) ---
#: Mínimo de muestras aceptadas (definido por la tarea del MVP).
MIN_SIGNAL_SAMPLES = 100
#: Máximo de muestras aceptadas (a 500 Hz son ~66 minutos).
MAX_SIGNAL_SAMPLES = 2_000_000
#: Frecuencia de muestreo mínima (Hz).
MIN_SAMPLE_RATE_HZ = 1
#: Frecuencia de muestreo máxima (Hz) — ECG típico 250–1000 Hz.
MAX_SAMPLE_RATE_HZ = 5000


class AnalyzeRequest(BaseModel):
    """Petición de ``POST /ecg/analyze`` (JSON)."""

    signal: list[float] = Field(
        min_length=MIN_SIGNAL_SAMPLES,
        max_length=MAX_SIGNAL_SAMPLES,
        description="Muestras de la señal ECG (un solo canal).",
        examples=[[0.1, 0.2, 0.15]],
    )
    sample_rate_hz: int = Field(
        ge=MIN_SAMPLE_RATE_HZ,
        le=MAX_SAMPLE_RATE_HZ,
        description="Frecuencia de muestreo en Hz.",
        examples=[500],
    )
    units: Literal["mV", "uV"] = Field(
        default="mV",
        description="Unidades de la señal: milivoltios o microvoltios.",
    )
    filename: str | None = Field(
        default=None,
        max_length=255,
        description="Nombre de archivo opcional (solo informativo).",
    )

    @field_validator("signal")
    @classmethod
    def _signal_finita(cls, values: list[float]) -> list[float]:
        """JSON permite ``NaN``/``Infinity`` en Python; los rechazamos."""
        for value in values:
            if not math.isfinite(value):
                raise ValueError(
                    "signal solo acepta valores finitos (sin NaN ni infinitos)"
                )
        return values

    @field_validator("filename")
    @classmethod
    def _filename_no_vacio(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("filename no puede estar vacío")
        return value


class ZonaAnomala(BaseModel):
    """Zona anómala de la señal (índices de muestra)."""

    inicio: int = Field(ge=0, description="Índice de inicio (inclusivo).")
    fin: int = Field(ge=0, description="Índice de fin (exclusivo).")
    tipo: str = Field(min_length=1, max_length=120)


class AnalysisResponse(BaseModel):
    """Respuesta de ``POST /ecg/analyze`` y ``GET /ecg/{id}``."""

    id: str
    ecg_record_id: str
    anomalia: str
    confianza: float = Field(ge=0.0, le=1.0)
    zonas_anomalas: list[ZonaAnomala]
    model_version: str


class HealthResponse(BaseModel):
    """Respuesta de ``GET /health`` (healthcheck de docker-compose)."""

    status: str
