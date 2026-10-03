"""Pruebas unitarias del modelo heurístico (``app/model.py``).

Cada rama se prueba con ``SignalFeatures`` construidas a
mano (sin generar señales): verifica la regla, la
confianza ∈ [0,1] y la forma de las zonas.
"""

from __future__ import annotations

import numpy as np
from app.model import (
    LABEL_AMPLITUD,
    LABEL_ARRHYTHMIA,
    LABEL_BRADYCARDIA,
    LABEL_NO_VALORABLE,
    LABEL_NORMAL,
    LABEL_RUIDO,
    LABEL_TACHYCARDIA,
    MODEL_VERSION,
    classify,
)
from app.signal import SignalFeatures


def _features(
    n_beats: int = 10,
    heart_rate_bpm: float | None = 70.0,
    rr_cv: float | None = 0.02,
    r_amplitude_mv: float | None = 1.0,
    noise_ratio: float = 0.05,
    n_samples: int = 5000,
) -> SignalFeatures:
    """Construye features sintéticas para probar reglas."""
    n = max(n_beats, 0)
    ventanas = [
        (i * 50, i * 50 + 80) for i in range(n)
    ]
    amplitudes = [r_amplitude_mv or 1.0] * n
    return SignalFeatures(
        n_samples=n_samples,
        sample_rate_hz=500,
        duration_s=n_samples / 500,
        n_beats=n,
        heart_rate_bpm=heart_rate_bpm,
        rr_cv=rr_cv,
        r_amplitude_mv=r_amplitude_mv,
        noise_ratio=noise_ratio,
        r_peak_indices=np.array([i * 50 for i in range(n)]),
        beat_windows=ventanas,
        beat_amplitudes_mv=amplitudes,
    )


def _check_zonas(zonas, n_samples: int) -> None:
    for zona in zonas:
        assert isinstance(zona["inicio"], int)
        isinstance(zona["fin"], int)
        assert 0 <= zona["inicio"] < zona["fin"] <= n_samples
        assert isinstance(zona["tipo"], str) and zona["tipo"]


def test_senal_no_valorable() -> None:
    deteccion = classify(_features(n_beats=1))
    assert deteccion.anomalia == LABEL_NO_VALORABLE
    assert deteccion.zonas == []
    assert 0.0 <= deteccion.confianza <= 1.0


def test_ruido_excesivo() -> None:
    deteccion = classify(_features(noise_ratio=0.5))
    assert deteccion.anomalia == LABEL_RUIDO
    assert deteccion.zonas, "el ruido cubre toda la señal"
    assert deteccion.zonas == [
        {"inicio": 0, "fin": 5000, "tipo": LABEL_RUIDO}
    ]
    _check_zonas(deteccion.zonas, 5000)
    assert 0.0 <= deteccion.confianza <= 1.0


def test_arritmia() -> None:
    deteccion = classify(_features(rr_cv=0.30, n_beats=10))
    assert deteccion.anomalia == LABEL_ARRHYTHMIA
    _check_zonas(deteccion.zonas, 5000)
    assert 0.60 <= deteccion.confianza <= 0.92


def test_arritmia_requiere_latidos_suficientes() -> None:
    """CV alto con pocos latidos no alcanza para arritmia."""
    deteccion = classify(_features(rr_cv=0.30, n_beats=3))
    assert deteccion.anomalia != LABEL_ARRHYTHMIA


def test_taquicardia() -> None:
    deteccion = classify(_features(heart_rate_bpm=120.0))
    assert deteccion.anomalia == LABEL_TACHYCARDIA
    assert deteccion.zonas == [
        {"inicio": 0, "fin": 5000, "tipo": LABEL_TACHYCARDIA}
    ]
    assert 0.60 <= deteccion.confianza <= 0.95


def test_bradicardia() -> None:
    deteccion = classify(_features(heart_rate_bpm=45.0))
    assert deteccion.anomalia == LABEL_BRADYCARDIA
    assert deteccion.zonas == [
        {"inicio": 0, "fin": 5000, "tipo": LABEL_BRADYCARDIA}
    ]
    assert 0.60 <= deteccion.confianza <= 0.95


def test_amplitud_alta_con_zonas() -> None:
    deteccion = classify(_features(r_amplitude_mv=5.0))
    assert deteccion.anomalia == LABEL_AMPLITUD
    assert deteccion.zonas
    _check_zonas(deteccion.zonas, 5000)
    assert all(z["tipo"] == "amplitud_alta" for z in deteccion.zonas)


def test_amplitud_baja_con_zonas() -> None:
    deteccion = classify(_features(r_amplitude_mv=0.02))
    assert deteccion.anomalia == LABEL_AMPLITUD
    assert deteccion.zonas
    assert all(z["tipo"] == "amplitud_baja" for z in deteccion.zonas)


def test_normal() -> None:
    deteccion = classify(_features())
    assert deteccion.anomalia == LABEL_NORMAL
    assert deteccion.zonas == []
    assert 0.50 <= deteccion.confianza <= 0.90


def test_version_del_modelo_constante() -> None:
    assert isinstance(MODEL_VERSION, str) and MODEL_VERSION


def test_confianza_acotada_en_todos_los_casos() -> None:
    casos = [
        _features(n_beats=1),
        _features(noise_ratio=0.9),
        _features(rr_cv=0.8, n_beats=20),
        _features(heart_rate_bpm=200.0),
        _features(heart_rate_bpm=20.0),
        _features(r_amplitude_mv=50.0),
        _features(r_amplitude_mv=0.001),
        _features(),
    ]
    for features in casos:
        deteccion = classify(features)
        assert 0.0 <= deteccion.confianza <= 1.0
        _check_zonas(deteccion.zonas, features.n_samples)
