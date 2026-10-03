"""Pruebas de ``POST /ecg/analyze``: validación, contrato y modelo."""

from __future__ import annotations

import math
import uuid

import pytest

from tests.backend.synthetic import synthetic_ecg

CONTRACT_KEYS = {
    "id",
    "ecg_record_id",
    "anomalia",
    "confianza",
    "zonas_anomalas",
    "model_version",
}


def _analyze(client, signal, sample_rate_hz=500, units="mV"):
    return client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": sample_rate_hz, "units": units},
    )


# ---------------------------------------------------------------------------
# Contrato y casos felices
# ---------------------------------------------------------------------------


def test_senal_normal_200(client) -> None:
    """Señal válida → 200 con anomalía + confianza ∈ [0,1]."""
    signal = synthetic_ecg(bpm=70)
    response = _analyze(client, signal)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == CONTRACT_KEYS
    assert body["anomalia"] == "normal"
    assert 0.0 <= body["confianza"] <= 1.0
    assert body["zonas_anomalas"] == []
    assert body["model_version"]
    # Los ids son UUID válidos.
    uuid.UUID(body["id"])
    uuid.UUID(body["ecg_record_id"])


def test_senal_taquicardia(client) -> None:
    signal = synthetic_ecg(bpm=120)
    response = _analyze(client, signal)
    assert response.status_code == 200
    assert response.json()["anomalia"] == "taquicardia"
    assert 0.0 <= response.json()["confianza"] <= 1.0


def test_senal_bradicardia(client) -> None:
    signal = synthetic_ecg(bpm=45)
    response = _analyze(client, signal)
    assert response.status_code == 200
    assert response.json()["anomalia"] == "bradicardia"


def test_senal_arritmia_con_zonas(client) -> None:
    """RR irregular → arritmia con zonas de latidos irregulares."""
    signal = synthetic_ecg(
        rr_pattern=[0.60, 1.00, 0.65, 0.95, 0.60, 1.00]
    )
    response = _analyze(client, signal)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "arritmia"
    assert body["zonas_anomalas"], "se esperaban zonas anómalas"
    n_samples = len(signal)
    for zona in body["zonas_anomalas"]:
        assert set(zona.keys()) == {"inicio", "fin", "tipo"}
        assert isinstance(zona["inicio"], int)
        assert isinstance(zona["fin"], int)
        assert 0 <= zona["inicio"] < zona["fin"] <= n_samples
        assert zona["tipo"]


def test_amplitud_alta(client) -> None:
    signal = synthetic_ecg(r_amplitude_mv=4.0)
    response = _analyze(client, signal)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "amplitud_anomala"
    assert body["zonas_anomalas"]


def test_amplitud_baja(client) -> None:
    signal = synthetic_ecg(r_amplitude_mv=0.05, noise_sigma_mv=0.005)
    response = _analyze(client, signal)
    assert response.status_code == 200
    assert response.json()["anomalia"] == "amplitud_anomala"


def test_ruido_excesivo(client) -> None:
    signal = synthetic_ecg(noise_sigma_mv=0.3)
    response = _analyze(client, signal)
    assert response.status_code == 200
    assert response.json()["anomalia"] == "ruido_excesivo"


def test_senal_plana_no_valorable(client) -> None:
    """Señal sin picos R → etiqueta 'senal_no_valorable', no 500."""
    response = _analyze(client, [0.0] * 300)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "senal_no_valorable"
    assert 0.0 <= body["confianza"] <= 1.0


def test_unidades_uv_equivalentes(client) -> None:
    """La misma señal en uV produce el mismo diagnóstico que en mV."""
    signal_mv = synthetic_ecg(bpm=120)
    signal_uv = [value * 1000.0 for value in signal_mv]
    resp_mv = _analyze(client, signal_mv, units="mV")
    resp_uv = _analyze(client, signal_uv, units="uV")
    assert resp_mv.status_code == resp_uv.status_code == 200
    assert resp_mv.json()["anomalia"] == resp_uv.json()["anomalia"]


def test_filename_opcional_se_acepta(client) -> None:
    signal = synthetic_ecg()
    response = client.post(
        "/ecg/analyze",
        json={
            "signal": signal,
            "sample_rate_hz": 500,
            "filename": "muestra_sintetica.csv",
        },
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Baja frecuencia de muestreo (D1): fs válida según el contrato
# (1-5000 Hz) que antes causaba 500 al construir el filtro.
# ---------------------------------------------------------------------------


def test_baja_frecuencia_muestreo_fs_50(client) -> None:
    """fs=50 Hz: la banda del filtro se adapta al Nyquist (25 Hz)."""
    signal = synthetic_ecg(sample_rate_hz=50, duration_s=10.0, bpm=120)
    response = _analyze(client, signal, sample_rate_hz=50)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "taquicardia"
    assert 0.0 <= body["confianza"] <= 1.0


def test_baja_frecuencia_muestreo_fs_90(client) -> None:
    """fs=90 Hz: borde alto clampado a 44.5 Hz (< Nyquist)."""
    signal = synthetic_ecg(sample_rate_hz=90, duration_s=10.0, bpm=70)
    response = _analyze(client, signal, sample_rate_hz=90)
    assert response.status_code == 200
    assert response.json()["anomalia"] == "normal"


@pytest.mark.parametrize("fs", [1, 2, 3])
def test_frecuencia_muestreo_minima_respuesta_valida(client, fs) -> None:
    """fs muy baja (1-3 Hz): respuesta 200 válida, nunca 500.

    La señal no puede representar un latido (el QRS es más
    estrecho que una muestra), así que la etiqueta es la
    mejor suposición de la heurística (ver docs/MODEL.md).
    """
    signal = synthetic_ecg(sample_rate_hz=fs, duration_s=150.0 / fs)
    assert len(signal) >= 100
    response = _analyze(client, signal, sample_rate_hz=fs)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["confianza"] <= 1.0
    assert isinstance(body["anomalia"], str) and body["anomalia"]


# ---------------------------------------------------------------------------
# Validación (422)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # Señal vacía
        {"signal": [], "sample_rate_hz": 500},
        # Demasiado corta (< 100 muestras)
        {"signal": [0.1] * 99, "sample_rate_hz": 500},
        # Valores no numéricos
        {"signal": [0.1, "abc", 0.2] * 50, "sample_rate_hz": 500},
        # Frecuencia de muestreo inválida
        {"signal": [0.1] * 100, "sample_rate_hz": 0},
        {"signal": [0.1] * 100, "sample_rate_hz": 100000},
        {"signal": [0.1] * 100, "sample_rate_hz": "abc"},
        # Unidades no soportadas
        {"signal": [0.1] * 100, "sample_rate_hz": 500, "units": "V"},
        # filename vacío
        {"signal": [0.1] * 100, "sample_rate_hz": 500, "filename": "  "},
        # Faltan campos obligatorios
        {"signal": [0.1] * 100},
        {},
        # NaN / infinitos como JSON crudo (httpx no los
        # serializa: se envía el literal NaN del estándar
        # extendido que Python acepta parsear).
        '{"signal": [' + "NaN," * 100 + '], "sample_rate_hz": 500}',
        '{"signal": [' + "Infinity," * 100 + '], "sample_rate_hz": 500}',
    ],
)
def test_senal_invalida_422(client, payload) -> None:
    if isinstance(payload, str):
        response = client.post(
            "/ecg/analyze",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.post("/ecg/analyze", json=payload)
    assert response.status_code == 422


def test_confianza_siempre_en_rango(client) -> None:
    """Para toda familia de señales, la confianza está en [0,1]."""
    cases = [
        synthetic_ecg(bpm=70),
        synthetic_ecg(bpm=120),
        synthetic_ecg(bpm=45),
        synthetic_ecg(rr_pattern=[0.6, 1.0, 0.65, 0.95]),
        synthetic_ecg(r_amplitude_mv=5.0),
        synthetic_ecg(noise_sigma_mv=0.4),
        [0.0] * 200,
    ]
    for signal in cases:
        response = _analyze(client, signal)
        assert response.status_code == 200
        confianza = response.json()["confianza"]
        assert 0.0 <= confianza <= 1.0
        assert math.isfinite(confianza)
