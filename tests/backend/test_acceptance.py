"""Verificación explícita de los criterios de aceptación (issue #5).

Se ejecuta con el doble de repositorio (no requiere PostgreSQL):
    python3 -m pytest tests/backend/test_acceptance.py -v
"""

from __future__ import annotations

import uuid

from tests.backend.synthetic import synthetic_ecg


def test_criterio_1_health_200(client) -> None:
    """[1] GET /health responde 200 {"status": "ok"}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_criterio_2_senal_valida(client) -> None:
    """[2] Señal válida → 200 con anomalia + confianza ∈ [0,1]."""
    signal = synthetic_ecg(bpm=110)
    response = client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": 500},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "taquicardia"
    assert 0.0 <= body["confianza"] <= 1.0


def test_criterio_3_senal_invalida_422(client) -> None:
    """[3] Señal inválida (vacía, corta, no numérica) → 422."""
    casos = [
        {"signal": [], "sample_rate_hz": 500},  # vacía
        {"signal": [0.1] * 99, "sample_rate_hz": 500},  # < 100
        {"signal": [0.1, "x"] * 50, "sample_rate_hz": 500},  # no numérico
    ]
    for payload in casos:
        response = client.post("/ecg/analyze", json=payload)
        assert response.status_code == 422, payload


def test_criterio_4_persistencia_guardar_leer(client) -> None:
    """[4] El resultado queda persistente y se lee con GET /ecg/{id}."""
    signal = synthetic_ecg(bpm=120)
    creado = client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": 500},
    ).json()

    response = client.get(f"/ecg/{creado['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == creado["id"]
    assert body["ecg_record_id"] == creado["ecg_record_id"]
    assert body["anomalia"] == creado["anomalia"]
    assert body["confianza"] == creado["confianza"]
    assert body["zonas_anomalas"] == creado["zonas_anomalas"]
    assert body["model_version"] == creado["model_version"]


def test_criterio_5_ids_y_formato_zonas(client) -> None:
    """[5] Formato exacto de la respuesta (ids UUID, zonas con ints)."""
    signal = synthetic_ecg(
        rr_pattern=[0.60, 1.00, 0.65, 0.95, 0.60, 1.00]
    )
    response = client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": 500},
    )
    assert response.status_code == 200
    body = response.json()
    uuid.UUID(body["id"])
    uuid.UUID(body["ecg_record_id"])
    assert body["zonas_anomalas"]
    for zona in body["zonas_anomalas"]:
        assert isinstance(zona["inicio"], int)
        assert isinstance(zona["fin"], int)
        assert isinstance(zona["tipo"], str)
        assert 0 <= zona["inicio"] < zona["fin"] <= len(signal)


def test_criterio_6_404_inexistente(client) -> None:
    """[6] GET /ecg/{id} inexistente → 404."""
    response = client.get(f"/ecg/{uuid.uuid4()}")
    assert response.status_code == 404
