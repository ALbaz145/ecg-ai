"""Pruebas de persistencia: flujo guardar → leer (con el doble de DB).

Demuestra que el resultado de ``POST /ecg/analyze`` queda
almacenado y es consultable con ``GET /ecg/{id}`` — la
misma lógica de la API, con un doble en memoria del
repositorio (no se requiere PostgreSQL).
"""

from __future__ import annotations

import uuid

from tests.backend.synthetic import synthetic_ecg


def test_guardar_y_leer_por_id_analisis(client) -> None:
    """Guardar → el id devuelto permite leer el análisis."""
    signal = synthetic_ecg(bpm=120)
    response = client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": 500},
    )
    assert response.status_code == 200
    creado = response.json()

    obtenido = client.get(f"/ecg/{creado['id']}")
    assert obtenido.status_code == 200
    body = obtenido.json()
    assert body["id"] == creado["id"]
    assert body["ecg_record_id"] == creado["ecg_record_id"]
    assert body["anomalia"] == creado["anomalia"] == "taquicardia"
    assert body["confianza"] == creado["confianza"]
    assert body["zonas_anomalas"] == creado["zonas_anomalas"]
    assert body["model_version"] == creado["model_version"]


def test_guardar_y_leer_por_id_registro(client) -> None:
    """``GET /ecg/{ecg_record_id}`` devuelve el análisis del registro."""
    signal = synthetic_ecg(bpm=70)
    creado = client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": 500},
    ).json()

    obtenido = client.get(f"/ecg/{creado['ecg_record_id']}")
    assert obtenido.status_code == 200
    assert obtenido.json()["id"] == creado["id"]


def test_segundo_analisis_es_nuevo_registro(client) -> None:
    """Dos análisis → dos registros distintos y consultables."""
    signal = synthetic_ecg()
    primero = client.post(
        "/ecg/analyze", json={"signal": signal, "sample_rate_hz": 500}
    ).json()
    segundo = client.post(
        "/ecg/analyze", json={"signal": signal, "sample_rate_hz": 250}
    ).json()

    assert primero["id"] != segundo["id"]
    assert primero["ecg_record_id"] != segundo["ecg_record_id"]
    for analisis in (primero, segundo):
        resp = client.get(f"/ecg/{analisis['id']}")
        assert resp.status_code == 200
        assert resp.json() == analisis


def test_get_analisis_inexistente_404(client) -> None:
    """UUID válido pero no almacenado → 404."""
    response = client.get(f"/ecg/{uuid.uuid4()}")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_get_id_no_uuid_404(client) -> None:
    """Identificador con formato inválido → 404 (no existe)."""
    response = client.get("/ecg/no-es-un-uuid")
    assert response.status_code == 404


def test_metadatos_guardados_en_el_doble(fake_repository, client) -> None:
    """El registro guarda los metadatos de la señal (contrato docs/DB.md)."""
    signal = synthetic_ecg()
    creado = client.post(
        "/ecg/analyze",
        json={
            "signal": signal,
            "sample_rate_hz": 500,
            "units": "uV",
            "filename": "sintetica_uv.csv",
        },
    ).json()

    record_id = uuid.UUID(creado["ecg_record_id"])
    record = fake_repository.records[record_id]
    assert record["filename"] == "sintetica_uv.csv"
    assert record["signal_metadata"] == {
        "sample_rate_hz": 500,
        "n_channels": 1,
        "units": "uV",
    }

    analysis = fake_repository.analyses[uuid.UUID(creado["id"])]
    assert analysis["model_version"]
    assert 0.0 <= analysis["confianza"] <= 1.0
