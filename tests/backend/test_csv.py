"""Pruebas de la subida de archivos CSV (``POST /ecg/analyze/file``)."""

from __future__ import annotations

import pytest

from tests.backend.synthetic import synthetic_ecg


def _csv(signal: list[float], header: bool = False) -> str:
    lines = (["tiempo,milivoltios"] if header else []) + [
        f"{value:.6f}" for value in signal
    ]
    return "\n".join(lines) + "\n"


def test_csv_sin_cabecera_200(client) -> None:
    """CSV de una columna (sin cabecera) → mismo análisis que JSON."""
    signal = synthetic_ecg(bpm=120)
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", _csv(signal), "text/csv")},
        data={"sample_rate_hz": 500, "units": "mV"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "taquicardia"
    assert 0.0 <= body["confianza"] <= 1.0


def test_csv_con_cabecera_200(client) -> None:
    """La primera fila no numérica (cabecera) se ignora."""
    signal = synthetic_ecg(bpm=70)
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", _csv(signal, header=True), "text/csv")},
        data={"sample_rate_hz": 500},
    )
    assert response.status_code == 200
    assert response.json()["anomalia"] == "normal"


def test_csv_usa_nombre_de_archivo(client) -> None:
    signal = synthetic_ecg()
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("mi_ecg.csv", _csv(signal), "text/csv")},
        data={"sample_rate_hz": 500},
    )
    assert response.status_code == 200


def test_csv_corto_422(client) -> None:
    """Menos de 100 muestras válidas → 422 (validación compartida)."""
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("corto.csv", _csv([0.1] * 50), "text/csv")},
        data={"sample_rate_hz": 500},
    )
    assert response.status_code == 422


def test_csv_vacio_422(client) -> None:
    """CSV sin valores numéricos → 422."""
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("vacio.csv", "hola\nmundo\n", "text/csv")},
        data={"sample_rate_hz": 500},
    )
    assert response.status_code == 422


def test_csv_sin_sample_rate_422(client) -> None:
    """Falta el campo obligatorio sample_rate_hz → 422."""
    signal = synthetic_ecg()
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", _csv(signal), "text/csv")},
        data={"units": "mV"},
    )
    assert response.status_code == 422


def test_csv_sample_rate_invalido_422(client) -> None:
    signal = synthetic_ecg()
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", _csv(signal), "text/csv")},
        data={"sample_rate_hz": 0},
    )
    assert response.status_code == 422


def test_csv_unidades_invalidas_422(client) -> None:
    signal = synthetic_ecg()
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", _csv(signal), "text/csv")},
        data={"sample_rate_hz": 500, "units": "V"},
    )
    assert response.status_code == 422


def test_csv_no_utf8_422(client) -> None:
    """Contenido que no es UTF-8 → 422."""
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", b"\xff\xfe\x00\x01", "text/csv")},
        data={"sample_rate_hz": 500},
    )
    assert response.status_code == 422


def test_csv_resultado_es_consultable(client) -> None:
    """El análisis de un CSV queda persistente y consultable."""
    signal = synthetic_ecg(bpm=45)
    creado = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", _csv(signal), "text/csv")},
        data={"sample_rate_hz": 500},
    ).json()
    obtenido = client.get(f"/ecg/{creado['id']}")
    assert obtenido.status_code == 200
    assert obtenido.json()["anomalia"] == "bradicardia"


@pytest.mark.parametrize("separator", [",", ";", "\t"])
def test_csv_separadores(client, separator) -> None:
    """Se aceptan coma, punto y coma y tab como separadores."""
    signal = synthetic_ecg(bpm=120)
    content = "\n".join(
        f"{value:.6f}{separator}0" for value in signal
    )
    response = client.post(
        "/ecg/analyze/file",
        files={"file": ("ecg.csv", content, "text/csv")},
        data={"sample_rate_hz": 500},
    )
    assert response.status_code == 200
    assert response.json()["anomalia"] == "taquicardia"
