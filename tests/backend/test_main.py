"""Pruebas del módulo de aplicación (``app/main.py``)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import _find_alembic_ini, app


def test_encuentra_alembic_ini() -> None:
    """El arranque localiza alembic.ini (contenedor o repo)."""
    ini = _find_alembic_ini()
    assert ini is not None
    assert ini.name == "alembic.ini"
    # El directorio del ini contiene las migraciones.
    assert (ini.parent / "db" / "alembic" / "versions").is_dir()


def test_lifespan_con_migraciones_omitidas(monkeypatch) -> None:
    """Con ``ECG_SKIP_MIGRATIONS`` la app arranca sin Alembic.

    Es la única forma de probar el lifespan sin PostgreSQL.
    """
    monkeypatch.setenv("ECG_SKIP_MIGRATIONS", "true")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_openapi_documenta_el_contrato() -> None:
    """Los endpoints y esquemas están en la docs (Swagger)."""
    response = TestClient(app).get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/ecg/analyze" in paths
    assert "/ecg/analyze/file" in paths
    assert "/ecg/{record_id}" in paths


def test_docs_de_la_api_estan_servidas() -> None:
    response = TestClient(app).get("/docs")
    assert response.status_code == 200
