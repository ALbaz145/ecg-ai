"""Pruebas del endpoint ``GET /health``."""

from __future__ import annotations

from app.main import app
from fastapi.testclient import TestClient


def test_health_responde_200_ok(client) -> None:
    """El healthcheck de docker-compose espera 200 + ``{"status": "ok"}``."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_no_requiere_base_de_datos() -> None:
    """/health es stateless: responde sin repositorio ni DB.

    Se usa ``TestClient(app)`` sin ``with`` para no disparar
    el lifespan (que ejecuta las migraciones Alembic).
    """
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
