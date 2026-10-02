"""Configuración de pruebas para tests/backend/.

- Ajusta ``sys.path`` para importar ``app`` (backend/) y
  ``db`` (raíz del repo) desde la raíz del repo.
- Provee el **doble en memoria** del repositorio (no se
  necesita PostgreSQL) y un cliente de prueba FastAPI.

Ejecución (desde la raíz del repo):

    python -m pytest tests/backend/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"

for _path in (str(REPO_ROOT), str(BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import uuid  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.repository import (  # noqa: E402
    EcgRepository,
    StoredAnalysis,
    get_repository,
    utc_now,
)


class FakeRepository:
    """Doble en memoria de ``EcgRepository``.

    Emula la semántica de persistencia (guardar → leer
    por id; último análisis por registro) sin tocar
    PostgreSQL. Los datos son ficticios/sintéticos.
    """

    def __init__(self) -> None:
        self.records: dict[uuid.UUID, dict[str, Any]] = {}
        self.analyses: dict[uuid.UUID, dict[str, Any]] = {}
        self._by_record: dict[uuid.UUID, list[uuid.UUID]] = {}

    def save_analysis(
        self,
        *,
        filename: str,
        signal_metadata: dict[str, Any],
        anomalia: str,
        confianza: float,
        model_version: str,
        zonas_anomalas: list[dict[str, Any]],
    ) -> StoredAnalysis:
        record_id = uuid.uuid4()
        analysis_id = uuid.uuid4()
        now = utc_now()
        self.records[record_id] = {
            "id": record_id,
            "filename": filename,
            "signal_metadata": signal_metadata,
            "created_at": now,
        }
        self.analyses[analysis_id] = {
            "id": analysis_id,
            "ecg_record_id": record_id,
            "anomalia": anomalia,
            "confianza": float(confianza),
            "model_version": model_version,
            "zonas_anomalas": [dict(z) for z in zonas_anomalas],
            "created_at": now,
        }
        self._by_record.setdefault(record_id, []).append(analysis_id)
        return StoredAnalysis(
            id=analysis_id,
            ecg_record_id=record_id,
            anomalia=anomalia,
            confianza=float(confianza),
            model_version=model_version,
            zonas_anomalas=[dict(z) for z in zonas_anomalas],
            created_at=now,
        )

    def get_analysis(self, analysis_id: uuid.UUID) -> StoredAnalysis | None:
        analysis = self.analyses.get(analysis_id)
        if analysis is None:
            return None
        return self._to_stored(analysis)

    def get_latest_analysis_for_record(
        self, record_id: uuid.UUID
    ) -> StoredAnalysis | None:
        ids = self._by_record.get(record_id, [])
        if not ids:
            return None
        return self._to_stored(self.analyses[ids[-1]])

    @staticmethod
    def _to_stored(analysis: dict[str, Any]) -> StoredAnalysis:
        return StoredAnalysis(
            id=analysis["id"],
            ecg_record_id=analysis["ecg_record_id"],
            anomalia=analysis["anomalia"],
            confianza=float(analysis["confianza"]),
            model_version=analysis["model_version"],
            zonas_anomalas=analysis["zonas_anomalas"],
            created_at=analysis["created_at"],
        )


@pytest.fixture
def fake_repository() -> FakeRepository:
    """Doble de repositorio compartido por las pruebas."""
    return FakeRepository()


@pytest.fixture
def client(fake_repository: FakeRepository) -> TestClient:
    """Cliente de prueba con el repositorio doble inyectado.

    Nota: se usa ``TestClient(app)`` **sin** ``with`` para
    no disparar el lifespan (que ejecuta ``alembic``).
    """
    app.dependency_overrides[get_repository] = (
        lambda: fake_repository
    )
    yield TestClient(app)
    app.dependency_overrides.pop(get_repository, None)


@pytest.fixture
def repository_type() -> type[EcgRepository]:
    """Tipo del repositorio (para anotaciones de pruebas)."""
    return EcgRepository
