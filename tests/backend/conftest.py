"""Configuración de pruebas para tests/backend/.

- Ajusta ``sys.path`` para importar ``app`` (backend/) y
  ``db`` (raíz del repo) desde la raíz del repo.
- Provee el **doble en memoria** del repositorio (no se
  necesita PostgreSQL) y un cliente de prueba FastAPI.
- **Aísla la suite de ``MODEL_PATH``** (D-x): salvo inyección
  explícita de un test, la rama por defecto de cada test es la
  heurística, venga de donde venga el env var (ver
  :func:`_aislamiento_model_path`).

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
from collections.abc import Iterator  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402
from app.main import app  # noqa: E402
from app.model_loader import MODEL_PATH_ENV, reset_model_cache  # noqa: E402
from app.repository import (  # noqa: E402
    EcgRepository,
    StoredAnalysis,
    get_repository,
    utc_now,
)
from fastapi.testclient import TestClient  # noqa: E402


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


@pytest.fixture(autouse=True)
def _aislamiento_model_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Aísla cada test de un ``MODEL_PATH`` externo (D-x, T11b/T11c).

    La suite asume la **rama heurística** por defecto: sin aislamiento,
    un ``MODEL_PATH`` apuntando a un artefacto válido —p. ej. el que
    genera el CLI de T11c o el entorno del shell— cambia las etiquetas
    de ``POST /ecg/analyze`` a AAMI y deja la suite en rojos
    (``assert 'arritmia_supraventricular' == 'normal'``). Antes de cada
    test se fija ``MODEL_PATH`` a una ruta **inexistente** dentro de
    ``tmp_path`` (cubre tanto el env var externo como un artefacto en
    la ruta por defecto ``data/models/baseline.joblib``) y se vacía la
    caché del loader, para que el siguiente ``get_real_model()`` lea el
    estado aislado; tras el test se vacía de nuevo para que ninguna
    caché se filtre al siguiente.

    **Orden de fixtures**: los ``autouse`` de mayor scope primero y,
    dentro del mismo scope, los ``autouse`` antes que el resto; aquí
    comparten la misma instancia *function-scoped* de ``monkeypatch``
    que los tests. Así esta fixture se ejecuta **antes** de los
    ``monkeypatch.setenv("MODEL_PATH", ...)`` de los tests que inyectan
    el modelo explícitamente (p. ej. ``client_con_modelo`` de
    ``test_inference.py`` o los de ``test_model_loader.py``): el
    ``setenv`` del test llega después y **gana**, sin tocar sus
    aserciones. Los tests del cuerpo también se ejecutan después, así
    que su ``setenv`` posterior igualmente prevalece.
    """
    reset_model_cache()
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "sin-modelo.joblib"))
    yield
    reset_model_cache()
