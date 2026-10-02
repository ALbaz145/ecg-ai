"""Capa de persistencia: repositorio sobre SQLAlchemy.

La API depende del **protocolo** ``EcgRepository`` (no de la
clase concreta). En producción se inyecta ``SqlAlchemyRepository``
(PostgreSQL vía ``db.models``); en las pruebas se inyecta un
doble en memoria (ver ``tests/backend/conftest.py``).

Los modelos ``EcgRecord`` / ``Analysis`` son la fuente de verdad
de ``db/models.py`` (ver docs/DB.md): las tablas se crean con
``alembic upgrade head``, nunca con ``create_all`` en producción.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Protocol

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_session
from db.models import Analysis, EcgRecord


@dataclass
class StoredAnalysis:
    """Resultado persistido (independiente del ORM).

    ``confianza`` se normaliza a ``float`` aquí porque PostgreSQL
    devuelve ``Decimal`` desde ``numeric(4,3)``.
    """

    id: uuid.UUID
    ecg_record_id: uuid.UUID
    anomalia: str
    confianza: float
    model_version: str
    zonas_anomalas: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime | None = None


class EcgRepository(Protocol):
    """Interfaz de persistencia del pipeline de análisis."""

    def save_analysis(
        self,
        *,
        filename: str,
        signal_metadata: dict[str, Any],
        anomalia: str,
        confianza: float,
        model_version: str,
        zonas_anomalas: list[dict[str, Any]],
    ) -> StoredAnalysis: ...

    def get_analysis(self, analysis_id: uuid.UUID) -> StoredAnalysis | None: ...

    def get_latest_analysis_for_record(
        self, record_id: uuid.UUID
    ) -> StoredAnalysis | None: ...


def _to_stored(record: EcgRecord, analysis: Analysis) -> StoredAnalysis:
    """Mapea la fila ORM al DTO de la API."""
    confianza = analysis.confianza
    if isinstance(confianza, Decimal):
        confianza = float(confianza)
    return StoredAnalysis(
        id=analysis.id,
        ecg_record_id=record.id,
        anomalia=analysis.anomalia,
        confianza=float(confianza),
        model_version=analysis.model_version,
        zonas_anomalas=list(analysis.zonas_anomalas),
        created_at=analysis.created_at,
    )


class SqlAlchemyRepository:
    """Implementación del repositorio sobre PostgreSQL."""

    def __init__(self, session: Session) -> None:
        self._session = session

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
        record = EcgRecord(
            filename=filename,
            signal_metadata=signal_metadata,
        )
        self._session.add(record)
        # flush: obtiene el id (server_default gen_random_uuid()).
        self._session.flush()

        analysis = Analysis(
            ecg_record_id=record.id,
            anomalia=anomalia,
            confianza=confianza,
            model_version=model_version,
            zonas_anomalas=zonas_anomalas,
        )
        self._session.add(analysis)
        self._session.flush()
        self._session.commit()

        return _to_stored(record, analysis)

    def get_analysis(self, analysis_id: uuid.UUID) -> StoredAnalysis | None:
        analysis = self._session.get(Analysis, analysis_id)
        if analysis is None:
            return None
        return _to_stored(analysis.ecg_record, analysis)

    def get_latest_analysis_for_record(
        self, record_id: uuid.UUID
    ) -> StoredAnalysis | None:
        stmt = (
            select(Analysis)
            .where(Analysis.ecg_record_id == record_id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        analysis = self._session.scalar(stmt)
        if analysis is None:
            return None
        return _to_stored(analysis.ecg_record, analysis)


def utc_now() -> datetime:
    """Timestamp UTC con zona (para el doble en memoria de tests)."""
    return datetime.now(timezone.utc)


def get_repository(session: Session = Depends(get_session)) -> EcgRepository:
    """Dependencia FastAPI: repositorio de producción (PostgreSQL).

    En las pruebas se sobreescribe con ``app.dependency_overrides``
    (ver ``tests/backend/conftest.py``).
    """
    return SqlAlchemyRepository(session)
