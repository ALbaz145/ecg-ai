"""Pruebas de configuración y mapeo del repositorio."""

from __future__ import annotations

import uuid
from decimal import Decimal

from app.config import resolve_database_url
from app.repository import _to_stored
from db.models import Analysis, EcgRecord


def test_resuelve_url_desde_database_url(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/d")
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    assert resolve_database_url() == "postgresql+psycopg2://u:p@h:5432/d"


def test_resuelve_url_desde_variables_postgres(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "ecg")
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cr3t@!")
    monkeypatch.setenv("POSTGRES_DB", "ecg")
    monkeypatch.setenv("POSTGRES_HOST", "db.ejemplo")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    url = resolve_database_url()
    assert url == (
        "postgresql+psycopg2://ecg:s3cr3t%40%21@db.ejemplo:5433/ecg"
    )


def test_resuelve_url_por_defecto(monkeypatch) -> None:
    for var in (
        "DATABASE_URL",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
    ):
        monkeypatch.delenv(var, raising=False)
    assert resolve_database_url() == (
        "postgresql+psycopg2://ecg:cambia_en_env@localhost:5432/ecg"
    )


def test_normaliza_esquema_postgres(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/d")
    assert resolve_database_url() == "postgresql+psycopg2://u:p@h/d"


def test_respeta_driver_explicito(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h/d")
    assert resolve_database_url() == "postgresql+psycopg://u:p@h/d"


def test_to_stored_convierte_decimal_a_float() -> None:
    """PostgreSQL devuelve ``Decimal`` desde ``numeric(4,3)``."""
    record_id = uuid.uuid4()
    analysis_id = uuid.uuid4()
    record = EcgRecord(
        id=record_id,
        filename="sintetica.csv",
        signal_metadata={"sample_rate_hz": 500, "n_channels": 1},
    )
    analysis = Analysis(
        id=analysis_id,
        ecg_record_id=record_id,
        anomalia="taquicardia",
        confianza=Decimal("0.950"),
        model_version="heuristic-mvp-0.1.0",
        zonas_anomalas=[{"inicio": 0, "fin": 10, "tipo": "taquicardia"}],
    )
    stored = _to_stored(record, analysis)
    assert stored.id == analysis_id
    assert stored.ecg_record_id == record_id
    assert stored.anomalia == "taquicardia"
    assert stored.confianza == 0.95
    assert isinstance(stored.confianza, float)
    assert stored.zonas_anomalas == [
        {"inicio": 0, "fin": 10, "tipo": "taquicardia"}
    ]
