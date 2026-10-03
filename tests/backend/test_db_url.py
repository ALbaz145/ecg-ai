"""Pruebas de la normalización de la URL de la DB (D2, T08).

Verifica que ``db.url.resolve_database_url`` (usado por
``db/alembic/env.py`` y ``backend/app/config.py``) normaliza el
esquema a ``postgresql+psycopg2://`` sin romper URLs con driver
explícito.
"""

from __future__ import annotations

from db.url import normalize_database_url, resolve_database_url


def test_normaliza_esquema_postgresql(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/d")
    assert resolve_database_url() == "postgresql+psycopg2://u:p@h:5432/d"


def test_normaliza_esquema_postgres(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h/d")
    assert resolve_database_url() == "postgresql+psycopg2://u:p@h/d"


def test_respeta_driver_psycopg2_explicito(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg2://u:p@h/d"
    )
    assert resolve_database_url() == "postgresql+psycopg2://u:p@h/d"


def test_respeta_otro_driver_explicito(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h/d")
    assert resolve_database_url() == "postgresql+psycopg://u:p@h/d"


def test_fallback_construido_usa_psycopg2(monkeypatch) -> None:
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


def test_normalize_database_url_unit() -> None:
    assert normalize_database_url("postgresql://u:p@h/d") == (
        "postgresql+psycopg2://u:p@h/d"
    )
    assert normalize_database_url("postgres://u:p@h/d") == (
        "postgresql+psycopg2://u:p@h/d"
    )
    assert normalize_database_url("postgresql+psycopg://u:p@h/d") == (
        "postgresql+psycopg://u:p@h/d"
    )
    # Sin esquema "://" no se toca.
    assert normalize_database_url("sqlite:///tmp.db") == (
        "sqlite:///tmp.db"
    )
