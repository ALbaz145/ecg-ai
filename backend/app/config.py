"""Configuración del backend.

La URL de conexión a PostgreSQL se resuelve igual que en
``db/alembic/env.py`` (ver docs/DB.md §1):

1. ``DATABASE_URL`` si está definida (docker-compose la inyecta al
   servicio ``backend``).
2. Si no, se construye desde ``POSTGRES_USER`` / ``POSTGRES_PASSWORD``
   / ``POSTGRES_DB`` con ``POSTGRES_HOST`` (por defecto ``localhost``)
   y ``POSTGRES_PORT`` (por defecto ``5432``).
"""

from __future__ import annotations

import os
from urllib.parse import quote_plus

# Valores por defecto (coinciden con .env.example y docker-compose.yml).
DEFAULT_POSTGRES_USER = "ecg"
DEFAULT_POSTGRES_PASSWORD = "cambia_en_env"
DEFAULT_POSTGRES_DB = "ecg"
DEFAULT_POSTGRES_HOST = "localhost"
DEFAULT_POSTGRES_PORT = "5432"


def resolve_database_url() -> str:
    """Devuelve la URL de conexión a la base de datos.

    El esquema se normaliza a ``postgresql+psycopg2://``
(dependencia del proyecto: ``psycopg2-binary``) porque
    en SQLAlchemy ≥ 2.1 el esquema ``postgresql://``
    por defecto apunta a psycopg 3.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return _normalize_scheme(url)

    user = os.getenv("POSTGRES_USER", DEFAULT_POSTGRES_USER)
    password = quote_plus(
        os.getenv("POSTGRES_PASSWORD", DEFAULT_POSTGRES_PASSWORD)
    )
    database = os.getenv("POSTGRES_DB", DEFAULT_POSTGRES_DB)
    host = os.getenv("POSTGRES_HOST", DEFAULT_POSTGRES_HOST)
    port = os.getenv("POSTGRES_PORT", DEFAULT_POSTGRES_PORT)
    return (
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def _normalize_scheme(url: str) -> str:
    """Añade el driver psycopg2 si la URL no lo especifica."""
    scheme, sep, rest = url.partition("://")
    if not sep or "+" in scheme:
        # Ya lleva driver explícito (o no es JDBC-like): tal cual.
        return url
    if scheme == "postgres":
        scheme = "postgresql"
    if scheme == "postgresql":
        return f"postgresql+psycopg2://{rest}"
    return url


def cors_origins() -> list[str]:
    """Orígenes permitidos por CORS (separados por comas en
    ``CORS_ORIGINS``). Por defecto solo el frontend local."""
    raw = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
