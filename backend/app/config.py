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

from db.url import (  # noqa: F401  (reexportados)
    DEFAULT_POSTGRES_DB,
    DEFAULT_POSTGRES_HOST,
    DEFAULT_POSTGRES_PASSWORD,
    DEFAULT_POSTGRES_PORT,
    DEFAULT_POSTGRES_USER,
    normalize_database_url,
    resolve_database_url,
)

# Alias histórico (se mantiene por compatibilidad con imports previos).
_normalize_scheme = normalize_database_url


def cors_origins() -> list[str]:
    """Orígenes permitidos por CORS (separados por comas en
    ``CORS_ORIGINS``). Por defecto solo el frontend local."""
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
