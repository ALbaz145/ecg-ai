"""Resolución y normalización de la URL de la base de datos.

Fuente única usada tanto por ``db/alembic/env.py`` como por
el backend (``backend/app/config.py``). Ver docs/DB.md §1.

La URL se resuelve así:

1. ``DATABASE_URL`` si está definida (docker-compose la inyecta al
   servicio ``backend``: ``postgresql://<user>:<pass>@.../<db>``).
2. Si no, se construye desde ``POSTGRES_USER`` / ``POSTGRES_PASSWORD``
   / ``POSTGRES_DB`` con ``POSTGRES_HOST`` (por defecto ``localhost``)
   y ``POSTGRES_PORT`` (por defecto ``5432``).

En ambos casos el esquema se normaliza a ``postgresql+psycopg2://``
(dependencia del proyecto: ``psycopg2-binary``) porque en
SQLAlchemy ≥ 2.1 el esquema ``postgresql://`` por defecto
apunta al driver ``psycopg`` (v3), que no está instalado.
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
    """Devuelve la URL de conexión normalizada a psycopg2."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return normalize_database_url(url)

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


def normalize_database_url(url: str) -> str:
    """Añade el driver psycopg2 si la URL no especifica driver.

    - ``postgresql://`` y ``postgres://`` → ``postgresql+psycopg2://``.
    - URLs con driver explícito (``postgresql+psycopg://``, …) se
      devuelven tal cual.
    """
    scheme, sep, rest = url.partition("://")
    if not sep or "+" in scheme:
        # Ya lleva driver explícito (o no es un esquema tipo URL).
        return url
    if scheme == "postgres":
        scheme = "postgresql"
    if scheme == "postgresql":
        return f"postgresql+psycopg2://{rest}"
    return url
