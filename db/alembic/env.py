"""Entorno de Alembic para ECG-AI.

Resuelve la URL de conexión en este orden:

1. ``DATABASE_URL`` si está definida (docker-compose la inyecta al
   servicio ``backend``: ``postgresql://<user>:<pass>@postgres:5432/<db>``).
2. Si no, construye la URL desde ``POSTGRES_USER`` / ``POSTGRES_PASSWORD``
   / ``POSTGRES_DB`` (valores por defecto de ``.env.example``) con
   ``POSTGRES_HOST`` (por defecto ``localhost``) para desarrollo
   fuera de Docker.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# La raíz del repo al sys.path para importar `db.models`
# (los modelos son la fuente de verdad del esquema).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db.models import Base  # noqa: E402  (import tras ajustar sys.path)
from db.url import resolve_database_url  # noqa: E402

# Objeto de configuración de Alembic (alembic.ini).
config = context.config

# Logging de Python desde alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata objetivo para la detección de cambios ('alembic revision --autogenerate').
target_metadata = Base.metadata


def _database_url() -> str:
    """Devuelve la URL de conexión (ver docstring del módulo).

    Normaliza el esquema a ``postgresql+psycopg2://`` para que
    SQLAlchemy no elija el driver psycopg 3 por defecto.
    """
    return resolve_database_url()


def run_migrations_offline() -> None:
    """Genera SQL a secas (``alembic upgrade head --sql``).

    No requiere conexión a ninguna DB: emite el DDL al stdout.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Ejecuta migraciones contra una DB real."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=False,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
