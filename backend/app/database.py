"""Engine y sesión de SQLAlchemy (PostgreSQL vía ``DATABASE_URL``).

En producción las tablas se crean con ``alembic upgrade head``
(ejecutado en el lifespan de la app, ver ``app/main.py``); **no** con
``Base.metadata.create_all`` (ver docs/DB.md §6).
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import resolve_database_url

DATABASE_URL = resolve_database_url()

# ``create_engine`` no abre conexión hasta la primera consulta, por lo
# que importar este módulo es seguro aunque la DB no esté levantada.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(
    bind=engine, autoflush=False, expire_on_commit=False, future=True
)


def get_session() -> Iterator[Session]:
    """Dependencia FastAPI: sesión por petición (se cierra al terminar)."""
    with SessionLocal() as session:
        yield session
