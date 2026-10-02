"""Aplicación FastAPI (punto de entrada: ``uvicorn app.main:app``).

Proyecto escolar, NO uso clínico. La API siempre se acompaña
del aviso: *Herramienta educativa, no es diagnóstico médico*.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# --- Bootstrap de importaciones -------------------------------------
# `db/` (modelos y migraciones Alembic) vive en la raíz del
# repo; en el contenedor el Dockerfile la copia a /app/db.
# Para desarrollo local (``uvicorn app.main:app`` desde
# backend/) garantizamos que el directorio que contiene ``db``
# esté en sys.path, igual que hace ``db/alembic/env.py``.
for _parent in (
    Path(__file__).resolve().parent,
    *Path(__file__).resolve().parents,
):
    if (_parent / "db" / "models.py").is_file():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.config import cors_origins  # noqa: E402
from app.repository import get_repository  # noqa: F401,E402  (dependencia)
from app.routers.ecg import router as ecg_router  # noqa: E402
from app.routers.health import router as health_router  # noqa: E402

logger = logging.getLogger("ecg_ai.backend")

#: Si es "1"/"true"/"yes", el arranque omite las migraciones
#: (útil en desarrollo con la DB ya migrada).
SKIP_MIGRATIONS_ENV = "ECG_SKIP_MIGRATIONS"


def _find_alembic_ini() -> Path | None:
    """Busca ``alembic.ini`` hacia arriba desde este módulo.

    En el contenedor: ``/app/alembic.ini`` (junto a ``app/``).
    En el repo (desarrollo): la raíz del repo.
    """
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "alembic.ini").is_file():
            return parent / "alembic.ini"
    return None


def _run_migrations() -> None:
    """Aplica ``alembic upgrade head`` en el arranque.

    Es el arranque de producción: las tablas se crean con
    Alembic, **nunca** con ``Base.metadata.create_all``
    (ver docs/DB.md §6). Si falla, la app no arranca
    (fail fast; docker compose reinicia el servicio).
    """
    if os.getenv(SKIP_MIGRATIONS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        logger.info(
            "Migraciones omitidas (%s está activo)",
            SKIP_MIGRATIONS_ENV,
        )
        return
    ini = _find_alembic_ini()
    if ini is None:
        raise RuntimeError(
            "No se encontró alembic.ini: no se pueden aplicar "
            "las migraciones en el arranque."
        )
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(ini.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "alembic upgrade head falló (¿está arrancada "
            "PostgreSQL?)\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    logger.info("Migraciones Alembic aplicadas (revisión head).")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Arranque: migraciones; apagado: nada que liberar."""
    _run_migrations()
    yield


app = FastAPI(
    title="ECG-AI API",
    version="0.1.0",
    description=(
        "API del MVP de ECG-AI. "
        "⚠️ Proyecto escolar, NO uso clínico: herramienta "
        "educativa, no es diagnóstico médico."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ecg_router)
