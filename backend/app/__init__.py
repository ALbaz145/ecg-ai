"""Aplicación backend de ECG-AI (API FastAPI + pipeline de análisis).

Proyecto escolar, NO uso clínico. Sin datos reales de pacientes.

Estructura:

- ``main``: creación de la app FastAPI (lifespan: migraciones Alembic).
- ``config`` / ``database``: resolución de ``DATABASE_URL`` y sesión SQLAlchemy.
- ``schemas``: modelos Pydantic (contrato de la API).
- ``repository``: capa de persistencia (protocolo + implementación SQLAlchemy).
- ``signal`` / ``model`` / ``pipeline``: procesamiento de señal, modelo
  heurístico y orquestación del pipeline.
- ``routers``: endpoints HTTP.
"""

__version__ = "0.1.0"
