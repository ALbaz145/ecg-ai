# Changelog — ECG-AI

Registro de cambios relevantes de la aplicación. Formato: `AAAA-MM-DD` · categoría · descripción.

## 2026-10-03

- **docs**: Fase 2 planificada en `docs/PLAN.md` (T09–T15) e issues #12–#18 creados: modelo real con MIT-BIH/wfdb, baseline sklearn con split por paciente, métricas, integración API y pruebas.
- **feat**: MVP completo mergeado a `main` (PR #11) — backend, frontend, DB, CI, +9,362/-34 líneas.
- **fix**: CORS permite `localhost:5173` (Vite) — D1, T07, issue relacionado T06.
- **fix**: Alembic normaliza driver `psycopg2` en `env.py` (SQLAlchemy ≥ 2.1) — D2, T08.
- **fix**: Sintaxis del Dockerfile de backend corregida.
- **docs**: T06 re-validada end-to-end por QA y aprobada.

## 2026-10-02

- **chore**: Infraestructura base con Docker/compose (T01, issue #1).
- **feat**: Esquema mínimo `ecg_records`/`analyses` con Alembic (T02, issue #3).
- **feat**: API de análisis de ECG `POST /ecg/analyze` (T03, issue #5).
- **docs**: Plan inicial del MVP creado (6 tareas, issues #1–#6).

## 2026-10-01

- **chore**: Repositorio inicial.
