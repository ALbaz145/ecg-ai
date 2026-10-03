# Plan del MVP — ECG-AI

_Fecha: 2026-10-02 · Milestone: MVP · Repo: [ALbaz145/ecg-ai](https://github.com/ALbaz145/ecg-ai)_

## Objetivo del MVP

Flujo completo: **subir un ECG → analizarlo con un modelo de IA → mostrar el resultado**
(anomalía detectada + nivel de confianza).

> Proyecto escolar, NO uso clínico. La UI siempre muestra:
> **"Herramienta educativa, no es diagnóstico médico"**.

## Flujo del MVP

1. El usuario carga una señal ECG (JSON/CSV) en la interfaz.
2. El frontend llama a `POST /ecg/analyze`.
3. El backend valida (Pydantic), normaliza, ejecuta el modelo y guarda en PostgreSQL.
4. Se muestra el resultado con confianza y la señal visualizada.

## Tareas

| ID  | Tarea                                | Responsable | Estado    | Issue                                   |
| --- | ------------------------------------ | ----------- | --------- | --------------------------------------- |
| T01 | Infraestructura base (Docker/compose) | @devops     | Hecha ✅ (QA: 7/7) | [#1](https://github.com/ALbaz145/ecg-ai/issues/1) |
| T02 | Esquema mínimo (ecg_records/analyses) | @db         | Hecha ✅ (QA: 8/8) | [#3](https://github.com/ALbaz145/ecg-ai/issues/3) |
| T03 | API de análisis (POST /ecg/analyze)   | @backend    | Hecha ✅ (QA: 8/8, D1-D4 fix) | [#5](https://github.com/ALbaz145/ecg-ai/issues/5) |
| T04 | UI de carga y resultado                | @frontend   | Hecha ✅ (QA: 60/60, lint 0 err, build OK) — cerrado 2026-10-03 | [#2](https://github.com/ALbaz145/ecg-ai/issues/2) |
| T05 | CI: lint + pruebas en PR               | @devops     | Hecha ✅ (QA: AC 6/6, sin cambios de lógica) — cerrado 2026-10-03 | [#4](https://github.com/ALbaz145/ecg-ai/issues/4) |
| T06 | Validación end-to-end del MVP          | @qa         | Hecha ✅ (QA aprobado 2026-10-03) | [#6](https://github.com/ALbaz145/ecg-ai/issues/6) |
| T07 | Fix CORS (D1): añadir localhost:5173    | @backend    | Hecho      | — |
| T08 | Fix alembic driver (D2): normalizar psycopg2 | @db  | Hecho      | — |

## Dependencias

```
T01 (infra) ──► T02 (db) ──► T03 (backend) ──► T04 (frontend)
                                                     │
                              T03 + T04 ─────────────► T05 (CI) ──► T06 (QA)
```

- T01 no tiene dependencias (arranque inmediato).
- Cada tarea, al finalizar, la valida **@qa** antes de cerrarse.

## Criterios de aceptación globales

- Una rama por tarea: `feature/<nombre>`. Nunca commit directo a `main`.
- Commits convencionales (`feat:`, `fix:`, `test:`, `chore:`).
- Todo cambio lleva pruebas.
- Sin datos reales de pacientes; solo datasets públicos (PTB-XL, MIT-BIH, etc.).
- La UI siempre muestra el aviso: "Herramienta educativa, no es diagnóstico médico".

## Registro de cambios

- **2026-10-02**: Plan inicial del MVP. 6 tareas y 6 issues creados en el repo
  (labels: `backend`, `frontend`, `db`, `devops`, `qa`, `mvp`).
- **2026-10-03**: Issues #2 (Frontend) y #4 (CI) re-validados por QA y cerrados.
  PR #11 mergeado a `main` con squash. 43 archivos, +9,362/-34 líneas.
- **2026-10-03**: T06 rechazada por QA. Defectos bloqueantes:
  - **D1**: CORS solo permite localhost:3000, Vite usa 5173 → @backend
  - **D2**: alembic env.py no normaliza driver psycopg2, falla con SQLAlchemy ≥ 2.1 → @db
  Prioridad: fix D1 y D2 antes de cerrar T06.
