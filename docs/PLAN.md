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

## Fase 2 — Modelo real con MIT-BIH

| ID  | Tarea                                          | Responsable | Estado  | Issue |
| --- | ---------------------------------------------- | ----------- | ------- | ----- |
| T15 | Dependencias y CI para ML (wfdb, sklearn)       | @devops     | En curso (desglosada T15a-d) | [#18](https://github.com/ALbaz145/ecg-ai/issues/18) |
| T09 | Pipeline de datos MIT-BIH con wfdb              | @backend    | Abierta | [#12](https://github.com/ALbaz145/ecg-ai/issues/12) |
| T10 | Baseline scikit-learn, split por paciente      | @backend    | Abierta | [#13](https://github.com/ALbaz145/ecg-ai/issues/13) |
| T13 | Validar métricas y matriz de confusión         | @qa         | Abierta | [#16](https://github.com/ALbaz145/ecg-ai/issues/16) |
| T11 | Integrar modelo real en POST /ecg/analyze      | @backend    | Abierta | [#14](https://github.com/ALbaz145/ecg-ai/issues/14) |
| T14 | Pruebas: la API usa el modelo real             | @qa         | Abierta | [#17](https://github.com/ALbaz145/ecg-ai/issues/17) |
| T12 | Documentar modelo en docs/MODEL.md             | @backend    | Abierta | [#15](https://github.com/ALbaz145/ecg-ai/issues/15) |

### Desglose T15 — Dependencias y CI para ML (padre #18)

| ID   | Subtarea | Responsable | Estado  | Issue |
| ---- | -------- | ----------- | ------- | ----- |
| T15a | Fijar deps ML en `backend/requirements.txt` (wfdb, scikit-learn, joblib con versión fija) | @devops | Hecha ⏳ pendiente QA (@devops: 4/4, rama `feature/ml-deps-t15a` 42ad5a5) | [#19](https://github.com/ALbaz145/ecg-ai/issues/19) |
| T15b | Docker build con nuevas deps (imagen backend construye y `/health` OK) | @devops | Abierta | [#20](https://github.com/ALbaz145/ecg-ai/issues/20) |
| T15c | CI + `.gitignore` dataset (lint+tests en verde, MIT-BIH y `*.pkl/joblib` ignorados) | @devops | Abierta | [#21](https://github.com/ALbaz145/ecg-ai/issues/21) |
| T15d | Validación QA de T15 (smoke imports + checklist #18) | @qa | Abierta | [#22](https://github.com/ALbaz145/ecg-ai/issues/22) |

Criterios por subtarea:
- **T15a**: `pip install -r backend/requirements.txt` OK en limpio; `import wfdb, sklearn, joblib` OK; sin romper `fastapi/numpy/scipy`; versiones fijadas (`==`).
- **T15b**: `docker build` backend OK; `compose up backend` healthy (`/health` 200); capa de deps aprovecha caché.
- **T15c**: CI `ruff + pytest` y `eslint + vitest` en verde; `git check-ignore` confirma `data/`, `*.dat`, `*.hea`, `*.atr`, `*.pkl`, `*.joblib` ignorados; dataset NO commiteado.
- **T15d**: @qa valida T15a-c contra los 4 AC de #18 antes de cerrar T15.

### Notas Fase 2

- **Baseline primero**: scikit-learn antes de cualquier red neuronal.
- **Split por paciente**: ningún paciente en train y test a la vez (evitar fuga).
- **Métricas obligatorias**: sensibilidad, especificidad, matriz de confusión por clase.
- Cero datos reales de pacientes propios: solo MIT-BIH público vía `wfdb`; el dataset no se commitea.
- Cada tarea la valida @qa antes de cerrarse.

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
