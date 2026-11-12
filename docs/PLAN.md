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
| T15 | Dependencias y CI para ML (wfdb, sklearn)       | @devops     | Hecha ✅ (QA final 4/4 2026-10-14, T15a-d cerradas) | [#18](https://github.com/ALbaz145/ecg-ai/issues/18) |
| T09 | Pipeline de datos MIT-BIH con wfdb              | @backend    | Hecha ✅ (QA final 5/5 2026-11-09, T09a-d cerradas) | [#12](https://github.com/ALbaz145/ecg-ai/issues/12) |
| T10 | Baseline scikit-learn, split por paciente      | @backend    | Hecha ✅ (QA: 5/5 sin defectos, 197bbaf, 255 passed) | [#13](https://github.com/ALbaz145/ecg-ai/issues/13) |
| T13 | Validar métricas y matriz de confusión         | @qa         | Hecha ✅ (QA aprobado, real 48 registros: acc 0,5158 sin fuga por registro — corregida a por sujeto en #61; control fuga +16 pts) | [#16](https://github.com/ALbaz145/ecg-ai/issues/16) |
| T11 | Integrar modelo real en POST /ecg/analyze      | @backend    | Cerrada ✅ (T11a-e + fixes + T14; #14 cerrado 2026-11-10) | [#14](https://github.com/ALbaz145/ecg-ai/issues/14) |
| T14 | Pruebas: la API usa el modelo real             | @qa         | Hecha ✅ (QA APROBADO, PR #58, 339 backend + 73 frontend) | [#17](https://github.com/ALbaz145/ecg-ai/issues/17) |
| T12 | Documentar modelo en docs/MODEL.md             | @backend    | Hecha ✅ (QA: APROBADO 5/5 AC, PR #60 + fixes PR #62) | [#15](https://github.com/ALbaz145/ecg-ai/issues/15) |
| T61 | Split por sujeto real (201/202) + re-validar métricas | @backend | Hecha ✅ (QA APROBADO re-verificación, PR #65, 350 tests; `MODEL_VERSION` 0.2.0, acc 0,5582) | [#61](https://github.com/ALbaz145/ecg-ai/issues/61) |
| B37 | Backlog menor T09: O-1, O-3, O-5, O-6 | @backend | Hecha ✅ (QA APROBADO con 6 mutaciones, PR #68, 364 tests; `mitbih_load` 100% cobertura) | [#37](https://github.com/ALbaz145/ecg-ai/issues/37) |
| D70 | Actualización integral de documentación post-Fase 2 | @backend/@frontend/@devops/@db | Hecha ✅ (QA APROBADO en 5 PRs: #71-#75; fixes D1-D5) | [#70](https://github.com/ALbaz145/ecg-ai/issues/70) |

### Desglose T11 — Integrar modelo real en POST /ecg/analyze (padre #14)

Decisiones de diseño (2026-11-09): **MODEL_PATH + script de entrenamiento**
(el artefacto no se commitea; fallback al heurístico si falta), **etiquetas
AAMI en español** en la respuesta (agregado por latido) y **resampleo al
fs del modelo** (360 Hz MIT-BIH) antes de inferir.

| ID   | Subtarea | Responsable | Estado  | Issue |
| ---- | -------- | ----------- | ------- | ----- |
| T11a | Loader del artefacto (`MODEL_PATH`, caché, fallback al heurístico) | @backend | Hecha ✅ (QA: APROBADO + D-1 cerrado, 270 tests, PR #49) | [#43](https://github.com/ALbaz145/ecg-ai/issues/43) |
| T11b | Inferencia real en `pipeline.run_analysis` (resample, AAMI, agregado) | @backend | Hecha ✅ (QA: APROBADO sin defectos, PR #51, 310 tests) | [#44](https://github.com/ALbaz145/ecg-ai/issues/44) |
| T11c | CLI `python -m app.train_model` (genera el artefacto, sin commitear) | @backend | Hecha ✅ (QA: APROBADO, PR #53, 331 tests; D-2 fix PR #56) | [#45](https://github.com/ALbaz145/ecg-ai/issues/45) |
| T11d | Docker: volumen `data/` + `MODEL_PATH` en compose | @devops | Hecha ✅ (QA: APROBADO, PR #54, up real con/sin artefacto) | [#46](https://github.com/ALbaz145/ecg-ai/issues/46) |
| T11e | Etiquetas AAMI en `frontend/src/utils/labels.js` | @frontend | Hecha ✅ (QA: APROBADO, PR #52, 73 tests; P3 fix PR #55) | [#47](https://github.com/ALbaz145/ecg-ai/issues/47) |

Criterios por subtarea:
- **T11a**: `get_real_model() -> BaselineModel | None` con caché por proceso y
  `reset_model_cache()` para tests; ausente/corrupto → `None` + warning,
  **sin romper el request** (fallback heurístico); pruebas con artefacto
  mínimo en `tmp_path`.
- **T11b**: con modelo → `model_version = baseline-sklearn-0.1.0`,
  `anomalia` ∈ {normal, arritmia_supraventricular, arritmia_ventricular,
  latido_fusion, clase_desconocida, senal_no_valorable}, zonas por latido
  no-N **en índices de la señal original**, confianza = media de
  `max(predict_proba)`; resampleo al fs del modelo (testeado 360/500/90 Hz);
  sin modelo → rama heurística intacta (244 tests pasan sin cambios).
- **T11c**: CLI con `--dest/--seed`, imprime métricas (T13), idempotente con
  dataset cacheado; `data/` y `*.joblib` siguen ignorados; tests con mocks.
- **T11d**: volumen `./data:/app/data` + `MODEL_PATH` en compose y
  `.env.example`; `docker compose config` OK; sin artefacto arranca igual.
- **T11e**: `ANOMALIAS`/`ZONAS_TIPO` ampliados con las claves nuevas;
  fallback `capitalizar` sigue funcionando; eslint + vitest verdes.

### Desglose T09 — Pipeline MIT-BIH (padre #12)

| ID   | Subtarea | Responsable | Estado  | Issue |
| ---- | -------- | ----------- | ------- | ----- |
| T09a | Descarga/cache MIT-BIH con `wfdb` (48 registros, a `data/`, sin commitear) | @backend | Hecha ✅ (QA: 5/5, 9bb4eee, 10 tests, 117 passed) | [#23](https://github.com/ALbaz145/ecg-ai/issues/23) |
| T09b | Loader registro → señal lead único + fs + anotaciones, ventanas ±0.4s (`signal.py`) | @backend | Hecha ✅ (QA: 5/5, 73f2c7f, 137 passed, D-1 mitigado) | [#24](https://github.com/ALbaz145/ecg-ai/issues/24) |
| T09c | Etiquetado AAMI + dataset serializable (X, y, paciente_id) | @backend | Hecha ✅ (QA: 2 rondas, D-1..D-4 corregidos, 200 passed) | [#25](https://github.com/ALbaz145/ecg-ai/issues/25) |
| T09d | Validación QA del pipeline (checklist #12) | @qa | Hecha ✅ (QA final 5/5, cierra #12) | [#26](https://github.com/ALbaz145/ecg-ai/issues/26) |

Criterios por subtarea:
- **T09a**: `download/cache` a `data/mit-bih/` con `wfdb`; re-ejecución usa caché; `data/` y `*.dat/*.hea/*.atr` siguen ignorados; tests con mocks (sin red).
- **T09b**: por registro devuelve señal lead 0, `fs` y anotaciones (muestra + símbolo); ventanas de latido compatibles con `BEAT_WINDOW_S=0.4s` de `signal.py`; tests con `wfdb.rdsamp/rdann` mockeados.
- **T09c**: mapeo de símbolos a clases AAMI (N/S/V/F/Q); `X` (n_latidos × ventana), `y`, `paciente_id`; `save/load` reproducible (npz/joblib); tests de formas y mapeo.
- **T09d**: @qa valida T09a-c contra los 5 AC de #12 antes de cerrar T09 (incluido “sin datos commiteados”).

### Desglose T15 — Dependencias y CI para ML (padre #18)

| ID   | Subtarea | Responsable | Estado  | Issue |
| ---- | -------- | ----------- | ------- | ----- |
| T15a | Fijar deps ML en `backend/requirements.txt` (wfdb, scikit-learn, joblib con versión fija) | @devops | Hecha ✅ (QA: 4/4 2026-10-14, rama `feature/ml-deps-t15a` 42ad5a5) | [#19](https://github.com/ALbaz145/ecg-ai/issues/19) |
| T15b | Docker build con nuevas deps (imagen backend construye y `/health` OK) | @devops | Hecha ✅ (QA aprobado, build OK, img 941MB, /health 200) | [#20](https://github.com/ALbaz145/ecg-ai/issues/20) |
| T15c | CI + `.gitignore` dataset (lint+tests en verde, MIT-BIH y `*.pkl/joblib` ignorados) | @devops | Hecha ✅ (QA aprobado, ruff clean, pytest 107 passed) | [#21](https://github.com/ALbaz145/ecg-ai/issues/21) |
| T15d | Validación QA de T15 (smoke imports + checklist #18) | @qa | Hecha ✅ (QA final 4/4, cierra #18) | [#22](https://github.com/ALbaz145/ecg-ai/issues/22) |

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
- **2026-11-05**: Fase 2 integrada en `main` (PRs #28-#31, CI verde, squash):
  docs de Fase 2 + deps ML de T15 (cerrada 2026-10-14, QA 4/4) + T09a.
- **2026-11-05**: T09a #23, T09b #24 y T09c #25 cerradas (QA aprobado; T09c
  tras 2 rondas con fixes D-1..D-4). PRs #30-#35. `pytest` 137 → 200 passed.
  Limpieza: 11 ramas integradas borradas, repo solo con `main`.
- **2026-11-05 (pausa)**: siguiente paso al reanudar → **T09d (#26, @qa)**:
  validación final de T09 contra los 5 AC de #12 → cierra #12 → T10 (#13).
- **2026-11-09**: **T09 cerrada** — T09d #26 y padre #12 APROBADOS por @qa
  (5/5, sin defectos; E2E mockeado download→load→build→save→load coherente,
  cobertura 96%, flujo MVP intacto). Backlog menor O-1/O-3/O-5/O-6 → issue #37.
- **2026-11-09**: **T10 cerrada** — baseline sklearn (#13, @qa 5/5 sin defectos):
  split por paciente, 14 features de signal.py, 255 tests. PRs #39-#40.
- **2026-11-09**: **T13 cerrada** — métricas validadas con ENTRENAMIENTO REAL
  de los 48 registros MIT-BIH (#16, @qa, 0 defectos): split sin fuga por
  registro (38/10 pacientes; la fuga residual 201/202 se corrigió en #61),
  accuracy 0,5158 / sens_macro 0,3584 / spec_macro 0,8823, control con fuga
  +16 pts demuestra que el split importa. Nota F: soporte 4 en test.
- **2026-11-09 (pausa)**: siguiente paso al reanudar → **T11 (#14, @backend)**:
  integrar el modelo real (baseline joblib de T10) en `POST /ecg/analyze` →
  luego T14 (#17, @qa) → T12 (#15, @backend) → cierra Fase 2. Backlog #37.
- **2026-11-09**: **T11 desglosada** en 5 sub-issues de #14 con decisiones
  de diseño fijadas: **MODEL_PATH + script `train_model`** (artefacto no
  commiteado, fallback al heurístico), **etiquetas AAMI en español**
  (agregado por latido, mapeo S/V/F/Q) y **resampleo al fs del modelo**
  (360 Hz; zonas en índices de la señal original). Issues: T11a #43,
  T11b #44, T11c #45 (T11d #46 @devops, T11e #47 @frontend, deps T11b).
  Orden: T11a → T11b → (T11c ‖ T11d) → T11e → T14 (#17, @qa).
- **2026-11-09**: **T11a cerrada** — loader del artefacto (#43, PR #49,
  @qa APROBADO con D-1 menor corregido y re-verificado): `get_real_model()`
  con caché thread-safe, `reset_model_cache()`, `None` + warning ante
  artefacto ausente/corrupto/ruta inválida (nunca excepción al request),
  270 tests. Nota: el baseline real era 255 tests (PLAN decía 244).
- **2026-11-10**: **T11b-e cerradas** (QA APROBADO en las 4; PRs #51, #53,
  #54, #52): inferencia real en `run_analysis` (resampleo `resample_poly` al
  fs del modelo, zonas en índices originales, vocabulario AAMI en español,
  confianza de `predict_proba`, umbral no-N 0,20, fallback heurístico
  intacto); CLI `train_model` (métricas stdout, sin red, exit 0/1/2);
  volumen `./data:/app/data:ro` + `MODEL_PATH` en compose (up real
  verificado con/sin artefacto); etiquetas AAMI en `labels.js` (consistencia
  exacta 11/11 y 10/10 con `inference.py`). Fixes posteriores: **D-x**
  (mayor, preexistente: suite aislada de `MODEL_PATH` con fixture autouse,
  17 failed → 0) y **D-2** (`~` en `--dest/--data-dir`) en PR #56; **P3**
  (comentarios `CLASE_AAMI` → constantes reales) en PR #55. 336 tests
  backend + 73 frontend en verde.
- **2026-11-10 (pausa)**: siguiente paso al reanudar → **T14 (#17, @qa)**:
  validación final de T11 (API con modelo real) → cierra #14 → T12 (#15,
  @backend, docs/MODEL.md) → Fase 2 completa. Backlog #37.
- **2026-11-10**: **T14 cerrada y T11 completa** — PR #58 (QA APROBADO):
  matriz AC #17 cubierta (versión real desde metadatos, E2E dataset→respuesta
  con etiqueta conocida a 360 y 500 Hz, fallback con warning en caplog),
  339 tests backend + 73 frontend. **Issues #17 y #14 cerrados**: la API
  usa el modelo real (`baseline-sklearn-0.1.0`) con fallback heurístico
  intacto. Siguiente: **T12 (#15, @backend)** → Fase 2 completa.
- **2026-11-10**: **T12 cerrada → FASE 2 COMPLETA** — `docs/MODEL.md`
  reestructurado (Parte I heurístico histórico / Parte II modelo real con
  dataset, split, features, métricas T13 con CM 5×5 y tabla por clase
  archivadas en #16, uso en la API, 11 limitaciones / Parte III hoja de
  ruta). @qa APROBADO (5/5 AC, cifras verificadas sin inventar); fixes
  D1-D4 en PR #62 (incl. corrección de la atribución de canales MLII/V5:
  solo 102/104/114 tienen V5 en canal 0). **Issues #15 cerrado.**
  Hallazgo nuevo → issue **#61**: fuga residual 201/202 (mismo sujeto
  partido por el split por registro; prioridad baja-media, retrain +
  re-validación QA). Tests: 339 backend + 73 frontend. Backlog #37.
- **2026-11-11**: **#61 cerrada — split por sujeto real** — PR #65 (QA
  APROBADO en re-verificación; primer veredicto RECHAZADO solo por docs,
  corregido en `510032d`/`83fe726`): `MULTI_RECORD_SUBJECTS = {"202": "201"}`
  aplicado dentro de `split_por_paciente` (identidad sin 201/202, determinismo
  y assert anti-fuga sobre registro **y** sujeto intactos; contrato de
  `paciente_id` sin cambios). Re-entrenamiento real reproducido bit a bit por
  @qa: **acc 0,5582 / sens_macro 0,5884 / spec_macro 0,8992** (T13:
  0,5158 / 0,3584 / 0,8823 con test distinto — la partición cambió: 5 de 10
  sujetos de test distintos), 201 y 202 ambos en train. `MODEL_VERSION`
  0.1.0 → **0.2.0** (los artefactos antiguos ya no cargan; regenerar con
  `python -m app.train_model`). `MODEL.md` §9/§11/§13.8/§14.2 + CHANGELOG
  actualizados (matiz "sin fuga por registro/sujeto", D-2 y O-1 de este PR
  de cierre), nota en #16. 350 tests backend + 73 frontend. Backlog #37.
- **2026-11-12**: **#37 cerrada — backlog T09** — PR #68 (QA APROBADO, 6
  mutaciones de verificación): O-5 (`.gitignore` ignora `*.npz` en cualquier
  ruta + `tests/test_gitignore.py` con skip si no hay git), O-3
  (`zip(..., strict=True)` en `mitbih_load` y en `model._amplitude_zones`,
  contrato de longitudes verificado por construcción), O-1 (tests de
  `p_signal.ndim == 1` y forma de señal inválida) y O-6 (tests de
  `_check_alignment` y del `continue` de `_purge_incomplete`). Cobertura:
  `mitbih_load.py` 100%, `mitbih_dataset.py` 99%. 350 → **364 tests**,
  ruff OK. Nota: el PR #67 (otro colaborador) quedó cerrado sin merge; el
  trabajo se hizo desde cero en #68. **Fase 2 sin backlog pendiente.**
- **2026-11-12**: **#70 cerrada — documentación integral post-Fase 2** —
  5 PRs (todos QA APROBADO): #71 `frontend/README.md` (labels AAMI,
  estructura real, comandos), #72 `docs/DB.md` (vocabulario dual
  heurístico+AAMI en `anomalia`/zonas, receta de migración online
  reescrita — D1: vía principal `docker compose run/exec backend alembic`,
  desde el host exige publicar el puerto 5432), #73 `backend/README.md`
  (árbol completo de módulos, contrato API con ambos vocabularios,
  `confianza` con semántica por modelo, sección de entrenamiento) +
  `MODEL.md` §12 (364 tests + nota cwd), #74 README raíz (frase stale
  "issue #1" eliminada, flujo de entrenamiento verificado de punta a
  punta con E2E docker) + `AGENTS.md` (estructura real) + `.env.example`
  (flujo corregido: `cd backend` escribía en `backend/data/` y el volumen
  monta `./data`) + `.dockerignore`/`frontend/.dockerignore` (los
  contextos de build ya no copian `data/`/`node_modules`) + healthcheck
  `127.0.0.1` (D2: frontend `unhealthy` por IPv6 → 3/3 healthy), #75 pie
  de UI coherente con modelo real (D4) + test. Vocabularios idénticos en
  5 puntos documentales; 364 tests backend + **74** frontend.
  **Issue #70 cerrado.**
