# Changelog — ECG-AI

Registro de cambios relevantes de la aplicación. Formato: `AAAA-MM-DD` · categoría · descripción.

## 2026-11-09

- **test**: T09 (pipeline MIT-BIH) cerrada — validación final de @qa 5/5 sin defectos (#26, #12): E2E mockeado download→load→build→save→load coherente entre módulos, cobertura 96%, flujo MVP sin cambios. Issues #23-#26 cerrados.
- **chore**: Backlog menor de observaciones T09 (`.gitignore` `*.npz` global, `zip` estricto, 2 tests de cobertura) → issue #37, no bloquea T10.
- **feat(model)**: Baseline scikit-learn con split por paciente (#13, @qa 5/5 sin defectos): `backend/app/baseline.py` — split de IDs anti-fuga, 14 features por latido reutilizando `signal.py`, Pipeline(Imputer→Scaler→LR balanced), métricas por clase AAMI (sensibilidad/especificidad/CM 5×5), joblib con `model_version`+`dataset_id` validados. Suite 200→255 tests (PRs #39-#40).
- **test**: Métricas del baseline validadas con entrenamiento REAL de los 48 registros MIT-BIH (#16): 112.552 latidos, split 38/10 pacientes sin fuga, accuracy 0,5158 · sens_macro 0,3584 · spec_macro 0,8823; control con fuga por muestra +16 pts de accuracy demuestra la ausencia de fuga; determinismo con `random_state=42` confirmado.

## 2026-11-05

- **chore(deps)**: Dependencias ML fijadas en `backend/requirements.txt` — `wfdb==4.3.1`, `scikit-learn==1.8.0`, `joblib==1.5.2` (T15a, #19) — e `.gitignore` de Fase 2: `*.dat, *.hea, *.atr, *.pkl, *.joblib, *.h5, *.onnx, models/, mlruns/` (T15c, #21). Integrados vía PR #29 (QA 4/4; T15 cerrada #18).
- **docs**: Fase 2 completa en `docs/PLAN.md` (objetivo, tareas T09-T15, desgloses T09a-d y T15a-d con criterios, estados y responsables) + `docs/CHANGELOG.md` (PRs #28, #31, #33, #35).
- **feat(data)**: Descarga y caché de MIT-BIH con `wfdb` (`backend/app/mitbih.py`): 48 registros canónicos verificados contra PhysioNet, re-descarga evitada, errores de red gestionados (T09a, #23, QA 5/5).
- **feat(data)**: Loader MIT-BIH (`backend/app/mitbih_load.py`): señal lead único, `fs`, anotaciones y ventanas ±0.4 s reutilizando `BEAT_WINDOW_S`; caché exige `.hea+.dat+.atr` (D-1) (T09b, #24, QA 5/5).
- **feat(data)**: Dataset AAMI serializable (`backend/app/mitbih_dataset.py`): mapeo N/S/V/F/Q con no-mapeados → Q, `X` float32 / `y` / `paciente_id` por latido, NPZ con `schema_version` + `dataset_id` sha256 revalidado; validación de dominio `y` (fixes D-1..D-4) (T09c, #25, QA 2 rondas).
- **chore**: Integración en `main` con CI verde (PRs #28-#35) y limpieza de 11 ramas ya mergeadas; el repo solo conserva `main`.
- **test**: Suite backend de 107 → **200 pruebas** (sin red, verificadas con `unshare -rn`).

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
