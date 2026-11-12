# Backend — ECG-AI (MVP)

> ⚠️ **Herramienta educativa, no es diagnóstico médico.**
> Proyecto escolar, NO uso clínico. Sin datos reales de
> pacientes.

API FastAPI que recibe una señal ECG, la analiza con el
**modelo real** —baseline scikit-learn entrenado con
MIT-BIH, `model_version = baseline-sklearn-0.2.0`, ver
`docs/MODEL.md` Parte II— y persiste el resultado en
PostgreSQL. Si falta el artefacto del modelo
(`data/models/baseline.joblib`), la API funciona igual
con el **heurístico de respaldo**
(`model_version = heuristic-mvp-0.1.0`, ver
`docs/MODEL.md` Parte I): avisa en los logs, pero **no es
un error** y ningún request falla por eso.

## Estructura

```
backend/
├── Dockerfile
├── requirements.txt
├── README.md            ← este archivo
└── app/
    ├── __init__.py      # paquete app (docstring de estructura)
    ├── main.py          # app FastAPI (uvicorn app.main:app;
    │                    #   migraciones Alembic en el arranque)
    ├── config.py        # resolución de DATABASE_URL / CORS
    ├── database.py      # engine + sesión SQLAlchemy
    ├── schemas.py       # modelos Pydantic (contrato API)
    ├── repository.py    # persistencia (protocolo + SQLAlchemy)
    ├── signal.py        # filtrado, normalización, picos R, features
    ├── model.py         # modelo HEURÍSTICO (fallback; heuristic-mvp-0.1.0)
    ├── model_loader.py  # carga del artefacto real (MODEL_PATH, caché
    │                    #   por proceso, fallback con warning)
    ├── inference.py     # inferencia real: resample al fs del modelo,
    │                    #   predicción por latido y agregado AAMI
    │                    #   (AAMI_TO_ANOMALIA / AAMI_TO_ZONA)
    ├── baseline.py      # modelo sklearn (entrenar/save/load;
    │                    #   baseline-sklearn-0.2.0)
    ├── train_model.py   # CLI de entrenamiento (python -m app.train_model)
    ├── mitbih.py        # descarga y caché del dataset MIT-BIH (wfdb)
    ├── mitbih_load.py   # lectura de registros MIT-BIH + ventanas ±0,4 s
    ├── mitbih_dataset.py # dataset etiquetado AAMI (serialización NPZ)
    ├── pipeline.py      # orquestación + parseo de CSV; decide modelo
    │                    #   real vs. heurístico (run_analysis)
    └── routers/
        ├── health.py    # GET /health
        └── ecg.py       # POST /ecg/analyze, /ecg/analyze/file,
                         # GET /ecg/{id}
```

## Ejecutar

### Con Docker (recomendado)

```bash
# desde la raíz del repo
cp .env.example .env
mkdir -p data            # volumen del modelo: si no existe, Docker lo crea como root
docker compose up -d --build
docker compose ps            # backend healthcheck usa GET /health
```

**En el arranque del contenedor se ejecuta
`alembic upgrade head`** (las tablas se crean con
Alembic, nunca con `create_all`). Si quieres saltar las
migraciones en el arranque (DB ya migrada), define
`ECG_SKIP_MIGRATIONS=true`.

### Modelo real: entrenar el artefacto

El artefacto **no se commitea** (`data/` y `*.joblib`
están en `.gitignore`): se genera bajo demanda en el
host. Requisitos en el host: **Python ≥ 3.11** y las
mismas dependencias que la API:

```bash
python -m pip install -r backend/requirements.txt
```

CLI (desde `backend/`):

```bash
cd backend
python -m app.train_model --help   # opciones: --dest, --seed, --records…
python -m app.train_model          # entrenamiento completo (los 48 registros)
```

- **Primera ejecución**: descarga MIT-BIH desde
  PhysioNet (**~100 MB**) a `data/mit-bih/` (gitignored).
  Es **idempotente**: lo ya cacheado no se vuelve a
  descargar, así que re-ejecutarlo no vuelve a bajar nada
  (funciona sin red si la caché está completa).
- Imprime las métricas por stdout (accuracy,
  sensibilidad/especificidad macro, matriz de confusión,
  `model_version`, `dataset_id`…) y la ruta escrita.
- Escribe el artefacto, por defecto en
  `data/models/baseline.joblib` (**relativa al directorio
  de trabajo** desde el que ejecutas el CLI).

**Atención al directorio de trabajo**: `--data-dir`,
`--dest` y `MODEL_PATH` se resuelven contra el cwd, y
docker-compose monta `./data` de la **raíz** del repo
(solo lectura, en `/app/data`). Por eso hay un comando
para cada forma de levantar la API:

| Dónde levantas la API | Cómo entrenar | Artefacto |
| --- | --- | --- |
| Docker (compose) | desde la **raíz**: `PYTHONPATH=backend python -m app.train_model` | `./data/models/baseline.joblib` (raíz) |
| Sin Docker (`uvicorn` desde `backend/`) | desde **`backend/`**: `python -m app.train_model` | `backend/data/models/baseline.joblib` |

**`MODEL_PATH`** (env var, ver `.env.example`) sobreescribe
la ruta del artefacto; por defecto
`data/models/baseline.joblib` relativa al WORKDIR (en el
contenedor: `/app/data/models/baseline.joblib`, montado
desde `./data` de la raíz). Se lee **en runtime**: si la
generas con la API ya arrancada, el siguiente request ya
la usa, sin reiniciar.

**Sin artefacto no hay error**: la API arranca y responde
200 con el heurístico (`model_version =
heuristic-mvp-0.1.0`) y emite un warning **por request**
en los logs:
`Modelo real no disponible en <ruta> (<motivo>); el pipeline seguirá con el heurístico.`

### Sin Docker (desarrollo)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# crea la DB y aplica migraciones (hay que ejecutarlo
# desde la raíz del repo):
cd .. && alembic upgrade head
# lanza la API (desde backend/):
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La URL de conexión se resuelve igual que en
`db/alembic/env.py`: `DATABASE_URL` si existe, si no
`POSTGRES_USER/PASSWORD/DB/HOST/PORT` (ver docs/DB.md §1).

## Pruebas

```bash
# desde la raíz del repo (sin PostgreSQL necesario:
# las pruebas usan un doble en memoria del repositorio)
ruff check backend/ tests/      # lint
python -m pytest tests/ -q      # suite completa
python -m pytest tests/backend/ -v   # solo backend, verbose
```

## API (contrato para @frontend)

Base: `http://localhost:8000` (ver `VITE_API_URL` en
`.env.example`). Docs interactivas: `GET /docs`.

### `GET /health`

```json
{"status": "ok"}
```

Solo healthcheck: **no** expone qué modelo está activo
(usa `model_version` de cada análisis o los logs; ver
más abajo).

### `POST /ecg/analyze` (JSON)

Petición:

```json
{
  "signal": [0.12, 0.18, -0.05, "..."],
  "sample_rate_hz": 500,
  "units": "mV",
  "filename": "ecg_001.csv"
}
```

| Campo | Tipo | Requerido | Validación |
| --- | --- | --- | --- |
| `signal` | `float[]` | sí | 100–2 000 000 muestras, valores finitos |
| `sample_rate_hz` | `int` | sí | 1–5000 Hz (a fs baja la banda del filtro se adapta al Nyquist; ver `docs/MODEL.md` §6) |
| `units` | `"mV" \| "uV"` | no | por defecto `"mV"` |
| `filename` | `string` | no | máx. 255 caracteres |

Respuesta `200` — **el esquema es idéntico en los dos
modelos**; cambian los valores de `anomalia`,
`zonas_anomalas[].tipo` y `model_version`. Con el
**modelo real** (artefacto presente):

```json
{
  "id": "3f6b... (UUID del análisis)",
  "ecg_record_id": "9c1e... (UUID del registro)",
  "anomalia": "arritmia_ventricular",
  "confianza": 0.812,
  "zonas_anomalas": [
    {"inicio": 1200, "fin": 2400, "tipo": "latido_ventricular"}
  ],
  "model_version": "baseline-sklearn-0.2.0"
}
```

Con el **heurístico** (fallback: sin artefacto o si la
inferencia real falla):

```json
{
  "id": "3f6b... (UUID del análisis)",
  "ecg_record_id": "9c1e... (UUID del registro)",
  "anomalia": "taquicardia",
  "confianza": 0.892,
  "zonas_anomalas": [
    {"inicio": 0, "fin": 5000, "tipo": "taquicardia"}
  ],
  "model_version": "heuristic-mvp-0.1.0"
}
```

- `anomalia` y `zonas_anomalas[].tipo` usan **uno de dos
  vocabularios** según `model_version`:

  | | Modelo real `baseline-sklearn-0.2.0` | Heurístico `heuristic-mvp-0.1.0` |
  | --- | --- | --- |
  | Cuándo | Hay artefacto válido en `MODEL_PATH` | Sin artefacto (o inferencia falló): fallback |
  | `anomalia` | `normal` \| `arritmia_supraventricular` \| `arritmia_ventricular` \| `latido_fusion` \| `clase_desconocida` \| `senal_no_valorable` | `normal` \| `taquicardia` \| `bradicardia` \| `arritmia` \| `amplitud_anomala` \| `ruido_excesivo` \| `senal_no_valorable` |
  | `zonas_anomalas[].tipo` | `latido_supraventricular` \| `latido_ventricular` \| `latido_fusion` \| `latido_desconocido` (una zona por latido no-N) | `taquicardia` \| `bradicardia` \| `ruido_excesivo` (zonas globales) \| `latido_irregular` \| `amplitud_baja` \| `amplitud_alta` |
  | Constantes | `AAMI_TO_ANOMALIA` / `AAMI_TO_ZONA` en `app/inference.py` | `LABEL_*` / `ZONE_*` en `app/model.py` |

  Con modelo real, la señal se clasifica **por latido** y
  se agrega a una etiqueta (fracción de latidos no-N <
  20 % → `normal`; dominante no-N en caso contrario; ver
  `docs/MODEL.md` §12). `normal` y `senal_no_valorable`
  existen en ambos vocabularios; el resto **no se mezclan
  entre modelos**.

- `confianza`: `float` en `[0.0, 1.0]` (redondeada a 3
  decimales). **Depende del modelo activo**:
  - **Heurístico**: función de la fuerza de la evidencia,
    **no es probabilidad calibrada** (`docs/MODEL.md` §4).
  - **Modelo real**: **sí deriva de `predict_proba`** —
    media de `max(predict_proba)` por latido sobre los
    latidos que sustentan la etiqueta (los no-N si es
    anómala, todos si es `normal`), recortada a [0,1];
    probabilidad del modelo **sin calibrar**
    (`docs/MODEL.md` §12).
- `zonas_anomalas`: índices de muestra enteros
  (`inicio` inclusivo, `fin` exclusivo) sobre la señal
  subida (con modelo real, ya mapeados desde la señal
  resampleada a la original); `[]` si no hay zonas.
- `model_version`: `baseline-sklearn-0.2.0` (modelo real)
  o `heuristic-mvp-0.1.0` (heurístico/fallback). **Cómo
  saber cuál está activo**:
  1. en cada respuesta (incluido `GET /ecg/{id}`): el
     campo `model_version` dice qué modelo dictó **ese**
     análisis;
  2. `GET /health` **no** lo expone (solo
     `{"status": "ok"}`);
  3. en los logs: sin artefacto hay un warning por
     request —
     `Modelo real no disponible en <ruta> (<motivo>); …`
     (logger `ecg_ai.backend`), visible con
     `docker compose logs backend` o la consola de
     uvicorn. Si ese warning no aparece y las respuestas
     traen `baseline-sklearn-0.2.0`, está activo el
     modelo real.
- **Fallback**: sin artefacto (o corrupto, ruta inválida
  o inferencia fallida) la respuesta sigue siendo **200**
  con el heurístico y su `model_version`, más el warning
  en los logs. **No es un error.**
- Entradas inválidas (señal vacía, < 100 muestras,
  valores no numéricos/NaN, frecuencia fuera de rango,
  unidades no soportadas) → **422** con el detalle de
  Pydantic.

### `POST /ecg/analyze/file` (multipart, CSV)

Campos de formulario: `file` (CSV de **una columna** de
valores; se acepta una fila de cabecera), `sample_rate_hz`,
`units` (opcional). Mismas validaciones y respuesta que
`POST /ecg/analyze`.

```bash
curl -X POST http://localhost:8000/ecg/analyze/file \
  -F "file=@senal.csv" -F "sample_rate_hz=500" -F "units=mV"
```

### `GET /ecg/{id}`

`{id}` puede ser el **id del análisis** o el
**id del registro** (`ecg_record_id`; devuelve el análisis
más reciente de ese registro).

- `200`: mismo cuerpo que `POST /ecg/analyze`.
- `404`: `{"detail": "Análisis no encontrado."}`.

## Notas de implementación

- **Persistencia**: `ecg_records` (metadatos de la señal;
  el MVP no guarda las muestras) y `analyses` (resultado,
  incluido `model_version`), vía SQLAlchemy sobre
  `db/models.py` (ver docs/DB.md). Las tablas se crean con
  `alembic upgrade head`.
- **Modelo**: conviven dos, con el mismo esquema de
  respuesta — el **real** (`baseline-sklearn-0.2.0`,
  baseline sklearn entrenado con MIT-BIH; artefacto
  `data/models/baseline.joblib`, nunca commiteado),
  activo cuando existe el artefacto, y el **heurístico**
  (`heuristic-mvp-0.1.0`, reglas a mano, no es
  diagnóstico médico) como **fallback** si el artefacto
  falta o falla. Ambos documentados en `docs/MODEL.md`
  (Parte I el heurístico, Parte II el modelo real con
  dataset, métricas y limitaciones).
- **CORS**: orígenes configurables con `CORS_ORIGINS`
  (por defecto `http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173`).
