# Backend — ECG-AI (MVP)

> ⚠️ **Herramienta educativa, no es diagnóstico médico.**
> Proyecto escolar, NO uso clínico. Sin datos reales de
> pacientes.

API FastAPI que recibe una señal ECG, la analiza con el
modelo (baseline heurístico, ver `docs/MODEL.md`) y
persiste el resultado en PostgreSQL.

## Estructura

```
backend/
├── Dockerfile
├── requirements.txt
├── README.md            ← este archivo
└── app/
    ├── main.py          # app FastAPI (uvicorn app.main:app)
    ├── config.py        # resolución de DATABASE_URL / CORS
    ├── database.py      # engine + sesión SQLAlchemy
    ├── schemas.py       # modelos Pydantic (contrato API)
    ├── repository.py    # persistencia (protocolo + SQLAlchemy)
    ├── signal.py        # filtrado, normalización, picos R
    ├── model.py         # modelo heurístico (ver docs/MODEL.md)
    ├── pipeline.py      # orquestación + parseo de CSV
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
docker compose up -d --build
docker compose ps            # backend healthcheck usa GET /health
```

**En el arranque del contenedor se ejecuta
`alembic upgrade head`** (las tablas se crean con
Alembic, nunca con `create_all`). Si quieres saltar las
migraciones en el arranque (DB ya migrada), define
`ECG_SKIP_MIGRATIONS=true`.

### Sin Docker (desarrollo)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# crea la DB y aplica migraciones (desde la raíz del repo):
alembic upgrade head
# lanza la API (desde backend/):
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

La URL de conexión se resuelve igual que en
`db/alembic/env.py`: `DATABASE_URL` si existe, si no
`POSTGRES_USER/PASSWORD/DB/HOST/PORT` (ver docs/DB.md §1).

## Pruebas

```bash
# desde la raíz del repo (sin PostgreSQL necesario:
# las pruebas usan un doble en memoria del repositorio)
python -m pytest tests/backend/ -v
```

## API (contrato para @frontend)

Base: `http://localhost:8000` (ver `VITE_API_URL` en
`.env.example`). Docs interactivas: `GET /docs`.

### `GET /health`

```json
{"status": "ok"}
```

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

Respuesta `200`:

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

- `anomalia`: `normal` | `taquicardia` | `bradicardia` |
  `arritmia` | `amplitud_anomala` | `ruido_excesivo` |
  `senal_no_valorable` (ver `docs/MODEL.md`).
- `confianza`: `float` en `[0.0, 1.0]` (heurística,
  **no es probabilidad calibrada**).
- `zonas_anomalas`: índices de muestra enteros
  (`inicio` inclusivo, `fin` exclusivo) sobre la señal
  subida; `[]` si no hay zonas.
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
  el MVP no guarda las muestras) y `analyses` (resultado),
  vía SQLAlchemy sobre `db/models.py` (ver docs/DB.md).
  Las tablas se crean con `alembic upgrade head`.
- **Modelo**: baseline heurístico documentado en
  `docs/MODEL.md` — no es un modelo entrenado ni diagnóstico
  médico.
- **CORS**: orígenes configurables con `CORS_ORIGINS`
  (por defecto `http://localhost:3000,http://127.0.0.1:3000`).
