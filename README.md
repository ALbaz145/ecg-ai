# ECG-AI

> ⚠️ **Herramienta educativa, no es diagnóstico médico.**
> Proyecto escolar para detectar anomalías en electrocardiogramas con IA.
> No usar en entornos clínicos ni con datos reales de pacientes.

Flujo del MVP: **subir un ECG (JSON/CSV) → analizarlo con un modelo de IA → mostrar el resultado** (anomalía detectada + nivel de confianza). El análisis usa el **modelo real** si existe el artefacto entrenado (`baseline-sklearn-0.2.0`) y, si no, **cae al modelo heurístico** (`heuristic-mvp-0.1.0`): ver [Modelo real](#modelo-real-entrenar-montar-y-verificar).

## Requisitos previos

- [Docker](https://www.docker.com/get-started) (Engine 24+)
- Docker Compose v2 (`docker compose`)
- Opcional — [Python](https://www.python.org/) 3.11+ con `pip`: solo para
  entrenar el modelo real en el host (ver
  [Modelo real](#modelo-real-entrenar-montar-y-verificar))

## Inicio rápido

```bash
# 1. Crear el archivo de entorno (nunca se sube al repo)
cp .env.example .env

# 2. Levantar todos los servicios
docker compose up -d --build

# 3. Verificar que todo está arriba y saludable
docker compose ps

# 4. Abrir la interfaz
#    http://localhost:3000  (y la API en http://localhost:8000/docs)
```

> **No hace falta entrenar nada para levantar la pila**: sin artefacto el
> backend funciona con el modelo heurístico (fallback sin errores). El
> entrenamiento del modelo real es **opcional** y se describe en
> [Modelo real](#modelo-real-entrenar-montar-y-verificar).

## Servicios y puertos

| Servicio   | Puerto (host) | Descripción                                        |
| ---------- | ------------- | -------------------------------------------------- |
| `frontend` | http://localhost:3000 | Interfaz React servida con nginx           |
| `backend`  | http://localhost:8000 | API FastAPI · docs en http://localhost:8000/docs |
| `postgres` | (solo red interna) | PostgreSQL 16, volumen persistente `pgdata` |

La base de datos **no** se expone al host por defecto (ver comentario en
`docker-compose.yml` si la necesitas desde el host en desarrollo).

## Verificación

```bash
docker compose ps                 # estado de los servicios
docker compose logs -f backend    # logs del backend
curl http://localhost:8000/docs   # API docs (Swagger)
```

## Modelo real (entrenar, montar y verificar)

El backend usa **el modelo real si hay artefacto**
(`data/models/baseline.joblib`, `model_version = baseline-sklearn-0.2.0`)
y, si no, **cae al heurístico** (`heuristic-mvp-0.1.0`) sin romper nada.
El artefacto **no se commitea** (gitignored): se genera en el host y
`docker-compose.yml` monta `./data` en `/app/data` (solo lectura), con la
ruta configurable vía `MODEL_PATH` (ver `.env.example`).

### Prerrequisitos (en el host)

- **Python 3.11+** (el Dockerfile del backend usa 3.11 y la CI 3.12; las
  wheels de `scikit-learn==1.8.0` cubren 3.11–3.14).
- Dependencias instaladas en el host:
  `python -m pip install -r backend/requirements.txt`
- **Red solo en la primera ejecución**: se descargan los 48 registros
  MIT-BIH de PhysioNet a `data/mit-bih/` (≈90–100 MB). Las re-ejecuciones
  son **idempotentes**: lo ya cacheado no se vuelve a bajar (sin caché el
  error explica además cómo preparar los datos a mano para seguir sin red).

### 1. Entrenar en el host

```bash
# Ejecutar desde la RAÍZ del repo: las rutas por defecto son relativas al
# directorio de trabajo y el compose monta ./data → /app/data.
# (Si entrenas dentro de backend/, el artefacto caería en
#  backend/data/models/ y el contenedor NO lo vería.)
mkdir -p data                 # si data/ no existe, créalo antes del compose up
PYTHONPATH=backend python -m app.train_model

# Alternativa equivalente desde backend/ (apunta el destino a la raíz):
#   cd backend && python -m app.train_model --dest ../data/models/baseline.joblib
```

El CLI imprime las métricas por stdout y la ruta escrita (exit 0) o el
motivo por stderr (exit 1). La ayuda siempre está disponible:
`cd backend && python -m app.train_model --help`.

### 2. Montar el artefacto

```bash
docker compose up -d --build   # el backend verá el artefacto vía el volumen
```

### 3. Verificar

```bash
docker compose ps                                  # servicios healthy
curl -s http://localhost:8000/health               # {"status":"ok"}

# El modelo activo sale en el análisis (la señal necesita ≥100 muestras):
python3 - <<'PY'
import json, math, urllib.request
sig = [0.1 + 0.05 * math.sin(2 * math.pi * 2 * i / 250) for i in range(500)]
req = urllib.request.Request(
    "http://localhost:8000/ecg/analyze",
    data=json.dumps({"signal": sig, "sample_rate_hz": 500}).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req))["model_version"])
PY
# Con artefacto: baseline-sklearn-0.2.0   ·   Sin artefacto: heuristic-mvp-0.1.0
```

**Sin artefacto no hay error**: si `data/models/baseline.joblib` no existe
(o no es legible), el backend arranca igual, avisa en los logs
(`docker compose logs backend | grep "Modelo real no disponible"`) y
responde con el modelo heurístico (`model_version = heuristic-mvp-0.1.0`).

> Si re-entrenas con la pila ya levantada, reinicia el backend
> (`docker compose restart backend`) para que recargue el artefacto
> (el modelo cargado queda en caché por proceso).

## Apagar

```bash
docker compose down        # detiene y elimina contenedores
docker compose down -v     # además borra el volumen de la DB (⚠️ datos)
```

## Variables de entorno

Todas las configuraciones viven en `.env` (creado desde `.env.example`):

| Variable         | Ejemplo                | Uso                          |
| ---------------- | ---------------------- | ---------------------------- |
| `POSTGRES_USER`  | `ecg`                  | Usuario de PostgreSQL        |
| `POSTGRES_PASSWORD` | `cambia_en_env`     | Contraseña de PostgreSQL     |
| `POSTGRES_DB`    | `ecg`                  | Nombre de la base de datos   |
| `MODEL_PATH`     | `data/models/baseline.joblib` | Ruta del artefacto del modelo (relativa al cwd del proceso: `/app` en el contenedor) |
| `VITE_API_URL`   | `http://localhost:8000` | URL de la API en el navegador |

`DATABASE_URL` se construye automáticamente en `docker-compose.yml` a partir de
las variables de PostgreSQL; solo defínela en `.env` si usas una DB externa.

> **Seguridad:** los secretos nunca van en el repo. `.env` está en `.gitignore`;
> en CI/CD se usan **GitHub Secrets**.

## Estructura del proyecto

```
.
├── backend/        # API FastAPI + modelo de IA  (Dockerfile incluido)
├── frontend/       # React + nginx               (Dockerfile incluido)
├── db/             # Migraciones y esquemas (Alembic)
├── docs/           # Documentación: PLAN, MODEL, DB, CHANGELOG
├── tests/          # Pruebas (backend/ y raíz)
├── data/           # NO versionada (gitignored): dataset MIT-BIH + artefacto
├── .github/        # CI: .github/workflows/ci.yml (ruff+pytest / eslint+vitest)
├── AGENTS.md       # Reglas del proyecto para los agentes
├── docker-compose.yml   # Servicios: postgres, backend, frontend
└── .env.example    # Plantilla de variables de entorno (el .env real nunca se sube)
```

`data/` solo aparece tras entrenar o descargar el dataset (ver
[Modelo real](#modelo-real-entrenar-montar-y-verificar)).

## Desarrollo y pruebas

Mismos comandos que ejecuta la CI en cada PR
(`.github/workflows/ci.yml`):

```bash
# Backend (desde la raíz del repo)
python -m pip install -r backend/requirements.txt   # dependencias
python -m pip install ruff==0.16.10                 # linter (la CI usa esta versión)
ruff check backend/ tests/                          # lint
python -m pytest tests/ -q                          # pruebas (no necesitan PostgreSQL)

# Frontend
cd frontend
npm ci                                              # primera vez
npm run lint
npm run test
```

La documentación de diseño y decisiones vive en `docs/`
([PLAN.md](docs/PLAN.md), [MODEL.md](docs/MODEL.md),
[DB.md](docs/DB.md), [CHANGELOG.md](docs/CHANGELOG.md)).

## Notas

- El backend asume la app en `backend/app/main.py` (uvicorn `app.main:app`).
- El frontend asume build de **Vite** (salida en `dist/`); si se usa CRA,
  ajustar el último `COPY` del `frontend/Dockerfile` a `build/`.
- Para el estado, las tareas y las reglas de colaboración, ver `AGENTS.md`
  y `docs/PLAN.md`; para el detalle del modelo, `docs/MODEL.md`.
