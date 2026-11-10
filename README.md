# ECG-AI

> ⚠️ **Herramienta educativa, no es diagnóstico médico.**
> Proyecto escolar para detectar anomalías en electrocardiogramas con IA.
> No usar en entornos clínicos ni con datos reales de pacientes.

Flujo del MVP: **subir un ECG (JSON/CSV) → analizarlo con un modelo de IA → mostrar el resultado** (anomalía detectada + nivel de confianza).

## Requisitos previos

- [Docker](https://www.docker.com/get-started) (Engine 24+)
- Docker Compose v2 (`docker compose`)

## Inicio rápido

```bash
# 1. Crear el archivo de entorno (nunca se sube al repo)
cp .env.example .env

# 2. Levantar todos los servicios
docker compose up -d --build

# 3. Verificar que todo está arriba y saludable
docker compose ps
```

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

## Modelo real (entrenar → montar → verificar)

El artefacto del modelo (`data/models/baseline.joblib`) **no se commitea**
(gitignored): se genera en el host y `docker-compose.yml` monta `./data`
en `/app/data` para que el contenedor lo lea (solo lectura), con la ruta
configurable vía `MODEL_PATH` (ver `.env.example`).

```bash
# 1. Entrenar en el host (CLI de T11c; escribe data/models/baseline.joblib)
mkdir -p data            # si data/ no existe, créalo antes del compose up
cd backend && python -m app.train_model

# 2. Levantar (el backend verá el artefacto vía el volumen)
docker compose up -d --build

# 3. Verificar: la respuesta trae model_version = baseline-sklearn-0.1.0
curl -s http://localhost:8000/health
# (esquema abreviado: la señal real necesita ≥100 muestras)
curl -s -X POST http://localhost:8000/ecg/analyze \
  -H 'Content-Type: application/json' \
  -d '{"signal": [0.1, 0.2, ...], "sample_rate_hz": 500}'
```

**Sin artefacto no hay error**: si `data/models/baseline.joblib` no existe
(o no es legible), el backend arranca igual, avisa en los logs
(`docker compose logs backend | grep "Modelo real no disponible"`) y
responde con el modelo heurístico (`model_version = heuristic-mvp-0.1.0`).

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
| `MODEL_PATH`     | `data/models/baseline.joblib` | Ruta del artefacto del modelo (relativa al WORKDIR; en el contenedor `/app`) |
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
├── db/             # Migraciones y esquemas
├── infra/          # Documentación y assets de infra
├── tests/          # Pruebas
├── docker-compose.yml
└── .env.example
```

## Notas

- El backend asume la app en `backend/app/main.py` (uvicorn `app.main:app`).
- El frontend asume build de **Vite** (salida en `dist/`); si se usa CRA,
  ajustar el último `COPY` del `frontend/Dockerfile` a `build/`.
- Aún no hay código de aplicación: este issue (#1) solo levanta la infraestructura.
