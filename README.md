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
