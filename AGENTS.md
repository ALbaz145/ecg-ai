# Proyecto: ECG-AI (proyecto escolar, NO uso clínico)
Detecta anomalías cardíacas en electrocardiogramas con IA en Python.
Interfaz accesible (WCAG AA) para personal médico.

## Estructura
- `backend/`  - FastAPI + modelo IA
- `frontend/` - React
- `db/`       - migraciones y esquemas
- `docs/`     - documentación del proyecto (PLAN, MODEL, DB, CHANGELOG)
- `tests/`    - pruebas
- `data/`     - gitignored: dataset MIT-BIH y artefacto del modelo (nunca se commitea)

En la raíz del repo (no hay carpeta `infra/`):
- `docker-compose.yml` - servicios `postgres`, `backend`, `frontend`
- `.env.example`       - plantilla de variables (los secretos van en `.env` o GitHub Secrets, nunca en el repo)
- `.github/workflows/` - CI: lint + pruebas en cada pull request

## Reglas globales
- Una rama por tarea: feature/<nombre>. Nunca commit directo a main.
- Commits convencionales (feat:, fix:, test:, chore:).
- Todo cambio lleva pruebas.
- Sin datos reales de pacientes. Solo datasets públicos.
- La UI siempre muestra el aviso: "Herramienta educativa, no es diagnóstico médico".
