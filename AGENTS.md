# Proyecto: ECG-AI (proyecto escolar, NO uso clínico)
Detecta anomalías cardíacas en electrocardiogramas con IA en Python.
Interfaz accesible (WCAG AA) para personal médico.

## Estructura
- `backend/`  - FastAPI + modelo IA
- `frontend/` - React
- `db/`       - migraciones y esquemas
- `infra/`    - Docker, CI/CD
- `tests/`    - pruebas

## Reglas globales
- Una rama por tarea: feature/<nombre>. Nunca commit directo a main.
- Commits convencionales (feat:, fix:, test:, chore:).
- Todo cambio lleva pruebas.
- Sin datos reales de pacientes. Solo datasets públicos.
- La UI siempre muestra el aviso: "Herramienta educativa, no es diagnóstico médico".
