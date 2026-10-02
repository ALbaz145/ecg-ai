# Base de datos — ECG-AI (MVP)

> Proyecto escolar, **NO uso clínico**.
> **Cero datos reales de pacientes**: todos los ejemplos de este
> documento son ficticios. La regla global del repo prohíbe subir
> datos reales; solo datasets públicos (PTB-XL, MIT-BIH, etc.)
> fuera de la DB.

Este documento describe el esquema mínimo del MVP
(subir ECG → analizar → mostrar resultado), cómo ejecutar las
migraciones y cómo @backend consume los modelos.

---

## 1. Conexión

PostgreSQL 16 (servicio `postgres` en `docker-compose.yml`).
Las credenciales vienen de `.env` (ver `.env.example`):

```env
POSTGRES_USER=ecg
POSTGRES_PASSWORD=cambia_en_env
POSTGRES_DB=ecg
```

La URL de conexión (`DATABASE_URL`) se resuelve así, tanto en
`db/alembic/env.py` como en el backend:

1. `DATABASE_URL` si está definida. `docker-compose.yml` ya la
   inyecta al servicio `backend`:
   `postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}`.
2. Si no, se construye desde `POSTGRES_USER` / `POSTGRES_PASSWORD` /
   `POSTGRES_DB` con `POSTGRES_HOST` (por defecto `localhost`) y
   `POSTGRES_PORT` (por defecto `5432`) — útil para desarrollo
   fuera de Docker.

> La DB **no** se expone al host en `docker-compose.yml`. Para
> inspeccionarla desde el host en desarrollo, descomenta el
> mapeo de puertos del servicio `postgres`.

## 2. Ejecutar migraciones

Alembic está configurado en `alembic.ini` (raíz del repo) con
`script_location = db/alembic`. Se ejecuta **desde la raíz**:

```bash
# Online (requiere la DB corriendo: docker compose up -d postgres)
alembic upgrade head          # aplica todas las migraciones
alembic downgrade -1          # revierte la última migración
alembic current               # revisión aplicada
alembic revision --autogenerate -m "..."   # genera nueva migración

# Offline (SIN DB: genera el SQL que se ejecutaría)
alembic upgrade head --sql
alembic downgrade 0001:base --sql   # en --sql hay que dar el rango explícito
```

Estructura:

```
alembic.ini                  # configuración (script_location = db/alembic)
db/
├── __init__.py
├── models.py                # modelos SQLAlchemy (fuente de verdad del esquema)
└── alembic/
    ├── env.py               # resuelve DATABASE_URL + target_metadata
    ├── script.py.mako       # plantilla de revisiones
    └── versions/
        └── 20261002_0001_ecg_records_and_analyses.py
```

Todas las migraciones son **reversibles** (`downgrade` explícito,
nunca `pass`).

## 3. Esquema

```
ecg_records                      analyses
┌──────────────────────────┐     ┌──────────────────────────────┐
│ id            uuid  PK   │◄────│ ecg_record_id  uuid  FK      │  ON DELETE
│ filename      text       │     │   (CASCADE)                    │  CASCADE
│ signal_metadata jsonb    │     │ id               uuid  PK     │
│ created_at    timestamptz│     │ anomalia        text         │
└──────────────────────────┘     │ confianza     numeric(4,3)   │  CHECK 0..1
       1 ──────────────── ∞      │ model_version   text         │
                                 │ zonas_anomalas  jsonb        │
                                 │ created_at      timestamptz  │
                                 └──────────────────────────────┘
```

> Nota: las tablas no tienen `users`/`patients` en el MVP (no hay
> login). Ver §7 "Evolución prevista".

### 3.1 `ecg_records` — señales ECG subidas

El MVP **no** guarda las muestras de la señal en la DB; solo los
metadatos (la señal la procesa el backend en memoria/objeto
storage futuro).

| Columna         | Tipo                     | Restricciones                              |
| --------------- | ------------------------ | ------------------------------------------ |
| `id`            | `uuid`                   | PK, `DEFAULT gen_random_uuid()` (PG ≥ 13)  |
| `filename`      | `text`                   | NOT NULL, no vacío (`btrim`)               |
| `signal_metadata` | `jsonb`                | NOT NULL, debe ser objeto con la forma del contrato (ver abajo) |
| `created_at`    | `timestamptz`            | NOT NULL, `DEFAULT now()`                  |

**Contrato de `signal_metadata`** (ejemplo **ficticio**):

```json
{"sample_rate_hz": 500, "n_channels": 1, "units": "mV"}
```

- `sample_rate_hz` (number, obligatorio): frecuencia de muestreo en Hz (> 0 lo valida el backend con Pydantic).
- `n_channels` (number, obligatorio): número de canales.
- `units` (string, opcional): unidades de la señal, p. ej. `"mV"`.

La DB impone que los campos existan y tengan el tipo JSON
correcto (CHECKs `jsonb_typeof`); la validación semántica
(rango > 0, unidades soportadas) es responsabilidad del backend.

### 3.2 `analyses` — resultados del modelo de IA

| Columna         | Tipo                     | Restricciones                              |
| --------------- | ------------------------ | ------------------------------------------ |
| `id`            | `uuid`                   | PK, `DEFAULT gen_random_uuid()`            |
| `ecg_record_id` | `uuid`                   | NOT NULL, FK → `ecg_records.id` **ON DELETE CASCADE** |
| `anomalia`      | `text`                   | NOT NULL, no vacío (etiqueta del modelo, p. ej. `"normal"`, `"fibrilacion_auricular"`) |
| `confianza`     | `numeric(4,3)`           | NOT NULL, **CHECK `confianza >= 0 AND confianza <= 1`** |
| `model_version` | `text`                   | NOT NULL, no vacío (p. ej. `"mvp-0.1.0"`) |
| `zonas_anomalas`| `jsonb`                  | NOT NULL, DEFAULT `'[]'::jsonb`, debe ser array |
| `created_at`    | `timestamptz`            | NOT NULL, `DEFAULT now()`                  |

**Contrato de `zonas_anomalas`** (ejemplo **ficticio**):

```json
[{"inicio": 1.2, "fin": 1.45, "tipo": "latido_prematuro"}]
```

`inicio`/`fin` son **segundos** relativos al inicio de la señal;
`tipo` es una etiqueta libre del modelo. La DB solo exige que sea
un array JSON; la validación de la forma de cada elemento la hace
el backend.

**Ejemplo de fila (datos inventados):**

| anomalia | confianza | model_version | zonas_anomalas |
| --- | --- | --- | --- |
| `latido_prematuro` | `0.987` | `mvp-0.1.0` | `[{"inicio": 1.2, "fin": 1.45, "tipo": "latido_prematuro"}]` |

### 3.3 Restricciones (CHECKs)

| Nombre | Tabla | Regla |
| --- | --- | --- |
| `ck_ecg_records_signal_metadata_is_object` | ecg_records | `jsonb_typeof(signal_metadata) = 'object'` |
| `ck_ecg_records_signal_metadata_sample_rate` | ecg_records | `jsonb_typeof(signal_metadata->'sample_rate_hz') = 'number'` |
| `ck_ecg_records_signal_metadata_n_channels` | ecg_records | `jsonb_typeof(signal_metadata->'n_channels') = 'number'` |
| `ck_ecg_records_signal_metadata_units` | ecg_records | `units` ausente o string |
| `ck_ecg_records_filename_not_blank` | ecg_records | `char_length(btrim(filename)) > 0` |
| `ck_analyses_confianza_range` | analyses | **`confianza >= 0 AND confianza <= 1`** |
| `ck_analyses_zonas_anomalas_is_array` | analyses | `jsonb_typeof(zonas_anomalas) = 'array'` |
| `ck_analyses_anomalia_not_blank` | analyses | `char_length(btrim(anomalia)) > 0` |
| `ck_analyses_model_version_not_blank` | analyses | `char_length(btrim(model_version)) > 0` |

## 4. Índices y por qué

| Índice | Definición | Justificación |
| --- | --- | --- |
| `idx_analyses_ecg_record_created` | `analyses(ecg_record_id, created_at DESC)` | **Compuesto** que cubre dos accesos con un solo índice: (a) joins/búsquedas por `ecg_record_id` (prefijo izquierdo — PostgreSQL puede usarlo aunque la consulta no mencione `created_at`), y (b) "últimos N análisis de un ECG" (`WHERE ecg_record_id = ? ORDER BY created_at DESC LIMIT n`) sin sort adicional. Sustituye al índice simple sobre la FK. Además acelera el `ON DELETE CASCADE` (el borrado en cascada busca hijos por la FK). |
| `idx_analyses_created_at` | `analyses(created_at DESC)` | Listado global de análisis recientes (historial, futuro endpoint `GET /analyses`). `DESC` alinea el índice con el caso de uso "recientes primero". |
| `idx_ecg_records_created_at` | `ecg_records(created_at DESC)` | Orden natural de subidas (listado de registros). |

Nota: en PostgreSQL un índice btree es recorrible en ambos
sentidos, por lo que `DESC` no cambia el rendimiento de consultas
simples, pero documenta la intención y beneficia a planes
`ORDER BY ... DESC LIMIT` con índices compuestos.

No se indexa `anomalia` (cardinalidad baja y sin consultas de
filtro previstas en el MVP). Reevaluar si aparece filtrado por
anomalía.

## 5. Decisiones de diseño

- **`uuid` con `gen_random_uuid()`** (no `serial`): IDs no
  predecibles (no enumerables), seguros para exponer en API, y
  `gen_random_uuid()` es nativo en PG ≥ 13 (sin extensión
  `pgcrypto`). Coste: 16 bytes vs 4 — irrelevante a escala escolar.
- **`numeric(4,3)` para `confianza`** (no `real`/`double`):
  exactitud decimal (0.987 se almacena exacto; en punto flotante
  0.987 no lo es), 3 decimales bastan para mostrar un porcentaje
  en la UI (99.9%). El CHECK garantiza [0, 1].
- **`timestamptz`** (no `timestamp`): almacena en UTC y evita
  ambigüedades de zona horaria.
- **`jsonb`** (no `json`): binario, indexable y con operadores
  `->`/`jsonb_typeof` usados en los CHECKs.
- **Nombres de tabla en inglés, columnas del dominio en español**
  (`anomalia`, `confianza`, `zonas_anomalas`): sigue la
  terminología de la tarea y del dominio del proyecto.
- **Sin `UNIQUE` en `filename`**: el usuario puede subir varios
  archivos con el mismo nombre; la unicidad la da el PK.

## 6. Modelos SQLAlchemy (`db/models.py`)

`db/models.py` es la **fuente de verdad** del esquema, junto a
las migraciones (según AGENTS.md, `db/` = migraciones y esquemas).

**@backend** los importa así (desde la raíz del repo):

```python
from db.models import Base, EcgRecord, Analysis

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)   # solo para tests; en prod usa alembic
```

Ejemplo de uso:

```python
registro = EcgRecord(
    filename="muestra_ficticia_001.json",   # nombre inventado
    signal_metadata={"sample_rate_hz": 500, "n_channels": 1, "units": "mV"},
)
session.add(registro)
session.flush()
session.add(Analysis(
    ecg_record_id=registro.id,
    anomalia="latido_prematuro",
    confianza=Decimal("0.987"),
    model_version="mvp-0.1.0",
    zonas_anomalas=[{"inicio": 1.2, "fin": 1.45, "tipo": "latido_prematuro"}],
))
session.commit()
```

### Imagen del backend: modelos y migraciones incluidos

El servicio `backend` de `docker-compose.yml` construye con
**contexto en la raíz** del repo (`context: .`, `dockerfile:
backend/Dockerfile`), y el `Dockerfile` copia en `/app`:

- `backend/` → `/app/` (código de la API)
- `db/` → `/app/db/` (modelos y migraciones Alembic)
- `alembic.ini` → `/app/alembic.ini`

Así `from db.models import ...` resuelve dentro del
contenedor y **el comando de migración en el contenedor es**:

```bash
docker compose exec backend alembic upgrade head
```

En producción conviene ejecutar `alembic upgrade head` en el
arranque del contenedor (no `Base.metadata.create_all`).
Las pruebas del esquema también corren desde la raíz del
repo (ven §8).

### Sincronización modelo ↔ migración

- `tests/test_db_schema.py` (unittest, **no necesita DB**) verifica
  columnas, FK con `ON DELETE CASCADE`, el CHECK de confianza,
  los índices, y que la migración y el metadata declaren las
  mismas tablas/índices/constraints (comparación por AST).
- Cuando haya DB: `alembic revision --autogenerate` debe producir
  diff vacío contra `target_metadata = Base.metadata`.

## 7. Evolución prevista (fuera del MVP)

El esquema completo del proyecto contempla, en milestones
posteriores (issues futuros):

- **`users`**: cuentas del personal médico (login, `password_hash`,
  `rol`). No en el MVP — no hay login.
- **`patients`**: **siempre anónimos/ficticios** (p. ej.
  `codigo_anonimo`, sin nombre/identificadores reales). El MVP no
  la necesita: un `ecg_record` es un análisis puntual.
- **`audit_log`**: trazabilidad de acciones (quién, cuándo, qué)
  sobre registros y análisis. Requiere `users`.

Añadirlas será una migración Alembic adicional reversible
(`down_revision` encadenado a `"0001"`).

## 8. Verificación realizada

Estado: **verificado offline (sin DB)** — no hay daemon de Docker
ni PostgreSQL en el entorno de autoría.

| Criterio | Resultado |
| --- | --- |
| `alembic upgrade head --sql` | ✅ Genera el DDL completo: CREATE TABLE `ecg_records` y `analyses`, FK `fk_analyses_ecg_record_id ... ON DELETE CASCADE`, CHECKs (incl. `ck_analyses_confianza_range`), 3 índices |
| `alembic downgrade 0001:base --sql` | ✅ Revierte todo: DROP INDEX (3) → DROP TABLE `analyses` → DROP TABLE `ecg_records` (orden correcto por FK) |
| FK e índices presentes | ✅ En el SQL generado y en `db/models.py` (test `test_fk_en_cascada`, `test_indices_presentes`) |
| CHECK confianza ∈ [0,1] | ✅ `ck_analyses_confianza_range` en migración, modelo y test `test_check_confianza_entre_0_y_1` |
| `docs/DB.md` existe | ✅ Este documento |
| Cero datos personales reales | ✅ Solo ejemplos ficticios (`muestra_ficticia_001.json`, metadatos inventados) |
| Pruebas (`python -m unittest discover -s tests -v`) | ✅ Ver ejecutado junto a esta rama |

Pendiente para QA (con Docker disponible): `docker compose up -d
postgres && alembic upgrade head && alembic downgrade -1 &&
alembic upgrade head` online.
