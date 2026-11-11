# Modelo de IA — ECG-AI

> ⚠️ **Herramienta educativa, NO es diagnóstico médico.**
> Proyecto escolar: sin validación clínica, sin datos de pacientes
> propios (solo datasets públicos). No usar con pacientes reales ni
> para tomar decisiones clínicas.

Este documento describe los **dos modelos** que conviven en la API
(`POST /ecg/analyze` responde siempre con el campo `model_version`,
que indica cuál de los dos se usó en cada análisis):

| | Parte I (§1–§6) | Parte II (§7–§13) |
| --- | --- | --- |
| Modelo | Heurístico MVP (reglas a mano) | Baseline scikit-learn **entrenado** |
| Versión | `heuristic-mvp-0.1.0` | `baseline-sklearn-0.2.0` |
| Entrenamiento | **Ninguno** (placeholder didáctico) | MIT-BIH, 112.552 latidos, 48 registros |
| Estado | Histórico; hoy, **fallback** de la API | Activo cuando existe el artefacto (Fase 2: T09–T14) |

- **Parte I** documenta el baseline heurístico del MVP tal como fue:
  un placeholder didáctico escrito a mano, que sigue intacto y se usa
  como respaldo cuando no hay artefacto entrenado (§12).
- **Parte II** documenta el **modelo real** de la Fase 2: dataset
  MIT-BIH, split por sujeto real, features, métricas reales,
  integración en la API y limitaciones (issues #12–#17 y #61).
- **Parte III (§14)** es la hoja de ruta actualizada (antes §7).

---

# Parte I — Modelo heurístico MVP (histórico)

## 1. ¿Qué es el modelo del MVP?

Un **detector baseline heurístico** (reglas sobre features
de la señal), **no un modelo entrenado**. No se entrena con
ningún dataset en este milestone: las reglas se escriben a
mano para aproximar lo que mediría un modelo entrenado en
datasets públicos, y estaban diseñadas para ser sustituidas
por un modelo real (hecho en la Fase 2: ver Parte II, §7 en
adelante). Hoy este detector es el **fallback** de la API
(§12) y este capítulo documenta lo que fue, sin borrar nada.

| Aspecto | MVP |
| --- | --- |
| Tipo | Baseline heurístico (reglas) |
| Entrenamiento | **Ninguno** (sin dataset de entrenamiento) |
| Versión | `heuristic-mvp-0.1.0` (constante en `backend/app/model.py`) |
| Entrada | Señal de un solo canal (lead único), ya muestreada (JSON/CSV) |
| Salida | Etiqueta de anomalía + confianza ∈ [0,1] + zonas anómalas |

## 2. Pipeline de señal (`backend/app/signal.py`)

1. **Conversión de unidades** a mV (`uV` → `/1000`).
2. **Filtrado** pasa-banda **0.5–45 Hz** (Butterworth orden
   4, `sosfiltfilt` sin fase; caída a `sosfilt` causal si la
   señal es demasiado corta). Banda clínica estándar:
   elimina deriva de línea base (< 0.5 Hz) y ruido de
   red/músculo (> 45 Hz).
3. **Normalización** z-score (para la detección de picos;
   las medidas de amplitud se hacen en mV).
4. **Detección de picos R**: `scipy.signal.find_peaks` con
   período refractario de 300 ms (máx. ~200 lpm) y umbral
   adaptativo: descarta picos por debajo del 40 % del
   percentil 90 de las alturas (en un ECG, R es la desviación
   más alta: R >> T > P; las ondas T, ~25 % de la altura del
   R, quedan excluidas).
5. **Segmentación**: ventana de ±0.4 s por pico R (un
   "latido"), con la amplitud R medida en ±0.06 s.

Este mismo pipeline reutiliza el modelo real (§12) para
detectar picos y ventanas sobre la señal resampleada.

## 3. Features (entrada de las reglas)

| Feature | Cómo se calcula | Qué aproxima |
| --- | --- | --- |
| Frecuencia cardíaca (lpm) | `60 / mediana(RR)`, RR = intervalos entre picos R | FC media de la anotación de referencia |
| CV de RR | `std(RR) / mean(RR)` | Regularidad del ritmo (arritmia) |
| Amplitud R (mV) | Mediana de `max\|x\|` en ±60 ms del R | Criterios de voltaje del QRS |
| Ratio de ruido | Energía > 40 Hz / energía total | Calidad de la señal |

## 4. Reglas de clasificación (`backend/app/model.py`)

En orden de prioridad (la primera que se cumple gana):

| # | Regla | Etiqueta (`anomalia`) | Umbral (didáctico) |
| --- | --- | --- | --- |
| 1 | < 2 picos R detectados | `senal_no_valorable` | — |
| 2 | Ratio de ruido > 0.30 | `ruido_excesivo` | energía HF > 30 % |
| 3 | CV de RR > 0.15 (≥ 6 latidos) | `arritmia` | irregularidad del ritmo |
| 4 | FC > 100 lpm | `taquicardia` | FC > 100 lpm |
| 5 | FC < 60 lpm | `bradicardia` | FC < 60 lpm |
| 6 | Amplitud R fuera de [0.1, 3.0] mV | `amplitud_anomala` | criterio de voltaje |
| 7 | Ninguna anterior | `normal` | — |

**Zonas anómalas** (`zonas_anomalas`): índices de muestra
(`inicio` inclusivo, `fin` exclusivo) sobre la señal
subida.

- `taquicardia` / `bradicardia` / `ruido_excesivo`: una
  zona que cubre toda la señal (el hallazgo es global).
- `arritmia`: ventana de cada latido cuyo RR se desvía
  > 20 % de la mediana (`tipo: latido_irregular`).
- `amplitud_anomala`: ventana de cada latido con amplitud
  fuera de rango (`tipo: amplitud_baja` / `amplitud_alta`).
- `normal` / `senal_no_valorable`: lista vacía.

**Confianza** (`confianza`): función heurística de la
fuerza de la evidencia (regularidad del ritmo, nº de
latidos, nivel de ruido, severidad de la desviación),
acotada a [0.50, 0.95] según el caso. **No es una
probabilidad calibrada**: solo ordena resultados. Se
almacena redondeada a 3 decimales (`numeric(4,3)`).

## 5. Dataset y métricas (del heurístico)

**Dataset de entrenamiento: ninguno.** El heurístico no
entrena ni evalúa un modelo de ML; por tanto **no hay
métricas de ML** (accuracy, sensibilidad, especificidad)
que reportar **para este detector**. Sé explícito sobre
esto para evitar una falsa sensación de validación.
(Las métricas reales del modelo entrenado están en §11.)

Lo que el detector sí aproxima (y cómo se verifica en las
pruebas, todo con **señales sintéticas** — ver
`tests/backend/synthetic.py`):

| Métrica aproximada | Comportamiento esperado | Cobertura en tests |
| --- | --- | --- |
| FC medida vs. FC real de la señal | Error < 2 lpm en señales limpias | `test_extract_features_frecuencia_cardiaca` |
| Detección de latidos | ±1 latido vs. latidos simulados | `test_detect_r_peaks_cuenta_latidos` |
| Clasificación por reglas | Etiqueta esperada en cada familia sintética | `test_analyze.py`, `test_model.py` |

**Datasets públicos de referencia**: en el MVP no se
descargaba ni almacenaba nada en el repo. En la Fase 2 se
adoptó [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/)
como dataset de entrenamiento real (§8); queda como
candidato futuro [PTB-XL](https://physionet.org/content/ptb-xl/)
(21 837 registros de 12 leads, 5 superclases diagnósticas).
Ningún dato real de paciente propio circula por el repo o la
DB: solo señales sintéticas en las pruebas y datasets
públicos descargados a `data/` (gitignored).

## 6. Limitaciones del heurístico (léelas antes de usar esto)

1. **No es diagnóstico médico.** No detecta infarto,
   fibrilación auricular real, isquemia, hipertrofia, etc.
   Las etiquetas son reglas sobre FC/RR/amplitud, no
   diagnósticos clínicos.
2. **Un solo lead.** El MVP analiza un canal; los criterios
   clínicos de voltaje y la mayoría de modelos usan 12 leads.
3. **Umbrales no validados.** 100/60 lpm, CV 0.15,
   amplitud [0.1, 3.0] mV son valores didácticos, no
   guías clínicas.
4. **Confianza no calibrada.** No es una probabilidad.
5. **Solo picos R positivos.** Un QRS invertido (negativo)
   no se detecta como latido.
6. **Ruido fuerte enmascara todo.** Con `ruido_excesivo`
   las demás medidas no son fiables (por eso la regla 2 va
   antes que las de ritmo).
7. **Longitud mínima 100 muestras** (0.2 s a 500 Hz): un
   ritmo no se puede juzgar en tan poco; la señal se
   etiqueta `senal_no_valorable`.
8. **Frecuencias de muestreo bajas** (fs < ~50 Hz): la
   banda del filtro se limita al Nyquist (p. ej. 0.5–24.5 Hz
   a fs=50) y el QRS pierde fidelidad; con fs ≤ 3 Hz la
   señal no puede representar un latido y cualquier etiqueta
   es poco fiable (la API responde 200 válida, no error).
9. **Sin lectura de formatos médicos** (WFDB/DICOM): la API
   recibe la señal ya muestreada en JSON/CSV (`wfdb` se usa
   solo para descargar el dataset de entrenamiento, §8; el
   soporte WFDB como *entrada* de la API queda en la hoja
   de ruta, §14).

Estas limitaciones siguen vigentes **cada vez que la API
usa el fallback heurístico** (§12).

---

# Parte II — Modelo real (baseline scikit-learn, Fase 2)

## 7. Qué es el modelo real

Primer **modelo entrenado** del proyecto (Fase 2, issues
#13–#17): clasifica **cada latido** del dataset AAMI de
MIT-BIH en una de 5 clases —`N`, `S`, `V`, `F`, `Q`— y la
API agrega esas predicciones a una etiqueta por señal (§12).

| Aspecto | Modelo real |
| --- | --- |
| Tipo | `Pipeline` scikit-learn: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression(class_weight="balanced")` |
| Entrada del clasificador | Ventana **cruda** de ±0.4 s centrada en el pico R (288 muestras @360 Hz) → 14 features |
| Versión | `baseline-sklearn-0.2.0` (constante `MODEL_VERSION` en `backend/app/baseline.py`) |
| Entrenamiento | MIT-BIH `mitdb` (48 registros → 47 sujetos), 112.552 latidos, split **por sujeto real** 37/10 (§9) |
| Métricas de test | accuracy 0,5582 · sensibilidad macro 0,5884 · especificidad macro 0,8992 (§11, desde #61) |
| Artefacto | `*.joblib` con metadatos validados; **nunca commiteado** (gitignored) |
| Código | `mitbih.py`/`mitbih_load.py`/`mitbih_dataset.py` (T09) · `baseline.py` (T10) · `model_loader.py`/`inference.py`/`train_model.py` (T11) |

## 8. Dataset: MIT-BIH Arrhythmia (T09, issues #23–#25)

- **Fuente**: [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
  en PhysioNet, base `mitdb`, descargada con **`wfdb==4.3.1`**
  (`backend/app/mitbih.py`: `wfdb.dl_database("mitdb", …)`).
  Dataset **público** (Open Data Commons Attribution License 1.0).
- **Registros**: los **48 registros canónicos** de `RECORDS`
  (constante `RECORDS` en `mitbih.py`, verificada contra
  PhysioNet 2026-11-05): `100`–`109`, `111`–`119` y `121`–`124`
  sin huecos internos (no existen `110`, `120`, `125`, `126`),
  más `200`–`234` con 10 huecos (no existen `204`, `206`,
  `211`, `216`, `218`, `224`, `225`, `226`, `227`, `229`):
  10 + 9 + 4 + 25 = **48 registros**. Cada registro aporta
  `.hea` + `.dat` + `.atr`.
- **Caché**: `data/mit-bih/` está dentro de `data/`, que está
  entero en `.gitignore` (los `*.npz` solo se ignoran ahí,
  dentro de `data/`; no hay patrón global `*.npz`: los
  patrones globales son `*.dat`/`*.hea`/`*.atr`/`*.joblib`):
  **el dataset y el artefacto nunca se commitean**. La caché
  es por registro (los 3 sufijos deben existir para darlo por
  cacheado).
- **Señal**: **un solo canal por registro** — índice
  `lead=0` (el canal **MLII en la mayoría de registros**;
  102, 104 y 114 tienen V5 en el canal 0, y PhysioNet advierte
  que la configuración de leads varía entre registros) — con
  `fs=360 Hz`, resolución 11 bits sobre 10 mV (grabación
  analógica original).
- **Contenido histórico** (PhysioNet): 48 extractos de ~30 min
  de **47 sujetos** grabados en el Beth Israel Hospital entre
  **1975 y 1979** y **distribuidos por primera vez en 1980**;
  ~110.000 anotaciones de latido revisadas por cardiólogos.
  23 registros se eligieron al azar y **25 a propósito para
  incluir arritmias poco comunes**, así que los resultados
  agrupados **no** estiman la prevalencia de arritmias en
  población general (PhysioNet, Usage Notes).
- **Etiquetado AAMI** (`backend/app/mitbih_dataset.py`):
  correspondencia AAMI EC57 (de Chazal et al., IEEE TBME 2004):
  `N ← {N, L, R, e, j}`, `S ← {A, a, J, S}`, `V ← {V, E}`,
  `F ← {F}`, `Q ← {/, f, Q}`; **cualquier símbolo no latido o
  desconocido → `Q`** (criterio explícito, nunca en silencio).
- **Ventanas**: ±0.4 s por anotación (`BEAT_WINDOW_S=0.4`);
  se **excluyen** las ventanas recortadas en los bordes (1–2
  latidos por registro, ~0,1 %), sin padding.
- **Dataset resultante** (entrenamiento de T13):
  **112.552 latidos**, 48 registros, `fs=360`, ancho 288,
  `dataset_id = 818a3a4ad239bf44` (sha256 de esquema +
  registros + lead + fs + `X`/`y`/`paciente_id`, revalidado
  al cargar el NPZ). Distribución global:
  **N 90.588 (80,5 %)**, S 2.781 (2,5 %), V 7.235 (6,4 %),
  F 802 (0,7 %), Q 11.146 (9,9 %).

## 9. Split por sujeto real y anti-fuga (T10/T13, issues #13, #16 y #61)

**`baseline.split_por_paciente`** particiona por **sujeto
real**: agrupa los IDs de paciente con `_agrupar_sujetos`
(constante `MULTI_RECORD_SUBJECTS`) y muestrea esos **sujetos**
(no los latidos): todos los latidos de un registro van juntos
a train o a test **y** los registros del mismo sujeto caen en
el mismo bloque.

- **Agrupación explícita y centralizada** (#61): en MIT-BIH
  hay **48 registros de 47 sujetos** y el único par que
  comparte sujeto es **201 y 202** (PhysioNet, Usage Notes:
  *«records 201 and 202 should be assigned to the same
  partition because they originate from the same subject»*;
  48 − 47 = exactamente un sujeto con dos registros). El
  mapeo es `MULTI_RECORD_SUBJECTS = {"202": "201"}` y vive
  en `backend/app/baseline.py` (sin lógica duplicada: lo
  aplica el propio `split_por_paciente`). Con IDs sin
  agrupar la función es **identidad** (regresión: los tests
  de T10 siguen intactos).
- `test_size=0.2` sobre sujetos → `ceil(0.2 × 47)` =
  **10 sujetos de test / 37 de train** (train nunca vacío).
- Determinista con `random_state=42`
  (`numpy.random.default_rng`).
- Verificación **anti-fuga dentro de la propia función**: si
  tras la partición un registro **o un sujeto** apareciera en
  ambos conjuntos (imposible por construcción), lanza
  `AssertionError`.
- **Resultado real** (re-entrenamiento de #61, 2026-11-11):
  **train 37 sujetos (38 registros) / 86.853 latidos y test
  10 sujetos (10 registros) / 25.699 latidos**, intersección
  de sujetos vacía. Registros de test: `105, 119, 200, 203,
  207, 212, 213, 231, 232, 234`; **201 y 202 van juntos a
  train** con la semilla 42.
- **Histórico T13** (antes de #61, split por ID de registro):
  train 38 registros / 88.723 latidos y test 10 registros /
  23.829 latidos, intersección vacía (reconstruida y
  verificada de forma independiente por @qa). Registros de
  test: `105, 122, 201, 207, 209, 214, 219, 231, 232, 234`
  — `201` en test y `202` en train: la **fuga residual de 1
  sujeto** que #61 corrigió.

**Justificación (por qué no partir por latido):** los
registros de MIT-BIH tienen correlación **intra-paciente**
(misma persona, misma derivación, mismo aparato): si un
paciente cae en train y test, el modelo memoriza al paciente
y el test deja de medir generalización — **fuga de datos**.

**Control que lo demuestra (informe de T13, issue #16):** un
split aleatorio **por muestra** (con fuga: los 48 pacientes
en train *y* test) da métricas artificialmente mejores:

| Métrica (test) | Split por paciente (el nuestro) | Split por muestra (con fuga) |
| --- | --- | --- |
| accuracy | 0,5158 | **0,6776** (+16 pts) |
| sensibilidad macro | 0,3584 | 0,6998 |
| especificidad macro | 0,8823 | 0,9157 |

Esos +16 pts de accuracy son **inflados por la fuga**, no
calidad real: por eso el proyecto usa solo el split por
sujeto. La fuga residual de 1 sujeto detectada en T12 (los
registros **201 y 202**, mismo sujeto, en bloques distintos
con la semilla 42) quedó **resuelta en #61** agrupando por
sujeto real: hoy el split es **sin fuga por sujeto**
(§13.8).

## 10. Features y modelo (T10, `backend/app/baseline.py`)

**14 features por latido** sobre la ventana cruda en mV
(orden = contrato de serialización, `FEATURE_NAMES`):

| Grupo | Features | Descripción |
| --- | --- | --- |
| Amplitud (mV) | `amp_r_mv`, `amp_max_mv`, `amp_min_mv`, `amp_pp_mv` | `amp_r_mv` = `max\|x\|` en ±60 ms del R (mismo criterio que `extract_features` de `signal.py`); el resto, de la ventana entera |
| Estadísticos | `media_mv`, `desv_mv`, `rms_mv`, `energia_mv2` | Media/desviación/RMS/energía del latido |
| Morphology | `asimetria`, `simetria_corr` | Asimetría entre mitades y correlación de Pearson izquierda vs. derecha invertida (QRS estrecho y simétrico ≈ 1) |
| Ritmo (RR/FC) | `rr_prev_s`, `rr_next_s`, `fc_bpm`, `n_picos` | Segundos al pico R vecino visible en la ventana, FC instantánea `60/min(rr)` y nº de picos R; picos detectados **reutilizando** `bandpass_filter` + `detect_r_peaks` de `signal.py` |

- **Limitación de los features de ritmo**: el vecino solo
  cabe si `RR < 2 × 0.4 s` (0,8 s ⇒ FC > 75 lpm); si no hay
  vecino los tres features son `NaN` y el `SimpleImputer`
  los rellena con la **mediana del train** (nunca con
  información del test).

**Modelo** (`_build_pipeline`):

```
SimpleImputer(strategy="median", keep_empty_features=True)
    → StandardScaler()
    → LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
```

**Por qué lineal** (justificación documentada en
`baseline.py`):

- **Rápido y entendible**: entrena en milisegundos sobre
  14 features y sus coeficientes dicen qué feature pesa —
  una referencia *interpretable* antes de redes neuronales.
- **Determinista**: solver `lbfgs` sin aleatoriedad;
  `random_state` fijo solo por trazabilidad.
- **`class_weight="balanced"`** compensa el desbalance AAMI
  (N domina; Q es escaso) sin re-muestrear.
- `RandomForestClassifier` **descartado por ahora** (más
  lento, modelo mayor y sobreajustado con tan pocas features
  de bajo nivel); probarlo queda en la hoja de ruta (§14).

**Serialización**: `save_model`/`load_model` con `joblib`
(`scikit-learn==1.8.0`, `joblib==1.5.2`) y metadatos
validados al cargar: `model_version`, `dataset_id`,
`sklearn_version`, clases AAMI, `random_state`,
`feature_names`, `fs`, etc.

## 11. Métricas reales (T13 y re-validación #61, issues #16 y #61)

Entrenamiento **real** con los 48 registros MIT-BIH. Cifras
**vigentes desde 2026-11-11** (re-entrenamiento tras #61:
split por **sujeto real**, `MODEL_VERSION =
baseline-sklearn-0.2.0`, mismo dataset — `dataset_id =
818a3a4ad239bf44`, 112.552 latidos). El entrenamiento de
T13 (2026-11-09, validado por @qa con 0 defectos y
determinismo confirmado) queda como **histórico** más abajo
y en el comentario de cierre de issue #16.

**Resultado en test** (25.699 latidos de 10 sujetos, nunca
vistos en train; con `random_state=42`, **201 y 202 van
juntos a train** — §9):

| Métrica | Valor (desde #61, 2026-11-11) | Histórico T13 (antes de #61) |
| --- | --- | --- |
| Exactitud (accuracy) | **0,5582** | 0,5158 |
| Sensibilidad macro | **0,5884** | 0,3584 |
| Especificidad macro | **0,8992** | 0,8823 |

> Al corregir el split, la **partición de test cambió**
> (47 sujetos en vez de 48 registros → otros 10 sujetos:
> `105, 119, 200, 203, 207, 212, 213, 231, 232, 234`), así
> que la comparación refleja el **reparto completo nuevo**,
> no solo la fuga corregida. El control con fuga completa de
> §9 (+16 pts) sigue siendo la referencia del efecto de una
> fuga sobre este dataset.

**Por clase** (`zero_division=0`; orden AAMI; test
2026-11-11, 25.699 latidos):

| Clase | Significado | Sensibilidad | Especificidad | Soporte (test) | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | latido normal | 0,5390 | 0,9520 | 19.931 | 10.743 | 277 | 9.188 | 5.491 |
| S | supraventricular | 0,4206 | 0,9036 | 1.600 | 673 | 2.323 | 927 | 21.776 |
| V | ventricular | 0,8758 | 0,8902 | 2.190 | 1.918 | 2.581 | 272 | 20.928 |
| F | fusión | 0,6192 | 0,8748 | 365 | 226 | 3.173 | 139 | 22.161 |
| Q | desconocida/artefacto | 0,4873 | 0,8755 | 1.613 | 786 | 2.999 | 827 | 21.087 |

**Matriz de confusión 5×5** (filas = clase real, columnas =
predicha, orden N S V F Q; test 2026-11-11):

| real \ predicha | N | S | V | F | Q |
| --- | --- | --- | --- | --- | --- |
| **N** | 10.743 | 2.158 | 1.798 | 3.075 | 2.157 |
| **S** | 142 | 673 | 50 | 30 | 705 |
| **V** | 44 | 32 | 1.918 | 64 | 132 |
| **F** | 79 | 5 | 50 | 226 | 5 |
| **Q** | 12 | 128 | 683 | 4 | 786 |

Distribución de predicciones en test: N 11.020, S 2.996,
V 4.499, F 3.399, Q 3.785 (las 5 clases reciben
predicciones: el baseline **no colapsa** a una sola clase).
Coherencia verificada: filas de la CM = soportes de la
tabla; trazón 14.346 / 25.699 = 0,5582 = accuracy; macros =
medias de las 5 filas.

**Lectura honesta**:

- Cifras: **accuracy 0,5582 · sens_macro 0,5884 ·
  spec_macro 0,8992** (stdout de
  `python -m app.train_model`, 2026-11-11; coherencia
  aritmética comprobada fila a fila, ver arriba); ver
  **issue #16** para el informe completo de T13 (con el CM
  histórico y el control anti-fuga de §9).
- **Nota T13-1 actualizada (#61)**: en el test de T13 la
  clase **F** tenía soporte **4** → `sens F = 0` no era
  significativa; con el reparto nuevo F tiene **365**
  latidos en test (`sens F = 0,6192`), mucho más fiable.
  Queda matizada, no eliminada: F sigue siendo escaso a
  nivel global (802 de 112.552, §13.3).
- Es un **baseline lineal intencionadamente simple**: unas
  métricas modestas son esperables y honestas; no mide
  capacidad diagnóstica ninguna.
- El modelo se equivoca sobre todo confundiendo N con
  F/V/Q/S (fila N de la CM: 3.075 N→F, 2.158 N→S, 2.157
  N→Q, 1.798 N→V) y es conservador con S y Q (recall 0,42 y
  0,49, especificidad ~0,90); destaca en V (sens 0,8758).

**Histórico T13 (2026-11-09, antes de #61):** test de
23.829 latidos de 10 registros (split por ID de registro,
con la fuga residual 201/202 de §13.8) → accuracy 0,5158 ·
sens_macro 0,3584 · spec_macro 0,8823. Tabla por clase y CM
5×5 completas en el **issue #16** (transcritas allí para
que la cita sea autosuficiente): N sens 0,5481 · S 0,1895 ·
V 0,5045 · F 0,0000 (soporte 4) · Q 0,5497.

## 12. Uso en la API (T11, issues #44–#47)

`run_analysis` (`backend/app/pipeline.py`) decide así qué
modelo responde:

1. `get_real_model()` (`model_loader.py`) intenta cargar el
   artefacto de `MODEL_PATH` (caché thread-safe por proceso).
2. **Si hay artefacto** → `infer_detection`
   (`backend/app/inference.py`):

   ```
   señal original (mV, fs_orig)
     → resampleo al fs del modelo (p. ej. 360 Hz, resample_poly:
       fase cero, anti-alias; si fs_orig == fs_model no se remuestrea;
       tope 4.000.000 de muestras, truncado por prefijo)
     → pasa-banda + z-score + picos R + ventanas ±0,4 s
       (app.signal, sobre la señal resampleada; SOLO ventanas
       completas, igual que el dataset de entrenamiento)
     → predict / predict_proba por latido
     → agregado a nivel de señal
     → zonas mapeadas de vuelta a índices de la señal ORIGINAL
   ```

   **Agregado por latido** (`aggregate_beats`):

   - `< 2` latidos evaluables → `senal_no_valorable`
     (confianza 0,0).
   - Fracción de latidos **no-N** `< 0,20`
     (`NON_N_FRACTION_THRESHOLD`) → `normal` sin zonas (un
     latido anómalo cada cinco se tolera como ruido de
     clasificación).
   - En caso contrario → la clase no-N **dominante por
     conteo**; en empate gana el orden canónico AAMI
     (`N < S < V < F < Q`). Una zona por latido no-N (hasta
     `MAX_ZONES = 500`), **ya en índices de la señal
     original**.

   **Vocabulario AAMI en español** (`AAMI_TO_ANOMALIA` /
   `AAMI_TO_ZONA`):

   | Clase AAMI | `anomalia` | `tipo` de zona |
   | --- | --- | --- |
   | N | `normal` | — (sin zona) |
   | S | `arritmia_supraventricular` | `latido_supraventricular` |
   | V | `arritmia_ventricular` | `latido_ventricular` |
   | F | `latido_fusion` | `latido_fusion` |
   | Q | `clase_desconocida` | `latido_desconocido` |

   **Confianza**: media de `max(predict_proba)` de los
   latidos que **sustentan** la etiqueta (los no-N si es
   anómala, todos si es `normal`), recortada a [0,1]. Es la
   probabilidad del modelo **sin calibrar**.

3. **Fallback al heurístico** (filosofía "nunca romper el
   request"): si el artefacto **no existe, es corrupto o la
   ruta es inválida** (loader → `None` + `logger.warning`), o
   si la inferencia real lanza una excepción, se usa el
   heurístico de Parte I con sus etiquetas y su versión.
   La respuesta incluye siempre `model_version`:
   `baseline-sklearn-0.2.0` con modelo real,
   `heuristic-mvp-0.1.0` con fallback.

**Entrega del artefacto** (nunca commiteado):

| Pieza | Detalle |
| --- | --- |
| Env var | `MODEL_PATH` (default `data/models/baseline.joblib`, relativa al WORKDIR); leída en runtime, `~` expandido |
| CLI | `python -m app.train_model` desde `backend/`: descarga (idempotente) → `build_dataset` → split por sujeto real → imprime métricas por stdout (accuracy, sensibilidad/especificidad macro, matriz de confusión…) y escribe el artefacto. Exit 0/1/2 |
| Compose | `docker-compose.yml`: `MODEL_PATH: ${MODEL_PATH:-data/models/baseline.joblib}` + volumen **`./data:/app/data:ro`** (el contenedor solo lee) |
| Git | `*.joblib` y `data/` en `.gitignore`; las pruebas escriben solo en `tmp_path` |

**Pruebas** (T14, issue #17): la suite verifica que la API
usa el modelo real (versión desde los metadatos del
artefacto, E2E dataset → respuesta a 360 y 500 Hz) y que el
fallback emite warning — **339 tests backend + 73 frontend**
en verde.

## 13. Limitaciones del modelo real (léelas antes de usar esto)

1. **No es diagnóstico médico.** Herramienta educativa, no
   uso clínico: no detecta infarto, isquemia, fibrilación
   auricular real, etc. Las clases son anotaciones AAMI de
   latidos, no diagnósticos clínicos.
2. **Un solo lead.** Tanto el entrenamiento como la
   inferencia usan un canal (`lead=0`, MLII en la mayoría de
   registros de mitdb); los
   criterios clínicos de voltaje y la mayoría de modelos
   usan 12 leads.
3. **Clases desbalanceadas.** N es el **80,5 %** del dataset
   (90.588 de 112.552); F solo 802 (0,7 %). En el test
   vigente (#61) F tiene soporte **365** (era **4** en T13:
   `sens F = 0` no era significativa entonces — Nota T13-1,
   §11). `class_weight="balanced"` ayuda
   pero no elimina el desbalance.
4. **Dataset antiguo.** Grabaciones de **1975-1979**
   (distribuidas por primera vez en 1980), cinta analógica,
   360 Hz y 11 bits; 48 registros de **47 sujetos** de un
   único hospital. Además, 25 de los 48 registros se
   seleccionaron a propósito por arritmias poco comunes: los
   resultados agrupados **no** reflejan prevalencias
   poblacionales (PhysioNet). La tecnología y la población
   de entonces no representan la práctica clínica actual.
5. **fs de entrenamiento 360 Hz.** El modelo se entrenó y
   evaluó a 360 Hz; en la API las señales entrantes se
   **resamplean** al fs del artefacto (§12), lo que puede
   degradar señales muy distintas (y existe el tope de
   4 M de muestras con truncado por prefijo).
6. **Métricas modestas.** Accuracy 0,5582 y sensibilidad
   macro 0,5884 en test por sujeto (§11): es un baseline
   lineal, falla con frecuencia (p. ej. la sensibilidad de Q
   es 0,4873 y la de N 0,5390). Un baseline, no un
   clasificador fiable.
7. **Agregado y confianza no validados clínicamente.** El
   umbral de agregado no-N 0,20 y el desempate por conteo
   son reglas de ingeniería documentadas, no criterios
   clínicos; la confianza es `max(predict_proba)` **sin
   calibrar** (misma advertencia que §4).
8. **Split por sujeto real (corregido en #61).** Hasta #61
   el split era por **ID de registro** y dejaba una fuga
   residual de 1 sujeto sobre 47 (PhysioNet: los registros
   **201 y 202** son del mismo sujeto y con la semilla 42
   iban a bloques distintos). Desde #61,
   `split_por_paciente` agrupa por sujeto real
   (`MULTI_RECORD_SUBJECTS = {"202": "201"}`): **47
   sujetos, `ceil(0.2 × 47) = 10` en test y sin fuga por
   sujeto** (§9). Al corregirlo las métricas pasaron de
   0,5158/0,3584/0,8823 a 0,5582/0,5884/0,8992, pero con
   test distinto: no atribuible solo a la fuga (§11).
9. **Calidad dependiente del artefacto.** Sin artefacto (o
   corrupto) la API responde con el **heurístico** (§1–§6):
   comprueba siempre `model_version` en la respuesta para
   saber qué modelo dictó el resultado.
10. **Sin validación clínica.** Ninguna de las métricas de
    §11 procede de evaluación clínica: es evaluación de
    ingeniería sobre un dataset público. Proyecto escolar;
    cero datos de pacientes propios en el repo o la DB.
11. **Las limitaciones del heurístico (§6)** siguen vigentes
    cuando la API usa el fallback.

---

# Parte III — Hoja de ruta

## 14. Hoja de ruta del modelo (actualizada — antes §7)

**Hecho (Fase 2, T09–T14, issues #12–#17 y #61)**:

1. ✅ Descarga/lectura/etiquetado de **MIT-BIH** con `wfdb`
   y dataset AAMI serializable (T09).
2. ✅ **Baseline sklearn** con split por paciente sin fuga
   (T10, #13).
3. ✅ **Métricas reales** (sensibilidad, especificidad y
   matriz de confusión por clase) en un split de test
   separado, validadas por @qa (T13, #16) — §11.
4. ✅ Integración en `POST /ecg/analyze` con fallback
   heurístico (T11, #14) y pruebas (T14, #17) — §12.
5. ✅ **Split por sujeto real** (registros 201/202 agrupados)
   y métricas re-validadas con re-entrenamiento (#61) —
   §9/§11.

**Pendiente** (en orden de idoneidad como baseline ya existe):

1. **Modelos no lineales** sobre el mismo split y las mismas
   métricas: RandomForest/XGBoost (documentado como
   descartado en §10) o redes neuronales (**1D-CNN** sobre la
   ventana cruda), siempre con comparación contra este
   baseline.
2. **Multi-lead y 12 leads**: entrenar con más derivaciones
   y probar [PTB-XL](https://physionet.org/content/ptb-xl/)
   (5 superclases) para superar el "un solo lead".
3. **Soporte WFDB/DICOM como entrada de la API** (hoy la
   API acepta JSON/CSV; `wfdb` solo alimenta el dataset).
4. **Evaluación adicional**: otros datasets públicos
   (p. ej. MIT-BIH Noise Stress Test para ruido) y
   calibración de la confianza.
5. Documentar aquí cada nueva versión: métricas, dataset,
   limitaciones y bump de `model_version`.

---

*Documentación del modelo: Parte I para el issue #5 (T03) y
Parte II para los issues #12–#17 (Fase 2, cerrada con T12,
issue #15) y #61 (split por sujeto real). Mantener sincronizada
con `backend/app/model.py`, `signal.py`, `baseline.py`,
`inference.py`, `model_loader.py`, `mitbih*.py` y
`train_model.py`.*
