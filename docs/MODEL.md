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
| Versión | `heuristic-mvp-0.1.0` | `baseline-sklearn-0.1.0` |
| Entrenamiento | **Ninguno** (placeholder didáctico) | MIT-BIH, 112.552 latidos, 48 registros |
| Estado | Histórico; hoy, **fallback** de la API | Activo cuando existe el artefacto (Fase 2: T09–T14) |

- **Parte I** documenta el baseline heurístico del MVP tal como fue:
  un placeholder didáctico escrito a mano, que sigue intacto y se usa
  como respaldo cuando no hay artefacto entrenado (§12).
- **Parte II** documenta el **modelo real** de la Fase 2: dataset
  MIT-BIH, split por paciente, features, métricas reales, integración
  en la API y limitaciones (issues #12–#17).
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
| Amplitud R (mV) | Mediana de `max(|x|)` en ±60 ms del R | Criterios de voltaje del QRS |
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
| Versión | `baseline-sklearn-0.1.0` (constante `MODEL_VERSION` en `backend/app/baseline.py`) |
| Entrenamiento | MIT-BIH `mitdb` (48 registros), 112.552 latidos, split **por paciente** 38/10 |
| Métricas de test | accuracy 0,5158 · sensibilidad macro 0,3584 · especificidad macro 0,8823 (§11) |
| Artefacto | `*.joblib` con metadatos validados; **nunca commiteado** (gitignored) |
| Código | `mitbih.py`/`mitbih_load.py`/`mitbih_dataset.py` (T09) · `baseline.py` (T10) · `model_loader.py`/`inference.py`/`train_model.py` (T11) |

## 8. Dataset: MIT-BIH Arrhythmia (T09, issues #23–#25)

- **Fuente**: [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/)
  en PhysioNet, base `mitdb`, descargada con **`wfdb==4.3.1`**
  (`backend/app/mitbih.py`: `wfdb.dl_database("mitdb", …)`).
  Dataset **público** (Open Data Commons Attribution License 1.0).
- **Registros**: los **48 registros canónicos** de `RECORDS`
  (constante `RECORDS` en `mitbih.py`, verificada contra
  PhysioNet 2026-11-05): `100`–`109`, `111`–`119`, `121`–`124`,
  `200`–`234` con huecos (no existen 110, 120, 125, 126, 204,
  211, 216, 218). Cada registro aporta `.hea` + `.dat` + `.atr`.
- **Caché**: `data/mit-bih/` (y todo `*.dat`/`*.hea`/`*.atr`/
  `*.joblib`/`*.npz`) está en `.gitignore`: **el dataset y el
  artefacto nunca se commitean**. La caché es por registro
  (los 3 sufijos deben existir para darlo por cacheado).
- **Señal**: **un solo canal por registro** — índice
  `lead=0` (el canal MLII en `mitdb`; PhysioNet advierte que
  la configuración de leads varía entre registros) — con
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

## 9. Split por paciente y anti-fuga (T10/T13, issues #13 y #16)

**`baseline.split_por_paciente`** particiona los **IDs de
paciente** (no los latidos): todos los latidos de un registro
van juntos a train o a test.

- `test_size=0.2` sobre pacientes → `ceil(0.2 × 48)` =
  **10 pacientes de test / 38 de train** (train nunca vacío).
- Determinista con `random_state=42`
  (`numpy.random.default_rng`).
- Verificación **anti-fuga dentro de la propia función**: si
  tras la partición un paciente apareciera en ambos conjuntos
  (imposible por construcción), lanza `AssertionError`.
- Resultado real de T13: **train 38 pacientes / 88.723 latidos
  y test 10 pacientes / 23.829 latidos**, intersección vacía
  (reconstruida y verificada de forma independiente por @qa).
  Pacientes de test: `105, 122, 201, 207, 209, 214, 219, 231,
  232, 234`.

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
paciente. Detalle menor: PhysioNet recomienda asignar los
registros **201 y 202 al mismo bloque** (pertenecen al mismo
sujeto); en el split con semilla 42, `202` va a train y `201`
a test (ver limitación §13.8).

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

## 11. Métricas reales (T13, issue #16)

Entrenamiento **real** con los 48 registros MIT-BIH validado
por @qa el 2026-11-09 (0 defectos): descarga ~49 s,
`build_dataset` 112.552 latidos, entrenamiento ~179 s y
**determinismo confirmado** (re-ejecución con `rs=42`
idéntica, 174 s). Las cifras agregadas coinciden exactamente
con el cierre de issue #16 y con `docs/CHANGELOG.md`.

**Resultado en test** (23.829 latidos de 10 pacientes,
nunca vistos en train):

| Métrica | Valor |
| --- | --- |
| Exactitud (accuracy) | **0,5158** |
| Sensibilidad macro | **0,3584** |
| Especificidad macro | **0,8823** |

**Por clase** (`zero_division=0`; orden AAMI):

| Clase | Significado | Sensibilidad | Especificidad | Soporte (test) | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | latido normal | 0,5481 | 0,9032 | 19.544 | 10.713 | 415 | 8.831 | 3.870 |
| S | supraventricular | 0,1895 | 0,8141 | 2.058 | 390 | 4.048 | 1.668 | 17.723 |
| V | ventricular | 0,5045 | 0,9074 | 775 | 391 | 2.135 | 384 | 20.919 |
| F | fusión | 0,0000 | 0,8790 | 4 | 0 | 2.882 | 4 | 20.943 |
| Q | desconocida/artefacto | 0,5497 | 0,9080 | 1.448 | 796 | 2.059 | 652 | 20.322 |

**Matriz de confusión 5×5** (filas = clase real, columnas =
predicha, orden N S V F Q):

| real \ predicha | N | S | V | F | Q |
| --- | --- | --- | --- | --- | --- |
| **N** | 10.713 | 3.925 | 1.575 | 2.827 | 504 |
| **S** | 337 | 390 | 53 | 2 | 1.276 |
| **V** | 51 | 5 | 391 | 52 | 276 |
| **F** | 1 | 0 | 0 | 0 | 3 |
| **Q** | 26 | 118 | 507 | 1 | 796 |

Distribución de predicciones en test: N 11.128, S 4.438,
V 2.526, F 2.882, Q 2.855 (las 5 clases reciben
predicciones: el baseline **no colapsa** a una sola clase).

**Lectura honesta**:

- Cifras: **accuracy 0,5158 · sens_macro 0,3584 ·
  spec_macro 0,8823**, verificadas a mano y con sklearn por
  @qa (coincidencia exacta); ver **issue #16** para el
  informe completo de validación (incluye este CM y el
  control anti-fuga de §9).
- **Nota T13-1** (issue #16): el soporte de **F en test es
  solo 4 latidos** → `sens F = 0` **no es estadísticamente
  significativa**; señálalo siempre que se reporte.
- Es un **baseline lineal intencionadamente simple**: unas
  métricas modestas son esperables y honestas; no mide
  capacidad diagnóstica ninguna.
- El modelo se equivoca sobre todo confundiendo N con
  F/S/Q (véase la fila N de la CM: 3.925 N predichos como S,
  2.827 como F…) y es conservador con S/V/Q (recall bajo,
  especificidad alta).

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
   `baseline-sklearn-0.1.0` con modelo real,
   `heuristic-mvp-0.1.0` con fallback.

**Entrega del artefacto** (nunca commiteado):

| Pieza | Detalle |
| --- | --- |
| Env var | `MODEL_PATH` (default `data/models/baseline.joblib`, relativa al WORKDIR); leída en runtime, `~` expandido |
| CLI | `python -m app.train_model` desde `backend/`: descarga (idempotente) → `build_dataset` → split por paciente → imprime métricas por stdout (accuracy, sensibilidad/especificidad macro, matriz de confusión…) y escribe el artefacto. Exit 0/1/2 |
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
   inferencia usan un canal (`lead=0`, MLII en mitdb); los
   criterios clínicos de voltaje y la mayoría de modelos
   usan 12 leads.
3. **Clases desbalanceadas.** N es el **80,5 %** del dataset
   (90.588 de 112.552); F solo 802 (0,7 %) y en test **F
   tiene soporte 4** → `sens F = 0` no es significativa
   (Nota T13-1, issue #16). `class_weight="balanced"` ayuda
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
6. **Métricas modestas.** Accuracy 0,5158 y sensibilidad
   macro 0,3584 en test por paciente (§11): es un baseline
   lineal, falla con frecuencia (p. ej. la sensibilidad de S
   es 0,1895). Un baseline, no un clasificador fiable.
7. **Agregado y confianza no validados clínicamente.** El
   umbral de agregado no-N 0,20 y el desempate por conteo
   son reglas de ingeniería documentadas, no criterios
   clínicos; la confianza es `max(predict_proba)` **sin
   calibrar** (misma advertencia que §4).
8. **Split por ID de registro.** PhysioNet recomienda que
   los registros **201 y 202** (mismo sujeto) caigan en el
   mismo bloque; con la semilla 42, `202` está en train y
   `201` en test, así que puede quedar una **fuga residual de
   1 sujeto sobre 47**. Menor (el control con fuga completa
   de §9 está a +16 pts), pero corregible (§14).
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

**Hecho (Fase 2, T09–T14, issues #12–#17)**:

1. ✅ Descarga/lectura/etiquetado de **MIT-BIH** con `wfdb`
   y dataset AAMI serializable (T09).
2. ✅ **Baseline sklearn** con split por paciente sin fuga
   (T10, #13).
3. ✅ **Métricas reales** (sensibilidad, especificidad y
   matriz de confusión por clase) en un split de test
   separado, validadas por @qa (T13, #16) — §11.
4. ✅ Integración en `POST /ecg/analyze` con fallback
   heurístico (T11, #14) y pruebas (T14, #17) — §12.

**Pendiente** (en orden de idoneidad como baseline ya existe):

1. **Modelos no lineales** sobre el mismo split y las mismas
   métricas: RandomForest/XGBoost (documentado como
   descartado en §10) o redes neuronales (**1D-CNN** sobre la
   ventana cruda), siempre con comparación contra este
   baseline.
2. **Corregir el split 201/202** agrupando por sujeto real
   (§13.8) y re-validar métricas.
3. **Multi-lead y 12 leads**: entrenar con más derivaciones
   y probar [PTB-XL](https://physionet.org/content/ptb-xl/)
   (5 superclases) para superar el "un solo lead".
4. **Soporte WFDB/DICOM como entrada de la API** (hoy la
   API acepta JSON/CSV; `wfdb` solo alimenta el dataset).
5. **Evaluación adicional**: otros datasets públicos
   (p. ej. MIT-BIH Noise Stress Test para ruido) y
   calibración de la confianza.
6. Documentar aquí cada nueva versión: métricas, dataset,
   limitaciones y bump de `model_version`.

---

*Documentación del modelo: Parte I para el issue #5 (T03) y
Parte II para los issues #12–#17 (Fase 2, cerrada con T12,
issue #15). Mantener sincronizada con
`backend/app/model.py`, `signal.py`, `baseline.py`,
`inference.py`, `model_loader.py`, `mitbih*.py` y
`train_model.py`.*
