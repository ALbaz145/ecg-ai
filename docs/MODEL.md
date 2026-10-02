# Modelo de IA — ECG-AI (MVP)

> ⚠️ **Herramienta educativa, NO es diagnóstico médico.**
> Este modelo es un **placeholder didáctico**: no está
> entrenado, no está validado clínicamente y no debe usarse
> con pacientes reales. Proyecto escolar.

## 1. ¿Qué es el modelo del MVP?

Un **detector baseline heurístico** (reglas sobre features
de la señal), **no un modelo entrenado**. No se entrena con
ningún dataset en este milestone: las reglas se escriben a
mano para aproximar lo que mediría un modelo entrenado en
datasets públicos, y están diseñadas para ser sustituidas
por un modelo real en hitos posteriores (ver §6).

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

## 3. Features (entrada de las reglas)

| Feature | Cómo se calcula | Qué aproxima |
| --- | --- | --- |
| Frecuencia cardíaca (lpm) | `60 / mediana(RR)`, RR = intervalos entre picos R | FC media de la anotación de referencia |
| CV de RR | `std(RR) / mean(RR)` | Regularidad del ritmo (arritmia) |
| Amplitud R (mV) | Mediana de `max(|x|)` en ±00 ms del R | Criterios de voltaje del QRS |
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

## 5. Dataset y métricas

**Dataset de entrenamiento: ninguno.** El MVP no entrena
ni evalúa un modelo de ML; por tanto **no hay métricas de
ML** (accuracy, sensibilidad, especificidad) que reportar.
Sé explícito sobre esto para evitar una falsa sensación de
validación.

Lo que el detector sí aproxima (y cómo se verifica en las
pruebas, todo con **señales sintéticas** — ver
`tests/backend/synthetic.py`):

| Métrica aproximada | Comportamiento esperado | Cobertura en tests |
| --- | --- | --- |
| FC medida vs. FC real de la señal | Error < 2 lpm en señales limpias | `test_extract_features_frecuencia_cardiaca` |
| Detección de latidos | ±1 latido vs. latidos simulados | `test_detect_r_peaks_cuenta_latidos` |
| Clasificación por reglas | Etiqueta esperada en cada familia sintética | `test_analyze.py`, `test_model.py` |

**Datasets públicos de referencia** (para futuras
versiones, no se descargan ni almacenan en este repo):
[PTB-XL](https://physionet.org/content/ptb-xl/)
(21 837 registros de 12 leads, 5 superclases diagnósticas)
y [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/)
(48 registros de 2 leads con latidos anotados). Ningún dato
real de paciente circula por el repo o la DB: solo señales
sintéticas en las pruebas.

## 6. Limitaciones (léelas antes de usar esto)

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
9. **Sin lectura de formatos médicos** (WFDB/DICOM): el MVP
   recibe la señal ya muestreada en JSON/CSV. El soporte de
   `wfdb` queda para un hito posterior.

## 7. Hoja de ruta del modelo (fuera del MVP)

1. Entrenar un modelo ligero (p. ej. 1D-CNN o XGBoost sobre
   features) con **PTB-XL** (superclases: normal,
   bloqueo de rama, ST-T, hipertrofia, misc.): señales
   públicas, sin datos reales de pacientes del centro.
2. Evaluar con métricas reales (sensibilidad/especificidad
   por clase, matriz de confusión) en un split de test
   separado y documentarlas aquí.
3. Añadir detección de arritmias con anotaciones de
   MIT-BIH (latido a latido).
4. Soporte de leads múltiples y formato WFDB.

---

*Documentación del modelo para el issue #5 (T03). Mantener
sincronizada con `backend/app/model.py` y
`backend/app/signal.py`.*
