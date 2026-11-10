"""Inferencia real en el pipeline (T11b, issue #44).

Cuando :func:`app.model_loader.get_real_model` devuelve un artefacto,
``run_analysis`` usa **este módulo** en lugar del heurístico de
``app.model``. Sin artefacto el módulo no se invoca y la rama
heurística queda intacta (mismas etiquetas y ``heuristic-mvp-0.1.0``).

Flujo (issue #44)::

    señal original (mV, fs_orig)
        → resampleo al fs del modelo (p. ej. 360 Hz)
        → pasa-banda + z-score + picos R + ventanas ±0,4 s   (app.signal)
        → predict / predict_proba por latido (BaselineModel)
        → agregado a nivel de señal (regla documentada abajo)
        → zonas mapeadas de vuelta a índices de la señal original

Decisiones de diseño
--------------------

**Todo el flujo de detección corre sobre la señal resampleada** (no
solo la inferencia). Motivos: (1) ``BeatFeatureExtractor`` convierte
muestras → segundos con ``meta["fs"]``, así que una ventana cortada
en otra ``fs`` produciría features de ritmo (``rr_*``/``fc_bpm``)
inconsistentes con el entrenamiento; (2) el ancho de ventana del
modelo (``2 * round(0.4 * fs_model)`` = 288 @360 Hz) es el del train
(T09c); (3) el propio issue pide detectar picos y ventanas sobre la
señal resampleada. Las **zonas** que salen en la respuesta se mapean
siempre a índices de la señal **original** (``0 ≤ inicio < fin ≤ n``)
con :func:`map_window_to_original`.

**Resampleo con ``scipy.signal.resample_poly``** (FIR polifásico):
las ``fs`` del contrato son enteras → la razón es racional
(``up/down`` reducidos con ``math.gcd``); es de fase cero (no desplaza
los picos R, que es lo que alinea las ventanas), no sufre el
"wrap-around" del FFT de :func:`scipy.signal.resample` en los bordes
y aplica automáticamente el FIR que corta alias (al **bajar** de fs)
e imágenes (al subir). Solo se
usa :func:`scipy.signal.resample` (FFT) como respaldo si ``fs_model``
no es un número entero (los artefactos actuales guardan 360.0). Si
``fs_orig == fs_model`` no se
remeuestra nada (copia identidad). Justificación elegida sobre
``resample`` puro: ver arriba; el coste polifásico es ``O(n)``.

**Tope de longitud** (:data:`MAX_RESAMPLED_SAMPLES`): el contrato
acepta hasta 2 M de muestras a 1 Hz, que sin tope serían 720 M de
muestras resampleadas (≈ 5,7 GB). Si el resultado excede el tope se
trunca la entrada por **prefijo** (las zonas siguen dentro de
``[0, n)`` porque el prefijo es una subseñal contigua).

**Ventanas completas**: se predicen solo las ventanas ±0,4 s **sin
recortar en los bordes** (``fin - inicio == 2 * round(0.4 * fs)``),
igual que T09c excluye las ventanas recortadas del dataset: una
ventana a medias no tiene el pico R centrado y sus features no
corresponden a lo entrenado. Con menos de 2 ventanas evaluables la
señal es ``senal_no_valorable`` (misma guarda que el heurístico).

**Agregado a nivel de señal** (:func:`aggregate_beats`):

1. ``< 2`` latidos evaluables → ``senal_no_valorable`` (confianza
   ``0.0``: sin latidos que sustenten la etiqueta, no hay evidencia
   del modelo que reportar).
2. Fracción de latidos con clase ``!= N`` **<**
   :data:`NON_N_FRACTION_THRESHOLD` (0,20) → ``normal`` **sin zonas**
   (los latidos no-N por debajo del umbral se consideran
   clasificación espuria); la confianza es la media de
   ``max(predict_proba)`` de **todos** los latidos.
3. En caso contrario → la clase no-N **dominante por conteo**; en
   empate gana la de menor índice en el orden canónico AAMI
   (``N < S < V < F < Q``, es decir ``S`` antes que ``V`` antes que
   ``F`` antes que ``Q``): desempate determinista y documentado. La
   confianza es la media de ``max(predict_proba)`` de los latidos
   **no-N** (los que sustentan la etiqueta). Una zona por latido no-N
   (hasta :data:`~app.model.MAX_ZONES`).

**Mapeo de clases**: ``N/S/V/F/Q`` → ``normal`` /
``arritmia_supraventricular`` / ``arritmia_ventricular`` /
``latido_fusion`` / ``clase_desconocida`` y zonas
``latido_supraventricular`` / ``latido_ventricular`` /
``latido_fusion`` / ``latido_desconocido`` (ver
:data:`AAMI_TO_ANOMALIA` / :data:`AAMI_TO_ZONA`).

**Robustez**: este módulo puede lanzar ``ValueError`` (modelo con
``fs`` inválido, metadatos incoherentes…); ``run_analysis`` lo
captura y cae al heurístico con ``logger.warning``, siguiendo la
filosofía de T11a de **nunca romper el request**. Una clase predicha
fuera del vocabulario AAMI (imposible con artefactos bien formados)
se trata como ``Q`` en vez de propagar un ``KeyError``.

Pruebas: ``tests/backend/test_inference.py``.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy import signal as sp_signal

from app.baseline import BaselineModel
from app.mitbih_dataset import AAMI_CLASSES
from app.model import LABEL_NO_VALORABLE, LABEL_NORMAL, MAX_ZONES, Detection
from app.signal import (
    BEAT_WINDOW_S,
    bandpass_filter,
    detect_r_peaks,
    extract_features,
    zscore,
)

#: Umbral de agregado: si la fracción de latidos con clase distinta de
#: ``N`` es **menor** que este valor, la señal completa es ``normal``
#: (con ``>=`` umbral manda la clase no-N dominante). 0,20 = un latido
#: anómalo cada cinco se tolera como ruido de clasificación.
NON_N_FRACTION_THRESHOLD = 0.20

#: Tope de muestras de la señal resampleada (~3,1 h a 360 Hz, 32 MB en
#: float64). Sin tope, el mínimo del contrato (1 Hz) forzaría cientos
#: de millones de muestras; ver módulo (truncado por prefijo).
MAX_RESAMPLED_SAMPLES = 4_000_000

#: Clase AAMI → ``anomalia`` de la respuesta (vocabulario en español).
AAMI_TO_ANOMALIA: dict[str, str] = {
    "N": "normal",
    "S": "arritmia_supraventricular",
    "V": "arritmia_ventricular",
    "F": "latido_fusion",
    "Q": "clase_desconocida",
}

#: Clase AAMI → ``zonas_anomalas[].tipo`` (una zona por latido no-N).
AAMI_TO_ZONA: dict[str, str] = {
    "S": "latido_supraventricular",
    "V": "latido_ventricular",
    "F": "latido_fusion",
    "Q": "latido_desconocido",
}

_AAMI_SET = frozenset(AAMI_CLASSES)


def resample_to_fs(arr: np.ndarray, fs_orig: float, fs_target: float) -> np.ndarray:
    """Remuestrea ``arr`` de ``fs_orig`` a ``fs_target`` Hz.

    - ``fs_orig == fs_target`` → copia idéntica (sin filtrar).
    - Razones enteras → :func:`scipy.signal.resample_poly` con
      ``up/down`` reducidos por ``gcd`` (fase cero, anti-alias al
      subir; ver módulo).
    - ``fs_target`` no entera → respaldo :func:`scipy.signal.resample`
      (FFT) con longitud ``round(n * fs_target / fs_orig)``.
    - Si la longitud de salida excede :data:`MAX_RESAMPLED_SAMPLES`,
      la entrada se trunca por **prefijo** antes de remuestrear (el
      resultado siempre cabe en el tope; ver módulo).

    Raises
    ------
    ValueError
        Si alguna ``fs`` no es finita y positiva.
    """
    if not np.isfinite(fs_orig) or fs_orig <= 0:
        raise ValueError(f"fs_orig inválida: {fs_orig!r} (debe ser finita y > 0).")
    if not np.isfinite(fs_target) or fs_target <= 0:
        raise ValueError(f"fs_target inválida: {fs_target!r} (debe ser finita y > 0).")
    if fs_orig == fs_target:
        return arr.copy()

    # Tope de memoria: truncar por prefijo mapea a [0, n) siempre.
    n_out = max(1, round(arr.size * fs_target / fs_orig))
    if n_out > MAX_RESAMPLED_SAMPLES:
        n_max = max(1, math.floor(MAX_RESAMPLED_SAMPLES * fs_orig / fs_target))
        arr = arr[:n_max]
        n_out = max(1, round(arr.size * fs_target / fs_orig))

    up = round(fs_target)
    down = round(fs_orig)
    if up >= 1 and down >= 1 and up == fs_target and down == fs_orig:
        divisor = math.gcd(up, down)
        return sp_signal.resample_poly(arr, up // divisor, down // divisor)
    return sp_signal.resample(arr, n_out)


def map_index_to_original(
    index_resampled: int, fs_orig: float, fs_target: float, n_original: int
) -> int:
    """Índice de la **señal original** para un índice resampleado.

    ``índice_original = round(index_resampled * fs_orig / fs_target)``
    acotado a ``[0, n_original - 1]``.
    """
    idx = round(index_resampled * fs_orig / fs_target)
    return min(max(idx, 0), max(n_original - 1, 0))


def map_window_to_original(
    start: int,
    end: int,
    fs_orig: float,
    fs_target: float,
    n_original: int,
) -> tuple[int, int]:
    """Ventana ``(inicio, fin)`` de dominio resampleado → original.

    Garantiza ``0 <= inicio < fin <= n_original`` (recortando en los
    bordes): el mapeo por redondeo puede "pegar" ``fin`` una muestra
    más allá del final o colapsar una ventana minúscula (fs originales
    muy bajas), así que se re-acota defensivamente.
    """
    ratio = fs_orig / fs_target
    ini = round(start * ratio)
    fin = round(end * ratio)
    ini = min(max(ini, 0), max(n_original - 1, 0))
    fin = min(max(fin, ini + 1), max(n_original, 1))
    return ini, fin


def aggregate_beats(
    preds: np.ndarray,
    confianza_latido: np.ndarray,
    zonas: list[tuple[int, int]],
) -> Detection:
    """Agrega las predicciones por latido a una detección de señal.

    Regla (documentada en el docstring del módulo): ``< 2`` latidos →
    ``senal_no_valorable``; fracción no-N ``<``
    :data:`NON_N_FRACTION_THRESHOLD` → ``normal``; si no, clase no-N
    dominante por conteo (empate → orden canónico AAMI). La
    ``confianza`` es la media de ``max(predict_proba)`` de los latidos
    que **sustentan** la etiqueta (los no-N si es anómala, todos si es
    ``normal``), recortada a ``[0, 1]``.

    Parameters
    ----------
    preds:
        Clase AAMI predicha por latido, forma ``(n_latidos,)``.
    confianza_latido:
        ``max(predict_proba)`` por latido, forma ``(n_latidos,)``.
    zonas:
        Ventana ``(inicio, fin)`` de cada latido **ya en índices de la
        señal original** (mismo orden que ``preds``).

    Raises
    ------
    ValueError
        Si las tres entradas no están alineadas por latido.
    """
    preds_arr = np.asarray([str(p) for p in np.asarray(preds).ravel()])
    conf_arr = np.asarray(confianza_latido, dtype=np.float64).ravel()
    if not (preds_arr.size == conf_arr.size == len(zonas)):
        raise ValueError(
            "Entradas desalineadas: preds, confianza_latido y zonas deben "
            f"tener la misma longitud (recibido: {preds_arr.size}, "
            f"{conf_arr.size}, {len(zonas)})."
        )
    # Una clase fuera del vocabulario AAMI (solo posible con un bundle
    # corrupto; load_model valida los metadatos pero no classes_) se
    # trata como Q en vez de romper el request.
    preds_arr = np.array([p if p in _AAMI_SET else "Q" for p in preds_arr])

    n_latidos = preds_arr.size
    if n_latidos < 2:
        return Detection(
            anomalia=LABEL_NO_VALORABLE,
            confianza=0.0,
            zonas=[],
            motivo=(
                f"solo {n_latidos} latido(s) evaluables: se necesitan >= 2 "
                "para juzgar el ritmo con el modelo"
            ),
        )

    no_n = preds_arr != "N"
    n_no_n = int(np.count_nonzero(no_n))
    fraccion = n_no_n / n_latidos

    # 1) Casi todo normal → señal normal (confianza sobre TODOS los latidos).
    if fraccion < NON_N_FRACTION_THRESHOLD:
        confianza = float(np.clip(conf_arr.mean(), 0.0, 1.0))
        return Detection(
            anomalia=LABEL_NORMAL,
            confianza=confianza,
            zonas=[],
            motivo=(
                f"{n_latidos} latidos clasificados, {n_no_n} no-N "
                f"({fraccion:.0%}) < umbral {NON_N_FRACTION_THRESHOLD:.0%}"
            ),
        )

    # 2) Clase no-N dominante por conteo; empate → orden canónico AAMI.
    conteos = Counter(str(p) for p in preds_arr[no_n])
    dominante = min(
        conteos,
        key=lambda clase: (
            -conteos[clase],
            AAMI_CLASSES.index(clase) if clase in _AAMI_SET else len(AAMI_CLASSES),
        ),
    )
    zonas_out: list[dict[str, int | str]] = [
        {"inicio": int(ini), "fin": int(fin), "tipo": AAMI_TO_ZONA[str(p)]}
        for p, (ini, fin) in zip(preds_arr, zonas, strict=True)
        if p != "N"
    ][:MAX_ZONES]
    # Confianza sobre los latidos no-N (los que sustentan la etiqueta).
    confianza = float(np.clip(conf_arr[no_n].mean(), 0.0, 1.0))
    return Detection(
        anomalia=AAMI_TO_ANOMALIA[dominante],
        confianza=confianza,
        zonas=zonas_out,
        motivo=(
            f"{n_latidos} latidos clasificados, {n_no_n} no-N "
            f"({fraccion:.0%}) >= umbral {NON_N_FRACTION_THRESHOLD:.0%}; "
            f"dominante {dominante} ({conteos[dominante]} latidos)"
        ),
    )


def infer_detection(
    model: BaselineModel, original_mv: np.ndarray, fs_orig: float
) -> Detection:
    """Clasifica la señal con el modelo real (ver flujo del módulo).

    Resamplea ``original_mv`` al ``fs`` de los metadatos del modelo,
    detecta picos R y ventanas ±0,4 s **sobre la resampleada**
    (reutilizando ``app.signal``), predice por latido y agrega con
    :func:`aggregate_beats`. Las zonas del ``Detection`` vuelven a
    índices de la señal **original**.

    Raises
    ------
    ValueError
        Si el ``fs`` de los metadatos es inválido o el modelo no puede
        predecir (p. ej. ventana más corta que el mínimo); el caller
        (``run_analysis``) cae al heurístico.
    """
    fs_target = float(model.meta["fs"])
    if not np.isfinite(fs_target) or fs_target <= 0:
        raise ValueError(f"fs del modelo inválida en meta: {fs_target!r}.")

    # --- Resampleo al fs del modelo -------------------------
    resampled = resample_to_fs(original_mv, fs_orig, fs_target)
    n_original = int(original_mv.size)

    # --- Filtrado, picos R y ventanas (app.signal) ----------
    filtered = bandpass_filter(resampled, fs_target)
    normalized = zscore(filtered)
    r_peaks = detect_r_peaks(normalized, fs_target)
    features = extract_features(
        original_mv=resampled,
        filtered=filtered,
        r_peaks=r_peaks,
        fs=fs_target,
        sample_rate_hz=round(fs_target),
    )

    # Solo ventanas completas (sin recorte en los bordes, como T09c).
    beat_win = max(1, round(BEAT_WINDOW_S * fs_target))
    ventanas = [
        (start, end)
        for start, end in features.beat_windows
        if end - start == 2 * beat_win
    ]
    if len(ventanas) < 2:
        return Detection(
            anomalia=LABEL_NO_VALORABLE,
            confianza=0.0,
            zonas=[],
            motivo=(
                f"{len(ventanas)} ventana(s) completa(s) de latido: se "
                "necesitan >= 2 para juzgar el ritmo con el modelo"
            ),
        )

    # --- Predicción por latido ------------------------------
    X = np.stack([resampled[start:end] for start, end in ventanas])
    preds = np.asarray(model.predict(X))
    proba = np.asarray(model.predict_proba(X))
    confianza_latido = np.max(proba, axis=1)

    # --- Zonas en índices de la señal ORIGINAL --------------
    zonas_orig = [
        map_window_to_original(start, end, fs_orig, fs_target, n_original)
        for start, end in ventanas
    ]
    return aggregate_beats(preds, confianza_latido, zonas_orig)
