"""Procesamiento de señal ECG: filtrado, normalización y features.

Pipeline de señal (paso 2 del flujo del MVP):

1. **Conversión de unidades** a milivoltios (mV).
2. **Filtrado** pasa-banda 0.5–45 Hz (Butterworth orden 4,
   ``sosfiltfilt`` sin fase; caída a ``sosfilt`` si la señal
   es demasiado corta).
3. **Normalización** z-score (solo para la detección de
   picos; las medidas de amplitud se hacen en mV).
4. **Detección de picos R** (``find_peaks`` con umbral
   adaptativo y período refractario de ~300 ms).
5. **Segmentación** en latidos (ventana ±0.4 s por pico R)
   y extracción de features.

Todo es numpy/scipy puro: sin dependencias de lectura de
formatos médicos (el MVP recibe la señal ya muestreada en
JSON/CSV; el soporte de ``wfdb`` queda para hitos
posteriores, ver docs/MODEL.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal

# Banda clínica típica del ECG (ver docs/MODEL.md).
BAND_LOW_HZ = 0.5
BAND_HIGH_HZ = 45.0
BUTTER_ORDER = 4
#: Frecuencia de corte para estimar el ruido de alta frecuencia.
NOISE_CUTOFF_HZ = 40.0
#: Período refractario entre latidos (s): máx. ~200 lpm.
REFRACTORY_S = 0.3
#: Semivida de la ventana de medición de amplitud (s).
AMP_WINDOW_S = 0.06
#: Ventana de segmentación por latido (s, a cada lado del R).
BEAT_WINDOW_S = 0.4


def to_millivolts(values: list[float], units: str) -> np.ndarray:
    """Convierte la señal a mV (``uV`` → mV dividiendo 1000)."""
    arr = np.asarray(values, dtype=np.float64)
    if units == "uV":
        arr = arr / 1000.0
    return arr


def bandpass_filter(arr: np.ndarray, fs: float) -> np.ndarray:
    """Filtro pasa-banda 0.5–45 Hz (Butterworth, orden 4).

    El borde alto se limita a ``Nyquist - 0.5 Hz``: con
    frecuencias de muestreo bajas (p. ej. 50–90 Hz,
    legítimas según el contrato) la banda se adapta en
    lugar de fallar. Si no cabe una banda usable
    (``Nyquist`` demasiado bajo), se devuelve la señal
    sin filtrar: el modelo la etiquetará como no
    valorable en vez de devolver un error 500.
    """
    nyquist = fs / 2.0
    high = min(BAND_HIGH_HZ, nyquist - 0.5)
    if high <= BAND_LOW_HZ:
        # Imposible filtrar (fs ≤ 2 Hz: ni un latido
        # cabe en la banda de Nyquist).
        return arr.copy()
    sos = sp_signal.butter(
        BUTTER_ORDER,
        [BAND_LOW_HZ, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    try:
        return sp_signal.sosfiltfilt(sos, arr)
    except ValueError:
        # Señal demasiado corta para el filtrado sin fase:
        # filtrado causal (peor, pero mejor que nada).
        return sp_signal.sosfilt(sos, arr)


def zscore(arr: np.ndarray) -> np.ndarray:
    """Normalización z-score (señal plana → ceros)."""
    std = float(np.std(arr))
    if std == 0.0 or not np.isfinite(std):
        return np.zeros_like(arr)
    return (arr - float(np.mean(arr))) / std


@dataclass
class RPeaks:
    """Picos R detectados (índices de muestra y alturas)."""

    indices: np.ndarray
    heights: np.ndarray


def detect_r_peaks(filtered: np.ndarray, fs: float) -> RPeaks:
    """Detecta picos R con umbral adaptativo.

    Dos pasadas: (1) picos con período refractario mínimo
    (~300 ms); (2) descarte de picos por debajo del 40 %
    del percentil 90 de las alturas de la primera pasada
    (adaptativo a la amplitud de la señal, no a un valor
    absoluto; ver nota en el código).
    """
    if filtered.size == 0:
        return RPeaks(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )
    refractory = max(1, int(round(REFRACTORY_S * fs)))
    peaks, _props = sp_signal.find_peaks(
        filtered, distance=refractory
    )
    if peaks.size == 0:
        return RPeaks(peaks, np.array([], dtype=np.float64))
    # Alturas de la primera pasada (find_peaks solo
    # devuelve "peak_heights" si se pidió "height").
    heights = np.asarray(filtered[peaks], dtype=np.float64)
    # Umbral adaptativo: en un ECG el pico R es la
    # desviación más alta (R >> T > P). Se toma el
    # percentil 90 de las alturas de la primera pasada
    # como referencia (robusto ante un pico de ruido
    # aislado, a diferencia del máximo) y se descartan
    # los picos por debajo del 40 % de esa referencia:
    # las ondas T (~25 % de la altura del R) quedan
    # excluidas y los picos R se conservan.
    reference = float(np.percentile(heights, 90))
    threshold = 0.4 * reference
    peaks, props = sp_signal.find_peaks(
        filtered, height=threshold, distance=refractory
    )
    heights = np.asarray(
        props.get("peak_heights", []), dtype=np.float64
    )
    return RPeaks(peaks, heights)


@dataclass
class SignalFeatures:
    """Features de la señal (entrada del modelo heurístico)."""

    n_samples: int
    sample_rate_hz: int
    duration_s: float
    n_beats: int
    #: Latidos por minuto (mediana de los intervalos RR).
    heart_rate_bpm: float | None
    #: Coeficiente de variación de los intervalos RR.
    rr_cv: float | None
    #: Amplitud típica del R en mV.
    r_amplitude_mv: float | None
    #: Fracción de energía por encima de ~40 Hz (ruido).
    noise_ratio: float
    #: Índices de muestra de los picos R.
    r_peak_indices: np.ndarray
    #: Ventanas (inicio, fin) de cada latido, en muestras.
    beat_windows: list[tuple[int, int]]
    #: Amplitud en mV de cada latido (mismo orden que las ventanas).
    beat_amplitudes_mv: list[float]


def _noise_ratio(arr: np.ndarray, fs: float) -> float:
    """Fracción de la energía de la señal en banda alta.

    La frecuencia de corte se limita a ``0.9 × Nyquist``
    para que el filtro sea válido a cualquier ``fs``:
    con frecuencias de muestreo bajas no hay contenido
    representable por encima de 40 Hz, así que se mide
    la banda alta disponible.
    """
    total = float(np.sum(arr * arr))
    if total <= 0.0:
        return 0.0
    nyquist = fs / 2.0
    cutoff = min(NOISE_CUTOFF_HZ, 0.9 * nyquist)
    sos = sp_signal.butter(
        BUTTER_ORDER, cutoff, btype="lowpass", fs=fs, output="sos"
    )
    try:
        low = sp_signal.sosfiltfilt(sos, arr)
    except ValueError:
        return 0.0
    hf = arr - low
    return float(np.sum(hf * hf) / total)


def extract_features(
    original_mv: np.ndarray,
    filtered: np.ndarray,
    r_peaks: RPeaks,
    fs: float,
    sample_rate_hz: int,
) -> SignalFeatures:
    """Segmenta la señal en latidos y extrae features."""
    n_samples = original_mv.size
    amp_win = max(1, int(round(AMP_WINDOW_S * fs)))
    beat_win = max(1, int(round(BEAT_WINDOW_S * fs)))

    beat_windows: list[tuple[int, int]] = []
    beat_amplitudes: list[float] = []
    for idx in r_peaks.indices:
        start = max(0, int(idx) - beat_win)
        end = min(n_samples, int(idx) + beat_win)
        if end - start < 1:
            continue
        beat_windows.append((start, end))
        lo = max(0, int(idx) - amp_win)
        hi = min(n_samples, int(idx) + amp_win)
        beat_amplitudes.append(
            float(np.max(np.abs(original_mv[lo:hi])))
        )

    rr_samples = np.diff(r_peaks.indices)
    rr_seconds = rr_samples / fs
    heart_rate_bpm: float | None = None
    rr_cv: float | None = None
    if rr_seconds.size >= 1:
        median_rr = float(np.median(rr_seconds))
        if median_rr > 0:
            heart_rate_bpm = 60.0 / median_rr
    if rr_seconds.size >= 3:
        mean_rr = float(np.mean(rr_seconds))
        if mean_rr > 0:
            rr_cv = float(np.std(rr_seconds) / mean_rr)

    r_amplitude_mv: float | None = None
    if beat_amplitudes:
        r_amplitude_mv = float(np.median(beat_amplitudes))

    return SignalFeatures(
        n_samples=n_samples,
        sample_rate_hz=sample_rate_hz,
        duration_s=n_samples / fs,
        n_beats=int(len(beat_windows)),
        heart_rate_bpm=heart_rate_bpm,
        rr_cv=rr_cv,
        r_amplitude_mv=r_amplitude_mv,
        noise_ratio=_noise_ratio(original_mv, fs),
        r_peak_indices=r_peaks.indices,
        beat_windows=beat_windows,
        beat_amplitudes_mv=beat_amplitudes,
    )
