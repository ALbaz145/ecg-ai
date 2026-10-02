"""Modelo MVP: detector baseline **heurístico** (placeholder didáctico).

⚠️ **No es un modelo entrenado ni un dispositivo médico.**
Es un conjunto de reglas sobre features de la señal (ver
``app/signal.py``) que aproxima lo que mediría un modelo
entrenado en datasets públicos (PTB-XL, MIT-BIH). La
documentación completa —umbrales, métricas y limitaciones—
está en ``docs/MODEL.md``.

Reglas (en orden de prioridad):

1. ``senal_no_valorable``: < 2 latidos detectados.
2. ``ruido_excesivo``: fracción de energía HF > 0.30.
3. ``arritmia``: CV de los intervalos RR > 0.15 (≥ 6 latidos).
4. ``taquicardia``: FC > 100 lpm.
5. ``bradicardia``: FC < 60 lpm.
6. ``amplitud_anomala``: amplitud R fuera de [0.1, 3.0] mV.
7. ``normal``: en caso contrario.

La confianza es una función heurística de la fuerza de la
evidencia (regularidad, nº de latidos, ruido); **no está
calibrada** y solo pretende ordenar resultados.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.signal import SignalFeatures

#: Versión del modelo (se guarda en cada análisis).
MODEL_VERSION = "heuristic-mvp-0.1.0"

# --- Umbrales (aproximaciones didácticas, NO validadas clínicamente) ---
HR_TACHYCARDIA_BPM = 100.0
HR_BRADYCARDIA_BPM = 60.0
RR_CV_ARRHYTHMIA = 0.15
R_AMP_LOW_MV = 0.1
R_AMP_HIGH_MV = 3.0
NOISE_RATIO_HIGH = 0.30
#: Desviación RR respecto a la mediana para marcar un latido irregular.
RR_DEVIATION = 0.20
#: Latidos mínimos para juzgar arritmia (estadística de RR).
MIN_BEATS_ARRHYTHMIA = 6
#: Nº máximo de zonas anómalas devueltas (evita respuestas enormes).
MAX_ZONES = 500

# Etiquetas del modelo.
LABEL_NORMAL = "normal"
LABEL_TACHYCARDIA = "taquicardia"
LABEL_BRADYCARDIA = "bradicardia"
LABEL_ARRHYTHMIA = "arritmia"
LABEL_AMPLITUD = "amplitud_anomala"
LABEL_RUIDO = "ruido_excesivo"
LABEL_NO_VALORABLE = "senal_no_valorable"

#: Tipos de zona anómala.
ZONE_IRREGULAR = "latido_irregular"
ZONE_AMP_LOW = "amplitud_baja"
ZONE_AMP_HIGH = "amplitud_alta"


@dataclass
class Detection:
    """Salida del modelo."""

    anomalia: str
    confianza: float
    zonas: list[dict[str, int | str]] = field(default_factory=list)
    motivo: str = ""


def _clamp_confidence(value: float, floor: float, cap: float) -> float:
    return max(floor, min(cap, value))


def _global_zone(n_samples: int, tipo: str) -> list[dict[str, int | str]]:
    if n_samples < 1:
        return []
    return [{"inicio": 0, "fin": n_samples, "tipo": tipo}]


def _confidence_normal(features: SignalFeatures) -> float:
    conf = 0.60
    if features.rr_cv is not None and features.rr_cv < 0.05:
        conf += 0.10
    if features.n_beats >= 10:
        conf += 0.10
    if features.noise_ratio < 0.10:
        conf += 0.05
    return _clamp_confidence(conf, 0.50, 0.90)


def _confidence_rhythm(features: SignalFeatures) -> float:
    conf = 0.75
    if features.rr_cv is not None:
        conf += 0.15 * (1.0 - min(features.rr_cv, 1.0))
    if features.n_beats >= 10:
        conf += 0.05
    return _clamp_confidence(conf, 0.60, 0.95)


def _confidence_arrhythmia(features: SignalFeatures) -> float:
    cv = features.rr_cv or 0.0
    conf = 0.70 + 0.20 * min(cv / 0.5, 1.0)
    return _clamp_confidence(conf, 0.60, 0.92)


def _confidence_amplitude(r_amplitude_mv: float) -> float:
    severity = 0.0
    if r_amplitude_mv < R_AMP_LOW_MV:
        severity = 1.0 - (r_amplitude_mv / R_AMP_LOW_MV)
    elif r_amplitude_mv > R_AMP_HIGH_MV:
        severity = min(
            (r_amplitude_mv - R_AMP_HIGH_MV) / R_AMP_HIGH_MV, 1.0
        )
    return _clamp_confidence(0.70 + 0.15 * severity, 0.60, 0.90)


def _irregular_beat_zones(
    features: SignalFeatures,
) -> list[dict[str, int | str]]:
    """Zonas de los latidos cuyo RR se desvía > 20 % de la mediana."""
    rr = features.r_peak_indices
    if rr.size < 2:
        return []
    intervals = rr[1:] - rr[:-1]
    median_rr = float(intervals.mean())
    if median_rr <= 0:
        return []
    zones: list[dict[str, int | str]] = []
    for i, interval in enumerate(intervals):
        if abs(float(interval) - median_rr) / median_rr > RR_DEVIATION:
            start, end = features.beat_windows[i + 1]
            zones.append(
                {"inicio": int(start), "fin": int(end), "tipo": ZONE_IRREGULAR}
            )
            if len(zones) >= MAX_ZONES:
                break
    return zones


def _amplitude_zones(
    features: SignalFeatures,
) -> list[dict[str, int | str]]:
    """Zonas de los latidos con amplitud R anómala."""
    zones: list[dict[str, int | str]] = []
    for (start, end), amp in zip(
        features.beat_windows, features.beat_amplitudes_mv
    ):
        if amp < R_AMP_LOW_MV:
            tipo: str = ZONE_AMP_LOW
        elif amp > R_AMP_HIGH_MV:
            tipo = ZONE_AMP_HIGH
        else:
            continue
        zones.append({"inicio": int(start), "fin": int(end), "tipo": tipo})
        if len(zones) >= MAX_ZONES:
            break
    return zones


def classify(features: SignalFeatures) -> Detection:
    """Clasifica la señal y devuelve la detección completa."""
    # 1. Señal sin ritmo medible.
    if features.n_beats < 2:
        return Detection(
            anomalia=LABEL_NO_VALORABLE,
            confianza=0.25,
            zonas=[],
            motivo=(
                "no se detectaron suficientes latidos para medir "
                "el ritmo (se necesitan >= 2 picos R)"
            ),
        )

    # 2. Ruido excesivo: las demás medidas no son fiables.
    if features.noise_ratio > NOISE_RATIO_HIGH:
        return Detection(
            anomalia=LABEL_RUIDO,
            confianza=_clamp_confidence(
                0.55 + 0.10 * min(features.noise_ratio, 1.0),
                0.50,
                0.70,
            ),
            zonas=_global_zone(features.n_samples, LABEL_RUIDO),
            motivo=(
                f"energía de alta frecuencia "
                f"({features.noise_ratio:.2f}) > {NOISE_RATIO_HIGH}"
            ),
        )

    # 3. Arritmia (irregularidad del ritmo).
    if (
        features.rr_cv is not None
        and features.rr_cv > RR_CV_ARRHYTHMIA
        and features.n_beats >= MIN_BEATS_ARRHYTHMIA
    ):
        zones = _irregular_beat_zones(features)
        return Detection(
            anomalia=LABEL_ARRHYTHMIA,
            confianza=_confidence_arrhythmia(features),
            zonas=zones,
            motivo=(
                f"variabilidad del intervalo RR (CV={features.rr_cv:.2f}) "
                f"> {RR_CV_ARRHYTHMIA}"
            ),
        )

    # 4/5. Ritmo global: taquicardia / bradicardia.
    if (
        features.heart_rate_bpm is not None
        and features.heart_rate_bpm > HR_TACHYCARDIA_BPM
    ):
        return Detection(
            anomalia=LABEL_TACHYCARDIA,
            confianza=_confidence_rhythm(features),
            zonas=_global_zone(features.n_samples, LABEL_TACHYCARDIA),
            motivo=(
                f"frecuencia cardíaca {features.heart_rate_bpm:.1f} lpm "
                f"> {HR_TACHYCARDIA_BPM:.0f} lpm"
            ),
        )
    if (
        features.heart_rate_bpm is not None
        and features.heart_rate_bpm < HR_BRADYCARDIA_BPM
    ):
        return Detection(
            anomalia=LABEL_BRADYCARDIA,
            confianza=_confidence_rhythm(features),
            zonas=_global_zone(features.n_samples, LABEL_BRADYCARDIA),
            motivo=(
                f"frecuencia cardíaca {features.heart_rate_bpm:.1f} lpm "
                f"< {HR_BRADYCARDIA_BPM:.0f} lpm"
            ),
        )

    # 6. Amplitud anómala del complejo QRS.
    if features.r_amplitude_mv is not None and (
        features.r_amplitude_mv < R_AMP_LOW_MV
        or features.r_amplitude_mv > R_AMP_HIGH_MV
    ):
        return Detection(
            anomalia=LABEL_AMPLITUD,
            confianza=_confidence_amplitude(features.r_amplitude_mv),
            zonas=_amplitude_zones(features),
            motivo=(
                f"amplitud R {features.r_amplitude_mv:.2f} mV fuera de "
                f"[{R_AMP_LOW_MV}, {R_AMP_HIGH_MV}] mV"
            ),
        )

    # 7. Sin anomalías detectadas por las reglas.
    return Detection(
        anomalia=LABEL_NORMAL,
        confianza=_confidence_normal(features),
        zonas=[],
        motivo="sin anomalías detectadas por las reglas del MVP",
    )
