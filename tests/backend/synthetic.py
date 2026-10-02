"""Generador de señales ECG **sintéticas** para las pruebas.

Ninguna señal proviene de pacientes reales: todo se
simula ( gaussianas para P/QRS/T + ruido + deriva de
línea base ). Ver docs/MODEL.md.
"""

from __future__ import annotations

import numpy as np


def _gaussian(
    n_samples: int,
    center: float,
    sigma_s: float,
    amplitude: float,
    fs: int,
) -> np.ndarray:
    """Campana gaussiana centrada en ``center`` (muestras)."""
    indices = np.arange(n_samples, dtype=np.float64)
    sigma_samples = sigma_s * fs
    return amplitude * np.exp(
        -((indices - center) ** 2) / (2.0 * sigma_samples**2)
    )


def synthetic_ecg(
    sample_rate_hz: int = 500,
    duration_s: float = 10.0,
    bpm: float = 70.0,
    r_amplitude_mv: float = 1.0,
    rr_jitter_s: float = 0.0,
    rr_pattern: list[float] | None = None,
    noise_sigma_mv: float = 0.02,
    baseline_wander_mv: float = 0.05,
    seed: int = 42,
) -> list[float]:
    """Genera una señal ECG sintética de un solo canal.

    Args:
        sample_rate_hz: frecuencia de muestreo.
        duration_s: duración en segundos.
        bpm: latidos por minuto (ignorado si hay ``rr_pattern``).
        r_amplitude_mv: amplitud del pico R en mV.
        rr_jitter_s: desviación típica del intervalo RR (s).
        rr_pattern: secuencia explícita de intervalos RR (s)
            para simular ritmos irregulares.
        noise_sigma_mv: ruido gaussiano (mV).
        baseline_wander_mv: deriva de línea base (mV).
        seed: semilla aleatoria (reproducibilidad).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sample_rate_hz * duration_s)
    signal = np.zeros(n_samples, dtype=np.float64)

    # --- Tren de latidos ------------------------------------
    beat_times: list[float] = []
    beat_time = 0.2  # primer latido a 0.2 s
    pattern_index = 0
    while beat_time < duration_s - 0.1:
        beat_times.append(beat_time)
        if rr_pattern is not None:
            interval = rr_pattern[pattern_index % len(rr_pattern)]
            pattern_index += 1
        else:
            interval = 60.0 / bpm
            if rr_jitter_s > 0:
                interval += float(rng.normal(0.0, rr_jitter_s))
        beat_time += max(interval, 0.05)

    # --- Forma de onda PQRST por latido ---------------------
    for beat_time in beat_times:
        center = beat_time * sample_rate_hz
        # Onda P
        signal += _gaussian(
            n_samples,
            center - 0.15 * sample_rate_hz,
            0.02,
            0.15 * r_amplitude_mv,
            sample_rate_hz,
        )
        # Complejo QRS (pico R)
        signal += _gaussian(
            n_samples,
            center,
            0.012,
            r_amplitude_mv,
            sample_rate_hz,
        )
        # Onda T
        signal += _gaussian(
            n_samples,
            center + 0.30 * sample_rate_hz,
            0.05,
            0.30 * r_amplitude_mv,
            sample_rate_hz,
        )

    # --- Deriva de línea base y ruido -----------------------
    time = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    signal += baseline_wander_mv * np.sin(2 * np.pi * 0.1 * time)
    if noise_sigma_mv > 0:
        signal += rng.normal(
            0.0, noise_sigma_mv, size=n_samples
        )

    return [float(value) for value in signal]
