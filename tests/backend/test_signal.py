"""Pruebas unitarias del procesamiento de señal (``app/signal.py``)."""

from __future__ import annotations

import numpy as np

from app.signal import (
    _noise_ratio,
    bandpass_filter,
    detect_r_peaks,
    extract_features,
    to_millivolts,
    zscore,
)

from tests.backend.synthetic import synthetic_ecg


def test_to_millivolts_convierte_uv() -> None:
    values = [1000.0, 2000.0, 3000.0]
    mv = to_millivolts(values, "uV")
    np.testing.assert_allclose(mv, [1.0, 2.0, 3.0])


def test_to_millivolts_mantiene_mv() -> None:
    values = [1.0, -1.0, 0.5]
    mv = to_millivolts(values, "mV")
    np.testing.assert_allclose(mv, values)


def test_bandpass_filter_elimina_cc() -> None:
    """Una señal con offset de corriente continua sale centrada en 0."""
    fs = 500
    arr = np.asarray(synthetic_ecg(sample_rate_hz=fs, duration_s=4.0)) + 2.0
    filtrada = bandpass_filter(arr, fs)
    # Se excluyen los bordes: el filtrado sin fase deja
    # transitorios en los extremos de la señal. Tolerancia
    # de 0.01 mV frente a una amplitud R de ~1 mV.
    centro = filtrada[500:-500]
    assert abs(float(np.mean(centro))) < 0.01


def test_bandpass_filter_atenúa_ruido_alta_frecuencia() -> None:
    """El ruido de alta frecuencia se atenúa con el pasa-banda."""
    fs = 500
    rng = np.random.default_rng(7)
    limpia = np.asarray(synthetic_ecg(sample_rate_hz=fs, duration_s=4.0))
    ruidosa = limpia + rng.normal(0.0, 0.5, size=limpia.size)
    energia_ruido_original = float(np.sum((ruidosa - limpia) ** 2))
    filtrada = bandpass_filter(ruidosa, fs)
    energia_restante = float(np.sum((filtrada - limpia) ** 2))
    assert energia_restante < energia_ruido_original


def test_bandpass_filter_baja_frecuencia_muestreo_no_falla() -> None:
    """D1: fs ≤ 90 Hz es válida; el filtro adapta la banda
    al Nyquist (o devuelve la señal sin filtrar) en lugar
    de lanzar ``ValueError``."""
    arr = np.asarray(
        synthetic_ecg(sample_rate_hz=50, duration_s=4.0)
    )
    for fs in (1, 2, 3, 50, 90):
        filtrada = bandpass_filter(arr, fs)
        assert filtrada.size == arr.size
        assert np.all(np.isfinite(filtrada))


def test_noise_ratio_baja_frecuencia_muestreo_no_falla() -> None:
    """D1: el estimador de ruido no falla con fs ≤ 80 Hz
    (antes el ``butter`` a 40 Hz lanzaba ``ValueError``)."""
    arr = np.asarray(
        synthetic_ecg(sample_rate_hz=50, duration_s=4.0)
    )
    for fs in (50, 80, 90):
        ratio = _noise_ratio(arr, fs)
        assert 0.0 <= ratio <= 1.0


def test_zscore_media_cero_desviacion_uno() -> None:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(arr)
    assert abs(float(np.mean(z))) < 1e-12
    assert abs(float(np.std(z)) - 1.0) < 1e-12


def test_zscore_senal_plana() -> None:
    z = zscore(np.zeros(100))
    np.testing.assert_array_equal(z, np.zeros(100))


def test_detect_r_peaks_cuenta_latidos() -> None:
    """La detección de picos R cuenta (casi) los latidos simulados."""
    fs = 500
    bpm = 70
    duracion = 10.0
    signal = np.asarray(
        synthetic_ecg(sample_rate_hz=fs, duration_s=duracion, bpm=bpm)
    )
    filtrada = bandpass_filter(signal, fs)
    normalizada = zscore(filtrada)
    picos = detect_r_peaks(normalizada, fs)
    esperados = int(duracion * bpm / 60)
    # Tolerancia de ±1 latido (bordes de la ventana).
    assert abs(picos.indices.size - esperados) <= 1
    assert picos.indices.size > 0
    assert all(0 <= i < signal.size for i in picos.indices)


def test_extract_features_frecuencia_cardiaca() -> None:
    fs = 500
    bpm = 100
    signal = np.asarray(
        synthetic_ecg(sample_rate_hz=fs, duration_s=10.0, bpm=bpm)
    )
    filtrada = bandpass_filter(signal, fs)
    normalizada = zscore(filtrada)
    picos = detect_r_peaks(normalizada, fs)
    features = extract_features(
        original_mv=signal,
        filtered=filtrada,
        r_peaks=picos,
        fs=fs,
        sample_rate_hz=fs,
    )
    assert features.n_beats > 0
    assert features.heart_rate_bpm is not None
    assert abs(features.heart_rate_bpm - bpm) < 2.0
    assert features.r_amplitude_mv is not None
    assert features.r_amplitude_mv > 0.5  # R simulado de 1.0 mV
    assert features.rr_cv is not None
    assert features.rr_cv < 0.05  # ritmo regular


def test_extract_features_ruido_alto_en_senal_ruidosa() -> None:
    fs = 500
    limpia = np.asarray(synthetic_ecg(sample_rate_hz=fs, duration_s=6.0))
    ruidosa = np.asarray(
        synthetic_ecg(sample_rate_hz=fs, duration_s=6.0, noise_sigma_mv=0.5)
    )
    features_limpia = extract_features(
        original_mv=limpia,
        filtered=bandpass_filter(limpia, fs),
        r_peaks=detect_r_peaks(zscore(bandpass_filter(limpia, fs)), fs),
        fs=fs,
        sample_rate_hz=fs,
    )
    features_ruidosa = extract_features(
        original_mv=ruidosa,
        filtered=bandpass_filter(ruidosa, fs),
        r_peaks=detect_r_peaks(zscore(bandpass_filter(ruidosa, fs)), fs),
        fs=fs,
        sample_rate_hz=fs,
    )
    assert features_ruidosa.noise_ratio > 0.30
    assert features_ruidosa.noise_ratio > features_limpia.noise_ratio


def test_extract_features_senal_plana() -> None:
    plana = np.zeros(500)
    features = extract_features(
        original_mv=plana,
        filtered=bandpass_filter(plana, 500.0),
        r_peaks=detect_r_peaks(zscore(plana), 500.0),
        fs=500.0,
        sample_rate_hz=500,
    )
    assert features.n_beats == 0
    assert features.heart_rate_bpm is None
    assert features.r_amplitude_mv is None
    assert features.noise_ratio == 0.0
