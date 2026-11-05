"""Pruebas del loader MIT-BIH y ventanas por anotación (T09b, #24).

Ninguna prueba toca la red ni necesita wfdb ni el dataset real:

- ``app.mitbih._load_wfdb`` (seam perezoso compartido con T09a) se
  sustituye por un doble (``FakeWfdb``) con ``monkeypatch``; el doble
  finge ``rdrecord``/``rdann`` y, cuando hace falta, ``dl_database``.
- La descarga se puede interceptar por completo parcheando
  ``app.mitbih.download_mitbih``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from app import mitbih, mitbih_load
from app import signal as signal_mod
from app.mitbih_load import RecordData, beat_windows, load_record
from app.signal import RPeaks, extract_features

#: Señal sintética de 2 canales x 10 muestras (canal 0 y canal 1).
P_SIGNAL = np.array(
    [
        [0.10, 1.10],
        [0.11, 1.11],
        [0.12, 1.12],
        [0.13, 1.13],
        [0.14, 1.14],
        [0.15, 1.15],
        [0.16, 1.16],
        [0.17, 1.17],
        [0.18, 1.18],
        [0.19, 1.19],
    ]
)
FS = 360.0
SAMPLES = [1586, 1620, 4000]
SYMBOLS = ["N", "A", "Q"]


class FakeRecord:
    """Doble de ``wfdb.Record`` (solo ``p_signal`` y ``fs``)."""

    def __init__(self, p_signal, fs: float) -> None:
        self.p_signal = p_signal
        self.fs = fs


class FakeAnnotation:
    """Doble de ``wfdb.Annotation`` (solo ``sample`` y ``symbol``)."""

    def __init__(self, sample, symbol) -> None:
        self.sample = np.asarray(sample, dtype=np.int64)
        self.symbol = list(symbol)


class FakeWfdb:
    """Doble de ``wfdb`` sin red.

    - ``dl_database(...)`` simula la descarga creando ``.hea``/``.dat``
      (y ``.atr`` salvo con ``omit_atr``, que replica una descarga
      interrumpida para la mitigación D-1).
    - ``rdrecord(...)``/``rdann(...)`` devuelven los dobles
      configurados y registran cada llamada.
    """

    def __init__(
        self,
        p_signal=P_SIGNAL,
        fs: float = FS,
        sample=SAMPLES,
        symbol=SYMBOLS,
        omit_atr: bool = False,
    ) -> None:
        self.p_signal = p_signal
        self.fs = fs
        self.sample = sample
        self.symbol = symbol
        self.omit_atr = omit_atr
        self.index_url_calls: list[tuple] = []
        self.dl_calls: list[dict] = []
        self.rdrecord_calls: list[tuple] = []
        self.rdann_calls: list[tuple] = []

    def set_db_index_url(self, *args) -> None:
        self.index_url_calls.append(args)

    def dl_database(self, db_dir: str, dl_dir: str, **kwargs) -> None:
        self.dl_calls.append({"db_dir": db_dir, "dl_dir": dl_dir, **kwargs})
        suffixes = tuple(
            suffix
            for suffix in mitbih._SUFFIXES
            if not (self.omit_atr and suffix == ".atr")
        )
        for record in kwargs["records"]:
            for suffix in suffixes:
                (Path(dl_dir) / f"{record}{suffix}").write_text("x", encoding="utf-8")

    def rdrecord(self, record_name: str, physical: bool = True):
        self.rdrecord_calls.append((record_name, physical))
        return FakeRecord(self.p_signal, self.fs)

    def rdann(self, record_name: str, extension: str):
        self.rdann_calls.append((record_name, extension))
        return FakeAnnotation(self.sample, self.symbol)


def _install_fake(monkeypatch, fake: FakeWfdb) -> dict[str, int]:
    """Inyecta el doble y devuelve el contador de cargas de wfdb."""
    loads = {"count": 0}

    def loader():
        loads["count"] += 1
        return fake

    monkeypatch.setattr(mitbih, "_load_wfdb", loader)
    return loads


def _make_cached(dest: Path, record: str = "100") -> Path:
    """Crea un registro cacheado (``.hea``+``.dat``+``.atr``) en ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    for suffix in mitbih._SUFFIXES:
        (dest / f"{record}{suffix}").write_text("x", encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# load_record
# ---------------------------------------------------------------------------


def test_load_record_devuelve_senal_lead0_fs_y_anotaciones(monkeypatch, tmp_path) -> None:
    """Forma del lead 0, fs float y anotaciones [(muestra, símbolo)]."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    record = load_record("100", dest=dest)

    assert isinstance(record, RecordData)
    assert record.record_id == "100"
    # Señal 1-D float del lead 0 con todas las muestras.
    assert isinstance(record.signal, np.ndarray)
    assert record.signal.ndim == 1
    assert record.signal.shape == (P_SIGNAL.shape[0],)
    assert np.issubdtype(record.signal.dtype, np.floating)
    np.testing.assert_allclose(record.signal, P_SIGNAL[:, 0])
    assert record.n_samples == P_SIGNAL.shape[0]
    # fs como float.
    assert isinstance(record.fs, float)
    assert record.fs == FS
    # Anotaciones como pares (int, str), en orden.
    assert record.annotations == [(1586, "N"), (1620, "A"), (4000, "Q")]
    assert all(isinstance(s, int) and isinstance(sym, str) for s, sym in record.annotations)
    # wfdb se llamó con la ruta local y el .atr, sin pn_dir (cero red).
    path = str(dest / "100")
    assert fake.rdrecord_calls == [(path, True)]
    assert fake.rdann_calls == [(path, "atr")]


def test_load_record_lead_alternativo(monkeypatch, tmp_path) -> None:
    """lead=1 devuelve la segunda columna de p_signal."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    record = load_record("100", lead=1, dest=dest)

    np.testing.assert_allclose(record.signal, P_SIGNAL[:, 1])


def test_load_record_descarga_si_no_hay_cache(monkeypatch, tmp_path) -> None:
    """Sin archivos locales descarga primero (reutiliza T09a) y luego lee."""
    fake = FakeWfdb()
    loads = _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"

    record = load_record("100", dest=dest)

    # Se invocó la descarga de T09a con solo ese registro…
    assert len(fake.dl_calls) == 1
    assert fake.dl_calls[0]["records"] == ["100"]
    assert fake.dl_calls[0]["dl_dir"] == str(dest)
    assert (dest / "100.hea").is_file() and (dest / "100.atr").is_file()
    # …y después se leyó el registro.
    assert fake.rdrecord_calls and fake.rdann_calls
    np.testing.assert_allclose(record.signal, P_SIGNAL[:, 0])
    # wfdb se cargó para descargar y para leer.
    assert loads["count"] >= 2


def test_load_record_cacheado_no_vuelve_a_descargar(monkeypatch, tmp_path) -> None:
    """Registro cacheado: cero descarga (el seam no se toca)."""
    fake = FakeWfdb()
    loads = _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no debería descargarse: el registro está cacheado")

    monkeypatch.setattr(mitbih, "download_mitbih", forbidden)

    record = load_record("100", dest=dest)

    assert record.annotations == [(1586, "N"), (1620, "A"), (4000, "Q")]
    assert loads == {"count": 1}  # una sola carga de wfdb (lectura)


def test_d1_sin_atr_dispara_descarga_en_load_record(monkeypatch, tmp_path) -> None:
    """D-1: .hea+.dat sin .atr no está cacheado → load_record lo descarga."""
    fake = FakeWfdb(omit_atr=True)
    _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"
    dest.mkdir()
    (dest / "100.hea").write_text("x", encoding="utf-8")
    (dest / "100.dat").write_text("x", encoding="utf-8")

    load_record("100", dest=dest)

    # La descarga se volvió a disparar (el .atr faltaba)…
    assert len(fake.dl_calls) == 1
    assert fake.dl_calls[0]["records"] == ["100"]
    # …y como el doble omite el .atr, la caché sigue incompleta:
    # la próxima llamada volverá a intentarlo (no se da por cacheado).
    assert mitbih.is_cached(dest, "100") is False


def test_record_id_desconocido_error_claro_sin_tocar_wfdb(monkeypatch, tmp_path) -> None:
    """IDs fuera de RECORDS → ValueError antes de leer o descargar."""
    fake = FakeWfdb()
    loads = _install_fake(monkeypatch, fake)

    with pytest.raises(ValueError, match="desconocido"):
        load_record("110", dest=tmp_path / "mit-bih")

    assert loads == {"count": 0}
    assert fake.dl_calls == []
    assert fake.rdrecord_calls == []


def test_lead_fuera_de_rango_error_claro(monkeypatch, tmp_path) -> None:
    """Lead negativo o mayor que el último canal → IndexError legible."""
    fake = FakeWfdb()  # 2 canales
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    with pytest.raises(IndexError, match="fuera de rango"):
        load_record("100", lead=2, dest=dest)
    with pytest.raises(IndexError, match="fuera de rango"):
        load_record("100", lead=-1, dest=dest)


def test_registro_vacio_sin_crash(monkeypatch, tmp_path) -> None:
    """Señal 0 muestras y 0 anotaciones → RecordData vacío, sin error."""
    fake = FakeWfdb(
        p_signal=np.empty((0, 2)),
        sample=[],
        symbol=[],
    )
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    record = load_record("100", dest=dest)

    assert record.n_samples == 0
    assert record.signal.shape == (0,)
    assert record.annotations == []
    assert record.fs == FS
    # Y el pipeline de ventanas no revienta con él.
    assert beat_windows(record.annotations, record.n_samples, record.fs) == []


def test_registro_sin_anotaciones(monkeypatch, tmp_path) -> None:
    """Anotaciones vacías (rdann sin muestras) → lista vacía, sin crash."""
    fake = FakeWfdb(sample=[], symbol=[])
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    record = load_record("100", dest=dest)

    assert record.annotations == []
    assert record.n_samples == P_SIGNAL.shape[0]
    assert beat_windows(record.annotations, record.n_samples, record.fs) == []


def test_fs_invalida_error_claro(monkeypatch, tmp_path) -> None:
    """Un header con fs <= 0 no genera ventanas absurdas: ValueError."""
    fake = FakeWfdb(fs=0.0)
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    with pytest.raises(ValueError, match="frecuencia de muestreo"):
        load_record("100", dest=dest)


def test_p_signal_ausente_error_claro(monkeypatch, tmp_path) -> None:
    """wfdb sin señal física legible → ValueError con el ID del registro."""
    fake = FakeWfdb(p_signal=None)
    _install_fake(monkeypatch, fake)
    dest = _make_cached(tmp_path / "mit-bih")

    with pytest.raises(ValueError, match="100"):
        load_record("100", dest=dest)


# ---------------------------------------------------------------------------
# beat_windows
# ---------------------------------------------------------------------------


def test_beat_windows_central_recorta_y_respeta_constante() -> None:
    """Ventana ±BEAT_WINDOW_S*fs alrededor de la anotación (fs=360 → ±144)."""
    half = round(signal_mod.BEAT_WINDOW_S * 360)  # 144 muestras
    windows = beat_windows([(500, "N")], n_samples=1000, fs=360.0)

    assert windows == [(500 - half, 500 + half)]
    assert windows == [(356, 644)]


def test_beat_windows_recorta_bordes_extremos() -> None:
    """Cerca de 0 y del final: los bordes se recortan a [0, n_samples)."""
    windows = beat_windows([(10, "N"), (990, "N")], n_samples=1000, fs=360.0)

    assert windows == [(0, 154), (846, 1000)]
    assert all(0 <= start < end <= 1000 for start, end in windows)


def test_beat_windows_descarta_ventanas_vacias() -> None:
    """Anotaciones fuera de la señal y registro sin muestras → sin ventanas."""
    # fs=100 → ±40 muestras; muestra 150 > 100 (fuera) y -50 (negativa).
    assert beat_windows([(150, "N"), (-50, "N")], n_samples=100, fs=100.0) == []
    # Mezcla: solo sobrevive la anotación dentro de la señal.
    assert beat_windows([(50, "N"), (150, "N")], n_samples=100, fs=100.0) == [(10, 90)]
    # Registro vacío.
    assert beat_windows([(10, "N")], n_samples=0, fs=360.0) == []


def test_beat_windows_usa_beat_window_s_de_signal(monkeypatch) -> None:
    """Coherencia: se usa la constante de app.signal, sin duplicar 0.4."""
    # Misma constante objeto (no hay un 0.4 redefinido en mitbih_load).
    assert mitbih_load.BEAT_WINDOW_S is signal_mod.BEAT_WINDOW_S
    assert mitbih_load.BEAT_WINDOW_S == 0.4
    # Y la función lee ese nombre en tiempo de ejecución.
    monkeypatch.setattr(mitbih_load, "BEAT_WINDOW_S", 1.0)
    windows = beat_windows([(500, "N")], n_samples=1000, fs=10.0)
    assert windows == [(490, 510)]  # ±1.0 s * 10 Hz = ±10 muestras


def test_beat_windows_coherente_con_extract_features() -> None:
    """La ventana por anotación coincide con la del pipeline (±0.4 s)."""
    original = np.zeros(1000)
    peaks = RPeaks(
        indices=np.array([500], dtype=np.int64),
        heights=np.array([1.0]),
    )
    features = extract_features(
        original, original.copy(), peaks, fs=360.0, sample_rate_hz=360
    )
    assert features.beat_windows == beat_windows([(500, "N")], 1000, 360.0)


def test_beat_windows_acepta_muestras_sueltas() -> None:
    """Además de (muestra, símbolo), acepta índices sueltos (r_peaks)."""
    windows = beat_windows([500, 10], n_samples=1000, fs=360.0)

    assert windows == [(356, 644), (0, 154)]


def test_beat_windows_fs_invalida_error() -> None:
    """fs no positiva ni finita → ValueError claro."""
    with pytest.raises(ValueError, match="fs"):
        beat_windows([(10, "N")], n_samples=100, fs=0.0)
    with pytest.raises(ValueError, match="fs"):
        beat_windows([(10, "N")], n_samples=100, fs=float("nan"))
