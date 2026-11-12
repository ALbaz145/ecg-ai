"""Lectura de registros MIT-BIH y segmentación en ventanas (T09b, #24).

Módulo de **datos**: convierte un registro ya cacheado por
``app.mitbih`` en la forma que consume el pipeline (señal 1-D de un
lead, ``fs`` y anotaciones) y calcula las ventanas por anotación.

Decisiones de diseño:

- **Módulo aparte de ``app.mitbih``** (y no dentro de él): T09a se
  ocupa de descarga/caché y su docstring gira en torno a eso; aquí
  añadimos lectura (``wfdb.rdrecord``/``rdann``) y segmentación
  (depende de ``app.signal``). Responsabilidades distintas, pruebas
  separadas y ``mitbih.py`` sin crecer.
- **Reutiliza T09a, sin duplicar lógica**: la descarga (si no hay
  caché) va por ``mitbih.download_mitbih(dest, records=[...])`` y la
  lectura usa el mismo seam perezoso ``mitbih._load_wfdb`` (un solo
  ``monkeypatch`` en las pruebas cubre descarga + lectura). La
  comprobación de caché es ``mitbih.is_cached`` (que desde la
  mitigación **D-1** exige ``.hea``+``.dat``+``.atr``, así un registro
  cacheado siempre tiene anotaciones para ``load_record``).
- **wfdb local**: los archivos están en ``dest``; ``rdrecord`` y
  ``rdann`` se llaman con la ruta local (sin ``pn_dir``), cero red.
- **Un solo lead**: la señal se devuelve como ``np.ndarray`` 1-D
  float del ``lead`` pedido (por defecto 0 = canal MLII en mitdb).
- **Registros vacíos**: no son error; se devuelve ``RecordData`` con
  señal de tamaño 0 y anotaciones vacías (el pipeline ya trata la
  señal vacía sin crash).
- **Ventanas**: usa la constante ``BEAT_WINDOW_S`` de ``app.signal``
  (0.4 s a cada lado del R) **importada, no redefinida**, con los
  bordes recortados a ``[0, n_samples)`` y las ventanas que queden
  vacías descartadas (igual que ``extract_features``).

El dataset es **público** (MIT-BIH, PhysioNet) y nunca se commitea.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from app import mitbih
from app.signal import BEAT_WINDOW_S


@dataclass
class RecordData:
    """Registro MIT-BIH cargado en memoria (T09b).

    Attributes
    ----------
    record_id:
        ID canónico del registro (p. ej. ``"100"``).
    signal:
        Señal 1-D float del lead seleccionado (``p_signal[:, lead]``),
        en las unidades físicas del archivo (mV en ``mitdb``). Puede
        tener tamaño 0 (registro vacío).
    fs:
        Frecuencia de muestreo en Hz.
    annotations:
        Anotaciones ``(muestra, símbolo)`` de ``.atr`` en orden
        temporal (p. ej. ``(2186, "N")``). Puede estar vacía.
    """

    record_id: str
    signal: np.ndarray
    fs: float
    annotations: list[tuple[int, str]] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        """Número de muestras de la señal cargada."""
        return int(self.signal.size)


def load_record(
    record_id: str,
    *,
    lead: int = 0,
    dest: Path = mitbih.DEFAULT_DEST,
) -> RecordData:
    """Carga un registro MIT-BIH cacheado (lo descarga si falta).

    Devuelve la señal del ``lead`` pedido como ``np.ndarray`` float
    1-D, su ``fs`` y las anotaciones ``(muestra, símbolo)`` del
    archivo ``.atr``. Si el registro no está cacheado en ``dest``
    (criterio ``hea``+``dat``+``atr``, mitigación D-1) se descarga
    primero reutilizando :func:`app.mitbih.download_mitbih`.

    Parameters
    ----------
    record_id:
        ID canónico de :data:`app.mitbih.RECORDS` (p. ej. ``"100"``).
    lead:
        Índice del canal a devolver (por defecto 0; en ``mitdb`` es
        MLII). Debe estar en ``[0, n_leads)``.
    dest:
        Directorio con los archivos locales (por defecto
        ``data/mit-bih``, gitignored).

    Returns
    -------
    RecordData
        Señal 1-D float, ``fs`` (float, Hz) y anotaciones. Un registro
        sin muestras o sin anotaciones devuelve arrays/listas vacías
        (no es error).

    Raises
    ------
    ValueError
        Si ``record_id`` no está en :data:`app.mitbih.RECORDS`, o si
        el header no trae ``fs`` válido (> 0) o señal física legible.
    IndexError
        Si ``lead`` está fuera de rango para el número de canales.
    ConnectionError
        Si hay que descargar y wfdb falla (ver
        :func:`app.mitbih.download_mitbih`).
    """
    if record_id not in set(mitbih.RECORDS):
        raise ValueError(
            f"Registro desconocido en {mitbih.MITBIH_DB}: {record_id!r}. "
            f"Usa IDs de RECORDS (p. ej. {mitbih.RECORDS[0]!r})."
        )

    dest = Path(dest)
    if not mitbih.is_cached(dest, record_id):
        # Reutiliza la descarga de T09a (misma caché, misma purga).
        mitbih.download_mitbih(dest=dest, records=[record_id])

    wfdb = mitbih._load_wfdb()  # seam perezoso compartido con T09a
    path = str(dest / record_id)

    record = wfdb.rdrecord(path, physical=True)
    p_signal = getattr(record, "p_signal", None)
    if p_signal is None:
        raise ValueError(
            f"Registro {record_id!r}: wfdb no devolvió señal física "
            "legible (p_signal is None)."
        )
    p_signal = np.asarray(p_signal, dtype=np.float64)
    if p_signal.ndim == 1:
        # Registro de un solo canal: (n,) -> (n, 1).
        p_signal = p_signal[:, np.newaxis]
    if p_signal.ndim != 2:
        raise ValueError(
            f"Registro {record_id!r}: forma de señal inválida {p_signal.shape}."
        )
    n_leads = p_signal.shape[1]
    if not (0 <= lead < n_leads):
        raise IndexError(
            f"Lead {lead} fuera de rango para {record_id!r}: "
            f"el registro tiene {n_leads} canal(es) (0..{n_leads - 1})."
        )

    fs = float(getattr(record, "fs", float("nan")))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            f"Registro {record_id!r}: frecuencia de muestreo inválida ({fs!r})."
        )

    annotation = wfdb.rdann(path, "atr")
    samples = np.asarray(getattr(annotation, "sample", []), dtype=np.int64).ravel()
    symbols = list(getattr(annotation, "symbol", []))
    # strict=True: si wfdb devolviera samples/symbols de distinta
    # longitud, zip lanza ValueError en vez de truncar en silencio
    # anotaciones (issue #37, O-3).
    annotations = [
        (int(sample), str(symbol))
        for sample, symbol in zip(samples, symbols, strict=True)
    ]

    return RecordData(
        record_id=record_id,
        signal=np.ascontiguousarray(p_signal[:, lead]),
        fs=fs,
        annotations=annotations,
    )


def beat_windows(
    annotations: Iterable[tuple[int, str] | int],
    n_samples: int,
    fs: float,
) -> list[tuple[int, int]]:
    """Ventanas ``(inicio, fin)`` por anotación, en muestras.

    Cada anotación (par ``(muestra, símbolo)`` o muestra suelta) genera
    una ventana de ``±BEAT_WINDOW_S * fs`` muestras (la constante
    **se importa de** ``app.signal``: 0.4 s a cada lado del R, sin
    duplicar el valor), con los bordes recortados a ``[0, n_samples)``.
    Las ventanas que tras el recorte queden vacías (anotación fuera de
    la señal o registro sin muestras) se descartan.

    Parameters
    ----------
    annotations:
        Anotaciones como ``(muestra, símbolo)`` o muestras sueltas
        (también sirve una lista de índices).
    n_samples:
        Longitud de la señal (para recortar los bordes).
    fs:
        Frecuencia de muestreo en Hz (para convertir segundos a
        muestras).

    Returns
    -------
    list[tuple[int, int]]
        Ventanas ordenadas de ``[inicio, fin)`` con
        ``0 <= inicio < fin <= n_samples``.

    Raises
    ------
    ValueError
        Si ``fs`` no es un valor finito positivo.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs debe ser un valor finito positivo (recibido: {fs!r}).")
    if n_samples <= 0:
        return []

    half = max(1, round(BEAT_WINDOW_S * fs))
    windows: list[tuple[int, int]] = []
    for annotation in annotations:
        sample = (
            annotation
            if isinstance(annotation, (int, np.integer))
            else annotation[0]
        )
        start = max(0, int(sample) - half)
        end = min(int(n_samples), int(sample) + half)
        if end > start:
            windows.append((start, end))
    return windows
