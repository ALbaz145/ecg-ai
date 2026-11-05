"""Etiquetado AAMI y dataset serializable a partir de MIT-BIH (T09c, #25).

Módulo de **datos** (tercero de la cadena MIT-BIH de la Fase 2):
``mitbih.py`` descarga/caché → ``mitbih_load.py`` lectura + ventanas →
**este módulo** etiqueta cada latido con una clase AAMI, lo acumula en
``X``/``y``/``paciente_id`` y lo serializa. Vive en un módulo aparte de
``mitbih_load`` porque la responsabilidad es distinta (construir el
dataset que consume T10, no leer registros): pruebas propias y
``mitbih_load`` sin crecer.

Decisiones de diseño (los criterios que el issue #25 exige documentar):

- **Mapeo AAMI** — :data:`MAP_AAMI` / :func:`to_aami`: 5 clases según
  la unidad de anotaciones de MIT-BIH (PhysioNet ``mitdb``,
  https://physionet.org/content/mitdb/1.0.0/) y la correspondencia
  AAMI EC57 clásica (ANSI/AAMI EC57, el mapeo usado en de Chazal
  et al., IEEE TBME 2004): ``N ← {N, L, R, e, j}``,
  ``S ← {A, a, J, S}``, ``V ← {V, E}``, ``F ← {F}``, ``Q ← {/, f, Q}``.
  **No mapeados → Q (criterio explícito, nunca en silencio)**: los
  símbolos no-latido reales de los 48 registros de ``mitdb`` (``!``,
  ``[``, ``]``, ``x``, ``|``, ``~``, ``+``, ``"``) y *cualquier* otro
  símbolo desconocido —p. ej. ``p`` y ``t``, que no aparecen en los
  ``.atr`` reales— se etiquetan ``Q`` (desconocido/artefacto) en vez de
  descartarse: se prefiere una fila supervisada ``Q`` a perder
  anotaciones sin dejar rastro. La **única** exclusión de filas es la
  ventana (abajo): ningún símbolo se descarta por su clase.
- **Ventanas de longitud fija** — ``beat_windows`` recorta contra los
  bordes de la señal, así que una ventana puede ser más corta que el
  nominal ``2 * round(BEAT_WINDOW_S * fs)`` muestras (p. ej. el primer
  y el último latido de cada registro). **Criterio elegido: excluir los
  latidos con ventana incompleta** (no padear ni recortar): la pérdida
  es de 1-2 latidos por registro (~0.1 %), no introduce artefactos de
  relleno (zero/edge padding) ni recorta información real dentro de la
  ventana, y ``X`` sale rectangular sin trucos. Ventana vacía (anotación
  fuera de la señal) → también excluida.
- **Formato de ``y``** — array de strings ``<U1`` con las letras
  ``N|S|V|F|Q`` (constante ordenada :data:`AAMI_CLASSES`): legible,
  autodescriptivo y serializable **sin pickle**; a diferencia de los
  códigos enteros 0-4, el orden no puede desincronizarse entre el
  archivo y el código que lo carga. T10 lo codifica con un dict sobre
  ``AAMI_CLASSES`` o con ``sklearn.preprocessing.LabelEncoder``. El
  dominio se valida en ``save_dataset`` y en ``load_dataset``
  (``ValueError`` que nombra las clases inválidas), igual que
  ``meta["aami_classes"]`` al cargar: un NPZ con clases fuera de las 5
  no se abre en silencio.
- **``paciente_id`` por latido** — array de strings con el ID del
  registro repetido en cada fila (``"100"``…): obligatorio para el
  split por paciente de T10 (no por archivo ni global).
- **Un solo ``fs``** — el ancho de fila depende de ``fs``; si los
  registros mezclaran frecuencias, ``build_dataset`` lanza
  ``ValueError`` (en ``mitdb`` los 48 registros son 360 Hz).
- **Dataset vacío** — no es error: si ningún latido pasa el filtro se
  devuelve un ``EcgDataset`` con arrays vacíos (``X`` de forma
  ``(0, ancho)``). Sí es error la lista ``records=[]`` (misma regla y
  mismo mensaje que T09a: ``None`` = los 48).
- **Serialización NPZ** — ``np.savez`` + ``np.load(allow_pickle=False)``
  en vez de joblib/pickle: sin ejecución de código al abrir (los arrays
  son dtypes nativos más un metadato JSON), sin acoplamiento a versiones
  de Python/sklearn y reabrible con ``numpy`` puro. Los metadatos
  guardan ``schema_version`` (:data:`SCHEMA_VERSION`), ``dataset_id``,
  registros, ``lead``, ``fs``, ``created_at`` y las clases AAMI.
  ``save_dataset`` → ``load_dataset`` devuelve un dataset idéntico
  (``==``).
- **Identificador de contenido** — ``dataset_id`` = sha256 recortado a
  16 hex de esquema + registros + lead + fs + ``X``/``y``/
  ``paciente_id``: determinista (mismo input → mismo id, testeado) y
  revalidado al cargar para detectar archivos alterados o metadatos
  incoherentes.

El dataset real (descargado de PhysioNet) vive en ``data/mit-bih`` y
los ``.npz`` en ``data/``: gitignored, **nunca se commitean**. Las
pruebas usan ``RecordData`` sintéticos con ``load_record`` mockeado
(sin red, sin wfdb, sin dataset real).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app import mitbih
from app.mitbih_load import beat_windows, load_record
from app.signal import BEAT_WINDOW_S

#: Versión del esquema de los archivos ``.npz`` (forma de los arrays y
#: claves de ``meta``). Cambiarla invalida los archivos antiguos.
SCHEMA_VERSION = 1

#: Clases AAMI en orden canónico. El índice es el código entero 0-4 si
#: T10 prefiere codificar ``y``; la validación usa las letras.
AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")

#: Símbolo de anotación MIT-BIH → clase AAMI.
#:
#: Fuente: unidad de anotaciones de ``mitdb`` (PhysioNet) y la
#: correspondencia de 5 clases AAMI EC57 clásica (ANSI/AAMI EC57, mapeo
#: usado en de Chazal et al., IEEE TBME 2004):
#:
#: - ``N`` (normal): N, L (BBB izquierdo), R (BBB derecho), e, j
#: - ``S`` (ectópico supraventricular): A, a, J, S
#: - ``V`` (ectópico ventricular): V, E
#: - ``F`` (fusión): F
#: - ``Q`` (desconocido): /, f, Q **+** símbolos no-latido reales de
#:   los 48 registros de ``mitdb`` (``!``, ``[``, ``]``, ``x``, ``|``,
#:   ``~``, ``+``, ``"``) y cualquier símbolo no listado, p. ej. ``p``
#:   y ``t``, que no aparecen en los ``.atr`` (ver :func:`to_aami`).
MAP_AAMI: dict[str, str] = {
    # N — latido normal.
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # S — ectópico supraventricular (auricular/nodo AV prematuro).
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # V — ectópico ventricular (incluido el latido tardío E).
    "V": "V",
    "E": "V",
    # F — fusión de latido normal y ventricular.
    "F": "F",
    # Q — desconocido/artefacto (latidos Q + no-latido + no mapeados).
    # No-latido real presente en los 48 .atr: ! [ ] x | ~ + " . Los
    # símbolos p y t no aparecen en mitdb; se mantienen aquí y también
    # caen en Q (como cualquier otro no listado, vía to_aami).
    "/": "Q",
    "f": "Q",
    "Q": "Q",
    "!": "Q",
    "[": "Q",
    "]": "Q",
    "x": "Q",
    "|": "Q",
    "~": "Q",
    "+": "Q",
    '"': "Q",
    "p": "Q",
    "t": "Q",
}

#: Clase de los símbolos **no mapeados**: criterio explícito del issue
#: #25 — se etiquetan ``Q`` (desconocido) y **nunca** se descartan en
#: silencio.
UNMAPPED_AAMI_CLASS = "Q"


def to_aami(symbol: str) -> str:
    """Clase AAMI de un símbolo de anotación MIT-BIH.

    Parameters
    ----------
    symbol:
        Símbolo del archivo ``.atr`` (p. ej. ``"N"``, ``"V"``).

    Returns
    -------
    str
        Una de :data:`AAMI_CLASSES`. Los símbolos ausentes de
        :data:`MAP_AAMI` —cualquier otro símbolo desconocido o no
        presente en los 48 ``.atr`` reales— devuelven
        :data:`UNMAPPED_AAMI_CLASS` (``"Q"``): se etiqueta en vez de
        descartar en silencio.
    """
    return MAP_AAMI.get(symbol, UNMAPPED_AAMI_CLASS)


@dataclass(eq=False)
class EcgDataset:
    """Dataset MIT-BIH etiquetado con AAMI, latido a latido (T09c).

    ``eq=False`` porque la igualdad por defecto de dataclass compararía
    los arrays con ``==`` (devuelve arrays, no ``bool``): se implementa
    :meth:`__eq__` a mano con ``np.array_equal`` para que
    ``save_dataset`` → ``load_dataset`` sea comparable con ``==``.

    Attributes
    ----------
    X:
        Ventanas de señal, forma ``(n_latidos, n_muestras)`` y dtype
        float32. ``n_muestras`` es fijo por dataset
        (``2 * round(BEAT_WINDOW_S * fs)``): los latidos con ventana
        recortada en el borde no están (criterio documentado en el
        docstring del módulo).
    y:
        Clase AAMI por latido, strings ``"N"|"S"|"V"|"F"|"Q"`` (ver
        :data:`AAMI_CLASSES`). Longitud = ``n_latidos``.
    paciente_id:
        ID del registro MIT-BIH **por latido** (``"100"``…), para el
        split por paciente de T10. Longitud = ``n_latidos``.
    records:
        Registros de origen en el orden procesado.
    lead:
        Índice del canal usado (0 = MLII en ``mitdb``).
    fs:
        Frecuencia de muestreo (Hz), única para todo el dataset.
    dataset_id:
        Identificador de contenido (:func:`_compute_dataset_id`);
        vacío si el dataset se construyó a mano sin calcularlo.
    created_at:
        Fecha UTC ISO 8601 de construcción; vacío si no se registró.
    """

    X: np.ndarray
    y: np.ndarray
    paciente_id: np.ndarray
    records: tuple[str, ...]
    lead: int
    fs: float
    dataset_id: str = ""
    created_at: str = ""

    @property
    def n_latidos(self) -> int:
        """Número de latidos (filas de ``X``/``y``/``paciente_id``)."""
        return int(self.X.shape[0])

    @property
    def n_muestras(self) -> int:
        """Muestras por latido (columnas de ``X``); 0 si ``X`` no es 2-D."""
        return int(self.X.shape[1]) if self.X.ndim == 2 else 0

    def __eq__(self, other: object) -> bool:
        """Igualdad con arrays por ``np.array_equal`` (ver nota de clase)."""
        if not isinstance(other, EcgDataset):
            return NotImplemented
        return bool(
            np.array_equal(self.X, other.X)
            and np.array_equal(self.y, other.y)
            and np.array_equal(self.paciente_id, other.paciente_id)
            and self.records == other.records
            and self.lead == other.lead
            and self.fs == other.fs
            and self.dataset_id == other.dataset_id
            and self.created_at == other.created_at
        )


def _window_width(fs: float) -> int:
    """Ancho nominal de ventana en muestras (latido sin recortar).

    Espejo exacto de la fórmula de ``mitbih_load.beat_windows``:
    ``2 * half`` con ``half = max(1, round(BEAT_WINDOW_S * fs))`` (la
    constante se importa de ``app.signal``, sin duplicar el 0.4).
    """
    return 2 * max(1, round(BEAT_WINDOW_S * fs))


def _compute_dataset_id(
    X: np.ndarray,
    y: np.ndarray,
    paciente_id: np.ndarray,
    records: tuple[str, ...],
    lead: int,
    fs: float,
) -> str:
    """Identificador de contenido: sha256 (16 hex) de los datos.

    Determinista y reproducible: mismo esquema, registros, lead, fs y
    arrays ⇒ mismo id (no incluye ``created_at``). Se recalcula al
    cargar para detectar archivos alterados o metadatos incoherentes.
    """
    digest = hashlib.sha256()
    digest.update(f"ecg-ai/mitbih-dataset/v{SCHEMA_VERSION}".encode("utf-8"))
    digest.update(b"\x00")
    digest.update("\x1f".join(records).encode("utf-8"))
    digest.update(f"\x00lead={lead}\x00fs={fs!r}\x00".encode("utf-8"))
    for array in (X, y, paciente_id):
        array = np.ascontiguousarray(array)
        digest.update(f"\x00{array.dtype}\x00{array.shape}\x00".encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _check_alignment(X: np.ndarray, y: np.ndarray, paciente_id: np.ndarray) -> None:
    """Valida la forma de los tres arrays del dataset (guardar/cargar).

    Cada array debe ser un ``ndarray`` con la forma y el dtype del
    contrato (``X`` 2-D float32; ``y``/``paciente_id`` 1-D de strings)
    y las tres filas alineadas. Los tipos no-ndarray también son
    ``ValueError`` (nunca ``AttributeError``/``TypeError``).
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError(
            f"X debe ser un ndarray 2-D (n_latidos por ventana); "
            f"recibido: {type(X).__name__} con forma "
            f"{getattr(X, 'shape', None)!r}."
        )
    if X.dtype != np.float32:
        raise ValueError(f"X debe ser float32; recibido: {X.dtype}.")
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError(
            f"y debe ser un ndarray 1-D (clases AAMI por latido); "
            f"recibido: {type(y).__name__} con forma "
            f"{getattr(y, 'shape', None)!r}."
        )
    if not isinstance(paciente_id, np.ndarray) or paciente_id.ndim != 1:
        raise ValueError(
            f"paciente_id debe ser un ndarray 1-D (ID por latido); "
            f"recibido: {type(paciente_id).__name__} con forma "
            f"{getattr(paciente_id, 'shape', None)!r}."
        )
    if y.dtype.kind not in "US" or paciente_id.dtype.kind not in "US":
        raise ValueError(
            "y y paciente_id deben ser arrays de strings (clases AAMI y "
            f"IDs de paciente); recibido: {y.dtype} y {paciente_id.dtype}."
        )
    if not (X.shape[0] == y.shape[0] == paciente_id.shape[0]):
        raise ValueError(
            "Filas desalineadas: X, y y paciente_id deben tener la misma "
            f"longitud (recibido: {X.shape[0]}, {y.shape[0]}, "
            f"{paciente_id.shape[0]})."
        )


def _check_y_domain(y: np.ndarray, *, origin: str) -> None:
    """Valida que ``y`` solo contenga clases de :data:`AAMI_CLASSES`.

    Parameters
    ----------
    y:
        Array 1-D de etiquetas (el contrato de forma ya está
        verificado por :func:`_check_alignment`).
    origin:
        De dónde salen los datos, para el mensaje (p. ej. ``"el
        dataset"`` o la ruta del archivo).

    Raises
    ------
    ValueError
        Si alguna etiqueta está fuera de :data:`AAMI_CLASSES`; el
        mensaje nombra las clases inválidas.
    """
    invalid = sorted(set(map(str, y)) - set(AAMI_CLASSES))
    if invalid:
        raise ValueError(
            f"Clases AAMI inválidas en {origin}: {invalid}. "
            f"Se admite únicamente {list(AAMI_CLASSES)} (AAMI_CLASSES)."
        )


def build_dataset(
    records: list[str] | None = None,
    *,
    dest: Path = mitbih.DEFAULT_DEST,
    lead: int = 0,
) -> EcgDataset:
    """Construye el dataset AAMI (``X``, ``y``, ``paciente_id``).

    Por cada registro: :func:`~app.mitbih_load.load_record` (lo descarga
    si falta de ``dest``), :func:`~app.mitbih_load.beat_windows` y
    extracción de la ventana de la señal por anotación, que se etiqueta
    con :func:`to_aami`.

    Criterios (issue #25): los símbolos no mapeados van a ``Q``; los
    latidos con ventana vacía o recortada en el borde se **excluyen**;
    ``paciente_id`` repite el ID del registro en cada fila.

    Parameters
    ----------
    records:
        Subconjunto de :data:`app.mitbih.RECORDS` en el orden pedido
        (duplicados eliminados). ``None`` = los 48 completos.
        La lista vacía es error (misma regla que ``download_mitbih``).
    dest:
        Directorio con los archivos locales (por defecto
        ``data/mit-bih``, gitignored); se pasa a ``load_record``.
    lead:
        Índice del canal a usar (por defecto 0 = MLII en ``mitdb``).

    Returns
    -------
    EcgDataset
        ``X`` float32 ``(n_latidos, 2 * round(BEAT_WINDOW_S * fs))``,
        ``y`` y ``paciente_id`` de longitud ``n_latidos``. Si ningún
        latido pasa el filtro, los arrays quedan vacíos (``X`` con
        forma ``(0, ancho)``): no es error.

    Raises
    ------
    ValueError
        Si ``records`` está vacío o contiene IDs fuera de
        ``RECORDS`` (validación fail-fast de T09a: antes de descargar
        nada), o si los registros mezclan ``fs`` distintos (las filas
        deben tener un ancho único).
    IndexError
        Si ``lead`` fuera de rango (propagado de ``load_record``).
    ConnectionError
        Si hay que descargar y wfdb falla (propagado de T09a).
    """
    # Misma validación/canonización que T09a: None → 48, dedup preservando
    # orden y IDs desconocidos → ValueError antes de tocar la red.
    requested = mitbih._resolve_records(records)
    dest = Path(dest)

    rows: list[np.ndarray] = []
    labels: list[str] = []
    patients: list[str] = []
    fs_value: float | None = None
    width = 0

    for record_id in requested:
        data = load_record(record_id, lead=lead, dest=dest)
        if fs_value is None:
            fs_value = float(data.fs)
            width = _window_width(fs_value)
        elif data.fs != fs_value:
            raise ValueError(
                f"fs distinto entre registros: {record_id} tiene {data.fs} Hz "
                f"y los anteriores {fs_value} Hz. El dataset exige un único "
                f"fs (en {mitbih.MITBIH_DB} los 48 registros usan 360 Hz)."
            )

        for sample, symbol in data.annotations:
            # beat_windows se invoca anotación por anotación: la versión
            # global descarta las ventanas vacías y rompería la alineación
            # símbolo↔ventana (una anotación fuera de la señal desplazaría
            # todas las etiquetas siguientes).
            window = beat_windows([(sample, symbol)], data.n_samples, data.fs)
            if not window:
                continue  # sin ventana válida (anotación fuera de la señal)
            start, end = window[0]
            if end - start != width:
                continue  # ventana recortada en el borde → excluida (criterio)
            rows.append(np.asarray(data.signal[start:end], dtype=np.float32))
            labels.append(to_aami(symbol))
            patients.append(record_id)

    if fs_value is None:
        # Inalcanzable: _resolve_records ya lanzó si records está vacío.
        raise ValueError("records no puede estar vacío (usa None para los 48).")

    X = np.vstack(rows) if rows else np.empty((0, width), dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    paciente_id = np.asarray(patients, dtype=str)
    records_out = tuple(requested)
    return EcgDataset(
        X=X,
        y=y,
        paciente_id=paciente_id,
        records=records_out,
        lead=int(lead),
        fs=float(fs_value),
        dataset_id=_compute_dataset_id(X, y, paciente_id, records_out, int(lead), fs_value),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_dataset(ds: EcgDataset, path: str | Path) -> Path:
    """Serializa el dataset en un archivo NPZ (sin pickle).

    Escribe **exactamente** en ``path`` (se usa un handle de archivo:
    ``np.savez`` con un path sin sufijo le añadiría ``.npz``) y devuelve
    esa misma ruta. Los arrays van como dtypes nativos y los metadatos
    JSON en la clave ``meta``: el archivo se reabre con
    ``np.load(..., allow_pickle=False)`` sin ejecutar código.

    Parameters
    ----------
    ds:
        Dataset a guardar; se valida antes de escribir (formas
        alineadas, ``X`` float32 2-D, ``y``/``paciente_id`` strings,
        clases de ``y`` dentro de :data:`AAMI_CLASSES` y ``dataset_id``
        coherente con el contenido si está informado).
    path:
        Ruta destino; los directorios padre se crean si faltan.

    Returns
    -------
    Path
        La ruta escrita.

    Raises
    ------
    ValueError
        Si el dataset no cumple el contrato de formas/dtype, ``y``
        contiene clases fuera de :data:`AAMI_CLASSES` o su
        ``dataset_id`` no coincide con el contenido (dataset modificado
        tras ``build_dataset``).
    """
    _check_alignment(ds.X, ds.y, ds.paciente_id)
    _check_y_domain(ds.y, origin="el dataset")
    if ds.dataset_id:
        expected = _compute_dataset_id(
            ds.X, ds.y, ds.paciente_id, ds.records, ds.lead, ds.fs
        )
        if expected != ds.dataset_id:
            raise ValueError(
                f"dataset_id {ds.dataset_id!r} no coincide con el contenido "
                f"calculado ({expected!r}): el dataset se modificó tras "
                "build_dataset; reconstrúyelo."
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": ds.dataset_id,
        "records": list(ds.records),
        "lead": int(ds.lead),
        "fs": float(ds.fs),
        "created_at": ds.created_at,
        "aami_classes": list(AAMI_CLASSES),
    }
    with path.open("wb") as fh:
        np.savez(
            fh,
            X=ds.X,
            y=ds.y,
            paciente_id=ds.paciente_id,
            meta=np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )
    return path


def load_dataset(path: str | Path) -> EcgDataset:
    """Carga un dataset AAMI serializado con :func:`save_dataset`.

    Parameters
    ----------
    path:
        Archivo NPZ escrito por ``save_dataset``.

    Returns
    -------
    EcgDataset
        Idéntico (``==``) al que se guardó: mismos arrays, dtypes,
        registros, ``lead``, ``fs``, ``dataset_id`` y ``created_at``.

    Raises
    ------
    ValueError
        Si falta alguna clave (``X``, ``y``, ``paciente_id``, ``meta``),
        los metadatos no son JSON, no son un objeto (dict) o sus campos
        ``lead``/``fs``/``records`` no son legibles, la
        ``schema_version`` no es la soportada (:data:`SCHEMA_VERSION`),
        ``meta["aami_classes"]`` no coincide con
        :data:`AAMI_CLASSES`, ``y`` contiene clases fuera de
        :data:`AAMI_CLASSES`, las formas no están alineadas o el
        ``dataset_id`` no coincide con el contenido (archivo alterado).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [
            key for key in ("X", "y", "paciente_id", "meta") if key not in data.files
        ]
        if missing:
            raise ValueError(f"NPZ incompleto en {path}: faltan {missing}.")
        X = data["X"]
        y = data["y"]
        paciente_id = data["paciente_id"]
        raw_meta = data["meta"]

    try:
        meta = json.loads(str(raw_meta.item() if raw_meta.ndim == 0 else raw_meta))
    except (json.JSONDecodeError, AttributeError, ValueError) as exc:
        raise ValueError(f"Metadatos ilegibles en {path}: {exc}.") from exc
    if not isinstance(meta, dict):
        # JSON válido pero no-objeto ("[]", 42…): el contrato es un dict.
        raise ValueError(
            f"Metadatos ilegibles en {path}: se esperaba un objeto JSON "
            f"(dict); recibido: {type(meta).__name__} ({meta!r})."
        )

    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Versión de schema no soportada en {path}: {version!r} "
            f"(este código espera {SCHEMA_VERSION})."
        )

    aami_classes = meta.get("aami_classes")
    if aami_classes != list(AAMI_CLASSES):
        raise ValueError(
            f"meta['aami_classes'] inesperado en {path}: {aami_classes!r}; "
            f"se esperaba {list(AAMI_CLASSES)} (AAMI_CLASSES)."
        )

    _check_alignment(X, y, paciente_id)
    _check_y_domain(y, origin=f"el archivo {path}")

    try:
        lead = int(meta.get("lead", 0))
        fs = float(meta.get("fs", float("nan")))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Metadatos inválidos en {path}: lead/fs ilegibles "
            f"({type(exc).__name__}: {exc})."
        ) from exc
    raw_records = meta.get("records", ())
    if not isinstance(raw_records, (list, tuple)) or not all(
        isinstance(record, str) for record in raw_records
    ):
        raise ValueError(
            f"Metadatos inválidos en {path}: records debe ser una lista "
            f"de strings; recibido: {raw_records!r}."
        )
    records = tuple(raw_records)
    dataset_id = str(meta.get("dataset_id", ""))
    if dataset_id:
        expected = _compute_dataset_id(X, y, paciente_id, records, lead, fs)
        if expected != dataset_id:
            raise ValueError(
                f"dataset_id {dataset_id!r} no coincide con el contenido "
                f"calculado ({expected!r}) en {path}: archivo alterado o "
                "metadatos incoherentes."
            )
    return EcgDataset(
        X=X,
        y=y,
        paciente_id=paciente_id,
        records=records,
        lead=lead,
        fs=fs,
        dataset_id=dataset_id,
        created_at=str(meta.get("created_at", "")),
    )
