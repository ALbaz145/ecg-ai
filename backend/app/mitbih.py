"""Descarga y caché del dataset MIT-BIH Arrhythmia (PhysioNet).

Módulo de **datos** (T09a, issue #23): solo descarga y cachea los
archivos WFDB públicos de ``mitdb`` en ``data/mit-bih/`` (gitignored).
La lectura de la señal y las anotaciones (T09b, issue #24) vive en
``app.mitbih_load``; el procesamiento de la señal en ``app.signal``.

Decisiones de diseño:

- **Mecanismo wfdb**: ``wfdb.dl_database("mitdb", dest, ...)``
  (wfdb==4.3.1). Es la vía canónica: resuelve desde el archivo
  ``RECORDS`` del servidor los ``.hea``/``.dat``/``.atr`` de cada
  registro (lee el header remoto para conocer los ``.dat`` reales) y
  los deja planos en ``dest`` (``keep_subdirs=False``).
- **Caché a nivel de registro** (mitigación **D-1**): si ``<rec>.hea``,
  ``<rec>.dat`` **y** ``<rec>.atr`` ya existen en ``dest``, el registro
  se considera cacheado y no se invoca wfdb en absoluto (segunda llamada
  = cero red, cero imports). Exigir también el ``.atr`` da un registro
  **completo por cacheado**: un proceso matado a mitad de descarga no
  deja un registro "cacheado" sin anotaciones (``load_record`` de
  ``app.mitbih_load`` las necesita).
- **wfdb perezoso**: se importa dentro de la descarga (``_load_wfdb``)
  para no acoplar el import de la app al paquete pesado y para que las
  pruebas inyecten un doble sin red ni wfdb instalado.
- **``base_url``**: se aplica con ``wfdb.set_db_index_url(base_url)``
  antes de descargar y se restaura el índice por defecto al terminar.
  Limitación de wfdb 4.3.1: la *versión* de la BD se resuelve siempre
  contra ``physionet.org/content`` (``PN_CONTENT_URL`` está fijada en
  el código), así que ``base_url`` redirige la descarga de archivos
  pero no el descubrimiento de versión.
- **Errores**: cualquier ``OSError`` de wfdb (red caída, 404, timeout…
  en wfdb 4.3.1 todo lo de red deriva de ``OSError``) se relanza como
  ``ConnectionError`` con mensaje útil; las descargas incompletas de
  la llamada fallida se purgan, mientras que la caché previa se
  conserva (reintentar solo baja lo que falte).

El dataset es **público** (MIT-BIH, PhysioNet) y nunca se commitea:
``data/``, ``*.dat``, ``*.hea`` y ``*.atr`` están en ``.gitignore``.
"""

from __future__ import annotations

from pathlib import Path

#: Registros canónicos de la MIT-BIH Arrhythmia Database (**48**),
#: tomados del archivo canónico ``RECORDS`` que wfdb usa internamente:
#: https://physionet.org/files/mitdb/1.0.0/RECORDS (verificado
#: 2026-11-05). No existen el 110, 120, 125, 126 (huecos en el
#: bloque 100-126) ni, dentro de 200-234, el 204, 206, 211, 216, 218,
#: 224, 225, 226, 227 y 229; sí existen 122 y 222. Ojo: una
#: enumeración "100-108, 109, 111-119, 120, 121, 123-126, ..."
#: daría 49 IDs e incluiría inexistentes.
RECORDS: list[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119",
    "121", "122", "123", "124",
    "200", "201", "202", "203", "205",
    "207", "208", "209", "210",
    "212", "213", "214", "215", "217",
    "219", "220", "221", "222", "223", "228",
    "230", "231", "232", "233", "234",
]

#: Slug de la base en PhysioNet.
MITBIH_DB = "mitdb"

#: Directorio por defecto de descarga (relativo al cwd; gitignored).
DEFAULT_DEST = Path("data/mit-bih")

#: Índice por defecto de wfdb (``wfdb.io.download.PN_INDEX_URL``).
DEFAULT_BASE_URL = "https://physionet.org/files/"

#: Sufijos de cada registro MIT-BIH: cabecera, señal y anotación.
_SUFFIXES = (".hea", ".dat", ".atr")


def _load_wfdb():
    """Importa ``wfdb`` de forma perezosa (solo al descargar).

    Devuelve el paquete ``wfdb`` (4.3.1). Las pruebas lo sustituyen
    con un doble vía ``monkeypatch`` en esta función, por lo que la
    suite nunca toca la red ni requiere wfdb instalado.
    """
    import wfdb

    return wfdb


def is_cached(dest: Path, record: str) -> bool:
    """¿Está ``record`` cacheado en ``dest``?

    Un registro se considera cacheado cuando su cabecera (``.hea``),
    su señal (``.dat``) **y** sus anotaciones (``.atr``) ya existen
    localmente: solo así tiene sentido para ``app.mitbih_load``.
    Exigir el ``.atr`` (mitigación **D-1**) evita dar por cacheado un
    registro cuya descarga se interrumpió tras ``.hea``/``.dat`` y que
    quedaría sin anotaciones.
    """
    return all(
        (dest / f"{record}{suffix}").is_file() for suffix in _SUFFIXES
    )


def _purge_incomplete(dest: Path, records: list[str]) -> None:
    """Elimina los archivos de ``records`` que no estén completos.

    Tras un fallo, un registro solo se conserva si tiene ``.hea``,
    ``.dat`` **y** ``.atr`` (descarga terminada, sirve para reanudar);
    si falta cualquiera, se borran sus tres archivos para no dejar
    medio-descargas que la caché daría por buenas.
    """
    for record in records:
        files = [dest / f"{record}{suffix}" for suffix in _SUFFIXES]
        if all(path.is_file() for path in files):
            continue
        for path in files:
            path.unlink(missing_ok=True)


def _resolve_records(records: list[str] | None) -> list[str]:
    """Normaliza y valida la lista de registros pedida."""
    requested = list(RECORDS) if records is None else list(dict.fromkeys(records))
    if not requested:
        raise ValueError("records no puede estar vacío (usa None para los 48).")
    known = set(RECORDS)
    unknown = [record for record in requested if record not in known]
    if unknown:
        raise ValueError(
            f"Registros desconocidos en {MITBIH_DB}: {unknown}. "
            f"Usa IDs de RECORDS (p. ej. {RECORDS[0]!r})."
        )
    return requested


def download_mitbih(
    dest: Path = DEFAULT_DEST,
    records: list[str] | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> Path:
    """Descarga (o reutiliza desde caché) registros MIT-BIH con wfdb.

    Idempotente: un registro cuyo ``.hea``, ``.dat`` y ``.atr`` ya
    existen en ``dest`` no se vuelve a descargar, así que la segunda
    llamada no vuelve a tocar wfdb ni la red.

    Parameters
    ----------
    dest:
        Directorio destino (por defecto ``data/mit-bih``, gitignored).
    records:
        Subconjunto de :data:`RECORDS`; ``None`` = los 48 completos.
    base_url:
        Índice base de PhysioNet que wfdb usará para leer los archivos
        (por defecto el de wfdb). Se restaura el índice por defecto al
        terminar, haya habido error o no.

    Returns
    -------
    Path
        El directorio ``dest`` con los registros disponibles.

    Raises
    ------
    ValueError
        Si ``records`` está vacío o contiene IDs fuera de :data:`RECORDS`.
    ConnectionError
        Si wfdb falla (red caída, URL inválida, servidor no
        disponible…). El mensaje incluye el error original; las
        descargas incompletas de la llamada se purgan y la caché
        previa se conserva, de modo que reintentar solo baja lo que
        falte.
    """
    requested = _resolve_records(records)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    pending = [record for record in requested if not is_cached(dest, record)]
    if not pending:
        # Caché completa: ni siquiera se importa wfdb.
        return dest

    wfdb = _load_wfdb()
    try:
        wfdb.set_db_index_url(base_url)
        wfdb.dl_database(
            MITBIH_DB,
            str(dest),
            records=pending,
            annotators=["atr"],
            keep_subdirs=False,
            overwrite=False,
        )
    # Se captura Exception a propósito: lo no-OSError se relanza tal cual
    # (encadenado) y lo de red se convierte en ConnectionError abajo.
    except Exception as exc:
        _purge_incomplete(dest, pending)
        if isinstance(exc, OSError):
            raise ConnectionError(
                f"No se pudo descargar MIT-BIH ({len(pending)} registros) "
                f"desde {base_url}: {exc}. Comprueba la conexión a "
                "PhysioNet y reintenta: la caché local se conserva y solo "
                "se volverá a bajar lo que falte."
            ) from exc
        raise
    finally:
        # Restaura el índice por defecto de wfdb (PN_INDEX_URL).
        wfdb.set_db_index_url()

    return dest
