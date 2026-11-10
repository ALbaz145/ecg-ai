"""CLI para generar el artefacto del modelo real bajo demanda (T11c, #45).

Script reproducible —``python -m app.train_model`` desde ``backend/`` (o con
``PYTHONPATH=backend`` / el contenedor de T11d)— que **crea el artefacto** del
baseline sin commitearlo: descarga (o reutiliza) el dataset MIT-BIH con T09,
entrena con T10 y serializa en la **misma ruta** que la API resolverá en
runtime (:func:`app.model_loader.resolve_model_path`, por defecto
``data/models/baseline.joblib``), de modo que entrenar y servir apuntan al
mismo archivo.

Flujo — sin lógica duplicada, todo se reutiliza de T09/T10/T11a:

1. :func:`app.mitbih.download_mitbih(dest=--data-dir, records=--records)` —
   **idempotente**: los registros ya cacheados (``.hea`` + ``.dat`` +
   ``.atr``) no se vuelven a descargar (cero red, cero import de wfdb), así
   que re-ejecutar el CLI con ``data/`` local no vuelve a bajar nada.
2. :func:`app.mitbih_dataset.build_dataset(records, dest=--data-dir,
   lead=--lead)` — ventanas etiquetadas AAMI (T09c).
3. :func:`app.baseline.train_from_dataset(ds, test_size=--test-size,
   random_state=--seed)` — split **por paciente** y métricas de T10/T13.
4. :func:`app.baseline.save_model(model, --dest)` — crea los directorios
   destino y **sobrescribe** el artefacto (re-ejecutar es seguro).

Al terminar imprime por **stdout** el contenido de
:meth:`app.baseline.Metrics.to_dict` línea a línea (``accuracy``,
``sensibilidad_macro``, ``especificidad_macro``, ``matriz_confusion``,
etc.) y la ruta escrita —trazabilidad pedida en el issue #45—. Nada se
commitea: ``data/`` y ``*.joblib`` están en ``.gitignore`` y las pruebas
(:mod:`tests.backend.test_train_model`) escriben **solo** en ``tmp_path``.

Errores (sin traceback crudo): cualquier fallo del flujo —red caída,
IDs de registro desconocidos, dataset inválido o vacío, destino no
escribible— se imprime por **stderr** con el motivo y ``main`` devuelve
``1``; el guard ``sys.exit(main())`` convierte eso en **código de salida 1**
al ejecutar ``python -m app.train_model``. Los **argumentos** inválidos
siguen el código estándar de argparse (2). Si la descarga falla con
``ConnectionError`` y **no hay ninguna caché local** en ``--data-dir``, el
mensaje incluye instrucciones para preparar los datos a mano y funcionar
sin red.

Uso::

    cd backend
    python -m app.train_model --help
    python -m app.train_model --records 100 101 200 --seed 42
    MODEL_PATH=/ruta/modelo.joblib python -m app.train_model

Limitaciones (las de T09/T10, ver ``docs/MODEL.md``): solo la MIT-BIH
Arrhythmia (``mitdb``, 48 registros, un lead por registro), baseline
``LogisticRegression`` sobre 14 features por latido y métricas **de test
por paciente**, no validación clínica. Herramienta educativa, no es
diagnóstico médico.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from app.baseline import Metrics, save_model, train_from_dataset
from app.mitbih import _SUFFIXES, DEFAULT_DEST, RECORDS, download_mitbih, is_cached
from app.mitbih_dataset import EcgDataset, build_dataset
from app.model_loader import resolve_model_path

#: Semilla por defecto (``random_state`` del split y del clasificador).
DEFAULT_SEED = 42

#: Fracción de pacientes de test por defecto (issue #13: split por paciente).
DEFAULT_TEST_SIZE = 0.2

#: Mensaje de ayuda cuando no hay red **ni** caché local (``--data-dir`` vacío).
_MENSAJE_SIN_RED = (
    "No hay ninguna caché local en {dir} y no se pudo descargar. "
    "Para preparar los datos a mano y funcionar sin red:\n"
    "  1. Baja los registros MIT-BIH (.hea, .dat y .atr) desde\n"
    "     https://physionet.org/content/mitdb/1.0.0/\n"
    "     (p. ej. con `wfdb.dl_database('mitdb', {dir})` o una copia manual)\n"
    "     y déjalos planos en {dir}.\n"
    "  2. O copia ahí una caché existente (p. ej. data/mit-bih de otra máquina).\n"
    "  3. Re-ejecuta el CLI **sin** --force: con la caché completa no hay descarga."
)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------


def _semilla(value: str) -> int:
    """``--seed`` como entero en ``[0, 2**32 - 1]`` (rango de sklearn)."""
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--seed debe ser un entero; recibido: {value!r}."
        ) from exc
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError(
            f"--seed debe estar en [0, 2**32 - 1]; recibido: {seed}."
        )
    return seed


def _fraccion(value: str) -> float:
    """``--test-size`` como fracción estricta de pacientes en ``(0, 1)``."""
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--test-size debe ser un número en (0, 1); recibido: {value!r}."
        ) from exc
    if not 0.0 < ratio < 1.0:
        raise argparse.ArgumentTypeError(
            f"--test-size debe estar en (0, 1); recibido: {value!r}."
        )
    return ratio


def _ruta(value: str) -> Path:
    """``--dest``/``--data-dir`` como ``Path`` con ``~`` expandido (D-2).

    Sin expandir, argparse dejaría el ``~`` literal: ``--dest
    '~/modelo.joblib'`` crearía un directorio ``./~/`` en el *cwd* en
    vez de escribir en el home. El default de ``--dest`` (``None`` →
    ``resolve_model_path()``) ya expandía ``~``; aquí se aplica la
    misma semántica a lo que llega por línea de comandos.
    """
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    """Parser del CLI (cada argumento documentado en su ``help``).

    El default de ``--dest`` es ``None`` y se resuelve **al ejecutar**
    con :func:`app.model_loader.resolve_model_path` (lee ``MODEL_PATH``
    en cada llamada, no al construir el parser), de modo que una env var
    fijada antes de invocar :func:`main` se respeta y entrenar y servir
    apunten a la misma ruta.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.train_model",
        description=(
            "Entrena el baseline del proyecto (T10) con el dataset MIT-BIH "
            "(T09) y escribe el artefacto .joblib que la API carga en runtime "
            "(T11a). Imprime las métricas por stdout (accuracy, "
            "sensibilidad/especificidad macro, matriz de confusión) y la ruta "
            "escrita. data/ y *.joblib están gitignored: el artefacto nunca "
            "se commitea."
        ),
        epilog=(
            "Códigos de salida: 0 = artefacto escrito; 1 = error (detalle por "
            "stderr); 2 = argumentos inválidos (argparse)."
        ),
    )
    parser.add_argument(
        "--dest",
        type=_ruta,
        default=None,
        help=(
            "Ruta del artefacto .joblib a escribir (se crean los directorios "
            "padre y se sobrescribe si ya existe; ~ se expande al home). "
            "Por defecto: $MODEL_PATH o data/models/baseline.joblib, vía "
            "model_loader.resolve_model_path (la misma resolución que usa "
            "la API en runtime)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=_semilla,
        default=DEFAULT_SEED,
        help=(
            f"Semilla (random_state) del split por paciente y del "
            f"clasificador, para reproducibilidad (por defecto {DEFAULT_SEED})."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=_fraccion,
        default=DEFAULT_TEST_SIZE,
        help=(
            "Fracción de pacientes destinada al conjunto de test, en (0, 1) "
            f"(por defecto {DEFAULT_TEST_SIZE}). El reparto es por paciente: "
            "ningún paciente aparece en train y test a la vez (sin fuga)."
        ),
    )
    parser.add_argument(
        "--records",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Subconjunto de registros MIT-BIH a usar (p. ej. "
            "--records 100 101 200). Por defecto los 48 completos; cualquier "
            "ID fuera de la base mitdb es error antes de tocar la red."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=_ruta,
        default=DEFAULT_DEST,
        help=(
            "Directorio con la caché del dataset MIT-BIH (.hea/.dat/.atr); "
            f"por defecto {DEFAULT_DEST} (gitignored; ~ se expande al home). "
            "Si falta un registro se descarga aquí; si ya está cacheado no "
            "hay red."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-descarga los registros pedidos: purga su caché local "
            "(.hea/.dat/.atr) antes de descargar, por lo que requiere red "
            "y, si falla, esos archivos locales se pierden. Por defecto la "
            "caché local se respeta (funciona sin red)."
        ),
    )
    parser.add_argument(
        "--lead",
        type=int,
        default=0,
        help=(
            "Índice del canal (lead) del registro a usar; por defecto 0 "
            "(MLII en mitdb)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Flujo
# ---------------------------------------------------------------------------


def _registros_pedidos(records: list[str] | None) -> list[str]:
    """``records`` canonizado (``None`` → los 48 de :data:`app.mitbih.RECORDS`)."""
    return list(RECORDS) if records is None else list(records)


def _purgar_cache(data_dir: Path, records: list[str] | None) -> None:
    """Borra la caché local de los registros pedidos (soporte de ``--force``).

    Reutiliza el mismo trípode de sufijos de T09a (:data:`app.mitbih._SUFFIXES`,
    privado del paquete: se importa en vez de duplicar la lista) para que
    ``download_mitbih`` vuelva a bajarlos. Los IDs desconocidos se ignoran
    aquí (los valida ``download_mitbih`` con su error habitual).
    """
    conocidos = set(RECORDS)
    pedidos = [record for record in _registros_pedidos(records) if record in conocidos]
    for record in pedidos:
        for suffix in _SUFFIXES:
            (data_dir / f"{record}{suffix}").unlink(missing_ok=True)


def _hay_cache(data_dir: Path, records: list[str] | None) -> bool:
    """¿Hay algún registro de los pedidos completo en ``data_dir``? (D-1)."""
    return any(is_cached(data_dir, record) for record in _registros_pedidos(records))


def _cargar_dataset(
    *,
    records: list[str] | None,
    data_dir: Path,
    lead: int,
    force: bool,
) -> EcgDataset:
    """Descarga/carga MIT-BIH (T09) y construye el dataset AAMI (T09c).

    Parameters
    ----------
    records:
        Subconjunto de registros (``None`` = los 48).
    data_dir:
        Caché local; se descarga aquí lo que falte.
    lead:
        Índice del canal para :func:`app.mitbih_dataset.build_dataset`.
    force:
        Purga la caché de los registros pedidos antes de descargar.

    Returns
    -------
    EcgDataset
        Dataset con al menos 1 latido.

    Raises
    ------
    ValueError
        Si ``records`` contiene IDs desconocidos (validación fail-fast de
        T09a, antes de tocar la red), si los registros mezclan ``fs`` o si
        el dataset resulta **vacío** (ningún latido pasó el filtro).
    ConnectionError
        Si hay que descargar y wfdb falla (propagado de T09a).
    """
    data_dir = Path(data_dir)
    if force:
        _purgar_cache(data_dir, records)
    # Idempotente: lo cacheado no se vuelve a descargar (ver app.mitbih).
    download_mitbih(dest=data_dir, records=records)
    dataset = build_dataset(records, dest=data_dir, lead=lead)
    if dataset.n_latidos == 0:
        origen = ", ".join(records) if records else "los 48 registros"
        raise ValueError(
            f"Dataset vacío tras build_dataset (0 latidos; {origen}, lead={lead}): "
            "ningún latido pasó el filtro de ventanas. Comprueba --records/--lead "
            f"y que la caché en {data_dir} esté completa."
        )
    return dataset


def _imprimir_error(
    exc: Exception, *, data_dir: Path, records: list[str] | None
) -> None:
    """Mensaje de error por stderr (nunca un traceback crudo)."""
    print(
        f"Error: no se pudo generar el artefacto ({type(exc).__name__}): {exc}",
        file=sys.stderr,
    )
    if isinstance(exc, ConnectionError) and not _hay_cache(data_dir, records):
        print(_MENSAJE_SIN_RED.format(dir=data_dir), file=sys.stderr)


def _imprimir_metricas(metricas: Metrics, escrito: Path) -> None:
    """Métricas de T10/T13 por stdout, una clave por línea + ruta escrita.

    Imprime **todas** las claves de :meth:`app.baseline.Metrics.to_dict`
    (accuracy, sensibilidad/especificidad macro, matriz de confusión,
    métricas por clase, ``model_version``/``dataset_id`` de trazabilidad)
    con sus nombres exactos, más la ruta final del artefacto.
    """
    print("Entrenamiento completado.")
    for key, value in metricas.to_dict().items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        print(f"{key}: {value}")
    print(f"Artefacto escrito en: {escrito}")


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el CLI: descarga/carga → entrena → serializa → imprime.

    Parameters
    ----------
    argv:
        Argumentos sin el nombre del programa (``None`` = ``sys.argv[1:]``).
        ``--help`` lo gestiona argparse con salida por stdout y exit 0.

    Returns
    -------
    int
        ``0`` si el artefacto se escribió; ``1`` si el flujo falló (el
        motivo va a stderr — ver :func:`_imprimir_error`). El módulo lo
        convierte en código de salida con ``sys.exit(main())``.
    """
    args = build_parser().parse_args(argv)
    try:
        destino: Path = args.dest if args.dest is not None else resolve_model_path()
        dataset = _cargar_dataset(
            records=args.records,
            data_dir=args.data_dir,
            lead=args.lead,
            force=args.force,
        )
        modelo, metricas = train_from_dataset(
            dataset,
            test_size=args.test_size,
            random_state=args.seed,
        )
        escrito = save_model(modelo, destino)
    # Frontera de CLI: ningún error del flujo produce traceback crudo (el
    # motivo y la clase van a stderr con exit 1). SystemExit/KeyboardInterrupt
    # (BaseException) siguen propagándose: argparse --help y Ctrl-C intactos.
    except Exception as exc:
        _imprimir_error(exc, data_dir=Path(args.data_dir), records=args.records)
        return 1
    _imprimir_metricas(metricas, escrito)
    return 0


if __name__ == "__main__":
    sys.exit(main())
