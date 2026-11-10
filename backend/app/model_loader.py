"""Loader del artefacto del modelo real (T11a, issue #43).

Resuelve la env var ``MODEL_PATH`` (por defecto
``data/models/baseline.joblib``), carga el bundle **una sola vez por
proceso** con :func:`app.baseline.load_model` y lo expone de forma
segura: si el archivo falta, no es legible o sus metadatos no
validan, la función devuelve ``None`` y emite ``logger.warning`` con
la **ruta intentada** y el **motivo**. **Nunca lanza excepción** al
caller, de modo que el pipeline (T11b) pueda caer al heurístico de
``app.model`` sin romper el request. Las pruebas viven en
``tests/backend/test_model_loader.py``.

Rutas: ``MODEL_PATH`` y *working directory*
    ``MODEL_PATH`` se lee en **cada** llamada de
    :func:`get_real_model` (no al importar el módulo), de modo que
    los tests pueden usar ``monkeypatch.setenv`` y T11d puede
    inyectarlo en docker-compose. Una ruta **relativa** se resuelve
    contra el *working directory* del proceso —misma convención que
    ``app.mitbih.DEFAULT_DEST``: en desarrollo local (raíz del repo)
    es ``data/models/baseline.joblib`` y en el contenedor
    ``/app/data/models/baseline.joblib`` con el volumen de T11d—.
    ``MODEL_PATH`` vacío o en blanco cuenta como no definido (default)
    y ``~`` se expande.

Caché: modelo sí, ``None`` no
    El modelo cargado se cachea en el módulo con un ``threading.Lock``
    (los endpoints síncronos corren en threadpool): una sola lectura
    del disco por proceso y **siempre el mismo objeto** mientras la
    ruta resuelta no cambie (si cambia ``MODEL_PATH``, se recarga para
    la nueva ruta). Cuando la carga **falla** no se cachea nada: la
    siguiente llamada reintenta (un ``stat`` + un intento de
    ``joblib.load`` sobre un archivo pequeño, despreciable frente al
    procesamiento de la señal por request).

    Se elige **reintentar** en vez de cachear el ``None`` porque el
    artefacto puede aparecer en caliente —el CLI de T11c o el volumen
    de T11d— sin reiniciar la API: así, en cuanto exista el archivo,
    el siguiente request ya usa el modelo real sin ``reset_model_cache()``
    ni restart. Coste asumido: un ``warning`` por request mientras el
    artefacto siga ausente, que además hace visible el fallback.
    Tras regenerar o sobreescribir el artefacto **con la misma ruta**,
    :func:`reset_model_cache` fuerza la relectura (tests y recarga
    explícita).

Por qué se captura ``Exception`` (y no solo ``FileNotFoundError`` /
``ValueError`` / ``OSError``)
    Un archivo corrupto no siempre produce esas excepciones: con
    ``joblib`` un contenido basura puede lanzar ``IndexError`` /
    ``EOFError`` y un bundle con forma equivocada ``ValueError``. El
    contrato de este módulo es **nunca propagar**, así que se captura
    ``Exception`` (no ``BaseException``: ``KeyboardInterrupt`` /
    ``SystemExit`` siguen propagándose) y el motivo —clase y
    mensaje— va al warning.

Uso (T11b)::

    model = get_real_model()
    if model is None:
        ...  # rama heurística intacta (app.model.classify)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from app.baseline import BaselineModel, load_model

logger = logging.getLogger("ecg_ai.backend")

#: Env var con la ruta al artefacto (leída en cada llamada a
#: :func:`get_real_model`; inyectada por T11d en docker-compose).
MODEL_PATH_ENV = "MODEL_PATH"

#: Ruta por defecto si ``MODEL_PATH`` no está definida (o vacía).
#: Relativa al *working directory*; ``*.joblib`` está gitignored:
#: el artefacto **nunca se commitea** (la entrega es T11c + T11d).
DEFAULT_MODEL_PATH = Path("data/models/baseline.joblib")

#: Caché por proceso (protegida por ``_LOCK``: los endpoints
#: síncronos de FastAPI corren en threadpool).
_LOCK = threading.Lock()
_cached_path: Path | None = None
_cached_model: BaselineModel | None = None


def resolve_model_path() -> Path:
    """Ruta del artefacto: ``MODEL_PATH`` o :data:`DEFAULT_MODEL_PATH`.

    Lee la env var en este momento (no al importar) y normaliza:
    valor vacío o en blanco → default; ``~`` expandido; lo demás se
    devuelve tal cual (las rutas relativas se resuelven contra el
    *working directory* al abrir el archivo; ver docstring del módulo).
    """
    raw = os.getenv(MODEL_PATH_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MODEL_PATH


def get_real_model() -> BaselineModel | None:
    """Modelo real cacheado, o ``None`` si no está disponible.

    Carga perezosa y **cacheada** de :data:`~app.baseline.MODEL_VERSION`
    desde la ruta de :func:`resolve_model_path` (una lectura por
    proceso; la segunda llamada con la misma ruta devuelve el mismo
    objeto sin tocar disco).

    Returns
    -------
    BaselineModel | None
        El modelo cargado, o ``None`` si el artefacto **no existe**,
        **no es un archivo regular** o **no es cargable** (contenido
        corrupto, metadatos inválidos, permisos…). En el caso
        ``None`` se emite un ``logger.warning`` (logger
        ``ecg_ai.backend``) con la ruta intentada y el motivo.

    Notes
    -----
    **Nunca lanza excepción**: el contrato del módulo es que el
    caller (T11b) pueda seguir con el heurístico. El ``None`` **no
    se cachea** (se reintenta en cada llamada, ver docstring del
    módulo); para vaciar la caché de un modelo ya cargado está
    :func:`reset_model_cache`.
    """
    path = resolve_model_path()
    global _cached_path, _cached_model
    with _LOCK:
        if _cached_model is not None and _cached_path == path:
            return _cached_model

        motivo: str
        if not path.exists():
            motivo = "el archivo no existe"
        elif not path.is_file():
            motivo = "la ruta no es un archivo regular"
        else:
            try:
                model = load_model(path)
            except Exception as exc:  # el contrato es nunca propagar (ver docstring)
                detalle = str(exc).strip()
                motivo = f"{type(exc).__name__}: {detalle}" if detalle else type(exc).__name__
            else:
                _cached_path = path
                _cached_model = model
                return model

        logger.warning(
            "Modelo real no disponible en %s (%s); el pipeline seguirá "
            "con el heurístico.",
            path,
            motivo,
        )
        return None


def reset_model_cache() -> None:
    """Vacía la caché del modelo para que la siguiente llamada re-lea.

    Lo usan las pruebas (aislamiento entre tests) y la recarga
    explícita tras regenerar el artefacto con la misma ruta. No hace
    falta para que un artefacto **nuevo** pase a usarse si antes no
    estaba: ese caso reintenta solo (ver docstring del módulo).
    """
    global _cached_path, _cached_model
    with _LOCK:
        _cached_path = None
        _cached_model = None
