"""Pruebas del loader del artefacto del modelo real (T11a, #43).

Datos **sintéticos aprendibles** (mismo criterio que
``test_baseline.py``): sin dataset real, sin wfdb, sin red. Los
``.joblib`` se escriben solo en ``tmp_path`` (nunca se commitean).

Cobertura frente a los criterios de aceptación de #43:

- carga OK desde ``MODEL_PATH`` con artefacto mínimo en ``tmp_path``;
- ausente → ``None`` + ``logger.warning`` con ruta y motivo;
- corrupto → ``None`` + ``logger.warning`` (sin excepción al caller);
- caché: la 2ª llamada no re-lee del disco (``joblib.load`` parcheado);
- ``reset_model_cache()`` fuerza la relectura;
- default de ``MODEL_PATH`` y override por env var;
- el ``None`` no se cachea: si el artefacto aparece después, la
  siguiente llamada lo carga (decisión de diseño del módulo).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from app import baseline, model_loader
from app.baseline import MODEL_VERSION, BaselineModel, save_model, train_baseline
from app.mitbih_dataset import AAMI_CLASSES
from app.model_loader import (
    DEFAULT_MODEL_PATH,
    MODEL_PATH_ENV,
    get_real_model,
    reset_model_cache,
    resolve_model_path,
)

FS = 360.0
WIDTH = 288  # 2 * round(BEAT_WINDOW_S * fs), ancho real de EcgDataset
CENTER = WIDTH // 2

_T = np.arange(WIDTH, dtype=np.float64)


# ---------------------------------------------------------------------------
# Datos sintéticos y artefacto mínimo
# ---------------------------------------------------------------------------


def _bump(center: float, sigma: float, amp: float) -> np.ndarray:
    """Onda gaussiana de ancho ``sigma`` y amplitud ``amp``."""
    return amp * np.exp(-0.5 * ((_T - center) / sigma) ** 2)


def _ventana_sintetica(clase: str, rng: np.random.Generator) -> np.ndarray:
    """Ventana sintética distinguible por clase (versión mínima)."""
    if clase == "N":
        base = _bump(CENTER, 6.0, 1.0)
    elif clase == "S":
        base = _bump(CENTER, 5.0, 0.55)
    elif clase == "V":
        base = _bump(CENTER, 16.0, 1.8) - 0.6 * _bump(CENTER + 45, 12.0, 1.0)
    elif clase == "F":
        base = _bump(CENTER, 6.0, 1.0) + _bump(CENTER + 35, 10.0, 0.9)
    elif clase == "Q":
        base = 0.08 * np.sin(2 * np.pi * 1.5 * _T / FS)
    else:  # pragma: no cover - defensa contra typos en los tests
        raise AssertionError(f"clase sintética desconocida: {clase!r}")
    return base + rng.normal(0.0, 0.02, WIDTH)


def _dataset_sintetico(
    n_pacientes: int = 3, repeticiones: int = 1, semilla: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, paciente_id)`` mínimo y aprendible (3 pacientes x 5 clases)."""
    rng = np.random.default_rng(semilla)
    filas: list[np.ndarray] = []
    etiquetas: list[str] = []
    pacientes: list[str] = []
    for p in range(n_pacientes):
        for clase in AAMI_CLASSES * repeticiones:
            filas.append(_ventana_sintetica(clase, rng).astype(np.float32))
            etiquetas.append(clase)
            pacientes.append(f"{100 + p}")
    return np.vstack(filas), np.asarray(etiquetas), np.asarray(pacientes)


@pytest.fixture(scope="module")
def modelo_entrenado() -> BaselineModel:
    """``BaselineModel`` entrenado una sola vez (datos fijos y rápidos)."""
    X, y, paciente_id = _dataset_sintetico()
    model, _ = train_baseline(X, y, paciente_id, fs=FS, dataset_id="t11a-loader")
    return model


@pytest.fixture(autouse=True)
def _cache_limpio() -> Iterator[None]:
    """Aísla cada test: empieza y termina sin caché de modelo."""
    reset_model_cache()
    yield
    reset_model_cache()


@pytest.fixture
def cargas_de_joblib(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Doble de ``joblib.load`` que registra cada lectura del disco."""
    original = baseline.joblib.load
    cargas: list[object] = []

    def _cargador(*args: object, **kwargs: object) -> object:
        cargas.append(args[0] if args else None)
        return original(*args, **kwargs)

    monkeypatch.setattr(baseline.joblib, "load", _cargador)
    return cargas


def _artefacto(modelo: BaselineModel, tmp_path: Path, nombre: str = "baseline.joblib") -> Path:
    """Guarda el modelo en ``tmp_path`` y devuelve su ruta."""
    return save_model(modelo, tmp_path / nombre)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Mensajes WARNING del logger ``ecg_ai.backend`` capturados."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "ecg_ai.backend"
    ]


# ---------------------------------------------------------------------------
# Carga OK
# ---------------------------------------------------------------------------


def test_carga_ok_desde_model_path(
    modelo_entrenado: BaselineModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _artefacto(modelo_entrenado, tmp_path)
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))

    model = get_real_model()

    assert isinstance(model, BaselineModel)
    assert model.model_version == MODEL_VERSION
    # Predict con 1 latido (contrato T11): (1, n_muestras) → 1 etiqueta AAMI.
    win = _ventana_sintetica("N", np.random.default_rng(0))
    pred = model.predict(win[None, :])
    assert pred.shape == (1,)
    assert str(pred[0]) in AAMI_CLASSES


def test_carga_ok_devuelve_el_mismo_objeto_en_cache(
    modelo_entrenado: BaselineModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _artefacto(modelo_entrenado, tmp_path)
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))

    primero = get_real_model()
    segundo = get_real_model()

    assert primero is not None
    assert segundo is primero  # misma lectura, mismo objeto


# ---------------------------------------------------------------------------
# Ausente / corrupto → None + warning
# ---------------------------------------------------------------------------


def test_ausente_devuelve_none_y_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "no-existe.joblib"
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))
    caplog.set_level(logging.WARNING, logger="ecg_ai.backend")

    model = get_real_model()

    assert model is None  # nunca lanza: el caller sigue con el heurístico
    avisos = _warnings(caplog)
    assert len(avisos) == 1
    assert str(path) in avisos[0]  # ruta intentada
    assert "no existe" in avisos[0]  # motivo


def test_corrupto_devuelve_none_y_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "baseline.joblib"
    path.write_bytes(b"esto no es un joblib valido")
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))
    caplog.set_level(logging.WARNING, logger="ecg_ai.backend")

    model = get_real_model()

    assert model is None  # sin excepción hacia el request
    avisos = _warnings(caplog)
    assert len(avisos) == 1
    assert str(path) in avisos[0]
    # joblib/pickle lanzan clases distintas según el contenido basura
    # (p. ej. IndexError, EOFError, ValueError…); el motivo lo incluye.
    assert any(
        tipo in avisos[0] for tipo in ("IndexError", "EOFError", "ValueError", "error")
    )


def test_directorio_como_model_path_devuelve_none_y_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path))  # un dir, no un archivo
    caplog.set_level(logging.WARNING, logger="ecg_ai.backend")

    assert get_real_model() is None
    avisos = _warnings(caplog)
    assert len(avisos) == 1
    assert str(tmp_path) in avisos[0]
    assert "archivo regular" in avisos[0]


def test_falla_no_se_cachea_reintenta_en_siguiente_llamada(
    modelo_entrenado: BaselineModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decisión de diseño: el ``None`` no se cachea (ver model_loader)."""
    path = tmp_path / "baseline.joblib"
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))

    assert get_real_model() is None  # aún no existe

    _artefacto(modelo_entrenado, tmp_path)  # aparece en caliente (T11c/T11d)

    assert isinstance(get_real_model(), BaselineModel)  # sin reset ni restart


# ---------------------------------------------------------------------------
# Caché y reset
# ---------------------------------------------------------------------------


def test_cache_segunda_llamada_no_vuelve_a_leer(
    modelo_entrenado: BaselineModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cargas_de_joblib: list[object],
) -> None:
    path = _artefacto(modelo_entrenado, tmp_path)
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))

    assert get_real_model() is not None
    assert len(cargas_de_joblib) == 1  # primera llamada: 1 lectura

    assert get_real_model() is not None
    assert len(cargas_de_joblib) == 1  # cacheada: no se re-lee


def test_reset_model_cache_fuerza_relectura(
    modelo_entrenado: BaselineModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cargas_de_joblib: list[object],
) -> None:
    path = _artefacto(modelo_entrenado, tmp_path)
    monkeypatch.setenv(MODEL_PATH_ENV, str(path))

    assert get_real_model() is not None
    reset_model_cache()
    assert get_real_model() is not None

    assert len(cargas_de_joblib) == 2  # el reset obliga a una lectura nueva


def test_reset_model_cache_sin_carga_previa_es_seguro() -> None:
    reset_model_cache()  # idempotente: no lanza ni deja estado inconsistente
    reset_model_cache()
    assert model_loader._cached_model is None
    assert model_loader._cached_path is None


# ---------------------------------------------------------------------------
# MODEL_PATH: default, override y semántica de ruta
# ---------------------------------------------------------------------------


def test_model_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_PATH_ENV, raising=False)

    assert DEFAULT_MODEL_PATH == Path("data/models/baseline.joblib")
    assert resolve_model_path() == DEFAULT_MODEL_PATH


def test_model_path_vacio_usa_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, "   ")

    assert resolve_model_path() == DEFAULT_MODEL_PATH


def test_model_path_override_por_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    otro = tmp_path / "otro" / "modelo.joblib"
    monkeypatch.setenv(MODEL_PATH_ENV, str(otro))

    assert resolve_model_path() == otro


def test_model_path_se_relee_en_cada_llamada(
    modelo_entrenado: BaselineModel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El override del env var se aplica sin reiniciar el proceso."""
    path = _artefacto(modelo_entrenado, tmp_path)
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "no-existe.joblib"))
    assert get_real_model() is None

    monkeypatch.setenv(MODEL_PATH_ENV, str(path))
    assert isinstance(get_real_model(), BaselineModel)


def test_model_path_relativo_resuelve_contra_cwd(
    modelo_entrenado: BaselineModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relativa al *working directory* (convención de app.mitbih)."""
    _artefacto(modelo_entrenado, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(MODEL_PATH_ENV, "baseline.joblib")

    assert resolve_model_path() == Path("baseline.joblib")
    assert isinstance(get_real_model(), BaselineModel)
