"""Regresión del aislamiento de la suite respecto a ``MODEL_PATH`` (D-x).

Repro de QA: con un **artefacto válido** en cualquier ruta,
``MODEL_PATH=/ruta/artefacto.joblib python -m pytest tests/ -q`` dejaba
17 tests en rojo (los que asumen la rama heurística recibían etiquetas
AAMI del modelo real: ``assert 'arritmia_supraventricular' == 'normal'``).

Estos tests garantizan que la fixture autouse
``tests/backend/conftest.py::_aislamiento_model_path`` mantiene el
aislamiento:

- un ``MODEL_PATH`` **externo** con artefacto válido (el del shell de
  QA, simulado escribiendo ``os.environ`` en scope de módulo, antes
  que las fixtures *function-scoped*) no contamina al ``client``
  estándar: ``POST /ecg/analyze`` sigue en ``heuristic-mvp-0.1.0``;
- la inyección **explícita** (``monkeypatch.setenv`` en el test, el
  patrón de ``test_inference.py``) sigue ganando y activa la rama real
  —mismo artefacto, mismo cliente—, lo que además prueba que el
  artefacto usado por los tests es **válido**;
- la repro literal de QA en un **subproceso** (``MODEL_PATH`` externo +
  tests estándar de la suite) termina en verde.

Datos sintéticos aprendibles, ``.joblib`` solo en ``tmp_path`` (nunca
se commitean), sin red y sin wfdb.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from app.baseline import (
    MODEL_VERSION,
    BaselineModel,
    load_model,
    save_model,
    train_baseline,
)
from app.mitbih_dataset import AAMI_CLASSES
from app.model_loader import MODEL_PATH_ENV, reset_model_cache, resolve_model_path

from tests.backend.synthetic import synthetic_ecg

REPO_ROOT = Path(__file__).resolve().parents[2]

HEURISTIC_VERSION = "heuristic-mvp-0.1.0"

FS = 360.0
WIDTH = 288
CENTER = WIDTH // 2
_T = np.arange(WIDTH, dtype=np.float64)


# ---------------------------------------------------------------------------
# Artefacto mínimo entrenado (mismo patrón que T10/T11a/T11b)
# ---------------------------------------------------------------------------


def _bump(center: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((_T - center) / sigma) ** 2)


def _ventana_sintetica(clase: str, rng: np.random.Generator) -> np.ndarray:
    if clase == "N":
        base = _bump(CENTER, 6.0, 1.0)
    elif clase == "S":
        base = _bump(CENTER, 5.0, 0.55)
    elif clase == "V":
        base = _bump(CENTER, 16.0, 1.8) - 0.6 * _bump(CENTER + 45, 12.0, 1.0)
    elif clase == "F":
        base = _bump(CENTER, 6.0, 1.0) + _bump(CENTER + 35, 10.0, 0.9)
    else:  # "Q"
        base = 0.08 * np.sin(2 * np.pi * 1.5 * _T / FS)
    return base + rng.normal(0.0, 0.02, WIDTH)


def _modelo_minimo() -> BaselineModel:
    """Baseline aprendible mínimo (6 pacientes x 5 clases x 4)."""
    rng = np.random.default_rng(7)
    filas: list[np.ndarray] = []
    etiquetas: list[str] = []
    pacientes: list[str] = []
    for p in range(6):
        for clase in AAMI_CLASSES * 4:
            filas.append(_ventana_sintetica(clase, rng).astype(np.float32))
            etiquetas.append(clase)
            pacientes.append(f"{100 + p}")
    model, _metrics = train_baseline(
        np.vstack(filas),
        np.asarray(etiquetas),
        np.asarray(pacientes),
        fs=FS,
        dataset_id="d-x-aislamiento",
    )
    return model


@pytest.fixture(scope="module")
def model_path_externo(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """``MODEL_PATH`` **externo** con artefacto **válido** (repro de QA).

    Fija la env var directamente con ``os.environ`` —sin ``monkeypatch``,
    que solo existe en scope de función— para simular el valor que el
    shell/CI pone al proceso. Es un fixture de **módulo**, así que se
    instancia antes que las fixtures *function-scoped* (incluida la
    autouse de aislamiento de ``conftest.py``), que son las que tienen
    que neutralizarlo. Se restaura el valor previo al terminar.
    """
    artefacto = save_model(
        _modelo_minimo(), tmp_path_factory.mktemp("modelo-externo") / "baseline.joblib"
    )
    original = os.environ.get(MODEL_PATH_ENV)
    os.environ[MODEL_PATH_ENV] = str(artefacto)
    try:
        yield artefacto
    finally:
        if original is None:
            os.environ.pop(MODEL_PATH_ENV, None)
        else:
            os.environ[MODEL_PATH_ENV] = original
        reset_model_cache()


def _post(client, signal, sample_rate_hz: int = 500) -> dict:
    response = client.post(
        "/ecg/analyze",
        json={"signal": signal, "sample_rate_hz": sample_rate_hz, "units": "mV"},
    )
    assert response.status_code == 200
    return response.json()


# ---------------------------------------------------------------------------
# El env var externo no contamina al cliente estándar
# ---------------------------------------------------------------------------


def test_model_path_externo_con_artefacto_valido_no_contamina_al_cliente(
    client, model_path_externo: Path
) -> None:
    """D-x: ``MODEL_PATH`` externo con artefacto válido → sigue heurístico.

    El artefacto externo es **válido** (carga con ``load_model``) y
    aun así un ``POST /ecg/analyze`` **sin inyección explícita** vuelve
    ``heuristic-mvp-0.1.0``: la fixture autouse de ``conftest.py``
    redirigió el env var a una ruta inexistente antes de que corriera
    el test (sin ella, ``resolve_model_path()`` seguiría apuntando al
    artefacto y la etiqueta sería AAMI —el fallo de QA).
    """
    assert isinstance(load_model(model_path_externo), BaselineModel)

    ruta_aislada = resolve_model_path()
    assert ruta_aislada != model_path_externo
    assert not ruta_aislada.exists()

    body = _post(client, synthetic_ecg(bpm=120))

    assert body["model_version"] == HEURISTIC_VERSION
    assert body["anomalia"] == "taquicardia"  # etiqueta del heurístico


def test_inyeccion_explicita_del_modelo_sigue_ganando(
    client, model_path_externo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contra-prueba del test anterior: inyección explícita → rama real.

    El mismo artefacto y el mismo ``client`` estándar, pero con el
    ``setenv`` explícito del patrón T11b (el que usan los tests de
    ``test_inference.py``), que la fixture de aislamiento respeta. Si
    este test fallara, el artefacto de ``model_path_externo`` sería
    inválido y el test anterior no probaría nada.
    """
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path_externo))
    reset_model_cache()

    body = _post(client, synthetic_ecg(bpm=120))

    assert body["model_version"] == MODEL_VERSION == "baseline-sklearn-0.1.0"


def test_repro_de_qa_con_model_path_externo_pasa_en_subproceso(
    model_path_externo: Path,
) -> None:
    """Repro literal de QA en subproceso: rama verde en un proceso limpio.

    Lanza ``python -m pytest`` con ``MODEL_PATH`` apuntando al artefacto
    válido y dos tests estándar que **sin aislamiento** fallaban
    (``test_analyze.py:41`` y su hermana de taquicardia). El código de
    salida 0 exige que la fixture de aislamiento exista y funcione
    también fuera del proceso que la define.
    """
    resultado = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/backend/test_analyze.py::test_senal_normal_200",
            "tests/backend/test_analyze.py::test_senal_taquicardia",
            "-q",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, MODEL_PATH_ENV: str(model_path_externo)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    assert "2 passed" in resultado.stdout
