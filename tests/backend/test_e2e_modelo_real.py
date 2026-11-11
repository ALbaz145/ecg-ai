"""Pruebas end-to-end de la API con el modelo real (T14, issue #17).

Camino completo, de punta a punta y **sin dobles del pipeline**::

    EcgDataset sintético → train_from_dataset (el helper del CLI de
    T11c) → save_model → artefacto en tmp_path → MODEL_PATH → loader
    (T11a) → POST /ecg/analyze → resampleo al fs del modelo →
    predicción por latido → agregado AAMI → AnalysisResponse

Señal de **clase conocida**: 12 latidos ``V`` sintéticos encadenados
(mismo patrón que ``test_inference.py``, cuyo artefacto se entrenaba
con ``train_baseline`` **directo**; aquí se usa la ruta
``EcgDataset → train_from_dataset`` del CLI de T11c). Si la API dejara
de usar el modelo real —o el CLI dejara de generar artefactos
compatibles—, estos tests fallan a la vez en la **etiqueta**
(``arritmia_ventricular`` no existe en el heurístico) y en la
**versión** del contrato.

Cubre los AC de #17.1/#17.2 con la pierna de **resampleo** (500 Hz →
360 Hz del modelo) incluida: ``test_inference.py`` solo verificaba
etiquetas conocidas a fs=360 (identidad) y el resample con señales de
etiqueta no predeterminable.

Mismos datos sintéticos y semillas que el resto de la suite (patrón
T10/T11): los ``.joblib`` viven solo en ``tmp_path`` y nunca se
commitean; sin dataset real, sin wfdb y sin red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from app.baseline import (
    MODEL_VERSION,
    save_model,
    train_from_dataset,
)
from app.inference import resample_to_fs
from app.mitbih_dataset import EcgDataset
from app.model_loader import MODEL_PATH_ENV

from tests.backend.test_inference import (
    _assert_contrato,
    _dataset_sintetico,
    _post,
    _senal_tiled,
)

FS_MODELO = 360.0
FS_CONTRATO = 500  # fs distinta a la del modelo → obliga al resampleo

#: Nº de latidos ``V`` de la señal de clase conocida.
N_LATIDOS_V = 12


@pytest.fixture(scope="module")
def artefacto_desde_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Artefacto entrenado con la ruta del CLI (T11c): ``EcgDataset`` → ``train_from_dataset``.

    Los datos son fijos y aprendibles (semilla 7, 10 pacientes x 5
    clases AAMI x 4), igual que en ``test_inference.py``. El modelo se
    serializa una sola vez por módulo en un ``tmp_path`` efímero.
    """
    X, y, paciente_id = _dataset_sintetico()
    dataset = EcgDataset(
        X=X,
        y=y,
        paciente_id=paciente_id,
        records=tuple(sorted(set(paciente_id.tolist()))),
        lead=0,
        fs=FS_MODELO,
        dataset_id="t14-e2e-modelo-real",
    )
    model, metrics = train_from_dataset(dataset)
    # Salud del entrenamiento (AC "pruebas con modelo real"): el split
    # por paciente evalúa latidos held-out y el dataset es aprendible,
    # así que la exactitud debe estar claramente por encima del azar
    # (5 clases → 0,20). Determinista: semilla y datos fijos.
    assert metrics.n_muestras > 0
    assert metrics.accuracy > 0.5
    assert len(metrics.matriz_confusion) == 5  # matriz 5x5 AAMI
    return save_model(model, tmp_path_factory.mktemp("e2e") / "baseline.joblib")


@pytest.fixture
def client_e2e(client, artefacto_desde_dataset: Path, monkeypatch: pytest.MonkeyPatch):
    """Cliente con ``MODEL_PATH`` apuntando al artefacto E2E de ``tmp_path``."""
    monkeypatch.setenv(MODEL_PATH_ENV, str(artefacto_desde_dataset))
    return client


def test_e2e_dataset_a_respuesta_version_y_etiqueta_reales(client_e2e) -> None:
    """[AC #17.1 + #17.2] request → respuesta con versión real y etiqueta conocida."""
    signal = _senal_tiled("V")  # 12 latidos V conocidos, fs = 360 Hz
    response = _post(client_e2e, signal, sample_rate_hz=360)

    assert response.status_code == 200
    body = response.json()
    _assert_contrato(body, len(signal))
    # Versión real del artefacto (metadatos), nunca la heurística.
    assert body["model_version"] == MODEL_VERSION == "baseline-sklearn-0.2.0"
    # Etiqueta coherente con la señal: anómala → con zonas.
    assert body["anomalia"] == "arritmia_ventricular"
    assert len(body["zonas_anomalas"]) == N_LATIDOS_V
    assert all(z["tipo"] == "latido_ventricular" for z in body["zonas_anomalas"])
    assert 0.0 < body["confianza"] <= 1.0


def test_e2e_con_resample_500hz_etiqueta_coherente(client_e2e) -> None:
    """[AC #17.2] La misma morfología a 500 Hz pasa por el resampleo (500 → 360)."""
    senal_360 = _senal_tiled("V")
    # Misma señal muestreada a 500 Hz (resampleo de fase cero, como la
    # que produciría un equipo a esa fs): la API debe resamplear al fs
    # del modelo, predecir y devolver una etiqueta coherente.
    senal_500 = resample_to_fs(np.asarray(senal_360), FS_MODELO, float(FS_CONTRATO))
    response = _post(client_e2e, senal_500.tolist(), sample_rate_hz=FS_CONTRATO)

    assert response.status_code == 200
    body = response.json()
    _assert_contrato(body, len(senal_500))
    assert body["model_version"] == MODEL_VERSION == "baseline-sklearn-0.2.0"
    assert body["anomalia"] == "arritmia_ventricular"
    # Zonas válidas en índices de la señal ORIGINAL de 500 Hz (ya
    # chequeados por _assert_contrato) y de tipo ventricular.
    assert body["zonas_anomalas"]
    assert all(z["tipo"] == "latido_ventricular" for z in body["zonas_anomalas"])
    assert len(body["zonas_anomalas"]) == N_LATIDOS_V
