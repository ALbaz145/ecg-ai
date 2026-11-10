"""Pruebas de la inferencia real en el pipeline (T11b, issue #44).

Artefacto **entrenado** con datos sintéticos aprendibles (mismo
criterio que ``test_baseline.py``/``test_model_loader.py``): sin
dataset real, sin wfdb, sin red; los ``.joblib`` se escriben solo en
``tmp_path`` (nunca se commitean). Además hay un doble de pipeline con
predicciones/probas **controladas** para verificar de forma exacta la
semántica de agregado, confianza y zonas.

Cobertura frente a los criterios de aceptación de #44:

- con artefacto: ``model_version = baseline-sklearn-0.1.0`` y
  ``anomalia`` del vocabulario AAMI en español; contrato
  ``AnalysisResponse`` intacto (mismas claves);
- ``zonas_anomalas``: ``0 <= inicio < fin <= n`` en la **señal
  original** y tipos del nuevo vocabulario;
- ``confianza ∈ [0,1]`` derivada de ``predict_proba`` (cambia según
  las probabilidades, no es una constante);
- señal a fs 360/500/90 coherente + ``resample_to_fs`` y mapeo de
  índices original ↔ resampleado;
- sin artefacto (o inferencia imposible): rama heurística intacta
  (fallback verificado con sus etiquetas y ``heuristic-mvp-0.1.0``);
- agregado: umbral 0,20, desempate por orden AAMI, ``< 2`` latidos →
  ``senal_no_valorable`` y ``MAX_ZONES``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import sklearn
from app import pipeline as ecg_pipeline
from app.baseline import (
    FEATURE_NAMES,
    BaselineModel,
    BeatFeatureExtractor,
    save_model,
    train_baseline,
)
from app.baseline import (
    MODEL_VERSION as ARTIFACT_VERSION,
)
from app.inference import (
    NON_N_FRACTION_THRESHOLD,
    aggregate_beats,
    map_index_to_original,
    map_window_to_original,
    resample_to_fs,
)
from app.mitbih_dataset import AAMI_CLASSES
from app.model import (
    LABEL_NO_VALORABLE,
    MAX_ZONES,
)
from app.model import (
    MODEL_VERSION as HEURISTIC_VERSION,
)
from app.model_loader import MODEL_PATH_ENV, reset_model_cache

from tests.backend.synthetic import synthetic_ecg

FS = 360.0
WIDTH = 288  # 2 * round(BEAT_WINDOW_S * fs), ancho real de T09c
CENTER = WIDTH // 2

_T = np.arange(WIDTH, dtype=np.float64)

#: Vocabulario AAMI en español que puede aparecer en ``anomalia``.
VOCAB_ANOMALIA = {
    "normal",
    "arritmia_supraventricular",
    "arritmia_ventricular",
    "latido_fusion",
    "clase_desconocida",
    LABEL_NO_VALORABLE,
}

#: Vocabulario de tipos de zona del modelo real.
VOCAB_ZONAS = {
    "latido_supraventricular",
    "latido_ventricular",
    "latido_fusion",
    "latido_desconocido",
}

CONTRACT_KEYS = {
    "id",
    "ecg_record_id",
    "anomalia",
    "confianza",
    "zonas_anomalas",
    "model_version",
}


@pytest.fixture(autouse=True)
def _cache_limpio() -> Iterator[None]:
    """Aísla cada test: empieza y termina sin caché de modelo.

    Evita que un artefacto cargado por estos tests contamine los del
    heurístico (y viceversa).
    """
    reset_model_cache()
    yield
    reset_model_cache()


# ---------------------------------------------------------------------------
# Datos sintéticos y artefacto mínimo entrenado
# ---------------------------------------------------------------------------


def _bump(center: float, sigma: float, amp: float) -> np.ndarray:
    """Onda gaussiana de ancho ``sigma`` y amplitud ``amp``."""
    return amp * np.exp(-0.5 * ((_T - center) / sigma) ** 2)


def _ventana_sintetica(clase: str, rng: np.random.Generator) -> np.ndarray:
    """Ventana sintética distinguible por clase (mismo patrón que T10/T11a)."""
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
    n_pacientes: int = 10, repeticiones: int = 4, semilla: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, paciente_id)`` aprendible: 10 pacientes x 5 clases x 4."""
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
    model, _metrics = train_baseline(
        X, y, paciente_id, fs=FS, dataset_id="t11b-inferencia"
    )
    return model


@pytest.fixture
def artefacto(modelo_entrenado: BaselineModel, tmp_path: Path) -> Path:
    """Artefacto serializado en ``tmp_path`` (nunca se commitea)."""
    return save_model(modelo_entrenado, tmp_path / "baseline.joblib")


@pytest.fixture
def client_con_modelo(client, artefacto: Path, monkeypatch: pytest.MonkeyPatch):
    """Cliente con ``MODEL_PATH`` apuntando al artefacto de ``tmp_path``."""
    monkeypatch.setenv(MODEL_PATH_ENV, str(artefacto))
    return client


def _senal_tiled(clase: str, n_latidos: int = 12, semilla: int = 5) -> list[float]:
    """Señal de ``n_latidos`` ventanas sintéticas encadenadas (fs=360).

    Cada ventana mide 288 muestras (±0,4 s @360 Hz) con el pico R en
    el centro, de modo que el pipeline detecta **exactamente** un pico
    por ventana y las ventanas que extrae coinciden con las de
    entrenamiento (verificado: 12 latidos → 12 ventanas evaluables).
    """
    rng = np.random.default_rng(semilla)
    return np.concatenate(
        [_ventana_sintetica(clase, rng) for _ in range(n_latidos)]
    ).tolist()


def _post(client, signal, sample_rate_hz: int = 500, units: str = "mV"):
    """``POST /ecg/analyze`` con el payload mínimo."""
    return client.post(
        "/ecg/analyze",
        json={
            "signal": signal,
            "sample_rate_hz": sample_rate_hz,
            "units": units,
        },
    )


def _assert_contrato(body: dict, n_samples: int) -> None:
    """Contrato ``AnalysisResponse`` intacto + zonas válidas."""
    assert set(body.keys()) == CONTRACT_KEYS
    assert 0.0 <= body["confianza"] <= 1.0
    assert body["anomalia"] in VOCAB_ANOMALIA
    for zona in body["zonas_anomalas"]:
        assert set(zona.keys()) == {"inicio", "fin", "tipo"}
        assert isinstance(zona["inicio"], int)
        assert isinstance(zona["fin"], int)
        assert 0 <= zona["inicio"] < zona["fin"] <= n_samples
        assert zona["tipo"] in VOCAB_ZONAS


# ---------------------------------------------------------------------------
# Doble de pipeline con predicciones/probas controladas
# ---------------------------------------------------------------------------


class _PipelineControlado:
    """Doble del ``Pipeline`` sklearn: predicciones y probabilidades fijas.

    ``predict``/``predict_proba`` ciclan sobre sus filas por posición
    (misma fila para la misma posición ``i % len``), de modo que la
    confianza y la etiqueta de la respuesta son **exactamente** las que
    el test declara, sin depender del entrenamiento.
    """

    def __init__(
        self,
        clases: tuple[str, ...],
        patron: list[str],
        probas: list[list[float]],
    ) -> None:
        if len(patron) != len(probas):
            raise ValueError("patron y probas deben tener la misma longitud")
        self.classes_ = np.asarray(clases)
        self._patron = np.asarray(patron)
        self._probas = np.asarray(probas, dtype=np.float64)

    def _filas(self, n: int) -> np.ndarray:
        return np.arange(n) % len(self._patron)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._patron[self._filas(len(X))]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probas[self._filas(len(X))]


def _modelo_controlado(
    monkeypatch: pytest.MonkeyPatch,
    patron: list[str],
    probas: list[list[float]],
    clases: tuple[str, ...] = ("N", "V"),
    meta_extra: dict | None = None,
) -> BaselineModel:
    """Modelo controlado inyectado en el pipeline (sin tocar disco).

    Parchea ``app.pipeline.get_real_model`` (el loader de T11a se
    prueba aparte en ``test_model_loader.py``).
    """
    meta: dict = {
        "model_version": ARTIFACT_VERSION,
        "dataset_id": "t11b-controlado",
        "sklearn_version": sklearn.__version__,
        "aami_classes": list(AAMI_CLASSES),
        "random_state": 42,
        "feature_names": list(FEATURE_NAMES),
        "fs": FS,
    }
    meta.update(meta_extra or {})
    model = BaselineModel(
        pipeline=_PipelineControlado(clases, patron, probas),
        extractor=BeatFeatureExtractor(FS),
        meta=meta,
    )
    monkeypatch.setattr(ecg_pipeline, "get_real_model", lambda: model)
    return model


# ---------------------------------------------------------------------------
# resample_to_fs (scipy) y mapeo de índices
# ---------------------------------------------------------------------------


def test_resample_identidad_si_misma_fs() -> None:
    arr = np.arange(100.0)
    out = resample_to_fs(arr, 360.0, 360.0)
    np.testing.assert_array_equal(out, arr)
    assert out is not arr  # copia: el caller puede mutar sin sorpresas


@pytest.mark.parametrize(
    ("fs_orig", "n", "esperado"),
    [
        (500, 5000, 3600),  # 10 s a 500 Hz → 10 s a 360 Hz
        (90, 900, 3600),  # subida x4 (polifásico)
        (360, 3600, 3600),  # sin remuestreo real (copia)
        (500, 3750, 2700),  # 7,5 s a 500 Hz → 7,5 s a 360 Hz
    ],
)
def test_resample_longitud_preservada(fs_orig: int, n: int, esperado: int) -> None:
    arr = np.ones(n, dtype=np.float64)
    out = resample_to_fs(arr, float(fs_orig), FS)
    assert out.size == esperado


def test_resample_preserva_amplitud_del_seno() -> None:
    """Pasa-banda (5 Hz) con amplitud ~1 tras 500 → 360 Hz."""
    fs_o = 500.0
    t = np.arange(int(10 * fs_o)) / fs_o
    seno = np.sin(2 * np.pi * 5.0 * t)
    out = resample_to_fs(seno, fs_o, FS)
    assert out.size == 3600
    interior = out[50:-50]  # evita el transitorio de borde del FIR
    assert float(np.max(np.abs(interior))) == pytest.approx(1.0, abs=0.05)


def test_resample_fs_no_entera_usa_respaldo_fft() -> None:
    arr = np.ones(1000, dtype=np.float64)
    out = resample_to_fs(arr, 500.0, 360.5)  # fs_model no entera → resample()
    assert out.size == round(1000 * 360.5 / 500.0)


def test_resample_tope_trunca_por_prefijo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con el tope ajustado, la entrada se recorta por prefijo."""
    monkeypatch.setattr("app.inference.MAX_RESAMPLED_SAMPLES", 10_000)
    arr = np.arange(1000.0)
    out = resample_to_fs(arr, 1.0, 360.0)  # sin tope serían 360 000 muestras
    assert out.size <= 10_000
    n_max = 10_000 // 360  # floor(MAX * fs_orig / fs_target)
    np.testing.assert_allclose(out, resample_to_fs(arr[:n_max], 1.0, 360.0))


@pytest.mark.parametrize(
    ("fs_orig", "fs_target"),
    [(0.0, 360.0), (500.0, -1.0), (np.nan, 360.0), (500.0, np.inf)],
)
def test_resample_fs_invalida_error(fs_orig: float, fs_target: float) -> None:
    with pytest.raises(ValueError, match="inválid"):
        resample_to_fs(np.ones(10), fs_orig, fs_target)


def test_map_window_identidad_misma_fs() -> None:
    assert map_window_to_original(144, 432, FS, FS, 5000) == (144, 432)


def test_map_window_500_a_360() -> None:
    # 288 muestras @360 Hz = 0,8 s = 400 muestras @500 Hz.
    assert map_window_to_original(0, 288, 500.0, FS, 5000) == (0, 400)
    assert map_window_to_original(144, 432, 500.0, FS, 5000) == (200, 600)


@pytest.mark.parametrize(
    ("fs_orig", "n_original"),
    [(1.0, 10), (90.0, 50), (5000.0, 100)],
)
def test_map_window_acota_en_bordes(fs_orig: float, n_original: int) -> None:
    """La ventana mapeada siempre cumple 0 <= inicio < fin <= n."""
    # Ventana al inicio, en medio y fuera por la derecha.
    for start, end in [(0, 288), (144, 432), (10_000, 10_288)]:
        ini, fin = map_window_to_original(start, end, fs_orig, FS, n_original)
        assert 0 <= ini < fin <= n_original


def test_map_index_roundtrip_aprox() -> None:
    """original → resampleado → original se recupera con error ≤ 1."""
    for idx in (0, 1, 777, 4999):
        resampled = round(idx * FS / 500.0)
        back = map_index_to_original(resampled, 500.0, FS, 5000)
        assert abs(back - idx) <= 1


# ---------------------------------------------------------------------------
# aggregate_beats: regla de agregado documentada
# ---------------------------------------------------------------------------


def test_agregado_menos_de_dos_latidos() -> None:
    uno = aggregate_beats(np.array(["N"]), np.array([0.9]), [(0, 10)])
    assert uno.anomalia == LABEL_NO_VALORABLE
    assert uno.confianza == 0.0
    assert uno.zonas == []

    cero = aggregate_beats(np.array([], dtype=str), np.array([]), [])
    assert cero.anomalia == LABEL_NO_VALORABLE
    assert cero.confianza == 0.0
    assert cero.zonas == []


def test_agregado_por_debajo_del_umbral_es_normal_sin_zonas() -> None:
    preds = np.array(["N"] * 9 + ["V"])  # 1/10 = 10 % < 20 %
    conf = np.array([0.9] * 9 + [0.4])
    zonas = [(i * 10, i * 10 + 5) for i in range(10)]
    det = aggregate_beats(preds, conf, zonas)
    assert det.anomalia == "normal"
    assert det.confianza == pytest.approx(float(conf.mean()))  # todos los latidos
    assert det.zonas == []  # los no-N bajo el umbral no generan zonas


def test_agregado_en_el_umbral_usa_dominante() -> None:
    preds = np.array(["N"] * 8 + ["V", "V"])  # 2/10 = 20 % ≥ umbral
    conf = np.array([0.9] * 8 + [0.5, 0.7])
    zonas = [(i * 10, i * 10 + 5) for i in range(10)]
    det = aggregate_beats(preds, conf, zonas)
    assert NON_N_FRACTION_THRESHOLD == 0.20
    assert det.anomalia == "arritmia_ventricular"
    assert det.confianza == pytest.approx(0.6)  # media de los no-N
    assert len(det.zonas) == 2
    assert all(z["tipo"] == "latido_ventricular" for z in det.zonas)


def test_agregado_desempate_por_orden_canonico_aami() -> None:
    preds = np.array(["S", "V"] * 5)  # empate 5-5 → gana S (orden N<S<V<F<Q)
    conf = np.full(10, 0.6)
    zonas = [(i * 10, i * 10 + 5) for i in range(10)]
    det = aggregate_beats(preds, conf, zonas)
    assert det.anomalia == "arritmia_supraventricular"
    assert det.confianza == pytest.approx(0.6)
    tipos = [z["tipo"] for z in det.zonas]
    assert tipos == ["latido_supraventricular", "latido_ventricular"] * 5


def test_agregado_una_zona_por_latido_no_n() -> None:
    preds = np.array(["N", "S", "V", "F", "Q"])
    conf = np.full(5, 0.9)
    zonas = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    det = aggregate_beats(preds, conf, zonas)
    # 4 no-N de 5 (80 %) → dominante por empate 1-1-1-1 → S (orden AAMI).
    assert det.anomalia == "arritmia_supraventricular"
    assert [(z["inicio"], z["fin"]) for z in det.zonas] == zonas[1:]
    assert [z["tipo"] for z in det.zonas] == [
        "latido_supraventricular",
        "latido_ventricular",
        "latido_fusion",
        "latido_desconocido",
    ]


def test_agregado_respeta_max_zones() -> None:
    n = MAX_ZONES + 100
    preds = np.array(["V"] * n)
    conf = np.full(n, 0.8)
    zonas = [(i, i + 5) for i in range(n)]
    det = aggregate_beats(preds, conf, zonas)
    assert len(det.zonas) == MAX_ZONES


def test_agregado_clase_fuera_de_vocabulario_es_q() -> None:
    det = aggregate_beats(
        np.array(["N", "X"]), np.array([0.9, 0.6]), [(0, 10), (10, 20)]
    )
    assert det.anomalia == "clase_desconocida"
    assert det.zonas[0]["tipo"] == "latido_desconocido"


def test_agregado_confianza_sigue_a_las_probabilidades() -> None:
    preds = np.array(["V", "V"])
    zonas = [(0, 10), (10, 20)]
    alta = aggregate_beats(preds, np.array([0.95, 0.9]), zonas)
    baja = aggregate_beats(preds, np.array([0.55, 0.5]), zonas)
    assert alta.confianza == pytest.approx(0.925)
    assert baja.confianza == pytest.approx(0.525)
    assert alta.confianza != baja.confianza


def test_agregado_entradas_desalineadas_error() -> None:
    with pytest.raises(ValueError, match="desalineadas"):
        aggregate_beats(np.array(["N", "N"]), np.array([0.9]), [(0, 1)])


# ---------------------------------------------------------------------------
# Integración con artefacto real (MODEL_PATH → loader → pipeline)
# ---------------------------------------------------------------------------


def test_artefacto_latidos_n_tiled_normal(client_con_modelo) -> None:
    signal = _senal_tiled("N")
    response = _post(client_con_modelo, signal, sample_rate_hz=360)
    assert response.status_code == 200
    body = response.json()
    _assert_contrato(body, len(signal))
    assert body["anomalia"] == "normal"
    assert body["model_version"] == ARTIFACT_VERSION == "baseline-sklearn-0.1.0"
    assert body["zonas_anomalas"] == []
    assert 0.0 < body["confianza"] <= 1.0


def test_artefacto_latidos_v_tiled_arrtimia_ventricular(client_con_modelo) -> None:
    signal = _senal_tiled("V")
    response = _post(client_con_modelo, signal, sample_rate_hz=360)
    assert response.status_code == 200
    body = response.json()
    _assert_contrato(body, len(signal))
    assert body["anomalia"] == "arritmia_ventricular"
    assert body["model_version"] == ARTIFACT_VERSION
    # Una zona por latido no-N (12 de 12), índices de la señal ORIGINAL
    # (fs=360 = fs del modelo → identidad: ventanas de 288 muestras).
    assert len(body["zonas_anomalas"]) == 12
    assert all(z["tipo"] == "latido_ventricular" for z in body["zonas_anomalas"])
    assert all(z["fin"] - z["inicio"] == WIDTH for z in body["zonas_anomalas"])
    inicios = [z["inicio"] for z in body["zonas_anomalas"]]
    assert inicios == sorted(inicios)


def test_artefacto_senal_plana_no_valorable(client_con_modelo) -> None:
    response = _post(client_con_modelo, [0.0] * 300, sample_rate_hz=500)
    assert response.status_code == 200
    body = response.json()
    _assert_contrato(body, 300)
    assert body["anomalia"] == LABEL_NO_VALORABLE
    assert body["confianza"] == 0.0  # sin latidos: sin evidencia del modelo
    assert body["zonas_anomalas"] == []
    assert body["model_version"] == ARTIFACT_VERSION


def test_artefacto_un_solo_latido_no_valorable(client_con_modelo) -> None:
    signal = _senal_tiled("N", n_latidos=1)  # 1 ventana evaluable (< 2)
    response = _post(client_con_modelo, signal, sample_rate_hz=360)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == LABEL_NO_VALORABLE
    assert body["confianza"] == 0.0
    assert body["model_version"] == ARTIFACT_VERSION


def test_artefacto_unidades_uv_equivalentes(client_con_modelo) -> None:
    signal_mv = _senal_tiled("V")
    signal_uv = [value * 1000.0 for value in signal_mv]
    resp_mv = _post(client_con_modelo, signal_mv, sample_rate_hz=360, units="mV")
    resp_uv = _post(client_con_modelo, signal_uv, sample_rate_hz=360, units="uV")
    assert resp_mv.status_code == resp_uv.status_code == 200
    body_mv, body_uv = resp_mv.json(), resp_uv.json()
    assert body_mv["anomalia"] == body_uv["anomalia"] == "arritmia_ventricular"
    assert body_mv["zonas_anomalas"] == body_uv["zonas_anomalas"]


@pytest.mark.parametrize("fs", [360, 500, 90])
def test_fs_validas_con_modelo(client_con_modelo, fs: int) -> None:
    """Señal a cualquier fs válida → resultado coherente (AC de #44)."""
    signal = synthetic_ecg(sample_rate_hz=fs, duration_s=10.0, bpm=70)
    response = _post(client_con_modelo, signal, sample_rate_hz=fs)
    assert response.status_code == 200
    body = response.json()
    _assert_contrato(body, len(signal))
    assert body["model_version"] == ARTIFACT_VERSION
    # La morfología PQRST sostiene latidos no-N: hay zonas que validar
    # (tipos del nuevo vocabulario, ya chequeados en _assert_contrato).
    assert body["zonas_anomalas"]


def test_mismo_resultado_en_fs_360_500_90(client_con_modelo) -> None:
    """La misma morfología a 360/500/90 Hz produce la misma etiqueta."""
    etiquetas = set()
    for fs in (360, 500, 90):
        signal = synthetic_ecg(sample_rate_hz=fs, duration_s=10.0, bpm=70)
        response = _post(client_con_modelo, signal, sample_rate_hz=fs)
        assert response.status_code == 200
        etiquetas.add(response.json()["anomalia"])
    assert len(etiquetas) == 1


# ---------------------------------------------------------------------------
# Semántica exacta con pipeline controlado (predicciones/probas fijas)
# ---------------------------------------------------------------------------


def test_confianza_derivada_de_predict_proba(client, monkeypatch) -> None:
    """Mismo patrón, distintas probabilidades → distinta confianza."""
    signal = _senal_tiled("N")  # 12 latidos evaluables
    _modelo_controlado(
        monkeypatch,
        patron=["N", "V"],  # alterna → 6 V de 12 (50 % ≥ umbral)
        probas=[[0.9, 0.1], [0.1, 0.9]],  # V con confianza 0.9
    )
    alta = _post(client, signal, sample_rate_hz=360)
    assert alta.status_code == 200
    body_alta = alta.json()

    _modelo_controlado(
        monkeypatch,
        patron=["N", "V"],
        probas=[[0.9, 0.1], [0.55, 0.45]],  # V con confianza 0.55
    )
    baja = _post(client, signal, sample_rate_hz=360)
    assert baja.status_code == 200
    body_baja = baja.json()

    # Misma etiqueta (mismas predicciones), confianza distinta y exacta.
    assert body_alta["anomalia"] == body_baja["anomalia"] == "arritmia_ventricular"
    assert body_alta["confianza"] == pytest.approx(0.9)
    assert body_baja["confianza"] == pytest.approx(0.55)
    assert body_alta["confianza"] > body_baja["confianza"]
    assert body_alta["model_version"] == ARTIFACT_VERSION
    # Una zona por latido no-N en índices de la señal original.
    assert len(body_alta["zonas_anomalas"]) == 6
    assert all(z["tipo"] == "latido_ventricular" for z in body_alta["zonas_anomalas"])
    n = len(signal)
    for zona in body_alta["zonas_anomalas"]:
        assert 0 <= zona["inicio"] < zona["fin"] <= n


def test_umbral_de_agregado_con_modelo_controlado(client, monkeypatch) -> None:
    """1 V de 12 (8,3 % < 20 %) → ``normal`` sin zonas, confianza media."""
    _modelo_controlado(
        monkeypatch,
        patron=["N"] * 11 + ["V"],
        probas=[[0.9, 0.1]] * 11 + [[0.45, 0.55]],
    )
    signal = _senal_tiled("N")
    response = _post(client, signal, sample_rate_hz=360)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "normal"
    # Media de max(predict_proba) de los 12 latidos, redondeada a 3 decimales.
    assert body["confianza"] == pytest.approx(round((11 * 0.9 + 0.55) / 12, 3))
    assert body["zonas_anomalas"] == []
    assert body["model_version"] == ARTIFACT_VERSION


def test_version_de_la_respuesta_sale_de_los_metadatos_del_artefacto(
    client, monkeypatch
) -> None:
    """``model_version`` = la de los **metadatos** del artefacto (AC #17.1).

    Además de ``== baseline-sklearn-0.1.0`` con artefactos reales, este
    test prueba que la versión **no está hardcodeada** en el pipeline:
    con un artefacto cuyos metadatos declaran otra versión, la
    respuesta refleja exactamente ``model.meta["model_version"]``. Si
    el pipeline dejara de usar el modelo real (o escribiera una
    constante en vez de leer los metadatos), fallaría aquí.
    """
    version_metadata = "baseline-sklearn-0.2.0"
    _modelo_controlado(
        monkeypatch,
        patron=["N"],
        probas=[[0.9, 0.1]],
        meta_extra={"model_version": version_metadata},
    )
    response = _post(client, _senal_tiled("N"), sample_rate_hz=360)
    assert response.status_code == 200
    assert response.json()["model_version"] == version_metadata


# ---------------------------------------------------------------------------
# Fallback: sin artefacto (o inferencia imposible) → heurístico intacto
# ---------------------------------------------------------------------------


def test_sin_artefacto_rama_heuristica_intacta(
    client,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``MODEL_PATH`` inexistente → **200 heurístico + warning** (AC #17.3).

    El diseño acordado es *fallback, no error*: el artefacto ausente no
    rompe el request. Aquí se cubren los dos aspectos del AC en la
    llamada de **la API** (no solo del loader): (1) ``POST
    /ecg/analyze`` responde ``200`` con la etiqueta y la versión
    heurísticas; (2) el loader emite su ``logger.warning`` con la ruta
    intentada durante el request (caplog), haciendo visible el fallback.
    """
    ausente = tmp_path / "no-existe.joblib"
    monkeypatch.setenv(MODEL_PATH_ENV, str(ausente))
    reset_model_cache()
    signal = synthetic_ecg(bpm=120)
    with caplog.at_level(logging.WARNING, logger="ecg_ai.backend"):
        response = _post(client, signal)
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "taquicardia"
    assert body["model_version"] == HEURISTIC_VERSION == "heuristic-mvp-0.1.0"
    avisos = [
        r.getMessage()
        for r in caplog.records
        if r.name == "ecg_ai.backend" and r.levelno == logging.WARNING
    ]
    assert any(
        "Modelo real no disponible" in mensaje and str(ausente) in mensaje
        for mensaje in avisos
    ), avisos


def test_inferencia_imposible_cae_al_heuristico(
    client, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Modelo con ``fs`` inválido en metadatos → warning + heurístico."""
    _modelo_controlado(
        monkeypatch,
        patron=["N"],
        probas=[[0.9, 0.1]],
        meta_extra={"fs": 0.0},  # infer_detection lanza ValueError
    )
    with caplog.at_level(logging.WARNING, logger="ecg_ai.backend"):
        response = _post(client, synthetic_ecg(bpm=120))
    assert response.status_code == 200
    body = response.json()
    assert body["anomalia"] == "taquicardia"  # la heurística salvó el request
    assert body["model_version"] == HEURISTIC_VERSION
    avisos = [
        r.getMessage()
        for r in caplog.records
        if r.name == "ecg_ai.backend" and r.levelno == logging.WARNING
    ]
    assert any("Inferencia real falló" in mensaje for mensaje in avisos)
