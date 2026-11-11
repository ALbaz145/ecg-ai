"""Pruebas del baseline scikit-learn y del split por paciente (T10, #13).

Datos **sintéticos aprendibles** (ondas gaussianas distinguibles por
clase): sin dataset real, sin wfdb, sin red y sin descargar MIT-BIH.
Los ``.joblib`` se escriben solo en ``tmp_path`` (nunca se commitean).

Cobertura frente a los criterios de aceptación de #13:

- split por paciente sin fuga (disjoint en pacientes **y** en índices,
  determinista, ``test_size`` respetado, errores claros) y **por sujeto
  real** (#61: los registros 201/202 de MIT-BIH, mismo sujeto, nunca
  en train y test a la vez);
- features reutilizando ``bandpass_filter``/``detect_r_peaks`` y las
  constantes de ``app.signal`` (RR/FC, amplitud, morphology);
- métricas por clase: sensibilidad, especificidad, soporte y matriz de
  confusión 5x5, con valores verificables a mano y ``zero_division``;
- serialización joblib con ``model_version`` + ``dataset_id``
  (round-trip idéntico y metadatos inválidos rechazados);
- formas del pipeline: fit/predict con distintos tamaños y 1 latido.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from app import baseline, signal
from app.baseline import (
    FEATURE_NAMES,
    MODEL_VERSION,
    BaselineModel,
    BeatFeatureExtractor,
    evaluate,
    load_model,
    metrics_from_predictions,
    save_model,
    split_por_paciente,
    train_baseline,
    train_from_dataset,
)
from app.mitbih import RECORDS
from app.mitbih_dataset import AAMI_CLASSES, EcgDataset

FS = 360.0
WIDTH = 288  # 2 * round(BEAT_WINDOW_S * fs), ancho real de EcgDataset
CENTER = WIDTH // 2  # el pico R queda en el centro de la ventana
REFRACTORY = max(1, round(0.3 * FS))  # espejo de signal.REFRACTORY_S en muestras
DATASET_ID = "0123456789abcdef"  # estilo sha256 recortado de T09c

_T = np.arange(WIDTH, dtype=np.float64)


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------


def _bump(center: float, sigma: float, amp: float) -> np.ndarray:
    """Onda gaussiana de ancho ``sigma`` y amplitud ``amp``."""
    return amp * np.exp(-0.5 * ((_T - center) / sigma) ** 2)


def _ventana_sintetica(clase: str, rng: np.random.Generator) -> np.ndarray:
    """Ventana sintética distinguible por clase (mV, ruido pequeño).

    Cada clase tiene una firma distinta en los features del baseline:
    ``N`` pico estrecho amp 1,0; ``S`` estrecho y bajo (0,55); ``V``
    ancho y alto (1,8, con deflexión negativa); ``F`` asimétrico (dos
    bumpes); ``Q`` deriva de baja amplitud sin QRS.
    """
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
    """``(X, y, paciente_id)`` sintético: cada paciente tiene las 5 clases.

    Un tercio de los latidos lleva un vecino R izquierdo y un cuarto
    uno derecho (distancias ≥ período refractario) para que los
    features de ritmo (``rr_*``/``fc_bpm``) tengan valores reales y no
    solo ``NaN`` imputados.
    """
    rng = np.random.default_rng(semilla)
    filas: list[np.ndarray] = []
    etiquetas: list[str] = []
    pacientes: list[str] = []
    for p in range(n_pacientes):
        paciente = f"{100 + p}"
        ganancia = float(rng.uniform(0.9, 1.1))
        for k, clase in enumerate(AAMI_CLASSES * repeticiones):
            win = _ventana_sintetica(clase, rng) * ganancia
            if k % 3 == 0:  # vecino izquierdo (RR visible)
                win = win + _bump(CENTER - (110 + (k * 7) % 31), 6.0, 0.8 * ganancia)
            if k % 4 == 0:  # vecino derecho (RR visible)
                win = win + _bump(CENTER + (112 + (k * 11) % 28), 6.0, 0.8 * ganancia)
            filas.append(win.astype(np.float32))
            etiquetas.append(clase)
            pacientes.append(paciente)
    return np.vstack(filas), np.asarray(etiquetas), np.asarray(pacientes)


def _feat(x: np.ndarray) -> dict[str, float]:
    """Features de una ventana como dict nombre → valor."""
    fila = BeatFeatureExtractor(FS).transform(np.asarray(x, dtype=np.float64)[None, :])[0]
    return dict(zip(FEATURE_NAMES, fila, strict=True))


@pytest.fixture(scope="module")
def entrenado() -> tuple[BaselineModel, baseline.Metrics]:
    """Modelo y métricas entrenados una sola vez (datos fijos y rápidos)."""
    X, y, paciente_id = _dataset_sintetico()
    return train_baseline(X, y, paciente_id, fs=FS, dataset_id=DATASET_ID)


# ---------------------------------------------------------------------------
# Split por paciente (anti-fuga)
# ---------------------------------------------------------------------------


def test_split_disjoint_y_cubre_todos_los_indices() -> None:
    """Los índices de train/test son disjuntos y su unión es todo."""
    _, _, paciente_id = _dataset_sintetico()
    train_idx, test_idx = split_por_paciente(paciente_id)

    assert set(train_idx.tolist()).isdisjoint(test_idx.tolist())
    union = np.sort(np.concatenate([train_idx, test_idx]))
    np.testing.assert_array_equal(union, np.arange(paciente_id.size))
    assert train_idx.dtype.kind in "iu" and test_idx.dtype.kind in "iu"
    assert np.all(np.diff(train_idx) > 0) and np.all(np.diff(test_idx) > 0)


def test_split_sin_fuga_de_pacientes() -> None:
    """Ningún paciente_id de test aparece en train ni al revés (#13)."""
    _, _, paciente_id = _dataset_sintetico()
    train_idx, test_idx = split_por_paciente(paciente_id)

    pacientes_train = set(paciente_id[train_idx].tolist())
    pacientes_test = set(paciente_id[test_idx].tolist())
    assert pacientes_train, "train sin pacientes"
    assert pacientes_test, "test sin pacientes"
    assert pacientes_train.isdisjoint(pacientes_test)  # sin fuga
    assert pacientes_train | pacientes_test == set(paciente_id.tolist())


def test_split_determinista_con_mismo_random_state() -> None:
    """Misma semilla ⇒ misma partición; semilla distinta ⇒ distinta."""
    _, _, paciente_id = _dataset_sintetico(n_pacientes=20)

    a = split_por_paciente(paciente_id, random_state=42)
    b = split_por_paciente(paciente_id, random_state=42)
    c = split_por_paciente(paciente_id, random_state=7)

    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1]), "semillas distintas dio el mismo test"


@pytest.mark.parametrize(
    ("n_pacientes", "test_size", "esperados_en_test"),
    [(10, 0.2, 2), (8, 0.25, 2), (4, 0.5, 2), (7, 0.3, 3)],
)
def test_split_respetar_test_size(
    n_pacientes: int, test_size: float, esperados_en_test: int
) -> None:
    """El nº de pacientes de test es ceil(test_size * n_pacientes)."""
    _, _, paciente_id = _dataset_sintetico(n_pacientes=n_pacientes, repeticiones=1)
    train_idx, test_idx = split_por_paciente(paciente_id, test_size=test_size)

    n_test = len(set(paciente_id[test_idx].tolist()))
    n_train = len(set(paciente_id[train_idx].tolist()))
    assert n_test == esperados_en_test
    assert n_train == n_pacientes - esperados_en_test


def test_split_train_nunca_vacio_con_test_size_extremo() -> None:
    """test_size alto acota el test para no dejar train sin pacientes."""
    _, _, paciente_id = _dataset_sintetico(n_pacientes=2, repeticiones=1)
    train_idx, test_idx = split_por_paciente(paciente_id, test_size=0.9)

    assert train_idx.size > 0 and test_idx.size > 0


def test_split_un_saciente_error_claro() -> None:
    """Con un solo paciente no hay split posible: ValueError explícito."""
    paciente_id = np.array(["100"] * 30)
    with pytest.raises(ValueError, match="al menos 2 pacientes"):
        split_por_paciente(paciente_id)


def test_split_ids_vacios_error() -> None:
    """paciente_id vacío → ValueError (no una partición vacía silenciosa)."""
    with pytest.raises(ValueError, match="vacío"):
        split_por_paciente(np.array([], dtype=str))


def test_split_ids_no_1d_error() -> None:
    """paciente_id 2-D → ValueError con la forma recibida."""
    with pytest.raises(ValueError, match="1-D"):
        split_por_paciente(np.array([["100"], ["101"]]))


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.5, 1.5])
def test_split_test_size_invalido_error(test_size: float) -> None:
    """test_size fuera de (0, 1) → ValueError."""
    with pytest.raises(ValueError, match="test_size"):
        split_por_paciente(np.array(["100", "101"]), test_size=test_size)


# ---------------------------------------------------------------------------
# Split por sujeto real: 201/202 (issue #61)
# ---------------------------------------------------------------------------


def _ids_mitdb(latidos_por_registro: int = 6) -> np.ndarray:
    """Los 48 IDs canónicos de ``mitdb`` (sin duplicar la lista: de
    ``app.mitbih.RECORDS``), cada uno repetido ``latidos_por_registro``
    veces como si fueran filas de ``paciente_id``."""
    return np.array(
        [registro for registro in RECORDS for _ in range(latidos_por_registro)]
    )


def test_multi_record_subjects_solo_el_par_201_202() -> None:
    """El mapeo es exactamente {202 → 201}: único par multi-sujeto (#61).

    PhysioNet: 48 registros de **47 sujetos** ⇒ exactamente un sujeto
    con dos registros, y ese par es 201/202 (Usage Notes de mitdb).
    """
    assert dict(baseline.MULTI_RECORD_SUBJECTS) == {"202": "201"}

    sujetos = baseline._agrupar_sujetos(np.array(RECORDS))
    assert len(np.unique(sujetos)) == 47  # 48 registros → 47 sujetos
    for registro, sujeto in zip(RECORDS, sujetos.tolist(), strict=True):
        esperado = baseline.MULTI_RECORD_SUBJECTS.get(registro, registro)
        assert sujeto == esperado, f"registro {registro} agrupado como {sujeto}"


@pytest.mark.parametrize("random_state", [0, 1, 7, 42, 123, 2024])
def test_split_201_202_nunca_en_train_y_test_a_la_vez(random_state: int) -> None:
    """201 y 202 (mismo sujeto) van siempre al mismo bloque (#61).

    Con la semilla 42, antes de #61 ``202`` iba a train y ``201`` a
    test: este test habría fallado con el split por ID de registro.
    """
    ids = _ids_mitdb()
    train_idx, test_idx = split_por_paciente(ids, random_state=random_state)

    en_train = set(ids[train_idx].tolist())
    en_test = set(ids[test_idx].tolist())
    assert ("201" in en_train) == ("202" in en_train)
    assert ("201" in en_test) == ("202" in en_test)
    # El sujeto está en exactamente uno de los dos bloques (anti-fuga).
    assert ("201" in en_train) ^ ("201" in en_test)
    # El resto de registros, sin agrupar: siguen disjuntos entre bloques.
    assert en_train.isdisjoint(en_test)


def test_split_201_202_determinista_y_anti_fuga_a_nivel_sujeto() -> None:
    """Misma semilla → misma partición; el assert anti-fuga cubre sujetos."""
    ids = _ids_mitdb()
    a = split_por_paciente(ids, random_state=42)
    b = split_por_paciente(ids, random_state=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])

    sujetos = baseline._agrupar_sujetos(ids)
    sujetos_train = set(sujetos[a[0]].tolist())
    sujetos_test = set(sujetos[a[1]].tolist())
    assert sujetos_train and sujetos_test
    assert sujetos_train.isdisjoint(sujetos_test)  # sin fuga por sujeto
    # 47 sujetos: n_test = ceil(0.2 * 47) = 10 (train: 37 sujetos).
    assert len(sujetos_test) == 10
    assert len(sujetos_train) == 37


def test_split_solo_201_202_es_un_sujeto_y_no_se_puede_partir() -> None:
    """201 y 202 agrupados = 1 solo sujeto → ValueError explícito."""
    ids = np.array(["201"] * 5 + ["202"] * 5)
    with pytest.raises(ValueError, match="al menos 2 pacientes"):
        split_por_paciente(ids)


def test_split_ids_sin_agrupar_identidad_y_test_size_igual() -> None:
    """Sin 201/202 la agrupación es identidad: comportamiento de T10 intacto."""
    _, _, paciente_id = _dataset_sintetico(n_pacientes=14, repeticiones=1)

    np.testing.assert_array_equal(baseline._agrupar_sujetos(paciente_id), paciente_id)

    train_idx, test_idx = split_por_paciente(paciente_id)
    assert len(set(paciente_id[test_idx].tolist())) == 3  # ceil(0.2 * 14)
    assert len(set(paciente_id[train_idx].tolist())) == 11


def test_train_baseline_split_interno_agrupa_sujetos(monkeypatch) -> None:
    """El split que usa ``train_baseline`` agrupa 201/202 (sin lógica duplicada).

    Se espía la llamada real a ``split_por_paciente`` desde
    ``train_baseline`` y se comprueba que el sujeto 201/202 cae en un
    único bloque, sea el que sea la semilla.
    """
    X, y, paciente_id = _dataset_sintetico()
    paciente_id = paciente_id.copy()
    paciente_id[paciente_id == "108"] = "201"
    paciente_id[paciente_id == "109"] = "202"

    real = baseline.split_por_paciente
    captura: dict[str, set[str]] = {}

    def espiar(ids: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        train_idx, test_idx = real(ids, **kwargs)
        captura["train"] = set(ids[train_idx].tolist())
        captura["test"] = set(ids[test_idx].tolist())
        return train_idx, test_idx

    monkeypatch.setattr(baseline, "split_por_paciente", espiar)
    _, metrics = train_baseline(X, y, paciente_id, fs=FS, dataset_id=DATASET_ID)

    assert captura["train"].isdisjoint(captura["test"])  # anti-fuga
    assert ("201" in captura["train"]) == ("202" in captura["train"])
    assert ("201" in captura["test"]) == ("202" in captura["test"])
    assert ("201" in captura["train"]) ^ ("201" in captura["test"])
    # 10 registros -> 9 sujetos -> ceil(0.2 * 9) = 2 sujetos de test.
    sujetos_test = {
        baseline.MULTI_RECORD_SUBJECTS.get(registro, registro)
        for registro in captura["test"]
    }
    assert len(sujetos_test) == 2
    assert metrics.n_muestras > 0


# ---------------------------------------------------------------------------
# Features por latido
# ---------------------------------------------------------------------------


def test_features_formas_y_nombres() -> None:
    """``transform`` devuelve (n_latidos, 14) float64 con los nombres."""
    X, _, _ = _dataset_sintetico(n_pacientes=2, repeticiones=1)
    feats = BeatFeatureExtractor(FS).transform(X)

    assert feats.shape == (X.shape[0], len(FEATURE_NAMES))
    assert feats.dtype == np.float64
    assert BeatFeatureExtractor(FS).feature_names == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 14


def test_features_rr_con_vecino_izquierdo() -> None:
    """RR al vecino izquierdo visible: (centro - pico) / fs (signal.py)."""
    x = _bump(CENTER, 6.0, 1.0) + _bump(CENTER - 120, 6.0, 1.5)
    feats = _feat(x)

    assert feats["rr_prev_s"] == pytest.approx(120 / FS, abs=0.01)
    assert np.isnan(feats["rr_next_s"])
    assert feats["fc_bpm"] == pytest.approx(60.0 / feats["rr_prev_s"])
    assert feats["n_picos"] == 2.0


def test_features_rr_con_vecino_derecho() -> None:
    """RR al vecino derecho visible y FC instantánea derivada."""
    x = _bump(CENTER, 6.0, 1.0) + _bump(CENTER + 130, 6.0, 1.5)
    feats = _feat(x)

    assert feats["rr_next_s"] == pytest.approx(130 / FS, abs=0.01)
    assert np.isnan(feats["rr_prev_s"])
    assert feats["fc_bpm"] == pytest.approx(60.0 / feats["rr_next_s"], rel=1e-3)


def test_features_rr_ignora_picos_muy_cercanos_al_centro() -> None:
    """Picos a <REFRACTORY_S del centro son el propio latido, no vecinos."""
    # Pico alto a 60 muestras (< período refractario): detect_r_peaks
    # descarta el del centro, pero la guardas por distancia evita un RR
    # falso de 0,17 s (FC 360 lpm, imposible).
    x = _bump(CENTER - 60, 6.0, 1.5)
    feats = _feat(x)

    assert np.isnan(feats["rr_prev_s"]) and np.isnan(feats["rr_next_s"])
    assert np.isnan(feats["fc_bpm"])


def test_features_amplitud_central_excluye_al_vecino() -> None:
    """amp_r_mv mide solo ±AMP_WINDOW_S; amp_max_mv, la ventana entera."""
    x = _bump(CENTER, 6.0, 1.0) + _bump(CENTER - 120, 6.0, 1.5)
    feats = _feat(x)

    assert feats["amp_r_mv"] == pytest.approx(1.0, abs=0.01)
    assert feats["amp_max_mv"] == pytest.approx(1.5, abs=0.01)
    assert feats["amp_pp_mv"] == pytest.approx(feats["amp_max_mv"] - feats["amp_min_mv"])
    assert feats["desv_mv"] > 0 and feats["energia_mv2"] > 0


def test_features_morphology_simetria_y_asimetria() -> None:
    """Un QRS simétrico da simetria_corr ≈1 y asimetria ≈0; F, al revés."""
    simetrica = _bump(CENTER, 6.0, 1.0)
    asimetrica = simetrica + _bump(CENTER + 35, 10.0, 0.9)

    f_sim = _feat(simetrica)
    f_asim = _feat(asimetrica)

    assert f_sim["simetria_corr"] > 0.95
    assert abs(f_asim["asimetria"]) > 0.3
    assert f_asim["simetria_corr"] < f_sim["simetria_corr"] - 0.2
    # Ventana puramente simétrica → asimetría (casi) nula.
    assert abs(f_sim["asimetria"]) < 0.1


def test_features_ventana_plana_sin_nan_indeseados() -> None:
    """Señal plana: solo los features de ritmo son NaN (guardas)."""
    feats = _feat(np.zeros(WIDTH))

    nan_esperados = {"rr_prev_s", "rr_next_s", "fc_bpm"}
    for nombre, valor in feats.items():
        if nombre in nan_esperados:
            assert np.isnan(valor)
        else:
            assert np.isfinite(valor), f"{nombre} no finito: {valor}"
    assert feats["n_picos"] == 0.0
    assert feats["simetria_corr"] == 0.0 and feats["asimetria"] == 0.0


@pytest.mark.parametrize("fs", [0, -360.0, float("nan"), float("inf")])
def test_features_fs_invalido_error(fs: float) -> None:
    """fs no finito/positivo → ValueError (criterio de beat_windows)."""
    with pytest.raises(ValueError, match="fs"):
        BeatFeatureExtractor(fs)


def test_features_reutiliza_signal_sin_duplicar_filtro(monkeypatch) -> None:
    """El extractor llama a bandpass_filter/detect_r_peaks de app.signal."""
    llamadas: list[tuple[str, float]] = []

    def filtro_espi(arr: np.ndarray, fs: float) -> np.ndarray:
        llamadas.append(("filtro", fs))
        return signal.bandpass_filter(arr, fs)

    def detector_espi(arr: np.ndarray, fs: float):
        llamadas.append(("detector", fs))
        return signal.detect_r_peaks(arr, fs)

    # baseline importa los símbolos: se sustituyen en su namespace.
    monkeypatch.setattr(baseline, "bandpass_filter", filtro_espi)
    monkeypatch.setattr(baseline, "detect_r_peaks", detector_espi)

    _feat(_bump(CENTER, 6.0, 1.0))

    assert [nombre for nombre, _ in llamadas] == ["filtro", "detector"]
    assert llamadas[0][1] == FS


# ---------------------------------------------------------------------------
# Entrenamiento y predicción
# ---------------------------------------------------------------------------


def test_train_formas_y_metricas_por_clase(entrenado) -> None:
    """El pipeline entrena: métricas 5x5, sens/esp en [0, 1], pred (n,)."""
    model, metrics = entrenado
    X, _, _ = _dataset_sintetico()

    assert isinstance(model, BaselineModel)
    assert [nombre for nombre, _ in model.pipeline.steps] == ["imputer", "scaler", "clf"]

    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert set(pred.tolist()) <= set(AAMI_CLASSES)

    assert set(metrics.clases) == set(AAMI_CLASSES)
    assert len(metrics.matriz_confusion) == 5
    assert all(len(fila) == 5 for fila in metrics.matriz_confusion)
    assert sum(sum(fila) for fila in metrics.matriz_confusion) == metrics.n_muestras
    for clase in AAMI_CLASSES:
        m = metrics.clases[clase]
        assert 0.0 <= m.sensibilidad <= 1.0
        assert 0.0 <= m.especificidad <= 1.0
        assert m.sensibilidad == pytest.approx(
            m.tp / (m.tp + m.fn), abs=1e-12
        ) or (m.tp + m.fn) == 0
        assert m.especificidad == pytest.approx(m.tn / (m.tn + m.fp), abs=1e-12)

    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.model_version == MODEL_VERSION
    assert metrics.dataset_id == DATASET_ID


def test_train_accuracy_decente_en_pacientes_nuevos(entrenado) -> None:
    """Los pacientes de test (no vistos) se clasifican bien (≥0,8)."""
    _, metrics = entrenado
    assert metrics.accuracy >= 0.8


def test_train_evalua_solo_sobre_pacientes_de_test(entrenado) -> None:
    """Las métricas cubren exactamente las filas del split de test."""
    _, metrics = entrenado
    _, _, paciente_id = _dataset_sintetico()
    _, test_idx = split_por_paciente(paciente_id)

    assert metrics.n_muestras == test_idx.size
    assert metrics.matriz_confusion != ()  # hubo evaluación real


def test_predict_distintos_tamanos_y_un_latido(entrenado) -> None:
    """predict acepta 1, pocos y muchos latidos; formas correctas."""
    model, _ = entrenado
    X, _, _ = _dataset_sintetico()

    for n in (1, 7, X.shape[0]):
        pred = model.predict(X[:n])
        assert pred.shape == (n,)
        assert pred.dtype.kind == "U"
        assert set(pred.tolist()) <= set(AAMI_CLASSES)

    # Inferencia de 1 latido (contrato T11): (1, n_muestras) → (1,).
    una = model.predict(X[:1])
    assert una.shape == (1,)
    assert una[0] in AAMI_CLASSES

    probas = model.predict_proba(X[:1])
    assert probas.shape == (1, len(model.clases))
    assert float(probas.sum()) == pytest.approx(1.0, abs=1e-6)
    assert list(model.clases) == sorted(model.clases.tolist())


def test_predict_ventana_demasiado_corta_error(entrenado) -> None:
    """Una ventana por debajo de MIN_MUESTRAS_VENTANA → ValueError."""
    model, _ = entrenado
    with pytest.raises(ValueError, match="demasiado corta"):
        model.predict(np.zeros((1, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Métricas (valores verificables a mano)
# ---------------------------------------------------------------------------


def test_metrics_manualmente_verificables() -> None:
    """Caso pequeño con TP/TN/FP/FN calculados a mano."""
    y_true = np.array(list("NNNNSSVVFQ"))
    y_pred = np.array(list("NNNSSSVFQQ"))

    metrics = metrics_from_predictions(y_true, y_pred)

    esperado = {  # clase: (sensibilidad, especificidad, soporte, tp, tn, fp, fn)
        "N": (3 / 4, 6 / 6, 4, 3, 6, 0, 1),
        "S": (2 / 2, 7 / 8, 2, 2, 7, 1, 0),
        "V": (1 / 2, 8 / 8, 2, 1, 8, 0, 1),  # ningún V predicho es falso
        "F": (0 / 1, 8 / 9, 1, 0, 8, 1, 1),
        "Q": (1 / 1, 8 / 9, 1, 1, 8, 1, 0),
    }
    for clase, (sens, esp, soporte, tp, tn, fp, fn) in esperado.items():
        m = metrics.clases[clase]
        assert m.sensibilidad == pytest.approx(sens)
        assert m.especificidad == pytest.approx(esp)
        assert (m.soporte, m.tp, m.tn, m.fp, m.fn) == (soporte, tp, tn, fp, fn)

    assert metrics.matriz_confusion == (
        (3, 1, 0, 0, 0),
        (0, 2, 0, 0, 0),
        (0, 0, 1, 1, 0),
        (0, 0, 0, 0, 1),
        (0, 0, 0, 0, 1),
    )
    assert metrics.accuracy == pytest.approx(0.7)
    assert metrics.n_muestras == 10


def test_metrics_clases_ausentes_zero_division() -> None:
    """Sin positivos reales la sensibilidad es 0.0 (zero_division=0)."""
    y_true = np.array(["N", "N", "S", "S", "V", "F"])  # sin Q
    y_pred = np.array(["N", "S", "S", "S", "V", "F"])

    metrics = metrics_from_predictions(y_true, y_pred)

    assert set(metrics.clases) == set(AAMI_CLASSES)  # las 5, aunque falte Q
    assert metrics.clases["Q"].soporte == 0
    assert metrics.clases["Q"].sensibilidad == 0.0
    assert metrics.clases["Q"].tp == 0
    for clase in AAMI_CLASSES:
        assert 0.0 <= metrics.clases[clase].especificidad <= 1.0


def test_metrics_sin_negativos_especificidad_zero_division() -> None:
    """Clase única en y_true: TN+FP=0 → especificidad 0.0, no NaN."""
    y = np.array(["N", "N", "N"])
    metrics = metrics_from_predictions(y, y)

    assert metrics.clases["N"].sensibilidad == 1.0
    assert metrics.clases["N"].especificidad == 0.0  # sin negativos
    assert metrics.clases["N"].tn == 0
    assert metrics.clases["V"].especificidad == 1.0  # sí hay negativos para V
    assert not any(
        np.isnan(m.sensibilidad) or np.isnan(m.especificidad)
        for m in metrics.clases.values()
    )


def test_metrics_to_dict_serializable(entrenado) -> None:
    """``to_dict`` es JSON válido y conserva el orden de AAMI_CLASSES."""
    _, metrics = entrenado
    data = json.loads(json.dumps(metrics.to_dict()))

    assert data["model_version"] == MODEL_VERSION
    assert data["dataset_id"] == DATASET_ID
    assert list(data["clases"]) == list(AAMI_CLASSES)
    assert len(data["matriz_confusion"]) == 5
    assert data["sensibilidad_macro"] == pytest.approx(
        np.mean([m.sensibilidad for m in metrics.clases.values()])
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "match"),
    [
        (np.array([]), np.array([]), "Sin muestras"),
        (np.array(["N", "S"]), np.array(["N"]), "desalineados"),
        (np.array([["N"]]), np.array([["N"]]), "1-D"),
        (np.array(["N", "X"]), np.array(["N", "N"]), "y_true"),
        (np.array(["N", "N"]), np.array(["N", "X"]), "y_pred"),
    ],
)
def test_metrics_errores(y_true: np.ndarray, y_pred: np.ndarray, match: str) -> None:
    """Entradas inválidas → ValueError con mensaje claro."""
    with pytest.raises(ValueError, match=match):
        metrics_from_predictions(y_true, y_pred)


def test_evaluate_y_desalineado_error(entrenado) -> None:
    """X con más filas que y → ValueError de alineación."""
    model, _ = entrenado
    X, y, _ = _dataset_sintetico(n_pacientes=2, repeticiones=1)
    with pytest.raises(ValueError, match="desalineadas"):
        evaluate(model, X, y[:-1])


# ---------------------------------------------------------------------------
# Errores de entrenamiento
# ---------------------------------------------------------------------------


def test_train_errores_de_contrato() -> None:
    """Formas/dominio inválidos → ValueError (nunca excepciones raras)."""
    X, y, paciente_id = _dataset_sintetico(n_pacientes=3, repeticiones=1)

    with pytest.raises(ValueError, match="2-D"):
        train_baseline(X[0], y, paciente_id, fs=FS)
    with pytest.raises(ValueError, match="demasiado corta"):
        train_baseline(X[:, :3], y, paciente_id, fs=FS)
    with pytest.raises(ValueError, match="desalineadas"):
        train_baseline(X, y[:-1], paciente_id, fs=FS)
    with pytest.raises(ValueError, match="AAMI"):
        train_baseline(X, np.where(y == "N", "Z", y), paciente_id, fs=FS)
    with pytest.raises(ValueError, match="al menos 2 pacientes"):
        train_baseline(X, y, np.full_like(paciente_id, "100"), fs=FS)
    with pytest.raises(ValueError, match="fs"):
        train_baseline(X, y, paciente_id, fs=0.0)
    with pytest.raises(ValueError, match="sin latidos"):
        train_baseline(
            np.zeros((0, WIDTH), dtype=np.float32),
            np.array([], dtype=str),
            np.array([], dtype=str),
            fs=FS,
        )


# ---------------------------------------------------------------------------
# Serialización joblib
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_conserva_metadatos_y_predicciones(entrenado, tmp_path) -> None:
    """Round-trip: metadatos intactos y predicciones idénticas."""
    model, metrics = entrenado
    path = save_model(model, tmp_path / "baseline.joblib")

    assert path.is_file()
    assert path.parent == tmp_path  # solo tmp_path, nunca el repo

    cargado = load_model(path)
    assert cargado.meta == model.meta
    assert cargado.model_version == MODEL_VERSION == model.model_version
    assert cargado.dataset_id == DATASET_ID == model.dataset_id
    assert cargado.meta["sklearn_version"]  # trazabilidad
    assert cargado.meta["aami_classes"] == list(AAMI_CLASSES)
    assert cargado.meta["random_state"] == 42
    assert cargado.meta["feature_names"] == list(FEATURE_NAMES)
    assert cargado.meta["fs"] == FS

    X, _, _ = _dataset_sintetico(semilla=99)  # datos distintos a los de train
    np.testing.assert_array_equal(cargado.predict(X), model.predict(X))
    np.testing.assert_allclose(cargado.predict_proba(X), model.predict_proba(X))

    recargado = evaluate(cargado, X, np.full(X.shape[0], "N"))
    assert recargado.model_version == MODEL_VERSION
    assert metrics.n_muestras > 0


def _mutar(path, cambio) -> None:
    """Aplica una mutación al bundle serializado (simula archivos malos)."""
    bundle = joblib.load(path)
    cambio(bundle)
    joblib.dump(bundle, path)


def _quitar_version(bundle: dict) -> None:
    bundle["meta"].pop("model_version", None)


def _version_incompatible(bundle: dict) -> None:
    bundle["meta"]["model_version"] = "baseline-sklearn-9.9.9"


def _clases_incompatibles(bundle: dict) -> None:
    bundle["meta"]["aami_classes"] = ["N", "S"]


def _features_incompatibles(bundle: dict) -> None:
    bundle["meta"]["feature_names"] = ["otro_feature"]


def _fs_invalido(bundle: dict) -> None:
    bundle["meta"]["fs"] = 0


def _meta_no_dict(bundle: dict) -> None:
    bundle["meta"] = "no-soy-un-dict"


def _sin_pipeline(bundle: dict) -> None:
    bundle.pop("pipeline", None)


@pytest.mark.parametrize(
    ("cambio", "match"),
    [
        (_quitar_version, "faltan"),
        (_version_incompatible, "model_version incompatible"),
        (_clases_incompatibles, "aami_classes"),
        (_features_incompatibles, "feature_names"),
        (_fs_invalido, "fs inválido"),
        (_meta_no_dict, "se esperaba un dict"),
        (_sin_pipeline, "Bundle de modelo inválido"),
    ],
    ids=["sin-version", "version", "clases", "features", "fs", "meta", "pipeline"],
)
def test_load_metadatos_invalidos_rechazados(entrenado, tmp_path, cambio, match) -> None:
    """Bundles con metadatos rotos → ValueError, nunca carga en silencio."""
    model, _ = entrenado
    path = save_model(model, tmp_path / "modelo.joblib")
    _mutar(path, cambio)

    with pytest.raises(ValueError, match=match):
        load_model(path)


def test_load_archivo_inexistente(entrenado, tmp_path) -> None:
    """Ruta inexistente → FileNotFoundError (propagado de joblib)."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "no-existe.joblib")


def test_save_model_rechaza_objeto_no_modelo(tmp_path) -> None:
    """save_model exige un BaselineModel (error claro, no AttributeErrors)."""
    with pytest.raises(ValueError, match="BaselineModel"):
        save_model(object(), tmp_path / "mal.joblib")


def test_save_model_rechaza_metadatos_incompletos(tmp_path) -> None:
    """Un modelo con meta incompleta no se serializa (falla al guardar)."""
    X, y, paciente_id = _dataset_sintetico(n_pacientes=3, repeticiones=1)
    model, _ = train_baseline(X, y, paciente_id, fs=FS)
    model.meta.pop("dataset_id", None)

    with pytest.raises(ValueError, match="faltan"):
        save_model(model, tmp_path / "mal.joblib")


# ---------------------------------------------------------------------------
# Flujo helper (T11)
# ---------------------------------------------------------------------------


def test_train_from_dataset_usa_metadatos_del_dataset(tmp_path) -> None:
    """train_from_dataset propaga dataset_id/fs y produce modelo usable."""
    X, y, paciente_id = _dataset_sintetico(n_pacientes=6, repeticiones=2)
    ds = EcgDataset(
        X=X,
        y=y,
        paciente_id=paciente_id,
        records=tuple(sorted(set(paciente_id.tolist()))),
        lead=0,
        fs=FS,
        dataset_id=DATASET_ID,
    )

    model, metrics = train_from_dataset(ds)

    assert model.meta["dataset_id"] == DATASET_ID
    assert model.meta["fs"] == FS
    assert model.meta["n_muestras_train"] + model.meta["n_muestras_test"] == ds.n_latidos
    assert metrics.n_muestras > 0
    assert set(metrics.clases) == set(AAMI_CLASSES)

    # Guardar/cargar el modelo del dataset: usable para 1 latido (T11).
    path = save_model(model, tmp_path / "desde-dataset.joblib")
    cargado = load_model(path)
    assert cargado.predict(X[:1]).shape == (1,)
