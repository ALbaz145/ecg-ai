"""Baseline scikit-learn con split por paciente (T10, issue #13).

Primer **modelo entrenado** del proyecto: clasifica cada latido del
dataset AAMI de T09c (:class:`~app.mitbih_dataset.EcgDataset`) en una
de las 5 clases ``N|S|V|F|Q``. Vive en ``baseline.py`` —y no en
``model.py``— porque la responsabilidad es distinta: ``model.py`` es el
detector **heurístico** del MVP (reglas a mano, sin entrenar, ver
``docs/MODEL.md``), ``pipeline.py`` orquesta el flujo del MVP y **este
módulo** se ocupa del ciclo de vida de un modelo *entrenado*: features
por latido, split, fit, métricas y serialización. Pruebas propias en
``tests/backend/test_baseline.py``; la API (T11) sigue sin tocarse.

Criterios del issue #13 y decisiones de diseño:

Features por latido (RR, amplitud, FC y morphology stats)
    Una fila de ``X`` es una ventana **cruda** de ±``BEAT_WINDOW_S``
    centrada en el pico R (el R queda exactamente en ``width // 2``:
    ``beat_windows`` recorta en ``anotación - round(BEAT_WINDOW_S*fs)``
    y T09c excluye las ventanas recortadas, así que el centro es
    fijo). Por cada ventana (:data:`FEATURE_NAMES`, 14 features):

    - **Amplitud**: ``amp_r_mv`` = ``max|x|`` en ±``AMP_WINDOW_S``
      alrededor del R —el mismo criterio de ``extract_features`` de
      ``app.signal``, aplicado a la ventana porque ``EcgDataset`` no
      guarda la señal completa— más ``amp_max_mv``/``amp_min_mv``/
      ``amp_pp_mv`` de la ventana entera.
    - **Estadísticos**: ``media_mv``, ``desv_mv``, ``rms_mv``,
      ``energia_mv2`` (varianza/energía del latido).
    - **Morphology**: ``asimetria`` (diferencia de medias entre
      mitades normalizada por la desviación) y ``simetria_corr``
      (correlación de Pearson entre la mitad izquierda y la derecha
      invertida: un QRS estrecho y simétrico da ≈1).
    - **Ritmo (RR/FC)**: ``rr_prev_s``/``rr_next_s`` = segundos hasta
      el pico R vecino visible en la ventana y ``fc_bpm`` =
      ``60 / min(rr)`` (FC instantánea). Se detectan con
      **``bandpass_filter`` + ``detect_r_peaks`` de ``app.signal``
      reutilizados, sin reimplementar el filtro ni el detector**;
      los vecinos se limitan a ≥``REFRACTORY_S`` del centro (un
      pico a <0,3 s del centro es el propio latido detectado unos
      muestras fuera, no un vecino). ``n_picos`` cuenta los R
      detectados (1 = solo el latido).

    Limitación documentada: el vecino solo cabe en la ventana si
    ``RR < 2 * BEAT_WINDOW_S`` (0,8 s ⇒ FC > 75 lpm); si no hay
    vecino los tres features de ritmo son ``NaN`` y el
    ``SimpleImputer`` del pipeline los rellena con la mediana del
    **train** (nunca con información del test). Todo lo demás se
    calcula sobre la ventana cruda en mV, igual que las medidas de
    amplitud de ``extract_features``.

Split train/test **por paciente** (:func:`split_por_paciente`)
    La partición se hace sobre los **IDs de paciente** (filas de
    ``paciente_id``), no sobre latidos: un paciente no puede caer en
    train y test a la vez (fuga de datos; los registros de MIT-BIH
    tienen correlación intra-paciente y el modelo memorizaría al
    paciente, no a la arritmia). Determinista con ``random_state``,
    sin solape (verificado en la propia función) y con ambos
    conjuntos nunca vacíos.

Modelo: ``LogisticRegression(class_weight="balanced")``
    Justificación (baseline simple y rápido, sklearn 1.8.0):

    - **Lineal y entrenable en milisegundos** sobre ~14 features; el
      baseline debe servir de referencia *entendible* antes de
      redes neuronales: sus coeficientes dicen qué feature pesa.
    - **Determinista**: ``lbfgs`` (solver por defecto) no usa
      aleatoriedad; ``random_state`` fija queda por trazabilidad.
    - **``class_weight="balanced"``** compensa el desbalance AAMI
      (``N`` es la gran mayoría de MIT-BIH; ``Q`` es escaso) sin
      re-muestrear datos.
    - ``RandomForestClassifier`` descartado por ahora: más lento,
      modelo mayor y sobreajustado con tan pocas features de bajo
      nivel; se puede probar como siguiente paso sin cambiar este
      módulo (la evaluación y la serialización son agnósticas).

    El ``Pipeline`` sklearn encadena ``SimpleImputer(median,
    keep_empty_features=True)`` → ``StandardScaler`` → clasificador:
    los features están en escalas muy distintas (``energia_mv2`` en
    mV² frente a ``rr_next_s`` en s) y el scaler es obligatorio para
    una lineal.

Métricas (:func:`evaluate`, :func:`metrics_from_predictions`)
    **Por clase** (las 5 de :data:`~app.mitbih_dataset.AAMI_CLASSES`,
    aunque falten en el test): sensibilidad (recall), especificidad
    (``TN/(TN+FP)``), soporte y los cuatro conteos TP/TN/FP/FN, más
    la matriz de confusión 5x5 (filas = real, columnas = predicho,
    en orden AAMI) y exactitud. Sin clases (o sin negativos) la
    división es 0,0 → ``0.0`` (criterio ``zero_division=0``).

Serialización (:func:`save_model` / :func:`load_model`)
    ``joblib`` con metadatos: ``model_version`` (:data:`MODEL_VERSION`,
    constante documentada; cambiar features/entrenamiento exige
    bumpirla), ``dataset_id`` (el sha256 de ``EcgDataset``),
    ``sklearn_version``, clases AAMI, ``random_state``,
    ``feature_names``, ``fs`` y extras de trazabilidad. La carga
    valida los metadatos mínimos (presencia, versión del modelo,
    clases y nombres de features) y reconstruye el extractor —que no
    tiene estado— en vez de picklearlo. ``*.joblib`` está en
    ``.gitignore``: el modelo entrenado **nunca se commitea** (las
    pruebas escriben solo en ``tmp_path``).

Uso típico (T11)::

    model, metrics = train_from_dataset(ds)      # ds: EcgDataset
    save_model(model, "data/models/baseline.joblib")
    model = load_model("data/models/baseline.joblib")
    etiqueta = model.predict(ventana[None, :])[0]  # 1 latido (1, n_muestras)

Pruebas: datos **sintéticos aprendibles** (sin dataset real, sin red);
el dataset MIT-BIH descargado vive en ``data/`` (gitignored).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.mitbih_dataset import AAMI_CLASSES, EcgDataset
from app.signal import (
    AMP_WINDOW_S,
    BEAT_WINDOW_S,
    REFRACTORY_S,
    bandpass_filter,
    detect_r_peaks,
)

#: Versión del modelo entrenado (issue #13: versión + identificador de
#: dataset en la serialización). Se guarda en los metadatos y
#: :func:`load_model` la exige: cualquier cambio que altere features,
#: arquitectura o entrenamiento debe **bumpirla** (p. ej.
#: ``baseline-sklearn-0.2.0``) para que los archivos antiguos fallen
#: con un error claro en vez de cargar en silencio con semántica nueva.
MODEL_VERSION = "baseline-sklearn-0.1.0"

#: Máximo RR medible con el vecino visible en la ventana: el pico R
#: vecino cabe dentro de ±``BEAT_WINDOW_S`` → ``RR < 2 * BEAT_WINDOW_S``.
RR_VISIBLE_MAX_S = 2.0 * BEAT_WINDOW_S

#: Ancho mínimo de ventana para que existan las dos mitades, la región
#: de amplitud y la correlación de simetría. Las ventanas reales de
#: ``EcgDataset`` tienen ``2 * round(BEAT_WINDOW_S * fs)`` = 288.
MIN_MUESTRAS_VENTANA = 4

#: Nombres y orden de los features; el orden es parte del contrato de
#: serialización (:func:`load_model` lo valida contra los metadatos).
FEATURE_NAMES: tuple[str, ...] = (
    # Amplitud (mV, sobre la ventana cruda).
    "amp_r_mv",
    "amp_max_mv",
    "amp_min_mv",
    "amp_pp_mv",
    # Estadísticos de la ventana (mV / mV²).
    "media_mv",
    "desv_mv",
    "rms_mv",
    "energia_mv2",
    # Morphology (simetría del QRS).
    "asimetria",
    "simetria_corr",
    # Ritmo (RR/FC), NaN si el vecino no cabe en la ventana.
    "rr_prev_s",
    "rr_next_s",
    "fc_bpm",
    "n_picos",
)

#: Metadatos mínimos que :func:`save_model` escribe y
#: :func:`load_model` exige presentes en el bundle.
REQUIRED_META_KEYS: tuple[str, ...] = (
    "model_version",
    "dataset_id",
    "sklearn_version",
    "aami_classes",
    "random_state",
    "feature_names",
    "fs",
)

#: Claves opcionales que solo aportan trazabilidad (no validan carga).
OPTIONAL_META_KEYS: tuple[str, ...] = (
    "clases_entrenadas",
    "n_muestras_train",
    "n_muestras_test",
    "created_at",
)


# ---------------------------------------------------------------------------
# Features por latido
# ---------------------------------------------------------------------------


def _check_ventanas(X: np.ndarray, *, origen: str = "X") -> np.ndarray:
    """Valida una matriz de ventanas ``(n_latidos, n_muestras)``.

    Devuelve ``X`` como ``ndarray``. Lanza ``ValueError`` (nunca
    ``IndexError``/``TypeError``) si no es 2-D, no es numérico, la
    ventana es más corta que :data:`MIN_MUESTRAS_VENTANA` o no hay
    latidos.
    """
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(
            f"{origen} debe ser un ndarray 2-D (n_latidos por ventana); "
            f"recibido: {type(X).__name__} con forma {arr.shape!r}."
        )
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{origen} debe ser numérico; recibido: {arr.dtype}.")
    if arr.shape[1] < MIN_MUESTRAS_VENTANA:
        raise ValueError(
            f"Ventana demasiado corta en {origen}: {arr.shape[1]} muestras "
            f"(mínimo {MIN_MUESTRAS_VENTANA}; las reales tienen "
            "2 * round(BEAT_WINDOW_S * fs))."
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{origen} sin latidos: se esperaba al menos 1 fila.")
    return arr


class BeatFeatureExtractor:
    """Extrae los 14 :data:`FEATURE_NAMES` de cada ventana de latido.

    Sin estado salvo ``fs`` (y sus constantes derivadas), así que
    ``transform`` es idéntico en entrenamiento e inferencia y el
    extractor **no se serializa**: :func:`load_model` lo reconstruye
    desde ``meta["fs"]`` (menos superficie de pickle y un bundle de
    solo ``meta`` + ``pipeline``).

    Parameters
    ----------
    fs:
        Frecuencia de muestreo de las ventanas (Hz), para convertir
        muestras → segundos en los features de ritmo. Debe ser finita
        y positiva (mismo criterio que ``mitbih_load.beat_windows``).

    Notes
    -----
    El pico R central se asume en ``width // 2`` (contrato de
    ``beat_windows``/T09c descrito en el docstring del módulo). Los
    vecinos de RR se detectan con ``bandpass_filter`` y
    ``detect_r_peaks`` **importados de** ``app.signal`` (no se
    reimplementa el filtro pasa-banda ni el umbral adaptativo) y se
    aceptan solo a ≥ ``REFRACTORY_S`` del centro.
    """

    def __init__(self, fs: float) -> None:
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"fs debe ser un valor finito positivo (recibido: {fs!r}).")
        self.fs = float(fs)
        #: Región de amplitud ±``AMP_WINDOW_S`` y separación mínima
        #: de vecino (``REFRACTORY_S``) en muestras, sin duplicar las
        #: constantes de ``app.signal``.
        self._amp_half = max(1, round(AMP_WINDOW_S * self.fs))
        self._refractory = max(1, round(REFRACTORY_S * self.fs))

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Nombres de los features en el orden de salida."""
        return FEATURE_NAMES

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Matrix de features ``(n_latidos, len(FEATURE_NAMES))`` float64."""
        arr = _check_ventanas(X, origen="X")
        out = np.empty((arr.shape[0], len(FEATURE_NAMES)), dtype=np.float64)
        for i in range(arr.shape[0]):
            out[i] = self._one(np.asarray(arr[i], dtype=np.float64))
        return out

    def _one(self, x: np.ndarray) -> np.ndarray:
        """Features de una ventana (ver :data:`FEATURE_NAMES` para el orden)."""
        n = x.size
        center = n // 2

        # Amplitud: región central ±AMP_WINDOW_S (mismo criterio que
        # extract_features de app.signal, aplicado a la ventana).
        lo = max(0, center - self._amp_half)
        hi = min(n, center + self._amp_half + 1)
        amp_r = float(np.max(np.abs(x[lo:hi])))
        amp_max = float(np.max(x))
        amp_min = float(np.min(x))

        # Estadísticos de la ventana (guardas ante señal plana).
        media = float(np.mean(x))
        desv = float(np.std(x))
        rms = float(np.sqrt(np.mean(x * x)))
        energia = float(np.sum(x * x))

        # Morphology: mitades izquierda/derecha respecto del R.
        left = x[:center]
        right = x[center:]
        media_l = float(np.mean(left)) if left.size else media
        media_r = float(np.mean(right)) if right.size else media
        asimetria = (media_r - media_l) / desv if desv > 0.0 else 0.0
        a = left - media_l
        b = right[::-1] - media_r
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        simetria = float(np.dot(a, b) / den) if den > 0.0 else 0.0

        # Ritmo: picos R de la ventana filtrada (app.signal, sin
        # duplicar lógica). Vecinos = picos a ≥REFRACTORY_S del centro.
        peaks = detect_r_peaks(bandpass_filter(x, self.fs), self.fs).indices
        previos = peaks[peaks <= center - self._refractory]
        siguientes = peaks[peaks >= center + self._refractory]
        rr_prev = float(center - int(previos.max())) / self.fs if previos.size else math.nan
        rr_next = float(int(siguientes.min()) - center) / self.fs if siguientes.size else math.nan
        cercanos = [v for v in (rr_prev, rr_next) if not math.isnan(v)]
        fc = 60.0 / min(cercanos) if cercanos else math.nan

        return np.array(
            [
                amp_r,
                amp_max,
                amp_min,
                amp_max - amp_min,
                media,
                desv,
                rms,
                energia,
                asimetria,
                simetria,
                rr_prev,
                rr_next,
                fc,
                float(peaks.size),
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Split por paciente
# ---------------------------------------------------------------------------


def split_por_paciente(
    paciente_id: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Partición train/test **por paciente** (sin fuga de datos).

    Muestrea **IDs de paciente** (no latidos) y devuelve los índices
    de *filas* de cada conjunto: todos los latidos de un paciente van
    juntos a train o a test. Determinista para un mismo
    ``random_state`` (``numpy.random.default_rng``) y con los índices
    ordenados ascendentemente.

    Parameters
    ----------
    paciente_id:
        Array 1-D con el ID del registro **por latido** (el campo
        ``paciente_id`` de ``EcgDataset``).
    test_size:
        Fracción de **pacientes** para test, en ``(0, 1)``. El número
        de pacientes de test es ``ceil(test_size * n_pacientes)``,
        acotado a ``n_pacientes - 1`` para que train nunca quede vacío.
    random_state:
        Semilla de la permutación (42 por defecto).

    Returns
    -------
    (train_idx, test_idx):
        Índices ``int64`` ordenados y disjuntos cuya unión cubre
        **todos** los índices de ``paciente_id``.

    Raises
    ------
    ValueError
        Si ``paciente_id`` no es 1-D o está vacío, hay un solo
        paciente (no se puede particionar) o ``test_size`` no está en
        ``(0, 1)``.
    AssertionError
        Defensivo: si tras la partición un paciente apareciera en los
        dos conjuntos (imposible por construcción; es la garantía
        anti-fuga del issue #13).
    """
    ids = np.asarray(paciente_id)
    if ids.ndim != 1:
        raise ValueError(
            f"paciente_id debe ser un ndarray 1-D; recibido con forma {ids.shape!r}."
        )
    if ids.size == 0:
        raise ValueError("paciente_id vacío: no hay pacientes que dividir.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size debe estar en (0, 1); recibido: {test_size!r}.")

    unicos = np.unique(ids)  # orden estable y determinista
    if unicos.size < 2:
        raise ValueError(
            "Se necesitan al menos 2 pacientes para el split por paciente "
            f"(recibido: {unicos.size} → {unicos.tolist()})."
        )

    n_test = math.ceil(test_size * int(unicos.size))
    n_test = min(n_test, int(unicos.size) - 1)  # train nunca vacío
    rng = np.random.default_rng(random_state)
    permutados = rng.permutation(unicos)
    pacientes_test = permutados[:n_test]
    pacientes_train = permutados[n_test:]

    train_idx = np.flatnonzero(np.isin(ids, pacientes_train))
    test_idx = np.flatnonzero(np.isin(ids, pacientes_test))

    # Verificación anti-fuga (defensiva) sobre la salida real.
    solape = set(ids[train_idx].tolist()) & set(ids[test_idx].tolist())
    if solape:
        raise AssertionError(
            f"Fuga de datos: pacientes en train y test a la vez: {sorted(solape)}."
        )
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def _check_entradas(
    X: np.ndarray, y: np.ndarray, paciente_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valida el trío ``(X, y, paciente_id)`` de :func:`train_baseline`.

    Mismos criterios que el contrato de ``EcgDataset``: ``X`` 2-D
    numérico, ``y``/``paciente_id`` 1-D alineados por fila y dominio
    de ``y`` dentro de :data:`AAMI_CLASSES` (las clases inválidas se
    nombran en el error, nunca en silencio).
    """
    arr = _check_ventanas(X, origen="X")
    y_arr = np.asarray(y)
    pid = np.asarray(paciente_id)
    if y_arr.ndim != 1 or pid.ndim != 1:
        raise ValueError(
            "y y paciente_id deben ser arrays 1-D (clases AAMI y IDs por "
            f"latido); recibido: formas {y_arr.shape!r} y {pid.shape!r}."
        )
    if not (arr.shape[0] == y_arr.shape[0] == pid.shape[0]):
        raise ValueError(
            "Filas desalineadas: X, y y paciente_id deben tener la misma "
            f"longitud (recibido: {arr.shape[0]}, {y_arr.shape[0]}, {pid.shape[0]})."
        )
    invalidas = sorted(set(map(str, y_arr)) - set(AAMI_CLASSES))
    if invalidas:
        raise ValueError(
            f"Clases AAMI inválidas en y: {invalidas}. Se admite únicamente "
            f"{list(AAMI_CLASSES)} (AAMI_CLASSES)."
        )
    return arr, y_arr, pid


def _build_pipeline(random_state: int) -> Pipeline:
    """``Pipeline`` sklearn: imputación de NaN → escalado → clasificador.

    - ``SimpleImputer(median, keep_empty_features=True)``: rellena los
      features de ritmo ``NaN`` (vecino no visible) con la mediana del
      train; ``keep_empty_features`` evita descartar columnas si en el
      train ningún latido tiene vecino visible (el número de features
      se mantiene estable entre ``fit`` y ``predict``).
    - ``StandardScaler``: obligatorio para la regresión lineal con
      features en escalas dispares (mV, mV², s, lpm, conteos).
    - ``LogisticRegression(class_weight="balanced", max_iter=1000)``:
      ver la justificación en el docstring del módulo.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_baseline(
    X: np.ndarray,
    y: np.ndarray,
    paciente_id: np.ndarray,
    *,
    fs: float,
    dataset_id: str = "",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[BaselineModel, Metrics]:
    """Entrena el baseline con split por paciente y lo evalúa en test.

    Flujo: valida entradas → :func:`split_por_paciente` → features de
    los latidos de train (:class:`BeatFeatureExtractor`) →
    ``fit`` del pipeline → :func:`evaluate` **sobre los pacientes de
    test** (nunca sobre train).

    Parameters
    ----------
    X, y, paciente_id:
        Contrato de ``EcgDataset`` (ventanas crudas, clases AAMI e ID
        de registro por latido).
    fs:
        Frecuencia de muestreo de las ventanas (Hz).
    dataset_id:
        Identificador de contenido del ``EcgDataset`` de origen (los
        metadatos del modelo lo guardan; vacío si se entrena a mano).
    test_size, random_state:
        Repartidos a :func:`split_por_paciente`.

    Returns
    -------
    (model, metrics):
        El modelo entrenado y las métricas **por clase** del conjunto
        de test (matriz de confusión 5x5, sensibilidad y
        especificidad).

    Raises
    ------
    ValueError
        Si las entradas violan el contrato (formas, dominio AAMI, fs)
        o el split por paciente no es posible (<2 pacientes).
    """
    X, y, paciente_id = _check_entradas(X, y, paciente_id)
    train_idx, test_idx = split_por_paciente(
        paciente_id, test_size=test_size, random_state=random_state
    )

    extractor = BeatFeatureExtractor(fs)
    pipeline = _build_pipeline(random_state)
    pipeline.fit(extractor.transform(X[train_idx]), y[train_idx])

    meta: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "dataset_id": str(dataset_id),
        "sklearn_version": sklearn.__version__,
        "aami_classes": list(AAMI_CLASSES),
        "random_state": int(random_state),
        "feature_names": list(FEATURE_NAMES),
        "fs": float(fs),
        "clases_entrenadas": [str(c) for c in pipeline.classes_],
        "n_muestras_train": int(train_idx.size),
        "n_muestras_test": int(test_idx.size),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    model = BaselineModel(pipeline=pipeline, extractor=extractor, meta=meta)
    metrics = evaluate(model, X[test_idx], y[test_idx])
    return model, metrics


def train_from_dataset(
    ds: EcgDataset,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[BaselineModel, Metrics]:
    """Convenience para T11: entrena desde un ``EcgDataset`` de T09c.

    Extrae ``X``/``y``/``paciente_id``/``fs`` y propaga
    ``ds.dataset_id`` a los metadatos del modelo. El resto es
    idéntico a :func:`train_baseline`.
    """
    return train_baseline(
        ds.X,
        ds.y,
        ds.paciente_id,
        fs=ds.fs,
        dataset_id=ds.dataset_id,
        test_size=test_size,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


@dataclass
class ClassMetrics:
    """Métricas de **una** clase AAMI (issue #13).

    Attributes
    ----------
    sensibilidad:
        ``TP / (TP + FN)`` (recall); ``0.0`` si no hay positivos
        reales (``zero_division=0``).
    especificidad:
        ``TN / (TN + FP)``; ``0.0`` si no hay negativos (clase única
        en ``y_true``, ``zero_division=0``).
    soporte:
        Nº de latidos reales de la clase.
    tp, tn, fp, fn:
        Conteos de la clase frente a todas las demás (uno-vs-resto).
    """

    sensibilidad: float
    especificidad: float
    soporte: int
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        """Versión serializable (JSON) de las métricas de la clase."""
        return {
            "sensibilidad": self.sensibilidad,
            "especificidad": self.especificidad,
            "soporte": self.soporte,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }


@dataclass
class Metrics:
    """Métricas de evaluación completas, por clase y globales.

    Attributes
    ----------
    clases:
        Las 5 clases de ``AAMI_CLASSES`` en orden canónico (también
        las ausentes del test, con ``soporte=0`` y sensibilidad
        ``0.0``).
    matriz_confusion:
        5x5 ``(n, n)`` tuple-of-tuples: filas = clase real, columnas =
        clase predicha, en orden ``AAMI_CLASSES``.
    accuracy:
        Exactitud global (fracción en ``[0, 1]``).
    n_muestras:
        Nº de latidos evaluados.
    model_version, dataset_id:
        Metadatos del modelo evaluado (vacíos si la evaluación fue
        ``metrics_from_predictions`` sin modelo).
    """

    clases: dict[str, ClassMetrics]
    matriz_confusion: tuple[tuple[int, ...], ...]
    accuracy: float
    n_muestras: int
    model_version: str
    dataset_id: str

    @property
    def sensibilidad_macro(self) -> float:
        """Media de la sensibilidad de las 5 clases."""
        return float(np.mean([c.sensibilidad for c in self.clases.values()]))

    @property
    def especificidad_macro(self) -> float:
        """Media de la especificidad de las 5 clases."""
        return float(np.mean([c.especificidad for c in self.clases.values()]))

    def to_dict(self) -> dict[str, Any]:
        """Versión serializable (JSON) completa de las métricas."""
        return {
            "model_version": self.model_version,
            "dataset_id": self.dataset_id,
            "n_muestras": self.n_muestras,
            "accuracy": self.accuracy,
            "sensibilidad_macro": self.sensibilidad_macro,
            "especificidad_macro": self.especificidad_macro,
            "matriz_confusion": [list(row) for row in self.matriz_confusion],
            "clases": {clase: c.to_dict() for clase, c in self.clases.items()},
        }


def metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_version: str = "",
    dataset_id: str = "",
) -> Metrics:
    """Calcula las métricas por clase a partir de predicciones.

    Núcleo de :func:`evaluate`, testeable con ``y_pred`` escrito a
    mano (valores verificables sin entrenar nada). La matriz de
    confusión se construye con ``labels=AAMI_CLASSES`` para que sea
    **siempre 5x5** aunque falten clases, y la sensibilidad con
    ``recall_score(..., zero_division=0)``.

    Raises
    ------
    ValueError
        Si las formas no cuadran, ``y_true`` está vacío o alguna
        etiqueta (real o predicha) está fuera de ``AAMI_CLASSES``.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if y_t.ndim != 1 or y_p.ndim != 1:
        raise ValueError(
            f"y_true e y_pred deben ser 1-D; recibido: {y_t.shape!r} y {y_p.shape!r}."
        )
    if y_t.shape != y_p.shape:
        raise ValueError(
            "y_true e y_pred desalineados: "
            f"{y_t.shape[0]} reales frente a {y_p.shape[0]} predichos."
        )
    if y_t.size == 0:
        raise ValueError("Sin muestras: no se pueden calcular métricas de 0 latidos.")
    for nombre, arr in (("y_true", y_t), ("y_pred", y_p)):
        invalidas = sorted(set(map(str, arr)) - set(AAMI_CLASSES))
        if invalidas:
            raise ValueError(
                f"Clases AAMI inválidas en {nombre}: {invalidas}. Se admite "
                f"únicamente {list(AAMI_CLASSES)} (AAMI_CLASSES)."
            )

    etiquetas = list(AAMI_CLASSES)
    cm = confusion_matrix(y_t, y_p, labels=etiquetas)
    sensibilidades = recall_score(y_t, y_p, labels=etiquetas, average=None, zero_division=0)
    total = int(cm.sum())

    clases: dict[str, ClassMetrics] = {}
    for i, clase in enumerate(etiquetas):
        tp = int(cm[i, i])
        soporte = int(cm[i].sum())
        fn = soporte - tp
        fp = int(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        den = tn + fp
        clases[clase] = ClassMetrics(
            sensibilidad=float(sensibilidades[i]),
            especificidad=float(tn / den) if den > 0 else 0.0,
            soporte=soporte,
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
        )

    return Metrics(
        clases=clases,
        matriz_confusion=tuple(tuple(int(v) for v in row) for row in cm),
        accuracy=float(np.mean(y_t == y_p)),
        n_muestras=int(total),
        model_version=model_version,
        dataset_id=dataset_id,
    )


def evaluate(model: BaselineModel, X: np.ndarray, y: np.ndarray) -> Metrics:
    """Evalúa ``model`` sobre ``(X, y)`` con métricas por clase.

    Predice con el modelo (extracción de features + pipeline) y
    devuelve :class:`Metrics`: sensibilidad, especificidad y soporte
    por cada una de las 5 clases, matriz de confusión 5x5 y
    exactitud. En el flujo de :func:`train_baseline` ``X``/``y`` son
    los latidos de los pacientes de **test**.

    Raises
    ------
    ValueError
        Si ``y`` no es 1-D, si ``X`` y ``y`` tienen longitudes
        distintas, si ``X`` viola el contrato de ventanas o si
        ``y_true`` está vacío / con clases fuera de ``AAMI_CLASSES``
        (ver ``metrics_from_predictions``).
    """
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"y debe ser un ndarray 1-D; recibido con forma {y_arr.shape!r}.")
    y_pred = model.predict(X)
    if y_pred.size != y_arr.size:
        raise ValueError(
            "Filas desalineadas: X produce "
            f"{y_pred.size} predicciones y y tiene {y_arr.size} etiquetas."
        )
    return metrics_from_predictions(
        y_arr,
        y_pred,
        model_version=model.model_version,
        dataset_id=model.dataset_id,
    )


# ---------------------------------------------------------------------------
# Modelo, serialización
# ---------------------------------------------------------------------------


@dataclass
class BaselineModel:
    """Baseline entrenado: ``extractor`` + ``pipeline`` sklearn + metadatos.

    Envuelve los dos objetos necesarios para predecir (la extracción de
    features y el ``Pipeline`` imputer→scaler→LogisticRegression) más
    el dict de metadatos de :func:`save_model`. Los metadatos se
    validan en :func:`save_model`/ :func:`load_model`, no aquí (el
    objeto también se construye a mano en pruebas).

    Attributes
    ----------
    pipeline:
        ``sklearn.pipeline.Pipeline`` entrenado (con ``classes_``).
    extractor:
        ``BeatFeatureExtractor`` con el ``fs`` con el que se entrenó.
    meta:
        Ver :data:`REQUIRED_META_KEYS` (más trazabilidad opcional).
    """

    pipeline: Pipeline
    extractor: BeatFeatureExtractor
    meta: dict[str, Any]

    @property
    def model_version(self) -> str:
        """Versión del modelo (:data:`MODEL_VERSION` si está informada)."""
        return str(self.meta.get("model_version", ""))

    @property
    def dataset_id(self) -> str:
        """``dataset_id`` del ``EcgDataset`` con el que se entrenó."""
        return str(self.meta.get("dataset_id", ""))

    @property
    def clases(self) -> np.ndarray:
        """Clases AAMI en el orden de ``predict_proba`` (las entrenadas)."""
        return np.asarray(self.pipeline.classes_)

    def features(self, X: np.ndarray) -> np.ndarray:
        """Matrix de features de las ventanas ``(n_latidos, n_muestras)``."""
        return self.extractor.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Clase AAMI predicha por ventana, forma ``(n_latidos,)``.

        Acepta también **un solo latido** con forma
        ``(1, n_muestras)`` —el caso T11 de inferencia—::

            model.predict(ventana[None, :])[0]
        """
        return np.asarray(self.pipeline.predict(self.extractor.transform(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades por clase, forma ``(n_latidos, n_clases)``.

        El orden de las columnas es :attr:`clases` (las presentes en
        el train; no necesariamente las 5).
        """
        return np.asarray(self.pipeline.predict_proba(self.extractor.transform(X)))


def _validate_meta(meta: Any, *, origen: str) -> dict[str, Any]:
    """Valida los metadatos mínimos de un bundle de modelo.

    Parameters
    ----------
    meta:
        Dict de metadatos (o cualquier otro objeto → error).
    origen:
        Descripción del origen para los mensajes (la ruta al cargar).

    Returns
    -------
    dict
        El propio ``meta`` validado.

    Raises
    ------
    ValueError
        Si no es dict, faltan claves de :data:`REQUIRED_META_KEYS`,
        ``model_version`` difiere de :data:`MODEL_VERSION`,
        ``aami_classes`` difiere de ``AAMI_CLASSES`` o
        ``feature_names`` difiere de :data:`FEATURE_NAMES` (si los
        features cambian, hay que bumpear ``MODEL_VERSION``).
    """
    if not isinstance(meta, dict):
        raise ValueError(
            f"Metadatos ilegibles en {origen}: se esperaba un dict; "
            f"recibido {type(meta).__name__}."
        )
    faltan = [clave for clave in REQUIRED_META_KEYS if clave not in meta]
    if faltan:
        raise ValueError(f"Metadatos incompletos en {origen}: faltan {faltan}.")
    if meta["model_version"] != MODEL_VERSION:
        raise ValueError(
            f"model_version incompatible en {origen}: el archivo se creó con "
            f"{meta['model_version']!r} y este código espera {MODEL_VERSION!r}."
        )
    if list(meta["aami_classes"]) != list(AAMI_CLASSES):
        raise ValueError(
            f"aami_classes inesperado en {origen}: {meta['aami_classes']!r}; "
            f"se esperaba {list(AAMI_CLASSES)} (AAMI_CLASSES)."
        )
    if list(meta["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError(
            f"feature_names incompatible en {origen}: los features del "
            "archivo no coinciden con los de este código (bump de "
            f"{MODEL_VERSION} pendiente si los features cambiaron)."
        )
    try:
        fs = float(meta["fs"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"fs ilegible en {origen}: {meta['fs']!r} ({type(exc).__name__})."
        ) from exc
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs inválido en {origen}: {meta['fs']!r}.")
    return meta


def save_model(model: BaselineModel, path: str | Path) -> Path:
    """Serializa el modelo entrenado con ``joblib``.

    Escribe el bundle ``{"meta": ..., "pipeline": ...}`` (el extractor
    no se serializa: es sin estado y se reconstruye con ``meta["fs"]``)
    y devuelve la ruta escrita. Los directorios padre se crean si
    faltan. Los metadatos se validan antes de escribir para no dejar
    archivos que luego no cargan.

    Parameters
    ----------
    model:
        Modelo de :func:`train_baseline` (o reconstruido por
        :func:`load_model`).
    path:
        Ruta destino (p. ej. ``data/models/baseline.joblib``,
        gitignored: ``*.joblib`` **nunca se commitea**).

    Returns
    -------
    Path
        La ruta escrita.

    Raises
    ------
    ValueError
        Si ``model`` no es un ``BaselineModel`` o sus metadatos no
        pasan :func:`_validate_meta`.
    """
    if not isinstance(model, BaselineModel):
        raise ValueError(
            f"Se esperaba un BaselineModel; recibido: {type(model).__name__}."
        )
    _validate_meta(model.meta, origen="el modelo en memoria")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"meta": dict(model.meta), "pipeline": model.pipeline}, path)
    return path


def load_model(path: str | Path) -> BaselineModel:
    """Carga un modelo serializado con :func:`save_model`.

    Valida los metadatos mínimos (:func:`_validate_meta`: presencia,
    ``model_version``, clases AAMI y ``feature_names``) y que el
    ``pipeline`` tenga ``predict``/``classes_``; después reconstruye
    el :class:`BeatFeatureExtractor` con ``meta["fs"]``.

    ``sklearn_version`` se guarda para **trazabilidad** pero no
    invalida la carga: joblib es compatible dentro de la misma línea
    de versiones (se recomienda regenerar el modelo al cambiar de
    scikit-learn).

    Raises
    ------
    FileNotFoundError
        Si ``path`` no existe (propagado de ``joblib``).
    ValueError
        Si el bundle no tiene la forma ``{"meta", "pipeline"}``, los
        metadatos no pasan la validación o el pipeline no es usable.
    """
    path = Path(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "meta" not in bundle or "pipeline" not in bundle:
        raise ValueError(
            f"Bundle de modelo inválido en {path}: se esperaba un dict con "
            f"las claves 'meta' y 'pipeline'; recibido: {type(bundle).__name__}."
        )
    meta = _validate_meta(bundle["meta"], origen=str(path))
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "classes_"):
        raise ValueError(
            f"pipeline inválido en {path}: sin predict/classes_ "
            f"({type(pipeline).__name__})."
        )
    return BaselineModel(
        pipeline=pipeline,
        extractor=BeatFeatureExtractor(float(meta["fs"])),
        meta=dict(meta),
    )
