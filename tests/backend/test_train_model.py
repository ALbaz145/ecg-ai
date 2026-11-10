"""Pruebas del CLI ``python -m app.train_model`` (T11c, issue #45).

Datos **sintéticos aprendibles** (mismo criterio que ``test_baseline.py``):
sin dataset real, sin wfdb y sin red — la descarga y ``build_dataset`` se
sustituyen por dobles con ``monkeypatch`` cuando toca entrenar, y los
``.joblib`` se escriben **solo en ``tmp_path``** (``data/`` y ``*.joblib``
siguen gitignored: ``git status`` queda limpio tras la suite).

Cobertura frente a los criterios de aceptación de #45:

- ``--help`` OK (con ``--dest``, ``--seed``, ``--test-size``…) vía
  ``main`` **y** vía subprocess real ``python -m app.train_model --help``;
- entrena con el dataset mockeado, escribe el artefacto en ``--dest``
  (creando directorios) y ``load_model`` lo carga;
- métricas por stdout (``Metrics.to_dict``: accuracy,
  sensibilidad/especificidad macro, matriz de confusión) + ruta escrita;
- ``--dest`` por defecto = ``resolve_model_path()`` (``MODEL_PATH`` y
  default relativo bajo el cwd);
- idempotente: re-ejecutar sobrescribe el artefacto y con la caché local
  completa no hay re-descarga (el seam ``_load_wfdb`` estalla si wfdb se
  llegara a importar);
- errores de descarga / dataset inválido o vacío / IDs desconocidos →
  motivo claro por stderr y código de salida 1 (sin traceback crudo),
  con el mensaje de "cómo preparar los datos" si no hay red **ni** caché;
- argumentos inválidos → código estándar de argparse (2).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from app import mitbih, train_model
from app.baseline import Metrics, load_model
from app.mitbih_dataset import AAMI_CLASSES, EcgDataset
from app.model_loader import resolve_model_path

FS = 360.0
WIDTH = 288  # 2 * round(BEAT_WINDOW_S * fs), ancho real de EcgDataset
CENTER = WIDTH // 2
#: Sufijos de la caché de un registro (espejo del trípode D-1 de T09a).
SUFFIXES = (".hea", ".dat", ".atr")

_T = np.arange(WIDTH, dtype=np.float64)


# ---------------------------------------------------------------------------
# Datos sintéticos (sin red, sin dataset real)
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
    n_pacientes: int = 6, repeticiones: int = 2, semilla: int = 7
) -> EcgDataset:
    """``EcgDataset`` sintético mínimo y aprendible.

    6 pacientes x 10 latidos (5 clases AAMI x 2 repeticiones) = 60 filas:
    con ``--test-size 0.5`` el split por paciente deja 30 de test.
    """
    rng = np.random.default_rng(semilla)
    filas: list[np.ndarray] = []
    etiquetas: list[str] = []
    pacientes: list[str] = []
    for p in range(n_pacientes):
        for clase in AAMI_CLASSES * repeticiones:
            filas.append(_ventana_sintetica(clase, rng).astype(np.float32))
            etiquetas.append(clase)
            pacientes.append(f"{100 + p}")
    return EcgDataset(
        X=np.vstack(filas),
        y=np.asarray(etiquetas),
        paciente_id=np.asarray(pacientes),
        records=tuple(f"{100 + p}" for p in range(n_pacientes)),
        lead=0,
        fs=FS,
        dataset_id="t11c-cli-sintetico",
    )


@pytest.fixture
def dataset_sintetico() -> EcgDataset:
    """Dataset sintético compartido por los tests con descarga mockeada."""
    return _dataset_sintetico()


@pytest.fixture
def cli_sin_red(
    monkeypatch: pytest.MonkeyPatch, dataset_sintetico: EcgDataset
) -> Iterator[tuple[list[dict[str, object]], list[dict[str, object]]]]:
    """Dobles de ``download_mitbih`` y ``build_dataset``: cero red, cero wfdb.

    Devuelve ``(descargas, construcciones)`` con las llamadas registradas
    para poder asertar qué recibió cada paso del flujo.
    """
    descargas: list[dict[str, object]] = []
    construcciones: list[dict[str, object]] = []

    def _descarga(dest: Path, records: list[str] | None = None, **kwargs: object) -> Path:
        descargas.append({"dest": Path(dest), "records": records})
        return Path(dest)

    def _construye(
        records: list[str] | None = None, *, dest: Path, lead: int = 0
    ) -> EcgDataset:
        construcciones.append({"records": records, "dest": Path(dest), "lead": lead})
        return dataset_sintetico

    monkeypatch.setattr(train_model, "download_mitbih", _descarga)
    monkeypatch.setattr(train_model, "build_dataset", _construye)
    yield descargas, construcciones


def _escribir_cache(data_dir: Path, records: list[str]) -> None:
    """Crea la caché completa (``.hea``+``.dat``+``.atr``) de cada registro."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        for suffix in SUFFIXES:
            (data_dir / f"{record}{suffix}").write_text("cache sintética")


# ---------------------------------------------------------------------------
# --help (AC: ejecutable con los argumentos esperados)
# ---------------------------------------------------------------------------


def test_help_sale_con_0_y_lista_los_argumentos(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` → exit 0, los argumentos del AC y la salida de métricas."""
    with pytest.raises(SystemExit) as excinfo:
        train_model.main(["--help"])
    assert excinfo.value.code == 0

    out = capsys.readouterr().out
    for argumento in ("--dest", "--seed", "--test-size", "--records", "--data-dir", "--force"):
        assert argumento in out
    # La salida de métricas está documentada en la ayuda.
    assert "accuracy" in out
    assert "sensibilidad" in out
    assert "matriz de confusión" in out
    assert "Códigos de salida" in out


def test_python_m_train_model_help_subprocess() -> None:
    """Subprocess real ``python -m app.train_model --help`` desde ``backend/``."""
    backend_dir = Path(__file__).resolve().parents[2] / "backend"
    resultado = subprocess.run(
        [sys.executable, "-m", "app.train_model", "--help"],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        timeout=120,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "--dest" in resultado.stdout
    assert "--seed" in resultado.stdout
    assert "--test-size" in resultado.stdout


# ---------------------------------------------------------------------------
# Flujo happy path: entrena, escribe y imprime métricas (AC)
# ---------------------------------------------------------------------------


def test_cli_entrena_y_escribe_el_artefacto(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """Dataset mockeado → artefacto en ``--dest`` (directorios creados) y cargable."""
    descargas, construcciones = cli_sin_red
    destino = tmp_path / "modelos" / "baseline.joblib"  # aún no existe
    data_dir = tmp_path / "mit-bih"

    codigo = train_model.main(["--dest", str(destino), "--data-dir", str(data_dir)])

    assert codigo == 0
    assert destino.is_file()  # save_model crea los directorios padre
    out = capsys.readouterr().out
    assert str(destino) in out
    # El dataset se construye con el --data-dir y los registros pedidos.
    assert descargas[0]["dest"] == data_dir
    assert construcciones[0]["dest"] == data_dir
    # El artefacto carga con T11a/T10 (load_model valida metadatos).
    modelo = load_model(destino)
    assert modelo.meta["random_state"] == 42
    assert modelo.meta["fs"] == FS


def test_metricas_completas_de_to_dict_por_stdout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stdout trae todas las claves de ``Metrics.to_dict`` con sus valores."""
    original = train_model.train_from_dataset
    capturado: dict[str, Metrics] = {}

    def _espia(ds: EcgDataset, **kwargs: object) -> tuple[object, Metrics]:
        modelo, metricas = original(ds, **kwargs)
        capturado["metricas"] = metricas
        return modelo, metricas

    monkeypatch.setattr(train_model, "train_from_dataset", _espia)
    destino = tmp_path / "modelo.joblib"

    codigo = train_model.main(["--dest", str(destino), "--data-dir", str(tmp_path / "d")])

    assert codigo == 0
    out = capsys.readouterr().out
    metricas = capturado["metricas"]
    assert metricas.to_dict().keys() <= set(
        line.split(":", 1)[0] for line in out.splitlines() if ":" in line
    )
    assert f"accuracy: {metricas.accuracy}" in out
    assert f"sensibilidad_macro: {metricas.sensibilidad_macro}" in out
    assert f"especificidad_macro: {metricas.especificidad_macro}" in out
    assert f"matriz_confusion: {json.dumps(metricas.to_dict()['matriz_confusion'])}" in out
    assert 0.0 <= metricas.accuracy <= 1.0
    assert f"Artefacto escrito en: {destino}" in out


def test_seed_y_test_size_se_propagan_al_entrenamiento(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """``--seed 7 --test-size 0.5`` → 30/30 latidos (3 de 6 pacientes al test)."""
    destino = tmp_path / "modelo.joblib"

    codigo = train_model.main(
        [
            "--dest",
            str(destino),
            "--data-dir",
            str(tmp_path / "d"),
            "--seed",
            "7",
            "--test-size",
            "0.5",
        ]
    )

    assert codigo == 0
    modelo = load_model(destino)
    assert modelo.meta["random_state"] == 7
    assert modelo.meta["n_muestras_train"] == 30
    assert modelo.meta["n_muestras_test"] == 30


# ---------------------------------------------------------------------------
# --dest por defecto = resolve_model_path() (AC)
# ---------------------------------------------------------------------------


def test_dest_por_defecto_lee_model_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin ``--dest``, el artefacto va a ``$MODEL_PATH`` (la misma que la API)."""
    custom = tmp_path / "custom" / "modelo.joblib"
    monkeypatch.setenv("MODEL_PATH", str(custom))

    codigo = train_model.main(["--data-dir", str(tmp_path / "mit-bih")])

    assert codigo == 0
    assert custom.is_file()
    assert str(custom) in capsys.readouterr().out


def test_dest_por_defecto_es_data_models_bajo_el_cwd(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin ``MODEL_PATH`` ni ``--dest`` → ``data/models/baseline.joblib``."""
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.chdir(tmp_path)  # las rutas relativas se resuelven en el cwd

    codigo = train_model.main([])

    assert codigo == 0
    esperado = resolve_model_path()
    assert esperado == Path("data/models/baseline.joblib")
    assert (tmp_path / esperado).is_file()
    assert f"Artefacto escrito en: {esperado}" in capsys.readouterr().out


def test_dest_explicito_tiene_prioridad_sobre_model_path(
    tmp_path: Path,
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dest`` manda sobre ``MODEL_PATH``."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "por-env.joblib"))
    destino = tmp_path / "por-arg.joblib"

    codigo = train_model.main(["--dest", str(destino), "--data-dir", str(tmp_path / "d")])

    assert codigo == 0
    assert destino.is_file()
    assert not (tmp_path / "por-env.joblib").exists()


# ---------------------------------------------------------------------------
# Idempotencia y caché (AC)
# ---------------------------------------------------------------------------


def test_re_ejecutar_sobrescribe_el_artefacto(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
) -> None:
    """Re-ejecutar sobrescribe el mismo archivo (sin duplicar artefactos)."""
    destino = tmp_path / "modelo.joblib"
    args = ["--dest", str(destino), "--data-dir", str(tmp_path / "mit-bih")]

    assert train_model.main(args) == 0
    primero = destino.read_bytes()
    capsys.readouterr()

    assert train_model.main([*args, "--seed", "99"]) == 0
    segundo = destino.read_bytes()

    assert segundo != primero  # sobrescrito (los metadatos cambian de semilla)
    assert load_model(destino).meta["random_state"] == 99
    # Un solo artefacto: nada más se creó junto a él.
    assert [path.name for path in tmp_path.iterdir()] == ["modelo.joblib"]


def test_cache_local_completa_no_re_descarga(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Con la caché completa, la descarga real de T09 no toca wfdb ni la red."""
    data_dir = tmp_path / "mit-bih"
    _escribir_cache(data_dir, ["100"])
    # build_dataset mockeado (leer el registro exigiría wfdb); la descarga
    # es la **real** de T09a: si intentara re-descargar, wfdb se importaría.
    dataset = _dataset_sintetico()

    def _no_wfdb() -> object:
        raise AssertionError("wfdb importado con caché completa: re-descarga indebida")

    monkeypatch.setattr(mitbih, "_load_wfdb", _no_wfdb)
    monkeypatch.setattr(
        train_model,
        "build_dataset",
        lambda records=None, *, dest, lead=0: dataset,
    )

    codigo = train_model.main(
        [
            "--records",
            "100",
            "--data-dir",
            str(data_dir),
            "--dest",
            str(tmp_path / "modelo.joblib"),
        ]
    )

    assert codigo == 0
    assert (tmp_path / "modelo.joblib").is_file()
    assert (data_dir / "100.hea").is_file()  # caché intacta


def test_force_purga_la_cache_antes_de_descargar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--force`` borra el trípode local de los registros pedidos (a mano)."""
    data_dir = tmp_path / "mit-bih"
    _escribir_cache(data_dir, ["100"])
    visto: dict[str, bool] = {}
    dataset = _dataset_sintetico()

    def _descarga(dest: Path, records: list[str] | None = None, **kwargs: object) -> Path:
        visto["cache_borrada"] = not mitbih.is_cached(Path(dest), "100")
        return Path(dest)

    monkeypatch.setattr(train_model, "download_mitbih", _descarga)
    monkeypatch.setattr(
        train_model,
        "build_dataset",
        lambda records=None, *, dest, lead=0: dataset,
    )

    codigo = train_model.main(
        [
            "--force",
            "--records",
            "100",
            "--data-dir",
            str(data_dir),
            "--dest",
            str(tmp_path / "modelo.joblib"),
        ]
    )

    assert codigo == 0
    assert visto["cache_borrada"] is True
    assert not (data_dir / "100.hea").exists()
    assert (tmp_path / "modelo.joblib").is_file()


# ---------------------------------------------------------------------------
# Errores → stderr claro y exit 1 (AC)
# ---------------------------------------------------------------------------


def test_descarga_fallida_exit_1_con_mensaje_y_sin_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin red ni caché → exit 1, motivo por stderr y cómo preparar los datos."""
    data_dir = tmp_path / "vacio"  # no existe / sin registros

    def _sin_red(**kwargs: object) -> Path:
        raise ConnectionError("red caída (simulada)")

    monkeypatch.setattr(train_model, "download_mitbih", _sin_red)
    destino = tmp_path / "modelo.joblib"

    codigo = train_model.main(["--dest", str(destino), "--data-dir", str(data_dir)])

    capturado = capsys.readouterr()
    assert codigo == 1
    assert "ConnectionError" in capturado.err
    assert "red caída" in capturado.err
    assert "Traceback" not in capturado.err
    # Sin caché: instrucciones para preparar los datos a mano.
    assert "physionet.org" in capturado.err
    assert str(data_dir) in capturado.err
    assert "sin" in capturado.err and "--force" in capturado.err
    # No escribe nada ni imprime métricas.
    assert not destino.exists()
    assert capturado.out == ""


def test_descarga_fallida_con_cache_local_no_pide_preparar_datos(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hay caché parcial (100 sí, 101 no) → error sin el mensaje de 'sin datos'."""
    data_dir = tmp_path / "mit-bih"
    _escribir_cache(data_dir, ["100"])

    def _sin_red(**kwargs: object) -> Path:
        raise ConnectionError("red caída (simulada)")

    monkeypatch.setattr(train_model, "download_mitbih", _sin_red)

    codigo = train_model.main(
        [
            "--records",
            "100",
            "101",
            "--data-dir",
            str(data_dir),
            "--dest",
            str(tmp_path / "modelo.joblib"),
        ]
    )

    capturado = capsys.readouterr()
    assert codigo == 1
    assert "red caída" in capturado.err
    assert "physionet.org" not in capturado.err  # no data/: no hace falta


def test_dataset_invalido_exit_1(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_dataset`` con error de datos (fs mixto) → exit 1 y motivo claro."""

    def _invalido(records: list[str] | None = None, *, dest: Path, lead: int = 0) -> EcgDataset:
        raise ValueError(
            "fs distinto entre registros: 100 tiene 360 Hz y los anteriores 100 Hz."
        )

    monkeypatch.setattr(train_model, "build_dataset", _invalido)
    destino = tmp_path / "modelo.joblib"

    codigo = train_model.main(["--dest", str(destino), "--data-dir", str(tmp_path / "d")])

    capturado = capsys.readouterr()
    assert codigo == 1
    assert "fs distinto" in capturado.err
    assert "ValueError" in capturado.err
    assert "Traceback" not in capturado.err
    assert not destino.exists()


def test_dataset_vacio_exit_1_con_mensaje(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli_sin_red: tuple[list[dict[str, object]], list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 latidos etiquetados → dataset inválido, exit 1 y pista de --records/--lead."""
    vacio = EcgDataset(
        X=np.empty((0, WIDTH), dtype=np.float32),
        y=np.asarray([], dtype=str),
        paciente_id=np.asarray([], dtype=str),
        records=("100",),
        lead=0,
        fs=FS,
        dataset_id="",
    )
    monkeypatch.setattr(train_model, "build_dataset", lambda *a, **k: vacio)

    codigo = train_model.main(
        ["--dest", str(tmp_path / "m.joblib"), "--data-dir", str(tmp_path / "d")]
    )

    capturado = capsys.readouterr()
    assert codigo == 1
    assert "Dataset vacío" in capturado.err
    assert "0 latidos" in capturado.err
    assert "--lead" in capturado.err
    assert "Traceback" not in capturado.err


def test_records_desconocidos_exit_1_sin_tocar_la_red(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID fuera de mitdb → validación fail-fast de T09a antes de crear nada."""
    data_dir = tmp_path / "mit-bih"

    def _no_wfdb() -> object:
        raise AssertionError("wfdb no debería importarse: la validación es previa")

    monkeypatch.setattr(mitbih, "_load_wfdb", _no_wfdb)

    codigo = train_model.main(
        [
            "--records",
            "999",
            "--dest",
            str(tmp_path / "modelo.joblib"),
            "--data-dir",
            str(data_dir),
        ]
    )

    capturado = capsys.readouterr()
    assert codigo == 1
    assert "999" in capturado.err
    assert "desconocidos" in capturado.err
    assert "Traceback" not in capturado.err
    assert not data_dir.exists()  # ni siquiera se creó el directorio
    assert not (tmp_path / "modelo.joblib").exists()


# ---------------------------------------------------------------------------
# Argumentos inválidos (argparse → exit 2, estándar)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argumentos",
    [
        ["--test-size", "0"],
        ["--test-size", "1.5"],
        ["--test-size", "ac"],
        ["--seed", "-1"],
        ["--records"],
    ],
)
def test_argumentos_invalidos_argparse_exit_2(argumentos: list[str]) -> None:
    """Valores fuera de rango → error de argparse (exit 2, con usage)."""
    with pytest.raises(SystemExit) as excinfo:
        train_model.main(argumentos)
    assert excinfo.value.code == 2
