"""Pruebas del dataset AAMI (X, y, paciente_id) y su serialización (T09c, #25).

Sin red, sin wfdb ni dataset real:

- ``app.mitbih_dataset.load_record`` se sustituye con ``monkeypatch``
  por un doble que devuelve ``RecordData`` sintéticos por ID (los
  argumentos ``lead``/``dest`` se registran para comprobar el paso).
- Los archivos NPZ se escriben en ``tmp_path``.

El ancho nominal de ventana se deriva de ``BEAT_WINDOW_S`` (importada
de ``app.signal``) para no duplicar el 0.4 en los tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from app import mitbih_dataset
from app.mitbih_dataset import (
    AAMI_CLASSES,
    MAP_AAMI,
    SCHEMA_VERSION,
    UNMAPPED_AAMI_CLASS,
    EcgDataset,
    build_dataset,
    load_dataset,
    save_dataset,
    to_aami,
)
from app.mitbih_load import RecordData, beat_windows
from app.signal import BEAT_WINDOW_S

FS = 360.0
HALF = round(BEAT_WINDOW_S * FS)  # 144 muestras a cada lado del R
WIDTH = 2 * HALF  # 288 muestras por latido (ancho nominal de X)
N_SAMPLES = 1000  # longitud sintética de cada señal


def _record(
    record_id: str,
    annotations: list[tuple[int, str]],
    *,
    n_samples: int = N_SAMPLES,
    fs: float = FS,
    start: float = 0.0,
) -> RecordData:
    """``RecordData`` sintético: rampa ``start + i`` con las anotaciones dadas."""
    signal = np.arange(n_samples, dtype=np.float64) + start
    return RecordData(
        record_id=record_id, signal=signal, fs=fs, annotations=list(annotations)
    )


def _install(
    monkeypatch,
    records: dict[str, RecordData],
    calls: list | None = None,
) -> None:
    """Sustituye ``mitbih_dataset.load_record`` por un doble en memoria."""

    def fake_load(record_id: str, *, lead: int = 0, dest: Path | None = None):
        if calls is not None:
            calls.append((record_id, lead, dest))
        return records[record_id]

    monkeypatch.setattr(mitbih_dataset, "load_record", fake_load)


def _sample_records() -> dict[str, RecordData]:
    """Dos registros con ventanas completas (todas dentro de la señal)."""
    return {
        "100": _record("100", [(200, "N"), (500, "A"), (800, "V")]),
        "101": _record("101", [(300, "F"), (600, "Q")], start=10_000.0),
    }


def _build_sample(monkeypatch, tmp_path) -> EcgDataset:
    _install(monkeypatch, _sample_records())
    return build_dataset(["100", "101"], dest=tmp_path)


def _rewrite_meta(path: Path, *, drop: tuple[str, ...] = (), **changes) -> None:
    """Reescribe solo la clave ``meta`` de un NPZ (para simular errores)."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    meta = json.loads(str(arrays["meta"].item()))
    for key in drop:
        meta.pop(key, None)
    meta.update(changes)
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    with path.open("wb") as fh:
        np.savez(fh, **arrays)


def _rewrite_raw_meta(path: Path, raw) -> None:
    """Sustituye la clave ``meta`` del NPZ por ``raw`` sin interpretarla."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["meta"] = np.array(raw)
    with path.open("wb") as fh:
        np.savez(fh, **arrays)


def _handmade_ds(**overrides) -> EcgDataset:
    """Dataset de 4 filas construido a mano (sin ``build_dataset``)."""
    fields: dict = {
        "X": np.zeros((4, WIDTH), dtype=np.float32),
        "y": np.array(["N"] * 4, dtype=str),
        "paciente_id": np.array(["100"] * 4, dtype=str),
        "records": ("100",),
        "lead": 0,
        "fs": FS,
    }
    fields.update(overrides)
    return EcgDataset(**fields)


# ---------------------------------------------------------------------------
# Mapeo símbolo → AAMI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol,expected",
    [
        # N — normal: N, L (BBB izquierdo), R (BBB derecho), e, j.
        ("N", "N"), ("L", "N"), ("R", "N"), ("e", "N"), ("j", "N"),
        # S — ectópico supraventricular: A, a, J, S.
        ("A", "S"), ("a", "S"), ("J", "S"), ("S", "S"),
        # V — ectópico ventricular: V, E (tardío).
        ("V", "V"), ("E", "V"),
        # F — fusión.
        ("F", "F"),
        # Q — desconocido: latidos Q (/ f Q) y no-latido real de mitdb
        # (! [ ] x | ~ + "); p/t no están en los .atr y van a Q igual.
        ("/", "Q"), ("f", "Q"), ("Q", "Q"),
        ("!", "Q"), ("[", "Q"), ("]", "Q"), ("x", "Q"), ("|", "Q"),
        ("~", "Q"), ("+", "Q"), ('"', "Q"), ("p", "Q"), ("t", "Q"),
    ],
)
def test_to_aami_mapeo_de_cada_clase(symbol: str, expected: str) -> None:
    """Mapeo completo símbolo→AAMI (unidad de anotaciones mitdb + EC57)."""
    assert to_aami(symbol) == expected


def test_to_aami_simbolo_no_mapeado_es_q() -> None:
    """Criterio explícito #25: lo desconocido → Q, nunca se descarta."""
    assert to_aami("Z") == "Q"
    assert to_aami("?") == "Q"
    assert to_aami("") == "Q"
    assert UNMAPPED_AAMI_CLASS == "Q"
    assert UNMAPPED_AAMI_CLASS in AAMI_CLASSES


def test_map_aami_cubre_exactamente_las_cinco_clases() -> None:
    """``MAP_AAMI`` solo emite clases de ``AAMI_CLASSES`` y las cubre todas."""
    assert AAMI_CLASSES == ("N", "S", "V", "F", "Q")
    assert set(MAP_AAMI.values()) == set(AAMI_CLASSES)
    # Los 5 símbolos Q-clásicos de AAMI están, además, en el propio mapa.
    for symbol in ("/", "f", "Q", "N", "V"):
        assert symbol in MAP_AAMI


# ---------------------------------------------------------------------------
# build_dataset: formas, etiquetas, paciente_id y criterios
# ---------------------------------------------------------------------------


def test_build_dataset_formas_dtype_y_alineacion(monkeypatch, tmp_path) -> None:
    """X 2-D float32 con n_latidos == len(y) == len(paciente_id)."""
    ds = _build_sample(monkeypatch, tmp_path)

    assert isinstance(ds, EcgDataset)
    assert ds.X.ndim == 2
    assert ds.X.shape == (5, WIDTH)
    assert ds.X.dtype == np.float32
    assert ds.y.shape == (5,)
    assert ds.paciente_id.shape == (5,)
    assert len(ds.y) == len(ds.paciente_id) == ds.X.shape[0]
    assert ds.n_latidos == 5
    assert ds.n_muestras == WIDTH
    assert ds.records == ("100", "101")
    assert ds.lead == 0
    assert ds.fs == FS
    assert ds.dataset_id  # identificador de contenido informado
    assert ds.created_at  # fecha de construcción informada


def test_build_dataset_valores_y_etiquetas_por_latido(monkeypatch, tmp_path) -> None:
    """``y`` con las clases AAMI y ``X`` con la ventana ±0.4 s en float32."""
    ds = _build_sample(monkeypatch, tmp_path)

    # Etiquetas en orden temporal por registro (símbolo A → clase S).
    np.testing.assert_array_equal(ds.y, ["N", "S", "V", "F", "Q"])
    # El ID del registro se repite por cada latido de ese registro.
    np.testing.assert_array_equal(ds.paciente_id, ["100"] * 3 + ["101"] * 2)
    # Cada fila es la ventana de la señal (rampa) alrededor de la anotación.
    np.testing.assert_array_equal(
        ds.X[0], np.arange(200 - HALF, 200 + HALF, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        ds.X[4], np.arange(10_000 + 600 - HALF, 10_000 + 600 + HALF, dtype=np.float32)
    )


def test_build_dataset_paciente_id_por_latido_varios_registros(
    monkeypatch, tmp_path
) -> None:
    """``paciente_id`` es un array por latido con los IDs correctos."""
    records = {
        "100": _record("100", [(200, "N"), (500, "N")]),
        "105": _record("105", [(300, "V")]),
        "202": _record("202", [(400, "S"), (700, "F"), (850, "A")]),
    }
    _install(monkeypatch, records)

    ds = build_dataset(["100", "105", "202"], dest=tmp_path)

    assert ds.n_latidos == 6
    np.testing.assert_array_equal(
        ds.paciente_id, ["100", "100", "105", "202", "202", "202"]
    )
    np.testing.assert_array_equal(ds.y, ["N", "N", "V", "S", "F", "S"])


def test_build_dataset_excluye_latido_con_ventana_recortada(
    monkeypatch, tmp_path
) -> None:
    """Criterio de ventanas: el latido del borde (incompleto) se excluye."""
    # Premisa: las ventanas de 10 y 990 quedan recortadas por los bordes.
    (s_start, s_end) = beat_windows([(10, "V")], N_SAMPLES, FS)[0]
    assert (s_end - s_start) < WIDTH
    records = {"100": _record("100", [(10, "V"), (500, "N"), (990, "L")])}
    _install(monkeypatch, records)

    ds = build_dataset(["100"], dest=tmp_path)

    # Solo sobrevive el latido central con ventana completa.
    assert ds.X.shape == (1, WIDTH)
    np.testing.assert_array_equal(ds.y, ["N"])
    np.testing.assert_array_equal(ds.paciente_id, ["100"])
    np.testing.assert_array_equal(
        ds.X[0], np.arange(500 - HALF, 500 + HALF, dtype=np.float32)
    )


def test_build_dataset_anotacion_fuera_de_senal_no_desalinea_etiquetas(
    monkeypatch, tmp_path
) -> None:
    """Una anotación sin ventana no desplaza las etiquetas (alineación)."""
    records = {
        "100": _record("100", [(5000, "V"), (500, "N"), (-200, "L"), (600, "A")])
    }
    _install(monkeypatch, records)

    ds = build_dataset(["100"], dest=tmp_path)

    # V está fuera de la señal y L es negativa: ambas sin ventana válida.
    # (el símbolo A de la última anotación es clase S según AAMI).
    assert ds.X.shape == (2, WIDTH)
    np.testing.assert_array_equal(ds.y, ["N", "S"])
    np.testing.assert_array_equal(ds.paciente_id, ["100", "100"])


def test_build_dataset_fs_distinto_error(monkeypatch, tmp_path) -> None:
    """Filas de ancho único: fs mezclado entre registros → ValueError."""
    records = {
        "100": _record("100", [(500, "N")]),
        "101": _record("101", [(500, "N")], fs=500.0),
    }
    _install(monkeypatch, records)

    with pytest.raises(ValueError, match="fs distinto"):
        build_dataset(["100", "101"], dest=tmp_path)


def test_build_dataset_records_vacio_error(monkeypatch, tmp_path) -> None:
    """``records=[]`` es error (misma regla que T09a: None = los 48)."""
    _install(monkeypatch, {})

    with pytest.raises(ValueError, match="vacío"):
        build_dataset([], dest=tmp_path)


def test_build_dataset_record_desconocido_error_antes_de_cargar(
    monkeypatch, tmp_path
) -> None:
    """IDs fuera de RECORDS → ValueError sin cargar ni descargar nada."""
    calls: list = []
    _install(monkeypatch, {}, calls)

    with pytest.raises(ValueError, match="desconocido"):
        build_dataset(["110"], dest=tmp_path)

    assert calls == []


def test_build_dataset_propaga_lead_y_dest(monkeypatch, tmp_path) -> None:
    """Cada registro se carga con el ``lead`` y ``dest`` pedidos."""
    calls: list = []
    _install(monkeypatch, {"100": _record("100", [(500, "N")])}, calls)
    dest = tmp_path / "mit-bih"

    build_dataset(["100"], dest=dest, lead=1)

    assert calls == [("100", 1, dest)]


def test_build_dataset_vacio_sin_crash(monkeypatch, tmp_path) -> None:
    """Sin latidos válidos → arrays vacíos (X con el ancho nominal), sin error."""
    records = {
        "100": _record("100", []),  # registro sin anotaciones
        "101": _record("101", [(5000, "N")]),  # anotación fuera de la señal
    }
    _install(monkeypatch, records)

    ds = build_dataset(["100", "101"], dest=tmp_path)

    assert ds.X.shape == (0, WIDTH)
    assert ds.X.dtype == np.float32
    assert ds.y.shape == (0,)
    assert ds.paciente_id.shape == (0,)
    assert ds.n_latidos == 0
    assert ds.records == ("100", "101")
    assert ds.dataset_id  # el id de contenido también se calcula


def test_dataset_id_determinista(monkeypatch, tmp_path) -> None:
    """Mismo input → mismo id; contenido distinto → id distinto."""
    _install(monkeypatch, _sample_records())
    ds1 = build_dataset(["100", "101"], dest=tmp_path)
    ds2 = build_dataset(["100", "101"], dest=tmp_path)

    assert ds1.dataset_id == ds2.dataset_id

    _install(monkeypatch, {"100": _record("100", [(200, "N"), (500, "V")])})
    ds3 = build_dataset(["100"], dest=tmp_path)

    assert ds3.dataset_id != ds1.dataset_id


def test_criterio_ancho_coherente_con_beat_windows() -> None:
    """El ancho nominal es el de ``beat_windows`` sin recortar (±0.4 s)."""
    (start, end) = beat_windows([(500, "N")], N_SAMPLES, FS)[0]
    assert end - start == mitbih_dataset._window_width(FS) == WIDTH
    # Y una ventana de borde es estrictamente más corta (por eso se excluye).
    (b_start, b_end) = beat_windows([(10, "N")], N_SAMPLES, FS)[0]
    assert (b_end - b_start) < WIDTH


# ---------------------------------------------------------------------------
# _check_alignment: tipos no-ndarray → ValueError (nunca AttributeError)
# ---------------------------------------------------------------------------


def test_check_alignment_rechaza_y_no_ndarray() -> None:
    """``y`` en lista → ValueError que nombra el campo (no AttributeError)."""
    X = np.zeros((2, WIDTH), dtype=np.float32)

    with pytest.raises(ValueError, match="y debe ser un ndarray"):
        mitbih_dataset._check_alignment(X, ["N", "N"], np.array(["100", "100"]))


def test_check_alignment_rechaza_paciente_id_no_ndarray() -> None:
    """``paciente_id`` en lista → ValueError que nombra el campo."""
    X = np.zeros((2, WIDTH), dtype=np.float32)

    with pytest.raises(ValueError, match="paciente_id debe ser un ndarray"):
        mitbih_dataset._check_alignment(X, np.array(["N", "N"]), ["100", "100"])


# ---------------------------------------------------------------------------
# _check_alignment: ramas defensivas de X y dtype de strings (O-6, #37)
# ---------------------------------------------------------------------------


def test_check_alignment_rechaza_x_no_ndarray() -> None:
    """``X`` en lista → ValueError (nunca AttributeError, O-6)."""
    with pytest.raises(ValueError, match="X debe ser un ndarray 2-D"):
        mitbih_dataset._check_alignment(
            [[0.0] * WIDTH] * 2, np.array(["N", "N"]), np.array(["100", "100"])
        )


def test_check_alignment_rechaza_x_no_2d() -> None:
    """``X`` ndarray 1-D → ValueError: el contrato exige 2 filas-dimensión."""
    X = np.zeros(WIDTH, dtype=np.float32)

    with pytest.raises(ValueError, match="X debe ser un ndarray 2-D"):
        mitbih_dataset._check_alignment(X, np.array(["N"]), np.array(["100"]))


def test_check_alignment_rechaza_x_no_float32() -> None:
    """``X`` float64 → ValueError: el dtype del contrato es float32 (O-6)."""
    X = np.zeros((2, WIDTH), dtype=np.float64)

    with pytest.raises(ValueError, match="X debe ser float32"):
        mitbih_dataset._check_alignment(
            X, np.array(["N", "N"]), np.array(["100", "100"])
        )


@pytest.mark.parametrize("campo", ["y", "paciente_id"], ids=["y", "paciente_id"])
def test_check_alignment_rechaza_dtype_no_string(campo: str) -> None:
    """``y``/``paciente_id`` numéricos → ValueError (O-6).

    Un NPZ con ``y`` de códigos enteros no debe pasar la validación de
    formas en silencio: el contrato de dtype es string.
    """
    X = np.zeros((2, WIDTH), dtype=np.float32)
    y = np.array(["N", "N"])
    paciente_id = np.array(["100", "100"])
    if campo == "y":
        y = np.array([0, 1])  # códigos enteros en vez de letras AAMI
    else:
        paciente_id = np.array([100, 101])

    with pytest.raises(ValueError, match="arrays de strings"):
        mitbih_dataset._check_alignment(X, y, paciente_id)


# ---------------------------------------------------------------------------
# save_dataset / load_dataset: NPZ con metadatos
# ---------------------------------------------------------------------------


def test_save_load_round_trip_identico(monkeypatch, tmp_path) -> None:
    """save → load devuelve un dataset ``==`` con los mismos dtypes."""
    ds = _build_sample(monkeypatch, tmp_path)

    path = save_dataset(ds, tmp_path / "dataset.npz")

    assert path.is_file()
    loaded = load_dataset(path)
    assert isinstance(loaded, EcgDataset)
    assert loaded == ds
    assert loaded.X.dtype == np.float32
    assert loaded.y.dtype == ds.y.dtype
    assert loaded.paciente_id.dtype == ds.paciente_id.dtype
    assert loaded.records == ds.records
    assert loaded.lead == ds.lead
    assert loaded.fs == ds.fs
    assert loaded.dataset_id == ds.dataset_id
    assert loaded.created_at == ds.created_at


def test_save_escribe_exactamente_en_el_path_dado(monkeypatch, tmp_path) -> None:
    """Sin sufijo ``.npz`` añadido: se escribe exactamente en ``path``."""
    ds = _build_sample(monkeypatch, tmp_path)
    plain = tmp_path / "sin-sufijo"

    saved = save_dataset(ds, plain)

    assert saved == plain
    assert plain.is_file()
    assert not (tmp_path / "sin-sufijo.npz").exists()


def test_save_incluye_metadatos_en_el_npz(monkeypatch, tmp_path) -> None:
    """Metadatos JSON: versión de schema, identificador y parámetros."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"].item()))

    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["dataset_id"] == ds.dataset_id
    assert meta["records"] == ["100", "101"]
    assert meta["lead"] == 0
    assert meta["fs"] == FS
    assert meta["created_at"] == ds.created_at
    assert meta["aami_classes"] == list(AAMI_CLASSES)


def test_load_version_de_schema_desconocida_error(monkeypatch, tmp_path) -> None:
    """Un archivo de otra versión de esquema → ValueError legible."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    _rewrite_meta(path, schema_version=SCHEMA_VERSION + 99)

    with pytest.raises(ValueError, match="schema"):
        load_dataset(path)


def test_load_contenido_alterado_error(monkeypatch, tmp_path) -> None:
    """El ``dataset_id`` revalidado detecta un NPZ manipulado."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["X"] = arrays["X"].copy()
    arrays["X"][0, 0] += 1.0  # cambia el contenido sin tocar los metadatos
    with path.open("wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(ValueError, match="dataset_id"):
        load_dataset(path)


def test_load_npz_incompleto_error(tmp_path) -> None:
    """Faltan claves obligatorias → ValueError que las nombra."""
    path = tmp_path / "roto.npz"
    with path.open("wb") as fh:
        np.savez(fh, X=np.zeros((1, WIDTH), dtype=np.float32))

    with pytest.raises(ValueError, match="faltan"):
        load_dataset(path)


def test_dataset_vacio_save_load(monkeypatch, tmp_path) -> None:
    """El dataset vacío también serializa y recupera sin crash."""
    _install(monkeypatch, {"100": _record("100", [])})
    ds = build_dataset(["100"], dest=tmp_path)

    path = save_dataset(ds, tmp_path / "vacio.npz")
    loaded = load_dataset(path)

    assert loaded == ds
    assert loaded.X.shape == (0, WIDTH)
    assert loaded.n_latidos == 0


def test_save_rechaza_dataset_desalineado(tmp_path) -> None:
    """Filas desalineadas → ValueError antes de escribir nada."""
    ds = EcgDataset(
        X=np.zeros((2, WIDTH), dtype=np.float32),
        y=np.array(["N"], dtype=str),
        paciente_id=np.array(["100", "100"], dtype=str),
        records=("100",),
        lead=0,
        fs=FS,
    )

    with pytest.raises(ValueError, match="desalineadas"):
        save_dataset(ds, tmp_path / "mal.npz")

    assert not (tmp_path / "mal.npz").exists()


def test_save_rechaza_dataset_id_desactualizado(monkeypatch, tmp_path) -> None:
    """Un dataset modificado tras ``build_dataset`` no se puede guardar."""
    ds = _build_sample(monkeypatch, tmp_path)
    ds.X[0, 0] += 1.0  # alteración posterior al cálculo del id

    with pytest.raises(ValueError, match="dataset_id"):
        save_dataset(ds, tmp_path / "stale.npz")


def test_eq_con_otro_tipo_es_falso(monkeypatch, tmp_path) -> None:
    """``==`` con algo que no es dataset no revienta (NotImplemented→False)."""
    ds = _build_sample(monkeypatch, tmp_path)

    assert (ds == "no-es-dataset") is False
    assert ds != "no-es-dataset"


# ---------------------------------------------------------------------------
# Contratos de validación: no-ndarray, dominio de y y metadatos (QA T09c)
# ---------------------------------------------------------------------------


def test_save_rechaza_y_no_ndarray(tmp_path) -> None:
    """``save_dataset`` con ``y`` en lista → ValueError (no AttributeError)."""
    ds = _handmade_ds(y=["N"] * 4)

    with pytest.raises(ValueError, match="y debe ser un ndarray"):
        save_dataset(ds, tmp_path / "lista.npz")

    assert not (tmp_path / "lista.npz").exists()


def test_save_rechaza_paciente_id_no_ndarray(tmp_path) -> None:
    """``save_dataset`` con ``paciente_id`` en lista → ValueError."""
    ds = _handmade_ds(paciente_id=["100"] * 4)

    with pytest.raises(ValueError, match="paciente_id debe ser un ndarray"):
        save_dataset(ds, tmp_path / "lista.npz")

    assert not (tmp_path / "lista.npz").exists()


def test_save_rechaza_clase_aami_invalida(tmp_path) -> None:
    """``y`` con clases fuera de ``AAMI_CLASSES`` → ValueError que las nombra."""
    ds = _handmade_ds(y=np.array(["Z", "W", "N", "N"], dtype=str))

    with pytest.raises(ValueError, match="Clases AAMI inválidas") as excinfo:
        save_dataset(ds, tmp_path / "malas.npz")

    message = str(excinfo.value)
    assert "Z" in message and "W" in message
    assert not (tmp_path / "malas.npz").exists()


@pytest.mark.parametrize("raw", ["[]", 42], ids=["lista-json", "entero"])
def test_load_meta_json_no_objeto_error(monkeypatch, tmp_path, raw) -> None:
    """JSON válido pero no-dict en ``meta`` → ValueError (no AttributeError)."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    _rewrite_raw_meta(path, raw)

    with pytest.raises(ValueError, match="objeto JSON"):
        load_dataset(path)


def test_load_meta_json_corrupto_error(monkeypatch, tmp_path) -> None:
    """JSON roto en ``meta`` → ValueError legible (Metadatos ilegibles)."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    _rewrite_raw_meta(path, '{"schema_version": ')

    with pytest.raises(ValueError, match="Metadatos ilegibles"):
        load_dataset(path)


@pytest.mark.parametrize(
    "changes",
    [{"lead": None}, {"records": 100}, {"fs": "abc"}],
    ids=["lead-null", "records-entero", "fs-cadena"],
)
def test_load_metadatos_tipos_invalidos_error(monkeypatch, tmp_path, changes) -> None:
    """``lead``/``fs``/``records`` con tipos malos → ValueError con contexto."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    _rewrite_meta(path, **changes)

    with pytest.raises(ValueError, match="Metadatos inválidos"):
        load_dataset(path)


def test_load_rechaza_clase_aami_invalida(monkeypatch, tmp_path) -> None:
    """Un NPZ con ``y`` fuera de ``AAMI_CLASSES`` no se carga en silencio."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["y"] = np.array(["Z", "W", "N", "N", "N"])
    with path.open("wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(ValueError, match="Clases AAMI inválidas") as excinfo:
        load_dataset(path)

    message = str(excinfo.value)
    assert "Z" in message and "W" in message


def test_load_rechaza_aami_classes_de_meta_alterado(monkeypatch, tmp_path) -> None:
    """``meta['aami_classes']`` alterado → ValueError (no se abre en silencio)."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    _rewrite_meta(path, aami_classes=["N", "S"])

    with pytest.raises(ValueError, match="aami_classes"):
        load_dataset(path)


def test_load_rechaza_aami_classes_ausente(monkeypatch, tmp_path) -> None:
    """``meta['aami_classes']`` ausente → ValueError (clave obligatoria)."""
    ds = _build_sample(monkeypatch, tmp_path)
    path = save_dataset(ds, tmp_path / "dataset.npz")
    _rewrite_meta(path, drop=("aami_classes",))

    with pytest.raises(ValueError, match="aami_classes"):
        load_dataset(path)
