"""Pruebas de descarga y caché MIT-BIH (T09a, issue #23; D-1 en #24).

Ninguna prueba toca la red: ``app.mitbih._load_wfdb`` se sustituye por
un doble (``FakeWfdb``) inyectado con ``monkeypatch``, de modo que la
suite tampoco necesita wfdb instalado. El dataset real **nunca** se
descarga durante los tests.

La mitigación **D-1** (issue #24) exige que la caché de un registro
incluya el ``.atr`` además del ``.hea`` y el ``.dat``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app import mitbih


class FakeWfdb:
    """Doble de ``wfdb`` sin red.

    - ``set_db_index_url(*args)`` registra cada invocación (una sin
      argumentos = restauración del índice por defecto).
    - ``dl_database(...)`` registra la llamada y simula la creación de
      archivos. Con ``error`` configurado solo escribe la cabecera
      (descarga incompleta) y luego lanza la excepción.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.index_url_calls: list[tuple] = []
        self.dl_calls: list[dict] = []

    def set_db_index_url(self, *args) -> None:
        self.index_url_calls.append(args)

    def dl_database(self, db_dir: str, dl_dir: str, **kwargs) -> None:
        self.dl_calls.append({"db_dir": db_dir, "dl_dir": dl_dir, **kwargs})
        for record in kwargs["records"]:
            # Modo fallo: solo la cabecera (descarga a medias).
            suffixes = (".hea",) if self.error is not None else mitbih._SUFFIXES
            for suffix in suffixes:
                (Path(dl_dir) / f"{record}{suffix}").write_text("x", encoding="utf-8")
        if self.error is not None:
            raise self.error


def _install_fake(monkeypatch, fake: FakeWfdb) -> dict[str, int]:
    """Inyecta el doble y devuelve el contador de cargas de wfdb."""
    loads = {"count": 0}

    def loader():
        loads["count"] += 1
        return fake

    monkeypatch.setattr(mitbih, "_load_wfdb", loader)
    return loads


# ---------------------------------------------------------------------------
# Lista canónica
# ---------------------------------------------------------------------------


def test_lista_canonica_tiene_48_ids_unicos() -> None:
    """La lista documentada tiene los 48 IDs canónicos, sin duplicados."""
    assert len(mitbih.RECORDS) == 48
    assert len(set(mitbih.RECORDS)) == 48
    assert all(isinstance(record, str) and record.isdigit() for record in mitbih.RECORDS)
    # IDs verificados en physionet.org/files/mitdb/1.0.0/RECORDS.
    assert {"100", "114", "122", "124", "217", "222", "234"} <= set(mitbih.RECORDS)
    # Estos IDs NO existen en mitdb (evita listas inventadas).
    assert not ({"110", "120", "125", "126", "204", "211"} & set(mitbih.RECORDS))


# ---------------------------------------------------------------------------
# Descarga feliz y caché
# ---------------------------------------------------------------------------


def test_descarga_ok_crea_archivos(monkeypatch, tmp_path) -> None:
    """La primera llamada invoca wfdb y deja .hea/.dat/.atr en destino."""
    fake = FakeWfdb()
    loads = _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"

    result = mitbih.download_mitbih(dest=dest)

    assert result == dest
    assert loads == {"count": 1}
    assert len(fake.dl_calls) == 1
    call = fake.dl_calls[0]
    assert call["db_dir"] == mitbih.MITBIH_DB
    assert call["dl_dir"] == str(dest)
    assert call["records"] == mitbih.RECORDS  # los 48
    assert call["annotators"] == ["atr"]
    assert call["keep_subdirs"] is False
    assert (dest / "100.hea").is_file()
    assert (dest / "100.dat").is_file()
    assert (dest / "100.atr").is_file()
    assert (dest / "234.hea").is_file() and (dest / "234.atr").is_file()


def test_segunda_llamada_usa_cache_sin_tocar_wfdb(monkeypatch, tmp_path) -> None:
    """Idempotente: la segunda llamada no llama a wfdb (cero red)."""
    fake = FakeWfdb()
    loads = _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"

    mitbih.download_mitbih(dest=dest)
    result = mitbih.download_mitbih(dest=dest)

    assert result == dest
    assert loads == {"count": 1}  # wfdb solo se cargó en la primera llamada
    assert len(fake.dl_calls) == 1
    assert fake.index_url_calls == [(mitbih.DEFAULT_BASE_URL,), ()]


def test_cache_parcial_solo_descarga_lo_faltante(monkeypatch, tmp_path) -> None:
    """Un registro ya cacheado se salta; solo se pide el pendiente."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"
    dest.mkdir()
    for suffix in mitbih._SUFFIXES:
        (dest / f"100{suffix}").write_text("x", encoding="utf-8")

    mitbih.download_mitbih(dest=dest, records=["100", "101"])

    assert len(fake.dl_calls) == 1
    assert fake.dl_calls[0]["records"] == ["101"]
    assert mitbih.is_cached(dest, "100") is True
    assert mitbih.is_cached(dest, "101") is True


def test_destino_por_defecto_es_data_mitbih(monkeypatch, tmp_path) -> None:
    """Sin argumentos descarga en data/mit-bih (relativo al cwd)."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)
    monkeypatch.chdir(tmp_path)

    result = mitbih.download_mitbih()

    assert result == Path("data/mit-bih")
    assert mitbih.DEFAULT_DEST == Path("data/mit-bih")
    assert (tmp_path / "data" / "mit-bih" / "100.hea").is_file()


def test_base_url_se_configura_y_restaura(monkeypatch, tmp_path) -> None:
    """base_url se aplica antes de descargar y se restaura al final."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)

    mitbih.download_mitbih(
        dest=tmp_path / "mit-bih",
        records=["100"],
        base_url="https://espejo.example/files/",
    )

    assert fake.index_url_calls == [("https://espejo.example/files/",), ()]


# ---------------------------------------------------------------------------
# Validación y errores
# ---------------------------------------------------------------------------


def test_records_desconocidos_sin_invocar_wfdb(monkeypatch, tmp_path) -> None:
    """IDs fuera de RECORDS → ValueError antes de tocar wfdb."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)

    with pytest.raises(ValueError, match="desconocidos"):
        mitbih.download_mitbih(dest=tmp_path / "mit-bih", records=["110"])

    assert fake.dl_calls == []


def test_records_vacio_error(monkeypatch, tmp_path) -> None:
    """Una lista vacía es un error (None = los 48)."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)

    with pytest.raises(ValueError, match="vacío"):
        mitbih.download_mitbih(dest=tmp_path / "mit-bih", records=[])

    assert fake.dl_calls == []


def test_error_de_red_excepcion_clara_y_cache_limpia(monkeypatch, tmp_path) -> None:
    """Fallo de red → ConnectionError con mensaje útil y sin descargas a medias."""
    fake = FakeWfdb(error=ConnectionError("red caída"))
    loads = _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"
    # Caché previa que debe conservarse pese al fallo.
    dest.mkdir()
    for suffix in mitbih._SUFFIXES:
        (dest / f"100{suffix}").write_text("x", encoding="utf-8")

    with pytest.raises(ConnectionError) as excinfo:
        mitbih.download_mitbih(dest=dest, records=["100", "101"])

    message = str(excinfo.value)
    assert "MIT-BIH" in message
    assert "red caída" in message  # encadena el error original
    assert excinfo.value.__cause__ is not None
    # La descarga incompleta de 101 se purga; la caché de 100 se conserva.
    assert not list(dest.glob("101.*"))
    assert mitbih.is_cached(dest, "100") is True
    # El índice de wfdb se restaura también en el camino de error.
    assert loads == {"count": 1}
    assert fake.index_url_calls[-1] == ()


def test_error_no_os_se_propaga_sin_envolver(monkeypatch, tmp_path) -> None:
    """Un bug interno de wfdb (no-OSError) no se disfraza de error de red."""
    fake = FakeWfdb(error=TypeError("bug interno"))
    _install_fake(monkeypatch, fake)

    with pytest.raises(TypeError, match="bug interno"):
        mitbih.download_mitbih(dest=tmp_path / "mit-bih", records=["101"])

    # Tampoco quedan descargas a medias del registro intentado.
    assert not list((tmp_path / "mit-bih").glob("101.*"))


# ---------------------------------------------------------------------------
# Mitigación D-1 (issue #24): la caché exige .hea + .dat + .atr
# ---------------------------------------------------------------------------


def test_d1_cache_requiere_los_tres_sufijos(tmp_path) -> None:
    """is_cached solo es True con .hea, .dat **y** .atr presentes."""
    dest = tmp_path / "mit-bih"
    dest.mkdir()
    for suffix in mitbih._SUFFIXES:
        (dest / f"100{suffix}").write_text("x", encoding="utf-8")
    assert mitbih.is_cached(dest, "100") is True

    # Falta el .atr (descarga interrumpida) → no cacheado (D-1).
    (dest / "100.atr").unlink()
    assert mitbih.is_cached(dest, "100") is False

    # Cualquier otro sufijo que falte también invalida la caché.
    (dest / "100.atr").write_text("x", encoding="utf-8")
    (dest / "100.dat").unlink()
    assert mitbih.is_cached(dest, "100") is False


def test_d1_sin_atr_se_vuelve_a_descargar(monkeypatch, tmp_path) -> None:
    """Un .hea+.dat sin .atr no está cacheado: wfdb se vuelve a invocar."""
    fake = FakeWfdb()
    _install_fake(monkeypatch, fake)
    dest = tmp_path / "mit-bih"
    dest.mkdir()
    # Simula un proceso matado tras cabecera y señal, sin anotaciones.
    (dest / "100.hea").write_text("x", encoding="utf-8")
    (dest / "100.dat").write_text("x", encoding="utf-8")

    mitbih.download_mitbih(dest=dest, records=["100", "101"])

    assert fake.dl_calls[0]["records"] == ["100", "101"]
    assert (dest / "100.atr").is_file()
    assert mitbih.is_cached(dest, "100") is True


def test_purge_incomplete_conserva_registro_completo(tmp_path) -> None:
    """O-6: un registro con .hea+.dat+.atr se conserva; el incompleto se borra."""
    dest = tmp_path / "mit-bih"
    dest.mkdir()
    for suffix in mitbih._SUFFIXES:
        (dest / f"100{suffix}").write_text("x", encoding="utf-8")
    # 101 quedó a medias: falta el .atr.
    (dest / "101.hea").write_text("x", encoding="utf-8")
    (dest / "101.dat").write_text("x", encoding="utf-8")

    mitbih._purge_incomplete(dest, ["100", "101"])

    assert mitbih.is_cached(dest, "100") is True
    assert not list(dest.glob("101.*"))
