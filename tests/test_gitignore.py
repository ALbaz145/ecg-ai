"""Verificación de `.gitignore` para artefactos de datos (issue #37, O-5).

Los datasets y artefactos ML del proyecto (`.npz`, `.dat`…) se bajan o
regeneran aparte y **nunca se commitean**. Este test comprueba con el
propio `git check-ignore` (patrón real, sin reimplementar la lógica de
exclusión) que un NPZ queda ignorado **en cualquier ruta**.

Si `git` no está en el PATH (imágenes Docker slim), el test se **salta**
(``pytest.skip``) en vez de fallar: la verificación de patrón no es
posible sin el binario.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

#: Raíz del repo (tests/test_gitignore.py → un nivel hasta la raíz).
REPO_ROOT = Path(__file__).resolve().parents[1]


def _check_ignore(path: str) -> int:
    """``git check-ignore -q <path>`` en la raíz del repo (0 = ignorado)."""
    if shutil.which("git") is None:
        pytest.skip("git no está en el PATH (entorno Docker slim)")
    result = subprocess.run(
        ["git", "check-ignore", "-q", path],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        # 128 = error fatal (p. ej. copia del código sin .git): no hay
        # repo contra el que verificar, se salta en vez de fallar.
        pytest.skip(
            f"git check-ignore no pudo ejecutarse para {path!r}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.returncode


@pytest.mark.parametrize(
    "path",
    [
        "ds.npz",  # NPZ en la raíz (repro del issue: antes exit 1)
        "data/foo.npz",  # NPZ dentro de data/ (regresión: seguía OK)
        "sub/dir/ds.npz",  # NPZ en cualquier subcarpeta
    ],
    ids=["raiz", "dentro-de-data", "subcarpeta"],
)
def test_gitignore_ignora_npz_en_cualquier_ruta(path: str) -> None:
    """``*.npz`` está en `.gitignore` sin anclarse a una carpeta."""
    assert _check_ignore(path) == 0, f"{path} debería estar ignorado"
