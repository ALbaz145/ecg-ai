"""Verifica que .gitignore protege los datasets NPZ en cualquier ruta (#37, O-5)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="requiere git y un checkout con .git",
)
@pytest.mark.parametrize("ruta", ["ds.npz", "backend/ds.npz", "data/ds.npz"])
def test_gitignore_ignora_npz_en_cualquier_ruta(ruta: str) -> None:
    """``git check-ignore`` devuelve 0 si la ruta está ignorada."""
    resultado = subprocess.run(
        ["git", "check-ignore", "-q", ruta],
        cwd=ROOT,
        check=False,
    )

    assert resultado.returncode == 0, f"{ruta} no está ignorada por .gitignore"
