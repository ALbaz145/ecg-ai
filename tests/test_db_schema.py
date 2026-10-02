"""Pruebas del esquema MVP (sin necesidad de DB).

Verifica que los modelos SQLAlchemy (`db/models.py`) y la
migración Alembic (`db/alembic/versions/...`) estén
sincronizados: mismas tablas, índices y restricciones.

Ejecución (desde la raíz del repo):
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import ast
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402

from db.models import Analysis, Base, EcgRecord  # noqa: E402

MIGRATION_FILE = (
    REPO_ROOT / "db" / "alembic" / "versions"
    / "20261002_0001_ecg_records_and_analyses.py"
)


def _migracion_ast() -> ast.Module:
    return ast.parse(MIGRATION_FILE.read_text(encoding="utf-8"))


def _nombres_migracion() -> dict[str, set[str]]:
    """Extrae tablas, índices y CHECK con nombre de la migración (AST)."""
    tablas: set[str] = set()
    indices: set[str] = set()
    checks: set[str] = set()
    for node in ast.walk(_migracion_ast()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr in {"create_table", "create_index"} and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                (tablas if func.attr == "create_table" else indices).add(arg0.value)
        if func.attr in {
            "CheckConstraint",
            "PrimaryKeyConstraint",
            "ForeignKeyConstraint",
        }:
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(
                    kw.value, ast.Constant
                ) and isinstance(kw.value.value, str):
                    checks.add(kw.value.value)
    return {"tablas": tablas, "indices": indices, "checks": checks}


class TestEsquemaMetadata(unittest.TestCase):
    """Verifica el metadata de los modelos."""

    def test_tablas_presentes(self) -> None:
        self.assertEqual(
            set(Base.metadata.tables),
            {"ecg_records", "analyses"},
        )

    def test_columnas_ecg_records(self) -> None:
        cols = EcgRecord.__table__.columns
        self.assertEqual(cols["id"].primary_key, True)
        self.assertEqual(cols["filename"].type.__class__, sa.Text)
        self.assertEqual(cols["signal_metadata"].type.__class__, postgresql.JSONB)
        self.assertEqual(cols["created_at"].type.timezone, True)
        self.assertEqual(cols["filename"].nullable, False)

    def test_columnas_analyses(self) -> None:
        cols = Analysis.__table__.columns
        self.assertEqual(cols["confianza"].type.__class__, sa.Numeric)
        self.assertEqual(
            (cols["confianza"].type.precision, cols["confianza"].type.scale),
            (4, 3),
        )
        self.assertEqual(cols["zonas_anomalas"].type.__class__, postgresql.JSONB)
        self.assertEqual(cols["anomalia"].nullable, False)
        self.assertEqual(cols["model_version"].nullable, False)

    def test_fk_en_cascada(self) -> None:
        fks = list(Analysis.__table__.foreign_keys)
        self.assertEqual(len(fks), 1)
        fk = fks[0]
        self.assertEqual(fk.target_fullname, "ecg_records.id")
        self.assertEqual(fk.ondelete, "CASCADE")

    def test_check_confianza_entre_0_y_1(self) -> None:
        checks = {
            c.name: str(c.sqltext) for c in Analysis.__table__.constraints
            if isinstance(c, sa.CheckConstraint)
        }
        self.assertIn("ck_analyses_confianza_range", checks)
        self.assertIn("confianza >= 0", checks["ck_analyses_confianza_range"])
        self.assertIn("confianza <= 1", checks["ck_analyses_confianza_range"])

    def test_indices_presentes(self) -> None:
        indices = {ix.name for ix in Base.metadata.tables["analyses"].indexes}
        self.assertIn("idx_analyses_ecg_record_created", indices)
        self.assertIn("idx_analyses_created_at", indices)
        indices_ecg = {ix.name for ix in Base.metadata.tables["ecg_records"].indexes}
        self.assertIn("idx_ecg_records_created_at", indices_ecg)

    def test_indices_compuestos(self) -> None:
        idx = next(
            ix
            for ix in Base.metadata.tables["analyses"].indexes
            if ix.name == "idx_analyses_ecg_record_created"
        )
        # `expressions` incluye columnas y expresiones textuales
        # (created_at DESC); `columns` solo columnas directas.
        exprs = idx.expressions
        self.assertEqual(len(exprs), 2)
        self.assertEqual(exprs[0].name, "ecg_record_id")
        self.assertIn("created_at DESC", str(exprs[1]))


class TestSincroniaModeloMigracion(unittest.TestCase):
    """La migración y los modelos deben describir lo mismo."""

    def test_tablas_iguales(self) -> None:
        self.assertEqual(
            _nombres_migracion()["tablas"],
            set(Base.metadata.tables),
        )

    def test_indices_iguales(self) -> None:
        indices_modelos = {
            ix.name for t in Base.metadata.tables.values() for ix in t.indexes
        }
        self.assertEqual(_nombres_migracion()["indices"], indices_modelos)

    def test_constraints_con_nombre_iguales(self) -> None:
        constraints_modelos = {
            c.name
            for t in Base.metadata.tables.values()
            for c in t.constraints
            if c.name is not None
        }
        # Todo constraint con nombre del modelo debe existir en la migración.
        self.assertTrue(
            constraints_modelos.issubset(_nombres_migracion()["checks"]),
            msg=(
                "Constraints del modelo ausentes en la migración: "
                f"{constraints_modelos - _nombres_migracion()['checks']}"
            ),
        )

    def test_revision_migracion_valida(self) -> None:
        """La migración declara down_revision=None (primera revisión)."""
        tree = _migracion_ast()
        down_revisions = {
            node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "down_revision"
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(down_revisions, {None})


if __name__ == "__main__":
    unittest.main()
