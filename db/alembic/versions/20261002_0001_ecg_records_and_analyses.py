"""Crea ecg_records y analyses (esquema mínimo del MVP)

Proyecto escolar, NO uso clínico. Sin datos reales de pacientes.

Tablas:
- ``ecg_records``: registros de señales ECG subidas (solo
  metadatos de la señal en el MVP).
- ``analyses``: resultados del modelo de IA, con FK en cascada
  hacia ``ecg_records``.

Revision ID: 0001
Revises:
Create Date: 2026-10-02 00:00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- ecg_records -------------------------------------------------
    op.create_table(
        "ecg_records",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("signal_metadata", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_ecg_records"),
        sa.CheckConstraint(
            "jsonb_typeof(signal_metadata) = 'object'",
            name="ck_ecg_records_signal_metadata_is_object",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(signal_metadata -> 'sample_rate_hz') = 'number'",
            name="ck_ecg_records_signal_metadata_sample_rate",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(signal_metadata -> 'n_channels') = 'number'",
            name="ck_ecg_records_signal_metadata_n_channels",
        ),
        sa.CheckConstraint(
            "(signal_metadata -> 'units') IS NULL "
            "OR jsonb_typeof(signal_metadata -> 'units') = 'string'",
            name="ck_ecg_records_signal_metadata_units",
        ),
        sa.CheckConstraint(
            "char_length(btrim(filename)) > 0",
            name="ck_ecg_records_filename_not_blank",
        ),
    )

    # --- analyses ----------------------------------------------------
    op.create_table(
        "analyses",
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("ecg_record_id", sa.Uuid(), nullable=False),
        sa.Column("anomalia", sa.Text(), nullable=False),
        sa.Column("confianza", sa.Numeric(4, 3), nullable=False),
        sa.Column("model_version", sa.Text(), nullable=False),
        sa.Column(
            "zonas_anomalas",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_analyses"),
        sa.ForeignKeyConstraint(
            ["ecg_record_id"],
            ["ecg_records.id"],
            name="fk_analyses_ecg_record_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "confianza >= 0 AND confianza <= 1",
            name="ck_analyses_confianza_range",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(zonas_anomalas) = 'array'",
            name="ck_analyses_zonas_anomalas_is_array",
        ),
        sa.CheckConstraint(
            "char_length(btrim(anomalia)) > 0",
            name="ck_analyses_anomalia_not_blank",
        ),
        sa.CheckConstraint(
            "char_length(btrim(model_version)) > 0",
            name="ck_analyses_model_version_not_blank",
        ),
    )

    # --- índices -----------------------------------------------------
    # Compuesto: cubre joins por registro (prefijo izquierdo) y el
    # listado "últimos análisis de un ECG" sin sort adicional.
    op.create_index(
        "idx_analyses_ecg_record_created",
        "analyses",
        ["ecg_record_id", sa.text("created_at DESC")],
        unique=False,
    )
    # Listado global de análisis recientes (historial futuro).
    op.create_index(
        "idx_analyses_created_at",
        "analyses",
        [sa.text("created_at DESC")],
        unique=False,
    )
    # Orden natural de subidas en ecg_records.
    op.create_index(
        "idx_ecg_records_created_at",
        "ecg_records",
        [sa.text("created_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_ecg_records_created_at", table_name="ecg_records")
    op.drop_index("idx_analyses_created_at", table_name="analyses")
    op.drop_index("idx_analyses_ecg_record_created", table_name="analyses")
    op.drop_table("analyses")
    op.drop_table("ecg_records")
