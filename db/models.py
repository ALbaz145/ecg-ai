"""Modelos SQLAlchemy del esquema mínimo del MVP de ECG-AI.

Proyecto escolar, NO uso clínico. Sin datos reales de pacientes.

Esta es la FUENTE DE VERDAD del esquema para @backend (importar desde
`db.models`). Las migraciones Alembic viven en `db/alembic/versions/` y
deben mantenerse en sincronía con estos modelos (ver docs/DB.md).

Tablas del MVP (ver docs/DB.md para el detalle completo):

- ``ecg_records``: señales ECG subidas (metadatos de la señal, no la señal
  en sí en el MVP).
- ``analyses``: resultados del modelo de IA para un registro ECG.

Evolución prevista (no incluidas en el MVP, ver docs/DB.md):
``users``, ``patients`` (siempre anónimos/ficticios) y ``audit_log``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarativa compartida por todos los modelos del proyecto."""


class EcgRecord(Base):
    """Registro de una señal ECG subida (metadatos; la señal va aparte).

    El MVP no almacena muestras en la DB: ``signal_metadata`` guarda el
    contrato de la señal (ver docs/DB.md):

    .. code-block:: json

        {"sample_rate_hz": 500, "n_channels": 1, "units": "mV"}

    Ejemplo ficticio (nunca datos reales de pacientes).
    """

    __tablename__ = "ecg_records"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    signal_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="ecg_record",
        cascade="all, delete",
        passive_deletes=True,  # la FK ya tiene ON DELETE CASCADE
        lazy="selectin",
    )

    __table_args__ = (
        CheckConstraint(
            "jsonb_typeof(signal_metadata) = 'object'",
            name="ck_ecg_records_signal_metadata_is_object",
        ),
        CheckConstraint(
            "jsonb_typeof(signal_metadata -> 'sample_rate_hz') = 'number'",
            name="ck_ecg_records_signal_metadata_sample_rate",
        ),
        CheckConstraint(
            "jsonb_typeof(signal_metadata -> 'n_channels') = 'number'",
            name="ck_ecg_records_signal_metadata_n_channels",
        ),
        CheckConstraint(
            "(signal_metadata -> 'units') IS NULL "
            "OR jsonb_typeof(signal_metadata -> 'units') = 'string'",
            name="ck_ecg_records_signal_metadata_units",
        ),
        CheckConstraint(
            "char_length(btrim(filename)) > 0",
            name="ck_ecg_records_filename_not_blank",
        ),
        Index("idx_ecg_records_created_at", "created_at", postgresql_ops={"created_at": "DESC"}),
    )


class Analysis(Base):
    """Resultado de un análisis de IA sobre un ``EcgRecord``.

    ``zonas_anomalas`` es un array JSON de segmentos anómalos (ejemplo
    ficticio)::

        [{"inicio": 120, "fin": 145, "tipo": "amplitud_baja"}]

    ``inicio``/``fin`` son **índices de muestra enteros** sobre la
    señal tal como fue subida (``inicio`` inclusivo, ``fin``
    exclusivo) y ``tipo`` es una etiqueta del modelo (p. ej.
    ``latido_irregular``, ``amplitud_baja``, ``amplitud_alta``).
    El CHECK de PostgreSQL solo exige que sea un array JSONB; la
    semántica de cada elemento la valida el backend con Pydantic.
    """

    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    ecg_record_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ecg_records.id", ondelete="CASCADE"),
        nullable=False,
    )
    anomalia: Mapped[str] = mapped_column(Text, nullable=False)
    confianza: Mapped[Decimal] = mapped_column(Numeric(4, 3), nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    zonas_anomalas: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    ecg_record: Mapped[EcgRecord] = relationship(back_populates="analyses")

    __table_args__ = (
        # Compuesto: cubre joins por registro (prefijo izquierdo) y el
        # listado "últimos análisis de un ECG" sin sort (ver docs/DB.md).
        Index(
            "idx_analyses_ecg_record_created",
            "ecg_record_id",
            sa.text("created_at DESC"),
        ),
        Index("idx_analyses_created_at", "created_at", postgresql_ops={"created_at": "DESC"}),
        CheckConstraint(
            "confianza >= 0 AND confianza <= 1",
            name="ck_analyses_confianza_range",
        ),
        CheckConstraint(
            "jsonb_typeof(zonas_anomalas) = 'array'",
            name="ck_analyses_zonas_anomalas_is_array",
        ),
        CheckConstraint(
            "char_length(btrim(anomalia)) > 0",
            name="ck_analyses_anomalia_not_blank",
        ),
        CheckConstraint(
            "char_length(btrim(model_version)) > 0",
            name="ck_analyses_model_version_not_blank",
        ),
    )
