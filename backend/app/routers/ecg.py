"""Router de ECG: análisis de señales y consulta de resultados."""

from __future__ import annotations

import uuid
from typing import Literal

from app.pipeline import analyze_result, parse_csv_values, run_analysis
from app.repository import EcgRepository, get_repository
from app.schemas import (
    MAX_SAMPLE_RATE_HZ,
    MIN_SAMPLE_RATE_HZ,
    AnalysisResponse,
    AnalyzeRequest,
)
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

#: Tamaño máximo de archivo CSV aceptado (25 MB).
MAX_CSV_BYTES = 25_000_000

router = APIRouter(tags=["ecg"])


@router.post("/ecg/analyze", response_model=AnalysisResponse)
def analyze_ecg(
    payload: AnalyzeRequest,
    repository: EcgRepository = Depends(get_repository),
) -> AnalysisResponse:
    """Analiza una señal ECG enviada como JSON.

    Ejemplo de petición:

    ```json
    {"signal": [0.1, 0.2, ...], "sample_rate_hz": 500, "units": "mV"}
    ```
    """
    stored = run_analysis(
        signal=payload.signal,
        sample_rate_hz=payload.sample_rate_hz,
        units=payload.units,
        filename=payload.filename,
        repository=repository,
    )
    return AnalysisResponse(**analyze_result(stored))


@router.post("/ecg/analyze/file", response_model=AnalysisResponse)
async def analyze_ecg_file(
    file: UploadFile = File(
        description="CSV con una columna de valores (cabecera opcional)."
    ),
    sample_rate_hz: int = Form(
        ge=MIN_SAMPLE_RATE_HZ, le=MAX_SAMPLE_RATE_HZ
    ),
    units: Literal["mV", "uV"] = Form(default="mV"),
    repository: EcgRepository = Depends(get_repository),
) -> AnalysisResponse:
    """Analiza un ECG subido como archivo CSV (multipart).

    El CSV debe tener **una columna** de valores numéricos;
    se acepta una fila de cabecera (se ignora). La
    validación es la misma que ``POST /ecg/analyze``.
    """
    content = await file.read()
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera {MAX_CSV_BYTES} bytes.",
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail="El archivo debe ser UTF-8.",
        ) from exc

    values = parse_csv_values(text)
    try:
        payload = AnalyzeRequest(
            signal=values,
            sample_rate_hz=sample_rate_hz,
            units=units,
            filename=file.filename,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=jsonable_encoder(exc.errors()),
        ) from exc

    stored = run_analysis(
        signal=payload.signal,
        sample_rate_hz=payload.sample_rate_hz,
        units=payload.units,
        filename=payload.filename,
        repository=repository,
    )
    return AnalysisResponse(**analyze_result(stored))


@router.get("/ecg/{record_id}", response_model=AnalysisResponse)
def get_ecg(
    record_id: str,
    repository: EcgRepository = Depends(get_repository),
) -> AnalysisResponse:
    """Devuelve un análisis almacenado.

    ``record_id`` puede ser el **id del análisis** (el campo
    ``id`` de la respuesta) o el **id del registro ECG**
    (``ecg_record_id``; en ese caso devuelve el análisis
    más reciente de ese registro). 404 si no existe.
    """
    try:
        uid = uuid.UUID(record_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail="Análisis no encontrado.",
        ) from exc

    stored = repository.get_analysis(uid)
    if stored is None:
        stored = repository.get_latest_analysis_for_record(uid)
    if stored is None:
        raise HTTPException(
            status_code=404,
            detail="Análisis no encontrado.",
        )
    return AnalysisResponse(**analyze_result(stored))
