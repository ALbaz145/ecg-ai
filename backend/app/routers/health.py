"""Router de salud (healthcheck de docker-compose)."""

from __future__ import annotations

from app.schemas import HealthResponse
from fastapi import APIRouter

router = APIRouter(tags=["salud"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck: responde ``{"status": "ok"}`` (HTTP 200)."""
    return HealthResponse(status="ok")
