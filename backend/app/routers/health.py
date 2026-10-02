"""Router de salud (healthcheck de docker-compose)."""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["salud"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck: responde ``{"status": "ok"}`` (HTTP 200)."""
    return HealthResponse(status="ok")
