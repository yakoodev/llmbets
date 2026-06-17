"""FastAPI entrypoint. v1: health + config sanity only; routes grow per roadmap."""
from __future__ import annotations

from fastapi import FastAPI

from app.config import settings

app = FastAPI(title=settings.app_name)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "polza_configured": settings.is_configured_polza,
        "telegram_configured": settings.is_configured_telegram,
    }
