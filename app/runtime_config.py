"""Cross-process runtime config (DB-backed) — overrides set at runtime (e.g. the
/model bot command) that all containers (bot, scheduler, api) must see.
"""
from __future__ import annotations

from app.db.models import RuntimeConfig
from app.db.session import SessionLocal


async def get_config(key: str, default: str | None = None) -> str | None:
    async with SessionLocal() as session:
        row = await session.get(RuntimeConfig, key)
        return row.value if row else default


async def set_config(key: str, value: str) -> None:
    async with SessionLocal() as session:
        row = await session.get(RuntimeConfig, key)
        if row:
            row.value = value
        else:
            session.add(RuntimeConfig(key=key, value=value))
        await session.commit()
