"""Outbound Telegram messaging (forecasts, summaries). Proxy from env only."""
from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.telegram.bot import build_bot

log = logging.getLogger("telegram.notify")


async def _raw_send(text: str, target: str, parse_mode: str) -> bool:
    """Low-level send with retry for transient proxy blips. Does NOT enqueue
    (used by both send_message and the outbox drainer)."""
    for attempt in range(4):
        bot = build_bot()
        try:
            await bot.send_message(target, text, parse_mode=parse_mode)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Telegram send attempt %d failed: %s: %s", attempt + 1, type(e).__name__, e
            )
        finally:
            await bot.session.close()
        await asyncio.sleep(2 * (attempt + 1))
    return False


async def send_message(
    text: str, chat_id: str | None = None, parse_mode: str = "HTML"
) -> bool:
    """Send to Telegram; on failure (proxy down) queue to the outbox so a
    scheduler job redelivers it later — nothing is lost on an outage."""
    target = chat_id or settings.telegram_chat_id
    if not settings.is_configured_telegram or target in ("", "replace_me"):
        log.warning("Telegram not configured — skip send.")
        return False
    if await _raw_send(text, target, parse_mode):
        return True
    from app.db.models import Outbox
    from app.db.session import SessionLocal

    async with SessionLocal() as session:
        session.add(Outbox(text=text, parse_mode=parse_mode))
        await session.commit()
    log.error("Telegram send failed — queued to outbox for redelivery.")
    return False
