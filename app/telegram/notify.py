"""Outbound Telegram messaging (forecasts, summaries). Proxy from env only."""
from __future__ import annotations

import logging

from app.config import settings
from app.telegram.bot import build_bot

log = logging.getLogger("telegram.notify")


async def send_message(
    text: str, chat_id: str | None = None, parse_mode: str = "HTML"
) -> bool:
    target = chat_id or settings.telegram_chat_id
    if not settings.is_configured_telegram or target in ("", "replace_me"):
        log.warning("Telegram not configured — skip send.")
        return False
    bot = build_bot()
    try:
        await bot.send_message(target, text, parse_mode=parse_mode)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Telegram send failed: %s: %s", type(e).__name__, e)
        return False
    finally:
        await bot.session.close()
