"""Outbound Telegram messaging (forecasts, summaries). Proxy from env only."""
from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.telegram.bot import build_bot

log = logging.getLogger("telegram.notify")


async def send_message(
    text: str, chat_id: str | None = None, parse_mode: str = "HTML"
) -> bool:
    """Send to Telegram, retrying transient proxy/network blips so a flaky
    proxy doesn't silently drop a forecast/result."""
    target = chat_id or settings.telegram_chat_id
    if not settings.is_configured_telegram or target in ("", "replace_me"):
        log.warning("Telegram not configured — skip send.")
        return False
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
    log.error("Telegram send failed after retries — message dropped.")
    return False
