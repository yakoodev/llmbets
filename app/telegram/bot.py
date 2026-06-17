"""Telegram bot (aiogram 3.x), polling mode.

Proxy is mandatory in this deployment (Telegram blocked locally) and is read
from TELEGRAM_PROXY_URL only — never hardcode it. socks5:// is supported via
aiohttp-socks.

Run:  python -m app.telegram.bot
"""
from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import Message

from app.config import settings

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("bot")


def build_bot() -> Bot:
    session = AiohttpSession(proxy=settings.telegram_proxy_url or None)
    return Bot(token=settings.telegram_bot_token, session=session)


dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "👋 CS2 prediction bot is up.\n"
        f"Your chat_id: <code>{message.chat.id}</code>\n"
        "Put this value into TELEGRAM_CHAT_ID in .env.",
        parse_mode="HTML",
    )


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    await message.answer(
        f"env: {settings.app_env}\n"
        f"polza configured: {settings.is_configured_polza}\n"
        "v1 scope: pure predictions (no odds yet)."
    )


async def main() -> None:
    if not settings.is_configured_telegram:
        log.error("TELEGRAM_BOT_TOKEN is not set — fill .env. Exiting.")
        return
    if not settings.telegram_proxy_url:
        log.warning("TELEGRAM_PROXY_URL is empty — Telegram may be unreachable here.")
    bot = build_bot()
    log.info("Starting polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
