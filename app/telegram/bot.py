"""Telegram bot (aiogram 3.x), polling mode.

Proxy is mandatory (Telegram blocked locally), read from TELEGRAM_PROXY_URL only.
Commands let you browse today's / upcoming predictions and system status.

Run:  python -m app.telegram.bot
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command
from aiogram.types import BotCommand, Message
from sqlalchemy import func, select

from app.config import settings
from app.db.models import Match, NewsEvent, NewsItem, Prediction, Team
from app.db.session import SessionLocal
from app.telegram.formatters import esc, prediction_line

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("bot")

MAX_LEN = 4000


def build_bot() -> Bot:
    session = AiohttpSession(proxy=settings.telegram_proxy_url or None)
    return Bot(token=settings.telegram_bot_token, session=session)


dp = Dispatcher()


def _authorized(message: Message) -> bool:
    cid = settings.telegram_chat_id
    if cid in ("", "replace_me"):
        return True  # not locked down yet
    return str(message.chat.id) == str(cid)


async def _reply(message: Message, text: str) -> None:
    await message.answer(text[:MAX_LEN], parse_mode="HTML")


async def _team_names(session, match) -> tuple[str, str]:
    a = await session.get(Team, match.team_a_id) if match.team_a_id else None
    b = await session.get(Team, match.team_b_id) if match.team_b_id else None
    return (a.name if a else "TBD"), (b.name if b else "TBD")


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await _reply(
        message,
        "👋 <b>CS2 prediction bot</b>\n"
        f"Твой chat_id: <code>{message.chat.id}</code>\n\n"
        "Команды: /today /upcoming /predictions /status /help",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await _reply(
        message,
        "<b>Команды</b>\n"
        "/today — прогнозы на матчи сегодня\n"
        "/upcoming — ближайшие матчи (48ч)\n"
        "/predictions — последние прогнозы\n"
        "/status — состояние системы\n"
        "/start — chat_id\n\n"
        "<i>Прогнозы и сверка результатов приходят автоматически.</i>",
    )


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    if not _authorized(message):
        return
    async with SessionLocal() as s:
        upcoming = await s.scalar(
            select(func.count()).select_from(Match).where(Match.status == "upcoming")
        )
        preds = await s.scalar(select(func.count()).select_from(Prediction))
        settled = await s.scalar(
            select(func.count()).select_from(Prediction).where(
                Prediction.was_correct.isnot(None)
            )
        )
        correct = await s.scalar(
            select(func.count()).select_from(Prediction).where(
                Prediction.was_correct.is_(True)
            )
        )
        news = await s.scalar(select(func.count()).select_from(NewsItem))
        events = await s.scalar(
            select(func.count()).select_from(NewsEvent).where(
                NewsEvent.event_type != "irrelevant"
            )
        )
    acc = f"{correct / settled * 100:.0f}%" if settled else "—"
    await _reply(
        message,
        "🩺 <b>Статус системы</b>\n"
        f"📅 Upcoming матчей: <b>{upcoming}</b>\n"
        f"🎯 Прогнозов: <b>{preds}</b> (сверено {settled}, точность {acc})\n"
        f"🗞 Новостей: <b>{news}</b> · значимых событий: <b>{events}</b>\n"
        f"⚙️ Модель: <code>{esc(settings.explain_model_tier)}</code> · горизонт {settings.prediction_horizon_hours}ч",
    )


async def _predictions_between(session, start, end):
    rows = list(
        await session.execute(
            select(Prediction, Match)
            .join(Match, Match.id == Prediction.match_id)
            .where(Match.scheduled_at >= start, Match.scheduled_at < end)
            .order_by(Match.scheduled_at.asc())
        )
    )
    lines = []
    for pred, match in rows[:25]:
        a, b = await _team_names(session, match)
        lines.append(
            prediction_line(
                a, b,
                float(pred.team_a_probability),
                float(pred.team_b_probability),
                match.scheduled_at,
                pred.risk_level,
            )
        )
    return lines


@dp.message(Command("today"))
async def cmd_today(message: Message) -> None:
    if not _authorized(message):
        return
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    async with SessionLocal() as s:
        lines = await _predictions_between(s, start, start + timedelta(days=1))
    if not lines:
        await _reply(message, "📅 На сегодня прогнозов пока нет.")
        return
    await _reply(message, f"📅 <b>Прогнозы на сегодня</b> ({len(lines)})\n\n" + "\n".join(lines))


@dp.message(Command("upcoming"))
async def cmd_upcoming(message: Message) -> None:
    if not _authorized(message):
        return
    now = datetime.now(timezone.utc)
    async with SessionLocal() as s:
        matches = list(
            await s.scalars(
                select(Match)
                .where(
                    Match.status == "upcoming",
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                    Match.scheduled_at >= now,
                    Match.scheduled_at < now + timedelta(hours=48),
                )
                .order_by(Match.scheduled_at.asc())
            )
        )
        lines = []
        for m in matches[:25]:
            a, b = await _team_names(s, m)
            pred = await s.scalar(
                select(Prediction).where(Prediction.match_id == m.id).limit(1)
            )
            when = m.scheduled_at.strftime("%d.%m %H:%M") if m.scheduled_at else "—"
            if pred:
                pa = float(pred.team_a_probability) * 100
                pb = float(pred.team_b_probability) * 100
                lines.append(f"🕒 {when} · <b>{esc(a)}</b> {pa:.0f}% — <b>{esc(b)}</b> {pb:.0f}%")
            else:
                lines.append(f"🕒 {when} · {esc(a)} vs {esc(b)} <i>(скоро)</i>")
    if not lines:
        await _reply(message, "⏳ Ближайших матчей (48ч) с составами нет.")
        return
    await _reply(message, f"⏳ <b>Ближайшие матчи (48ч)</b>\n\n" + "\n".join(lines))


@dp.message(Command("predictions"))
async def cmd_predictions(message: Message) -> None:
    if not _authorized(message):
        return
    async with SessionLocal() as s:
        rows = list(
            await s.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .order_by(Prediction.created_at.desc())
                .limit(12)
            )
        )
        lines = []
        for pred, match in rows:
            a, b = await _team_names(s, match)
            mark = ""
            if pred.was_correct is not None:
                mark = " ✅" if pred.was_correct else " ❌"
            lines.append(
                prediction_line(
                    a, b,
                    float(pred.team_a_probability),
                    float(pred.team_b_probability),
                    match.scheduled_at,
                    pred.risk_level,
                )
                + mark
            )
    if not lines:
        await _reply(message, "Прогнозов пока нет.")
        return
    await _reply(message, "🎯 <b>Последние прогнозы</b>\n\n" + "\n".join(lines))


async def _set_commands(bot: Bot) -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="today", description="Прогнозы на сегодня"),
            BotCommand(command="upcoming", description="Ближайшие матчи (48ч)"),
            BotCommand(command="predictions", description="Последние прогнозы"),
            BotCommand(command="status", description="Состояние системы"),
            BotCommand(command="help", description="Список команд"),
        ]
    )


async def main() -> None:
    if not settings.is_configured_telegram:
        log.error("TELEGRAM_BOT_TOKEN is not set — fill .env. Exiting.")
        return
    if not settings.telegram_proxy_url:
        log.warning("TELEGRAM_PROXY_URL is empty — Telegram may be unreachable here.")
    bot = build_bot()
    await _set_commands(bot)
    log.info("Starting polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
