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
from app.db.models import (
    Match,
    NewsEvent,
    NewsItem,
    Prediction,
    SchedulerLock,
    Team,
    TeamRating,
)
from app.db.session import SessionLocal
from app.llm.client import llm
from app.runtime_config import get_config, set_config
from app.telegram.formatters import (
    esc,
    format_prediction_list,
    format_results_summary,
    team_name,
)

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
        "🤖 <b>Команды</b>\n\n"
        "📅 /today — прогнозы на сегодня (МСК)\n"
        "📅 /yesterday — прогнозы за вчера\n"
        "📅 /tomorrow — прогнозы на завтра\n"
        "⏳ /upcoming — ближайшие 48 часов\n"
        "🎯 /predictions — последние прогнозы\n"
        "📊 /results — итоги сыгранных матчей\n"
        "📈 /accuracy — точность модели\n"
        "💰 /balance — тестовый баланс\n"
        "🏅 /top — топ команд по Elo\n"
        "🩺 /status — состояние системы\n"
        "🧠 /model — показать/сменить модель LLM\n"
        "🆔 /start — chat_id\n\n"
        "<i>Время — МСК. Прогнозы и сверка результатов приходят автоматически. "
        "🏆 — предсказанный победитель.</i>",
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
        locks = list(await s.scalars(select(SchedulerLock)))
    acc = f"{correct / settled * 100:.0f}%" if settled else "—"

    now = datetime.now(timezone.utc)

    def _ago(dt) -> str:
        if not dt:
            return "не было"
        mins = (now - dt).total_seconds() / 60
        return f"{mins:.0f}м назад" if mins < 120 else f"{mins / 60:.1f}ч назад"

    last_any = max((lk.last_finished_at for lk in locks if lk.last_finished_at), default=None)
    if last_any and (now - last_any).total_seconds() < 1200:
        health = f"🟢 активен ({_ago(last_any)})"
    elif last_any:
        health = f"🔴 молчит ({_ago(last_any)})"
    else:
        health = "🔴 ещё не отмечался"
    jobs_txt = "\n".join(
        f"• {esc(lk.job_name)}: {_ago(lk.last_finished_at)}"
        for lk in sorted(locks, key=lambda x: x.job_name)
    ) or "—"

    await _reply(
        message,
        "🩺 <b>Статус системы</b>\n"
        f"📅 Upcoming матчей: <b>{upcoming}</b>\n"
        f"🎯 Прогнозов: <b>{preds}</b> (сверено {settled}, точность {acc})\n"
        f"🗞 Новостей: <b>{news}</b> · значимых событий: <b>{events}</b>\n"
        f"⚙️ Модель: <code>{esc(settings.explain_model_tier)}</code> · горизонт {settings.prediction_horizon_hours}ч\n"
        f"⏱ Планировщик: {health}\n"
        f"<blockquote expandable>{jobs_txt}</blockquote>",
    )


@dp.message(Command("model"))
async def cmd_model(message: Message) -> None:
    if not _authorized(message):
        return
    parts = (message.text or "").split()
    cur_chat = await get_config("chat_model", settings.polza_chat_model)
    cur_fast = await get_config("fast_model", settings.polza_fast_model)

    if len(parts) == 1:
        await _reply(
            message,
            "🧠 <b>Модели LLM</b>\n"
            f"chat (основная): <code>{esc(cur_chat)}</code>\n"
            f"fast (прогнозы/новости): <code>{esc(cur_fast)}</code>\n\n"
            "Сменить основную: <code>/model openai/gpt-5.5</code>\n"
            "Сменить fast: <code>/model fast openai/gpt-4.1-mini</code>\n\n"
            f"<i>Объяснения прогнозов сейчас идут на tier «{esc(settings.explain_model_tier)}» "
            f"(= {esc(cur_fast if settings.explain_model_tier == 'fast' else cur_chat)}).</i>",
        )
        return

    if parts[1].lower() == "fast" and len(parts) >= 3:
        key, new_model, label = "fast_model", parts[2].strip(), "fast"
    else:
        key, new_model, label = "chat_model", parts[1].strip(), "основная (chat)"

    try:
        models = await llm.list_models()
    except Exception:  # noqa: BLE001
        models = []
    if models and new_model not in models:
        await _reply(
            message,
            f"❌ Модель <code>{esc(new_model)}</code> не найдена у провайдера.\n"
            "Пример корректного id: <code>openai/gpt-5.5</code>",
        )
        return

    await set_config(key, new_model)
    await _reply(
        message,
        f"✅ Модель <b>{label}</b>: <code>{esc(new_model)}</code>\n"
        "Применится ко всем новым вызовам (бот + планировщик).",
    )


def _msk_day_bounds(offset: int = 0):
    """UTC bounds for the MSK calendar day `offset` days from today."""
    now = datetime.now(timezone.utc)
    msk_mid = (now + timedelta(hours=3)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = msk_mid - timedelta(hours=3) + timedelta(days=offset)
    return start, start + timedelta(days=1)


_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)


async def _items(session, rows) -> list[dict]:
    # one row per MATCH — keep the latest prediction (a match can carry >1:
    # re-predict on news, or a transient race) so the list never shows duplicates
    latest: dict = {}
    for pred, match in rows:
        cur = latest.get(match.id)
        if cur is None or (pred.created_at or _MIN_DT) > (cur[0].created_at or _MIN_DT):
            latest[match.id] = (pred, match)
    ordered = sorted(
        latest.values(),
        key=lambda pm: pm[1].scheduled_at or _MIN_DT,
    )
    items = []
    for pred, match in ordered:
        a, b = await _team_names(session, match)
        odds = pred.fair_odds or {}
        items.append(
            {
                "a": a,
                "b": b,
                "pa": float(pred.team_a_probability),
                "pb": float(pred.team_b_probability),
                "when": match.scheduled_at,
                "risk": pred.risk_level,
                "settled": pred.was_correct,
                "odds_a": odds.get("market_team_a"),
                "odds_b": odds.get("market_team_b"),
            }
        )
    return items


def _between(start, end):
    return (
        select(Prediction, Match)
        .join(Match, Match.id == Prediction.match_id)
        .where(Match.scheduled_at >= start, Match.scheduled_at < end)
        .order_by(Match.scheduled_at.asc())
    )


@dp.message(Command("today"))
async def cmd_today(message: Message) -> None:
    if not _authorized(message):
        return
    start, end = _msk_day_bounds(0)
    async with SessionLocal() as s:
        items = await _items(s, list(await s.execute(_between(start, end))))
    await _reply(
        message,
        format_prediction_list("📅 Прогнозы на сегодня (МСК)", items, "На сегодня прогнозов нет."),
    )


@dp.message(Command("tomorrow"))
async def cmd_tomorrow(message: Message) -> None:
    if not _authorized(message):
        return
    start, end = _msk_day_bounds(1)
    async with SessionLocal() as s:
        items = await _items(s, list(await s.execute(_between(start, end))))
    await _reply(
        message,
        format_prediction_list("📅 Прогнозы на завтра (МСК)", items, "На завтра прогнозов пока нет."),
    )


@dp.message(Command("yesterday"))
async def cmd_yesterday(message: Message) -> None:
    if not _authorized(message):
        return
    start, end = _msk_day_bounds(-1)
    async with SessionLocal() as s:
        items = await _items(s, list(await s.execute(_between(start, end))))
    await _reply(
        message,
        format_prediction_list("📅 Прогнозы за вчера (МСК)", items, "За вчера прогнозов нет."),
    )


@dp.message(Command("upcoming"))
async def cmd_upcoming(message: Message) -> None:
    if not _authorized(message):
        return
    now = datetime.now(timezone.utc)
    async with SessionLocal() as s:
        rows = list(
            await s.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .where(
                    Match.status == "upcoming",
                    Match.scheduled_at >= now,
                    Match.scheduled_at < now + timedelta(hours=48),
                )
                .order_by(Match.scheduled_at.asc())
            )
        )
        items = await _items(s, rows)
    await _reply(
        message,
        format_prediction_list("⏳ Ближайшие прогнозы (48ч, МСК)", items, "Ближайших спрогнозированных матчей нет."),
    )


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
                .limit(15)
            )
        )
        items = await _items(s, rows)
    await _reply(message, format_prediction_list("🎯 Последние прогнозы", items, "Прогнозов пока нет."))


@dp.message(Command("results"))
async def cmd_results(message: Message) -> None:
    if not _authorized(message):
        return
    async with SessionLocal() as s:
        rows = list(
            await s.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .where(Prediction.was_correct.isnot(None))
                .order_by(Prediction.settled_at.desc())
                .limit(12)
            )
        )
        results = []
        for pred, match in rows:
            a, b = await _team_names(s, match)
            winner = (
                await s.get(Team, match.winner_team_id)
                if match.winner_team_id
                else None
            )
            on_a = pred.predicted_winner_team_id == match.team_a_id
            results.append(
                {
                    "team_a": a,
                    "team_b": b,
                    "winner": winner.name if winner else "—",
                    "predicted": a if on_a else b,
                    "prob": float(pred.team_a_probability if on_a else pred.team_b_probability) * 100,
                    "correct": pred.was_correct,
                    "brier": float(pred.brier_score or 0),
                }
            )
    if not results:
        await _reply(message, "📊 Сыгранных прогнозов пока нет.")
        return
    await _reply(message, format_results_summary(results))


@dp.message(Command("accuracy"))
async def cmd_accuracy(message: Message) -> None:
    if not _authorized(message):
        return
    async with SessionLocal() as s:
        total = await s.scalar(select(func.count()).select_from(Prediction))
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
        avg_brier = await s.scalar(
            select(func.avg(Prediction.brier_score)).where(
                Prediction.brier_score.isnot(None)
            )
        )
    if not settled:
        await _reply(
            message,
            "📈 Сыгранных прогнозов пока нет — точность появится после первых матчей.",
        )
        return
    await _reply(
        message,
        "📈 <b>Точность модели</b>\n"
        f"Всего прогнозов: <b>{total}</b>\n"
        f"Сыграно: <b>{settled}</b>\n"
        f"Угадано: <b>{correct}</b> (<b>{correct / settled * 100:.0f}%</b>)\n"
        f"Средний Brier: <b>{float(avg_brier or 0):.3f}</b> <i>(меньше — лучше)</i>",
    )


@dp.message(Command("balance"))
async def cmd_balance(message: Message) -> None:
    if not _authorized(message):
        return
    from app.paper import balance
    from app.telegram.formatters import format_balance

    await _reply(message, format_balance(await balance()))


@dp.message(Command("balances"))
async def cmd_balances(message: Message) -> None:
    if not _authorized(message):
        return
    from app.paper import strategy_balances
    from app.telegram.formatters import esc

    rows = await strategy_balances()
    if not rows:
        await _reply(message, "💰 Стратегии ещё не считались.")
        return
    lines = ["💰 <b>Балансы по тактикам</b> (старт 1000)"]
    for r in rows:
        arrow = "🟢" if r["pnl"] >= 0 else "🔴"
        lines.append(
            f"{arrow} <b>{esc(r['strategy'])}</b> — <b>{r['balance']:.0f}</b> "
            f"(P&L {r['pnl']:+.0f}, ROI {r['roi']:+.1f}%, {r['won']}/{r['bets']}✓)\n"
            f"   <i>{esc(r['desc'])}</i>"
        )
    await _reply(message, "\n".join(lines))


@dp.message(Command("top"))
async def cmd_top(message: Message) -> None:
    if not _authorized(message):
        return
    async with SessionLocal() as s:
        rows = list(
            await s.execute(
                select(TeamRating, Team)
                .join(Team, Team.id == TeamRating.team_id)
                .order_by(TeamRating.elo.desc())
                .limit(15)
            )
        )
    if not rows:
        await _reply(message, "🏅 Рейтинги ещё не посчитаны.")
        return
    lines = ["🏅 <b>Топ команд по Elo</b>"]
    for i, (r, t) in enumerate(rows, 1):
        lines.append(
            f"{i}. <b>{team_name(t.name)}</b> — {float(r.elo):.0f} <i>({r.matches_played} м.)</i>"
        )
    await _reply(message, "\n".join(lines))


async def _set_commands(bot: Bot) -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="today", description="Прогнозы на сегодня (МСК)"),
            BotCommand(command="yesterday", description="Прогнозы за вчера (МСК)"),
            BotCommand(command="tomorrow", description="Прогнозы на завтра (МСК)"),
            BotCommand(command="upcoming", description="Ближайшие 48ч"),
            BotCommand(command="predictions", description="Последние прогнозы"),
            BotCommand(command="results", description="Итоги сыгранных"),
            BotCommand(command="accuracy", description="Точность модели"),
            BotCommand(command="balance", description="Тестовый баланс"),
            BotCommand(command="top", description="Топ команд по Elo"),
            BotCommand(command="status", description="Состояние системы"),
            BotCommand(command="model", description="Сменить модель LLM"),
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
