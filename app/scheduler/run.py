"""APScheduler entrypoint — the autonomous loop.

Jobs (intervals from env): news pipeline, match schedule, results, predictions
(with Telegram notify), settle + post-mortem, and a daily Elo/roster refresh.

Run:  python -m app.scheduler.run
"""
from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.collectors.pandascore import (
    collect_results,
    collect_rosters,
    collect_upcoming,
)
from app.collectors.player_news import collect_player_news
from app.config import settings
from app.postmortem.analyzer import run_postmortems, settle_predictions
from app.postmortem.daily_review import run_daily_review
from app.prediction.elo import rebuild_ratings
from app.prediction.engine import predict_upcoming, repredict_on_critical_news
from app.processing.pipeline import run_news_pipeline

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("scheduler")


def _job(coro_fn, name: str):
    """Wrap a coroutine fn so one failure never kills the scheduler."""

    async def runner():
        log.info("job start: %s", name)
        try:
            await coro_fn()
            log.info("job ok: %s", name)
        except Exception as e:  # noqa: BLE001
            log.exception("job failed: %s: %s", name, e)

    runner.__name__ = name
    return runner


async def _settle_and_postmortem():
    await settle_predictions(notify=True)
    await run_postmortems()


async def _daily_refresh():
    await collect_rosters()
    await rebuild_ratings()


def build_scheduler() -> AsyncIOScheduler:
    sched = AsyncIOScheduler(timezone="UTC", job_defaults={"max_instances": 1, "coalesce": True})
    m = 60  # seconds per minute

    sched.add_job(
        _job(run_news_pipeline, "news_pipeline"),
        "interval",
        minutes=settings.news_collect_interval_minutes,
    )
    sched.add_job(
        _job(collect_upcoming, "match_schedule"),
        "interval",
        minutes=settings.match_schedule_interval_minutes,
    )
    sched.add_job(
        _job(collect_results, "results"),
        "interval",
        minutes=settings.result_collect_interval_minutes,
    )
    sched.add_job(
        _job(lambda: predict_upcoming(notify=True), "predictions"),
        "interval",
        minutes=settings.prediction_interval_minutes,
    )
    sched.add_job(
        _job(_settle_and_postmortem, "settle_postmortem"),
        "interval",
        minutes=15,
    )
    sched.add_job(
        _job(lambda: repredict_on_critical_news(notify=True), "repredict_critical"),
        "interval",
        minutes=20,
    )
    sched.add_job(
        _job(collect_player_news, "player_news"), "interval", hours=3
    )
    sched.add_job(_job(_daily_refresh, "daily_refresh"), "interval", hours=24)
    sched.add_job(
        _job(run_daily_review, "daily_review"),
        "cron",
        hour=settings.daily_review_hour_utc,
        minute=0,
    )
    return sched


async def main() -> None:
    sched = build_scheduler()
    sched.start()
    log.info(
        "Scheduler started: news/%dm matches/%dm results/%dm predict/%dm",
        settings.news_collect_interval_minutes,
        settings.match_schedule_interval_minutes,
        settings.result_collect_interval_minutes,
        settings.prediction_interval_minutes,
    )
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
