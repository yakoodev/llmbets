"""APScheduler entrypoint — the autonomous loop.

Jobs (intervals from env): news pipeline, match schedule, results, predictions
(with Telegram notify), settle + post-mortem, and a daily Elo/roster refresh.

Run:  python -m app.scheduler.run
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.collectors.bo3 import collect_results, collect_upcoming
from app.collectors.pandascore import collect_rosters
from app.collectors.player_news import collect_player_news
from app.config import settings
from app.db.models import SchedulerLock
from app.db.session import SessionLocal
from app.paper import rebuild_ledger
from app.postmortem.analyzer import run_postmortems, settle_predictions
from app.postmortem.daily_review import run_daily_review
from app.prediction.elo import rebuild_ratings
from app.odds import refresh_odds_for_upcoming
from app.prediction.engine import predict_upcoming, repredict_on_critical_news
from app.processing.pipeline import run_news_pipeline
from app.telegram.outbox import drain as drain_outbox

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("scheduler")


async def _record(name: str, phase: str) -> None:
    """Persist job run state to scheduler_locks (observability — /status reads it)."""
    try:
        async with SessionLocal() as s:
            lock = await s.get(SchedulerLock, name)
            now = datetime.now(timezone.utc)
            if lock is None:
                lock = SchedulerLock(job_name=name)
                s.add(lock)
            if phase == "start":
                lock.last_started_at = now
                lock.status = "running"
            else:
                lock.last_finished_at = now
                lock.status = phase
            await s.commit()
    except Exception:  # noqa: BLE001
        pass  # bookkeeping must never break a job


def _job(coro_fn, name: str):
    """Wrap a coroutine fn so one failure never kills the scheduler."""

    async def runner():
        log.info("job start: %s", name)
        await _record(name, "start")
        try:
            await coro_fn()
            log.info("job ok: %s", name)
            await _record(name, "ok")
        except Exception as e:  # noqa: BLE001
            log.exception("job failed: %s: %s", name, e)
            await _record(name, "failed")

    runner.__name__ = name
    return runner


async def _settle_and_postmortem():
    await settle_predictions(notify=True)
    # recompute the whole ledger so balance self-heals if a result was corrected
    # (% staking compounds — a flipped historical bet shifts every later balance)
    await rebuild_ledger()
    await run_postmortems()


async def _daily_refresh():
    try:
        await collect_rosters()
    except Exception as e:  # noqa: BLE001 — rosters are enrichment; Elo must still rebuild
        log.warning("collect_rosters failed, continuing to Elo: %s", e)
    await rebuild_ratings()


def build_scheduler() -> AsyncIOScheduler:
    # misfire_grace_time: a run missed during PC sleep still fires on resume.
    # coalesce: collapse a backlog into one run. Jobs also fire ~on startup
    # (next_run_time below) so a restart/wake self-heals immediately.
    sched = AsyncIOScheduler(
        timezone="UTC",
        job_defaults={"max_instances": 1, "coalesce": True, "misfire_grace_time": 3600},
    )
    soon = datetime.now(timezone.utc)

    # (fn, name, interval-kwargs, startup-offset seconds)
    jobs = [
        (drain_outbox, "outbox", {"minutes": 3}, 5),
        (collect_results, "results", {"minutes": settings.result_collect_interval_minutes}, 10),
        (refresh_odds_for_upcoming, "odds", {"minutes": 30}, 70),
        (lambda: predict_upcoming(notify=True), "predictions", {"minutes": settings.prediction_interval_minutes}, 20),
        (_settle_and_postmortem, "settle_postmortem", {"minutes": 15}, 30),
        (collect_upcoming, "match_schedule", {"minutes": settings.match_schedule_interval_minutes}, 40),
        (lambda: repredict_on_critical_news(notify=True), "repredict_critical", {"minutes": 20}, 50),
        (run_news_pipeline, "news_pipeline", {"minutes": settings.news_collect_interval_minutes}, 60),
        (collect_player_news, "player_news", {"hours": 3}, 80),
        (_daily_refresh, "daily_refresh", {"hours": 24}, 100),
    ]
    for fn, name, interval, offset in jobs:
        sched.add_job(
            _job(fn, name),
            "interval",
            **interval,
            next_run_time=soon + timedelta(seconds=offset),
        )
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
