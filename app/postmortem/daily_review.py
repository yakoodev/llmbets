"""Daily self-review — once a day, reflect over settled results: what worked,
what didn't, why → store lessons in persistent memory (daily_reviews) and feed
them into future predictions.

CLI:  python -m app.postmortem.daily_review [lookback_hours]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config import settings
from app.db.models import (
    DailyReview,
    Match,
    Postmortem,
    Prediction,
    PredictionSnapshot,
    Team,
)
from app.db.session import SessionLocal
from app.llm.client import llm
from app.llm.prompts import load_prompt, render

log = logging.getLogger("postmortem.daily_review")


async def run_daily_review(lookback_hours: int = 24, notify: bool = True):
    from app.telegram.formatters import format_daily_review
    from app.telegram.notify import send_message

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=lookback_hours)
    review_date = now.date()

    async with SessionLocal() as session:
        rows = list(
            await session.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .where(
                    Prediction.was_correct.isnot(None),
                    Prediction.settled_at >= since,
                )
                .order_by(Prediction.settled_at.asc())
            )
        )
        if not rows:
            if notify:
                await send_message(
                    "🧠 <b>Дневной разбор</b>\n"
                    "За последние сутки сыгранных матчей с прогнозами не было — "
                    "выводов пока нет."
                )
            log.info("daily_review: no settled predictions in window")
            return None

        n = len(rows)
        correct = sum(1 for p, _ in rows if p.was_correct)
        accuracy = correct / n
        avg_brier = sum(float(p.brier_score or 0) for p, _ in rows) / n

        matches_payload = []
        for pred, match in rows[:40]:
            snap = await session.get(PredictionSnapshot, pred.snapshot_id)
            pm = await session.scalar(
                select(Postmortem).where(Postmortem.prediction_id == pred.id)
            )
            team_a = await session.get(Team, match.team_a_id)
            team_b = await session.get(Team, match.team_b_id)
            feats = snap.feature_snapshot if snap else {}
            matches_payload.append(
                {
                    "match": f"{team_a.name} vs {team_b.name}",
                    "tier": match.tier,
                    "risk": pred.risk_level,
                    "prob_a": float(pred.team_a_probability),
                    "was_correct": pred.was_correct,
                    "brier": float(pred.brier_score or 0),
                    "elo_a": feats.get("elo_a"),
                    "elo_b": feats.get("elo_b"),
                    "recent_form_a": feats.get("recent_form_a"),
                    "recent_form_b": feats.get("recent_form_b"),
                    "news_signal_a": feats.get("news_signal_a"),
                    "news_signal_b": feats.get("news_signal_b"),
                    "failure_reasons": (pm.suspected_failure_reasons if pm else None),
                }
            )

        payload = {
            "date": str(review_date),
            "stats": {
                "settled": n,
                "correct": correct,
                "accuracy": round(accuracy, 3),
                "avg_brier": round(avg_brier, 4),
            },
            "matches": matches_payload,
        }
        prompt = load_prompt("daily_review")
        try:
            data = await llm.chat_json(
                "Верни только валидный JSON по схеме.",
                render(prompt["template"], input_json=payload),
                tier=settings.daily_review_model_tier,
                temperature=0.3,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("daily_review LLM failed: %s", e)
            data = {"what_worked": [], "what_failed": [], "why": [], "lessons": []}

        review = await session.scalar(
            select(DailyReview).where(DailyReview.review_date == review_date)
        )
        conclusions = {
            "what_worked": data.get("what_worked", []),
            "what_failed": data.get("what_failed", []),
            "why": data.get("why", []),
            "lessons": data.get("lessons", []),
        }
        if review is None:
            review = DailyReview(review_date=review_date)
            session.add(review)
        review.predictions_settled = n
        review.correct = correct
        review.accuracy = round(accuracy, 3)
        review.avg_brier = round(avg_brier, 4)
        review.conclusions = conclusions
        review.raw_llm_output = data
        await session.commit()

        snapshot = {
            "date": str(review_date),
            "settled": n,
            "correct": correct,
            "accuracy": accuracy,
            "avg_brier": avg_brier,
            "conclusions": conclusions,
        }
    if notify:
        await send_message(format_daily_review(snapshot))
    log.info("daily_review: %d settled, acc %.2f", n, accuracy)
    return snapshot


async def latest_lessons(session, limit: int = 6) -> list[str]:
    """Recent daily-review lessons — global self-memory for future predictions."""
    reviews = list(
        await session.scalars(
            select(DailyReview).order_by(DailyReview.review_date.desc()).limit(3)
        )
    )
    lessons: list[str] = []
    for r in reviews:
        for l in (r.conclusions or {}).get("lessons", []) or []:
            lessons.append(l)
    return lessons[:limit]


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    await run_daily_review(hours)


if __name__ == "__main__":
    asyncio.run(_main())
