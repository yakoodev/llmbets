"""Prediction engine — Elo baseline + news context + LLM explanation.

Per TZ: the LLM explains, it does NOT compute probabilities. Numbers come from
Elo; the LLM only narrates and flags risks, grounded in retrieved news.

CLI:
  python -m app.prediction.engine all          # predict upcoming, no send
  python -m app.prediction.engine all --notify  # predict + send to Telegram
  python -m app.prediction.engine <match_id> [--notify]
"""
from __future__ import annotations

import asyncio
import logging
import math
import sys

from sqlalchemy import select

from app.config import settings
from app.db.models import (
    Match,
    MatchRelevanceLink,
    NewsEvent,
    NewsItem,
    Postmortem,
    Prediction,
    PredictionSnapshot,
    Team,
    TeamRating,
)
from app.db.session import SessionLocal
from app.llm.client import llm
from app.llm.prompts import load_prompt, render
from app.prediction.elo import BASE_ELO, expected_score
from app.prediction.features import news_signal, recent_form

log = logging.getLogger("prediction.engine")
MODEL_VERSION = "elo-form-news-v0.2"

# Blend weights (logit space): Elo is the anchor, form/news only nudge.
W_FORM = 0.8
W_NEWS = 0.6


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _confidence(mp_a: int, mp_b: int, pa: float) -> tuple[float, str]:
    data = min(mp_a, mp_b)
    margin = abs(pa - 0.5)
    conf = min(1.0, 0.35 + 0.30 * min(data, 20) / 20 + 0.7 * margin)
    if data < 5 or conf < 0.5:
        risk = "high"
    elif conf < 0.7:
        risk = "medium"
    else:
        risk = "low"
    return round(conf, 2), risk


async def _gather_news(session, match_id) -> list[dict]:
    rows = await session.execute(
        select(NewsEvent, NewsItem)
        .join(NewsItem, NewsItem.id == NewsEvent.news_item_id)
        .join(MatchRelevanceLink, MatchRelevanceLink.news_item_id == NewsItem.id)
        .where(MatchRelevanceLink.match_id == match_id)
        .order_by(NewsEvent.importance.desc().nullslast())
        .limit(8)
    )
    out = []
    for ev, item in rows:
        out.append(
            {
                "news_item_id": str(item.id),
                "event_type": ev.event_type,
                "summary": ev.summary,
                "source_quality": ev.source_quality,
                "impact_direction": ev.prediction_impact_direction,
                "title": item.title,
                "url": item.url,
            }
        )
    return out


async def predict_match(session, match: Match) -> Prediction | None:
    if not (match.team_a_id and match.team_b_id):
        return None
    ra_row = await session.get(TeamRating, match.team_a_id)
    rb_row = await session.get(TeamRating, match.team_b_id)
    ra = float(ra_row.elo) if ra_row else BASE_ELO
    rb = float(rb_row.elo) if rb_row else BASE_ELO
    mp_a = ra_row.matches_played if ra_row else 0
    mp_b = rb_row.matches_played if rb_row else 0

    p_elo = expected_score(ra, rb)
    form_a, fn_a = await recent_form(session, match.team_a_id)
    form_b, fn_b = await recent_form(session, match.team_b_id)
    sig_a, sig_b, news_details = await news_signal(session, match)

    logit_final = (
        _logit(p_elo) + W_FORM * (form_a - form_b) + W_NEWS * (sig_a - sig_b)
    )
    pa = _sigmoid(logit_final)
    pb = 1.0 - pa
    conf, risk = _confidence(mp_a, mp_b, pa)

    team_a = await session.get(Team, match.team_a_id)
    team_b = await session.get(Team, match.team_b_id)
    news = await _gather_news(session, match.id)
    lessons = await _past_lessons(session, match.team_a_id, match.team_b_id)

    feature_snapshot = {
        "elo_a": round(ra, 1),
        "elo_b": round(rb, 1),
        "matches_played_a": mp_a,
        "matches_played_b": mp_b,
        "prob_elo_a": round(p_elo, 4),
        "recent_form_a": round(form_a, 3),
        "recent_form_b": round(form_b, 3),
        "form_matches_a": fn_a,
        "form_matches_b": fn_b,
        "news_signal_a": round(sig_a, 3),
        "news_signal_b": round(sig_b, 3),
        "news_details": news_details,
        "prob_a": round(pa, 4),
        "prob_b": round(pb, 4),
    }
    snapshot = PredictionSnapshot(
        match_id=match.id,
        model_version=MODEL_VERSION,
        prompt_versions={"prediction_explainer": "1.0.0"},
        feature_snapshot=feature_snapshot,
        retrieved_news_ids=[__import__("uuid").UUID(n["news_item_id"]) for n in news],
    )
    session.add(snapshot)
    await session.flush()

    explanation = await _explain(
        match, team_a, team_b, feature_snapshot, conf, risk, news, lessons
    )

    pred = Prediction(
        snapshot_id=snapshot.id,
        match_id=match.id,
        team_a_probability=round(pa, 4),
        team_b_probability=round(pb, 4),
        confidence=conf,
        risk_level=risk,
        predicted_winner_team_id=match.team_a_id if pa >= 0.5 else match.team_b_id,
        feature_drivers=feature_snapshot,
        explanation=explanation,
    )
    session.add(pred)
    await session.commit()
    log.info(
        "predicted %s vs %s -> %.0f%%/%.0f%% (%s)",
        team_a.name,
        team_b.name,
        pa * 100,
        pb * 100,
        risk,
    )
    return pred


async def _past_lessons(session, team_a_id, team_b_id) -> list[str]:
    """Retrieve improvement hypotheses from past post-mortems on these teams
    (RAG-lite memory: the agent leans on its own prior conclusions)."""
    rows = list(
        await session.scalars(
            select(Postmortem)
            .join(Match, Match.id == Postmortem.match_id)
            .where(
                (Match.team_a_id.in_([team_a_id, team_b_id]))
                | (Match.team_b_id.in_([team_a_id, team_b_id]))
            )
            .order_by(Postmortem.created_at.desc())
            .limit(5)
        )
    )
    lessons: list[str] = []
    for pm in rows:
        for h in pm.model_improvement_hypotheses or []:
            lessons.append(h)
    return lessons[:8]


async def _explain(match, team_a, team_b, features, conf, risk, news, lessons) -> dict:
    prompt = load_prompt("prediction_explainer")
    payload = {
        "match": {
            "team_a": team_a.name,
            "team_b": team_b.name,
            "tournament": match.tournament_name,
            "format": match.format,
            "tier": match.tier,
        },
        "model_probabilities": {
            "team_a": features["prob_a"],
            "team_b": features["prob_b"],
        },
        "elo": {"team_a": features["elo_a"], "team_b": features["elo_b"]},
        "matches_in_history": {
            "team_a": features["matches_played_a"],
            "team_b": features["matches_played_b"],
        },
        "recent_form_winrate": {
            "team_a": features["recent_form_a"],
            "team_b": features["recent_form_b"],
        },
        "news_signal": {
            "team_a": features["news_signal_a"],
            "team_b": features["news_signal_b"],
            "detail": features["news_details"],
        },
        "confidence": conf,
        "risk_level": risk,
        "retrieved_news": news,
        "past_lessons": lessons,
    }
    try:
        return await llm.chat_json(
            "Верни только валидный JSON по схеме.",
            render(prompt["template"], input_json=payload),
            tier="chat",
            temperature=0.3,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("explain failed for match %s: %s", match.id, e)
        return {"short_summary": "", "main_reasons": [], "risks": [], "data_quality_warnings": ["LLM explanation unavailable"]}


async def predict_upcoming(notify: bool = False) -> int:
    from app.telegram.notify import send_message
    from app.telegram.formatters import format_forecast

    count = 0
    async with SessionLocal() as session:
        matches = list(
            await session.scalars(
                select(Match).where(
                    Match.status == "upcoming",
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                )
            )
        )
        for m in matches:
            already = await session.scalar(
                select(Prediction.id).where(Prediction.match_id == m.id)
            )
            if already:
                continue
            pred = await predict_match(session, m)
            if pred and notify:
                team_a = await session.get(Team, m.team_a_id)
                team_b = await session.get(Team, m.team_b_id)
                text = format_forecast(m, team_a, team_b, pred)
                await send_message(text)
                pred.notified_at = __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                )
                await session.commit()
            count += 1
    log.info("predict_upcoming: created %d predictions", count)
    return count


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    args = sys.argv[1:]
    notify = "--notify" in args
    target = next((a for a in args if not a.startswith("--")), "all")
    if target == "all":
        n = await predict_upcoming(notify=notify)
        print(f"Created {n} predictions (notify={notify}).")
    else:
        async with SessionLocal() as session:
            match = await session.get(Match, __import__("uuid").UUID(target))
            if not match:
                print("Match not found.")
                return
            pred = await predict_match(session, match)
            if pred and notify:
                from app.telegram.notify import send_message
                from app.telegram.formatters import format_forecast

                team_a = await session.get(Team, match.team_a_id)
                team_b = await session.get(Team, match.team_b_id)
                await send_message(format_forecast(match, team_a, team_b, pred))
            print("Prediction created.")


if __name__ == "__main__":
    asyncio.run(_main())
