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
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

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
from app.odds import capture_odds
from app.prediction.elo import BASE_ELO, expected_score
from app.prediction.features import head_to_head, news_signal, odds_drift, recent_form

log = logging.getLogger("prediction.engine")
MODEL_VERSION = "elo-form-news-market-learned-v0.4"

# Blend weights (logit space): Elo is the anchor, form/news only nudge.
W_FORM = 0.8
W_NEWS = 0.6
W_H2H = 0.5
W_DRIFT = 0.4
W_STANDIN = 0.5  # penalty for a team playing with a stand-in
W_STRENGTH = 0.45  # roster player-rating diff — the model's own strength read
# A sharp bookmaker's de-vigged probability is the single strongest signal.
# Blend it in heavily so our model stops contradicting the market and losing —
# it may still deviate (value), but anchored to the market, not free-floating.
W_MARKET = 0.35  # market is a CORRECTION now, not the driver — the model's own
# Elo/strength/form analysis leads; odds still pull it (user choice 2026-06-30)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    x = min(max(x, -30.0), 30.0)
    return 1.0 / (1.0 + math.exp(-x))


# ── Learnable model: a flat logistic over these feature components. Calibration
# fits the weights on settled results (engine reads them from runtime_config);
# PRIOR ≈ the current hand-tuned blend so a small sample stays close to it. ──
FEATURE_KEYS = ("elo", "form", "news", "h2h", "drift", "standin", "market", "mappool", "strength")
PRIOR_WEIGHTS = {
    "bias": 0.0, "elo": 0.30, "form": 0.22, "news": 0.17,
    "h2h": 0.14, "drift": 0.11, "standin": 0.14, "market": 0.60,
    "mappool": 0.30, "strength": 0.30,
}


def feature_x_from_raw(p_elo, form_a, form_b, fn_a, fn_b, sig_a, sig_b,
                       h2h_a, h2h_n, drift_a, standin_a, standin_b, market_p_a,
                       mappool=0.0, strength=0.0) -> dict:
    fw = min(min(fn_a, fn_b), 10) / 10.0
    hw = min(h2h_n, 6) / 6.0
    return {
        "elo": _logit(p_elo),
        "form": (form_a - form_b) * fw,
        "news": sig_a - sig_b,
        "h2h": (h2h_a - 0.5) * hw,
        "drift": drift_a,
        "standin": -(standin_a - standin_b),
        "market": _logit(float(market_p_a)) if (market_p_a is not None and 0.0 < float(market_p_a) < 1.0) else 0.0,
        "mappool": float(mappool or 0.0),
        "strength": float(strength or 0.0),
    }


def feature_x_from_snapshot(fd: dict) -> dict:
    return feature_x_from_raw(
        fd.get("prob_elo_a", 0.5), fd.get("recent_form_a", 0.5), fd.get("recent_form_b", 0.5),
        fd.get("form_matches_a", 0), fd.get("form_matches_b", 0),
        fd.get("news_signal_a", 0.0), fd.get("news_signal_b", 0.0),
        fd.get("h2h_winrate_a", 0.5), fd.get("h2h_matches", 0), fd.get("odds_drift_a", 0.0),
        1.0 if fd.get("standin_a") else 0.0, 1.0 if fd.get("standin_b") else 0.0,
        fd.get("market_prob_a"), fd.get("mappool", 0.0), fd.get("strength", 0.0),
    )


def learned_logit(weights: dict, x: dict) -> float:
    return weights.get("bias", 0.0) + sum(weights.get(k, 0.0) * x[k] for k in FEATURE_KEYS)


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
    h2h_a, h2h_n = await head_to_head(session, match.team_a_id, match.team_b_id)
    drift_a = await odds_drift(session, match.id, match.team_a_id)
    from app.collectors.bo3_maps import map_pool_adv
    mappool = await map_pool_adv(session, match.team_a_id, match.team_b_id)
    _ta = await session.get(Team, match.team_a_id)
    _tb = await session.get(Team, match.team_b_id)
    strength = (
        float(_ta.strength) - float(_tb.strength)
        if (_ta and _tb and _ta.strength is not None and _tb.strength is not None)
        else 0.0
    )

    # signals weighted by sample size — thin data must move the number less
    form_w = min(min(fn_a, fn_b), 10) / 10.0
    h2h_w = min(h2h_n, 6) / 6.0
    standin_a = 1.0 if match.team_a_standin else 0.0
    standin_b = 1.0 if match.team_b_standin else 0.0
    logit_model = (
        _logit(p_elo)
        + W_FORM * (form_a - form_b) * form_w
        + W_NEWS * (sig_a - sig_b)
        + W_H2H * (h2h_a - 0.5) * h2h_w
        + W_DRIFT * drift_a
        - W_STANDIN * (standin_a - standin_b)  # stand-in weakens that team
        + W_STRENGTH * strength  # roster player-rating edge (team_a − team_b)
    )
    # uncertainty shrinkage: pull toward 50/50 when Elo history is thin,
    # and extra for bo1 (single-map = much higher variance)
    shrink = 0.5 + 0.5 * (min(min(mp_a, mp_b), 15) / 15.0)
    if (match.format or "") == "bo1":
        shrink *= 0.85
    logit_model *= shrink

    # Market prior: capture real bookmaker odds now and blend the de-vigged
    # market probability in heavily (W_MARKET). The sharp market is usually right;
    # this stops the model favouring teams the market has as underdogs and losing.
    odds = await capture_odds(session, match)
    market_p_a = (odds or {}).get(match.team_a_id, {}).get("implied")
    from app.runtime_config import get_config

    _lw = await get_config("learned_weights")  # weights fit on settled history
    if _lw:
        import json
        x = feature_x_from_raw(
            p_elo, form_a, form_b, fn_a, fn_b, sig_a, sig_b,
            h2h_a, h2h_n, drift_a, standin_a, standin_b, market_p_a, mappool, strength,
        )
        logit_final = learned_logit(json.loads(_lw), x)
    elif market_p_a is not None and 0.0 < float(market_p_a) < 1.0:
        _wm = await get_config("w_market")  # self-calibrated; falls back to prior
        w_market = float(_wm) if _wm else W_MARKET
        logit_final = w_market * _logit(float(market_p_a)) + (1.0 - w_market) * logit_model
    else:
        logit_final = logit_model
    pa = _sigmoid(logit_final)
    pb = 1.0 - pa
    conf, risk = _confidence(mp_a, mp_b, pa)

    team_a = await session.get(Team, match.team_a_id)
    team_b = await session.get(Team, match.team_b_id)
    news = await _gather_news(session, match.id)
    from app.postmortem.daily_review import latest_lessons

    lessons = (await latest_lessons(session)) + await _past_lessons(
        session, match.team_a_id, match.team_b_id
    )

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
        "h2h_winrate_a": round(h2h_a, 3),
        "h2h_matches": h2h_n,
        "odds_drift_a": round(drift_a, 4),
        "standin_a": bool(match.team_a_standin),
        "standin_b": bool(match.team_b_standin),
        "shrink": round(shrink, 3),
        "mappool": round(mappool, 4),  # map-pool advantage (team_a) — the map-strength edge
        "strength": round(strength, 4),  # roster-rating diff (team_a − team_b) — team-strength signal
        # pre-blend model prob, kept so refresh_odds can re-apply the market prior
        # in place once 1xBet posts a line that wasn't available at predict time
        "model_prob_a": round(_sigmoid(logit_model), 4),
        "market_prob_a": round(float(market_p_a), 4) if market_p_a else None,
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
    await session.flush()
    if odds:  # already captured above for the market prior — reuse it
        pred.fair_odds = {
            "market_team_a": odds.get(match.team_a_id, {}).get("odds"),
            "market_team_b": odds.get(match.team_b_id, {}).get("odds"),
        }
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
        "final_probabilities": {
            "team_a": features["prob_a"],
            "team_b": features["prob_b"],
        },
        # the final prob is the model blended with the bookmaker line (market is
        # the dominant driver) — so the explanation reflects WHY, not just Elo/form
        "bookmaker_implied_prob_team_a": features.get("market_prob_a"),
        "pre_market_model_prob_team_a": features.get("model_prob_a"),
        "elo": {"team_a": features["elo_a"], "team_b": features["elo_b"]},
        "matches_in_history": {
            "team_a": features["matches_played_a"],
            "team_b": features["matches_played_b"],
        },
        "recent_form_winrate": {
            "team_a": features["recent_form_a"],
            "team_b": features["recent_form_b"],
        },
        "head_to_head_winrate_team_a": features["h2h_winrate_a"],
        "h2h_matches": features["h2h_matches"],
        "market_odds_drift_team_a": features["odds_drift_a"],
        "stand_in": {"team_a": features["standin_a"], "team_b": features["standin_b"]},
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
            tier=settings.explain_model_tier,
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
        horizon = datetime.now(timezone.utc) + timedelta(
            hours=settings.prediction_horizon_hours
        )
        matches = list(
            await session.scalars(
                select(Match)
                .where(
                    Match.status == "upcoming",
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                    Match.scheduled_at <= horizon,
                )
                .order_by(Match.scheduled_at.asc().nullslast())
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
                if await send_message(text):
                    pred.notified_at = datetime.now(timezone.utc)
                    await session.commit()
            count += 1
    log.info("predict_upcoming: created %d predictions", count)
    return count


async def repredict_on_critical_news(notify: bool = True) -> int:
    """Refresh a match's prediction when a CRITICAL news item (stand-in, illness,
    roster change…) arrives AFTER the last prediction — the core "react to a
    last-minute story" behaviour. Replaces the prior unsettled prediction."""
    from app.db.models import MatchRelevanceLink, NewsItem
    from app.telegram.formatters import format_forecast
    from app.telegram.notify import send_message

    count = 0
    async with SessionLocal() as session:
        rows = list(
            await session.execute(
                select(Match, func.max(Prediction.created_at))
                .join(Prediction, Prediction.match_id == Match.id)
                .where(
                    Match.status == "upcoming",
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                )
                .group_by(Match.id)
            )
        )
        for match, last_pred_at in rows:
            fresh = await session.scalar(
                select(func.count())
                .select_from(MatchRelevanceLink)
                .join(NewsItem, NewsItem.id == MatchRelevanceLink.news_item_id)
                .where(
                    MatchRelevanceLink.match_id == match.id,
                    NewsItem.is_critical.is_(True),
                    NewsItem.created_at > last_pred_at,
                )
            )
            if not fresh:
                continue
            # drop the stale (unsettled) prediction + its snapshot, then re-run
            old = list(
                await session.scalars(
                    select(Prediction).where(
                        Prediction.match_id == match.id,
                        Prediction.was_correct.is_(None),
                    )
                )
            )
            snap_ids = [p.snapshot_id for p in old]
            for p in old:
                await session.delete(p)
            await session.flush()
            for sid in snap_ids:
                snap = await session.get(PredictionSnapshot, sid)
                if snap:
                    await session.delete(snap)
            await session.commit()

            pred = await predict_match(session, match)
            if pred and notify:
                team_a = await session.get(Team, match.team_a_id)
                team_b = await session.get(Team, match.team_b_id)
                text = (
                    "🔄 Обновлённый прогноз — свежая критичная новость\n\n"
                    + format_forecast(match, team_a, team_b, pred)
                )
                if await send_message(text):
                    pred.notified_at = datetime.now(timezone.utc)
                    await session.commit()
            count += 1
    log.info("repredict_on_critical_news: refreshed %d predictions", count)
    return count


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    args = sys.argv[1:]
    notify = "--notify" in args
    target = next((a for a in args if not a.startswith("--")), "all")
    if target == "all":
        n = await predict_upcoming(notify=notify)
        print(f"Created {n} predictions (notify={notify}).")
    elif target == "critical":
        n = await repredict_on_critical_news(notify=notify)
        print(f"Refreshed {n} predictions on critical news (notify={notify}).")
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
