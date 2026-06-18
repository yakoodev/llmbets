"""Self-check + post-mortem.

settle_predictions: once a match is finished, score the prediction (correct?,
Brier) and notify the outcome.
run_postmortems: LLM diagnosis from the PRE-MATCH snapshot (no hindsight) →
postmortems table; conclusions feed future predictions as memory.

CLI:  python -m app.postmortem.analyzer
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.db.models import (
    Match,
    Postmortem,
    Prediction,
    PredictionSnapshot,
    Team,
)
from app.db.session import SessionLocal
from app.llm.client import llm
from app.llm.prompts import load_prompt, render

log = logging.getLogger("postmortem")


async def settle_predictions(notify: bool = True) -> int:
    from app.paper import place_paper_bet
    from app.telegram.formatters import format_results_summary
    from app.telegram.notify import send_message

    results: list[dict] = []
    async with SessionLocal() as session:
        rows = list(
            await session.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .where(
                    Prediction.was_correct.is_(None),
                    # A decided match HAS a winner — settle on that alone, never
                    # mind the status string (robust to any source's status set).
                    Match.winner_team_id.isnot(None),
                )
            )
        )
        if not rows:
            return 0
        if notify:
            await send_message(
                f"📊 Начинаю сверку результатов: {len(rows)} матч(ей)…"
            )
        for pred, match in rows:
            outcome_a = 1.0 if match.winner_team_id == match.team_a_id else 0.0
            pa = float(pred.team_a_probability)
            pred.brier_score = round((pa - outcome_a) ** 2, 4)
            pred.was_correct = pred.predicted_winner_team_id == match.winner_team_id
            pred.settled_at = datetime.now(timezone.utc)
            await place_paper_bet(session, pred)
            team_a = await session.get(Team, match.team_a_id)
            team_b = await session.get(Team, match.team_b_id)
            winner = await session.get(Team, match.winner_team_id)
            on_a = pred.predicted_winner_team_id == match.team_a_id
            results.append(
                {
                    "team_a": team_a.name,
                    "team_b": team_b.name,
                    "winner": winner.name,
                    "predicted": team_a.name if on_a else team_b.name,
                    "prob": float(pa if on_a else pred.team_b_probability) * 100,
                    "correct": pred.was_correct,
                    "brier": float(pred.brier_score),
                }
            )
        await session.commit()
    if notify and results:
        await send_message(format_results_summary(results))
    log.info("settle_predictions: settled %d", len(results))
    return len(results)


async def run_postmortems(limit: int = 20, notify: bool = True) -> int:
    from app.telegram.formatters import format_postmortem
    from app.telegram.notify import send_message

    prompt = load_prompt("postmortem_analyzer")
    done = 0
    async with SessionLocal() as session:
        preds = list(
            await session.scalars(
                select(Prediction)
                .where(Prediction.was_correct.isnot(None))
                .where(
                    Prediction.id.notin_(select(Postmortem.prediction_id))
                )
                .limit(limit)
            )
        )
        if not preds:
            return 0
        if notify:
            await send_message(
                f"🧠 Разбираю что было: работа над ошибками по {len(preds)} прогноз(ам)…"
            )
        for pred in preds:
            match = await session.get(Match, pred.match_id)
            snapshot = await session.get(PredictionSnapshot, pred.snapshot_id)
            team_a = await session.get(Team, match.team_a_id)
            team_b = await session.get(Team, match.team_b_id)
            winner = await session.get(Team, match.winner_team_id)
            payload = {
                "match": {"team_a": team_a.name, "team_b": team_b.name},
                "pre_match_features": snapshot.feature_snapshot if snapshot else {},
                "prediction": {
                    "team_a_probability": float(pred.team_a_probability),
                    "team_b_probability": float(pred.team_b_probability),
                    "confidence": float(pred.confidence) if pred.confidence else None,
                    "risk_level": pred.risk_level,
                    "explanation": pred.explanation,
                },
                "actual_winner": winner.name,
                "was_correct": pred.was_correct,
            }
            try:
                data = await llm.chat_json(
                    "Верни только валидный JSON по схеме.",
                    render(prompt["template"], input_json=payload),
                    tier=settings.postmortem_model_tier,
                    temperature=0.3,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("postmortem failed for %s: %s", pred.id, e)
                continue
            session.add(
                Postmortem(
                    prediction_id=pred.id,
                    match_id=match.id,
                    prediction_was_correct=pred.was_correct,
                    suspected_failure_reasons=data.get("suspected_failure_reasons"),
                    data_quality_issues=data.get("data_quality_issues"),
                    model_improvement_hypotheses=data.get(
                        "model_improvement_hypotheses"
                    ),
                    confidence_in_diagnosis=data.get("confidence_in_diagnosis"),
                    raw_llm_output=data,
                )
            )
            await session.commit()
            done += 1
            if notify:
                await send_message(
                    format_postmortem(team_a, team_b, winner, pred, data)
                )
    log.info("run_postmortems: created %d", done)
    return done


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    s = await settle_predictions()
    p = await run_postmortems()
    print(f"Settled {s}, post-mortems {p}.")


if __name__ == "__main__":
    asyncio.run(_main())
