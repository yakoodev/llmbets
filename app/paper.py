"""Paper-betting test balance.

No market odds from bo3, so each settled prediction is treated as a flat-stake
bet on the predicted winner at the model's FAIR odds (1/prob). Balance =
start + Σ pnl. This is a calibration test, not real-market value — swap to real
odds once a bookmaker/odds source is wired in.
"""
from __future__ import annotations

from sqlalchemy import func, select

from app.config import settings
from app.db.models import PaperBet, Prediction
from app.db.session import SessionLocal


async def place_paper_bet(session, pred: Prediction) -> PaperBet | None:
    """Create the paper bet for a settled prediction (idempotent per prediction)."""
    if pred.was_correct is None:
        return None
    exists = await session.scalar(
        select(PaperBet.id).where(PaperBet.prediction_id == pred.id)
    )
    if exists:
        return None
    fav_prob = max(float(pred.team_a_probability), float(pred.team_b_probability))
    fav_prob = min(max(fav_prob, 0.01), 0.99)
    odds = round(1.0 / fav_prob, 3)
    stake = settings.paper_stake
    won = bool(pred.was_correct)
    pnl = round(stake * (odds - 1.0), 2) if won else -stake
    bet = PaperBet(
        prediction_id=pred.id,
        match_id=pred.match_id,
        selection_team_id=pred.predicted_winner_team_id,
        stake=stake,
        odds=odds,
        result="won" if won else "lost",
        pnl=pnl,
        settled_at=pred.settled_at,
    )
    session.add(bet)
    return bet


async def place_for_settled(session) -> int:
    """Backfill paper bets for any settled prediction that lacks one."""
    preds = list(
        await session.scalars(
            select(Prediction)
            .where(Prediction.was_correct.isnot(None))
            .where(Prediction.id.notin_(select(PaperBet.prediction_id)))
        )
    )
    n = 0
    for pred in preds:
        if await place_paper_bet(session, pred):
            n += 1
    await session.commit()
    return n


async def balance() -> dict:
    async with SessionLocal() as session:
        total = await session.scalar(select(func.count()).select_from(PaperBet)) or 0
        won = await session.scalar(
            select(func.count()).select_from(PaperBet).where(PaperBet.result == "won")
        ) or 0
        pnl = await session.scalar(select(func.coalesce(func.sum(PaperBet.pnl), 0.0))) or 0.0
        staked = total * settings.paper_stake
    pnl = float(pnl)
    return {
        "start": settings.paper_start_balance,
        "balance": round(settings.paper_start_balance + pnl, 2),
        "pnl": round(pnl, 2),
        "bets": total,
        "won": won,
        "lost": total - won,
        "stake": settings.paper_stake,
        "roi": round(pnl / staked * 100, 1) if staked else 0.0,
    }
