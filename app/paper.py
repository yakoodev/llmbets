"""Paper-betting test balance.

No market odds from bo3, so each settled prediction is treated as a flat-stake
bet on the predicted winner at the model's FAIR odds (1/prob). Balance =
start + Σ pnl. This is a calibration test, not real-market value — swap to real
odds once a bookmaker/odds source is wired in.
"""
from __future__ import annotations

from sqlalchemy import func, select

from app.config import settings
from app.db.models import Match, PaperBet, Prediction
from app.db.session import SessionLocal
from app.odds import latest_odds


async def place_paper_bet(session, pred: Prediction) -> PaperBet | None:
    """Create the paper bet for a settled prediction (idempotent per prediction).

    With market odds: value bet on the side whose model prob beats market implied
    prob by >= min_edge, at MARKET odds. Without odds: flat bet on the model
    favourite at FAIR odds (calibration fallback)."""
    if pred.was_correct is None:
        return None
    if await session.scalar(select(PaperBet.id).where(PaperBet.prediction_id == pred.id)):
        return None

    match = await session.get(Match, pred.match_id)
    pa, pb = float(pred.team_a_probability), float(pred.team_b_probability)
    stake = settings.paper_stake
    odds_map = await latest_odds(session, pred.match_id)

    if match and match.team_a_id in odds_map and match.team_b_id in odds_map:
        oa, ob = odds_map[match.team_a_id], odds_map[match.team_b_id]
        edge_a, edge_b = pa - oa["implied"], pb - ob["implied"]
        if edge_a >= edge_b:
            selection, odds_used, edge = match.team_a_id, oa["odds"], edge_a
        else:
            selection, odds_used, edge = match.team_b_id, ob["odds"], edge_b
        if edge < settings.min_edge:
            return None  # no value vs the market → no bet
    else:
        fav_prob = min(max(max(pa, pb), 0.01), 0.99)
        odds_used = round(1.0 / fav_prob, 3)
        selection = pred.predicted_winner_team_id

    won = (match.winner_team_id == selection) if match else False
    pnl = round(stake * (odds_used - 1.0), 2) if won else -stake
    bet = PaperBet(
        prediction_id=pred.id,
        match_id=pred.match_id,
        selection_team_id=selection,
        stake=stake,
        odds=odds_used,
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
