"""Paper-betting test balance — % of bank staking.

Stake = paper_stake_pct of the CURRENT balance (compounds). We bet only when
the model has VALUE vs the market (model prob − implied prob >= min_edge) at
market odds; without odds we fall back to flat fair-odds on the favourite.
Balance = start + Σ pnl (each pnl already computed from the running stake, so
the sum telescopes correctly). Process bets in chronological (settle) order.
"""
from __future__ import annotations

from sqlalchemy import delete, func, select

from app.config import settings
from app.db.models import Match, PaperBet, Prediction
from app.db.session import SessionLocal
from app.odds import latest_odds


async def _current_balance(session) -> float:
    pnl = await session.scalar(select(func.coalesce(func.sum(PaperBet.pnl), 0.0))) or 0.0
    return settings.paper_start_balance + float(pnl)


async def place_paper_bet(session, pred: Prediction) -> PaperBet | None:
    """Place the paper bet for a settled prediction (idempotent per prediction).

    We bet on OUR predicted winner; the bookmaker odds only size the payout (use
    the market odds for our pick if available, else the model's fair odds). So a
    correct prediction = winning bet. Stake = % of current balance (compounds;
    relies on autoflush so prior bets in this session count in the balance)."""
    if pred.was_correct is None:
        return None
    if await session.scalar(select(PaperBet.id).where(PaperBet.prediction_id == pred.id)):
        return None

    match = await session.get(Match, pred.match_id)
    selection = pred.predicted_winner_team_id  # bet on our predicted winner
    odds_map = await latest_odds(session, pred.match_id)

    if selection and selection in odds_map:
        odds_used = odds_map[selection]["odds"]  # market odds for our pick
    else:
        sel_prob = 0.5
        if match:
            sel_prob = float(
                pred.team_a_probability
                if selection == match.team_a_id
                else pred.team_b_probability
            )
        odds_used = round(1.0 / min(max(sel_prob, 0.01), 0.99), 3)

    balance = await _current_balance(session)
    stake = round(settings.paper_stake_pct * balance, 2)
    if stake <= 0:
        return None

    won = bool(pred.was_correct)  # selection IS the predicted winner
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


async def rebuild_ledger() -> int:
    """Wipe and recompute every paper bet from scratch in settle order (needed
    when staking changes — % staking compounds, so order matters)."""
    async with SessionLocal() as session:
        await session.execute(delete(PaperBet))
        await session.flush()
        preds = list(
            await session.scalars(
                select(Prediction)
                .where(Prediction.was_correct.isnot(None))
                .order_by(
                    Prediction.settled_at.asc().nullslast(),
                    Prediction.created_at.asc(),
                )
            )
        )
        n = 0
        for pred in preds:
            if await place_paper_bet(session, pred):
                n += 1
        await session.commit()
    return n


async def place_for_settled(session) -> int:
    """Backfill bets for settled predictions lacking one (in settle order)."""
    preds = list(
        await session.scalars(
            select(Prediction)
            .where(Prediction.was_correct.isnot(None))
            .where(Prediction.id.notin_(select(PaperBet.prediction_id)))
            .order_by(Prediction.settled_at.asc().nullslast())
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
        pnl = float(await session.scalar(select(func.coalesce(func.sum(PaperBet.pnl), 0.0))) or 0.0)
        staked = float(await session.scalar(select(func.coalesce(func.sum(PaperBet.stake), 0.0))) or 0.0)
    return {
        "start": settings.paper_start_balance,
        "balance": round(settings.paper_start_balance + pnl, 2),
        "pnl": round(pnl, 2),
        "bets": total,
        "won": won,
        "lost": total - won,
        "stake_pct": settings.paper_stake_pct * 100,
        "roi": round(pnl / staked * 100, 1) if staked else 0.0,
    }
