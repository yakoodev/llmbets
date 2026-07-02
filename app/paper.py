"""Paper-betting test balance — % of bank staking.

Stake = paper_stake_pct of the CURRENT balance (compounds). We bet only when
the model has VALUE vs the market (model prob − implied prob >= min_edge) at
market odds; without odds we fall back to flat fair-odds on the favourite.
Balance = start + Σ pnl (each pnl already computed from the running stake, so
the sum telescopes correctly). Process bets in chronological (settle) order.
"""
from __future__ import annotations

from sqlalchemy import delete, func, select

from datetime import datetime, timezone

from app.config import settings
from app.db.models import Match, PaperBet, Prediction
from app.db.session import SessionLocal
from app.odds import latest_odds
from app.runtime_config import get_config


async def _ledger_since(session) -> datetime | None:
    """Optional cutoff: only bet predictions settled at/after this time. Set on a
    balance reset (runtime_config 'paper_ledger_since', ISO) so the ledger starts
    fresh from the reset point instead of replaying all history."""
    raw = await get_config("paper_ledger_since")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


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
    selection = pred.predicted_winner_team_id  # bet on our predicted winner
    if selection is None:
        return None  # no pick → no bet
    if await session.scalar(select(PaperBet.id).where(PaperBet.prediction_id == pred.id)):
        return None
    # one bet per MATCH: a match can carry >1 settled prediction (re-predict,
    # corrections) — never double-stake the same real outcome.
    if await session.scalar(select(PaperBet.id).where(PaperBet.match_id == pred.match_id)):
        return None

    match = await session.get(Match, pred.match_id)
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
    await session.flush()  # next bet's running balance must include this one
    return bet


async def rebuild_ledger() -> int:
    """Wipe and recompute every paper bet from scratch in settle order (needed
    when staking changes — % staking compounds, so order matters)."""
    async with SessionLocal() as session:
        await session.execute(delete(PaperBet))
        await session.flush()
        since = await _ledger_since(session)
        q = select(Prediction).where(Prediction.was_correct.isnot(None))
        if since is not None:
            q = q.where(Prediction.settled_at >= since)
        preds = list(
            await session.scalars(
                q.order_by(
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
    since = await _ledger_since(session)
    q = (
        select(Prediction)
        .where(Prediction.was_correct.isnot(None))
        .where(Prediction.id.notin_(select(PaperBet.prediction_id)))
    )
    if since is not None:
        q = q.where(Prediction.settled_at >= since)
    preds = list(await session.scalars(q.order_by(Prediction.settled_at.asc().nullslast())))
    n = 0
    for pred in preds:
        if await place_paper_bet(session, pred):
            n += 1
    await session.commit()
    return n


async def _build_events(session, since) -> list[dict]:
    """One bettable event per match (latest prediction), in settle order. Each
    strategy in app.strategies decides whether/how to bet it."""
    q = select(Prediction).where(Prediction.was_correct.isnot(None))
    if since is not None:
        q = q.where(Prediction.settled_at >= since)
    preds = list(await session.scalars(q.order_by(Prediction.created_at.asc())))
    latest: dict = {}
    for p in preds:
        if p.predicted_winner_team_id is not None:
            latest[p.match_id] = p  # later created_at wins → newest prediction/match
    events = []
    for p in latest.values():
        match = await session.get(Match, p.match_id)
        if not match:
            continue
        sel = p.predicted_winner_team_id
        prob = float(p.team_a_probability if sel == match.team_a_id else p.team_b_probability)
        odds_map = await latest_odds(session, p.match_id)
        if sel in odds_map:
            odds_used = float(odds_map[sel]["odds"])
            mkt = odds_map[sel].get("implied")
            mkt = float(mkt) if mkt else None
        else:
            odds_used = round(1.0 / min(max(prob, 0.01), 0.99), 3)
            mkt = None
        events.append(
            {
                "pred_id": p.id, "match_id": p.match_id, "selection": sel,
                "odds": odds_used, "prob": prob, "mkt": mkt, "risk": p.risk_level,
                "won": bool(p.was_correct), "settled_at": p.settled_at,
            }
        )
    events.sort(key=lambda e: e["settled_at"] or datetime.min.replace(tzinfo=timezone.utc))
    return events


async def rebuild_strategy_ledgers() -> int:
    """Wipe + replay every strategy over the settled-prediction stream."""
    from app.db.models import StrategyBet
    from app.strategies import STRATEGIES, simulate

    async with SessionLocal() as session:
        await session.execute(delete(StrategyBet))
        await session.flush()
        since = await _ledger_since(session)
        events = await _build_events(session, since)
        start = settings.paper_start_balance
        for name, spec in STRATEGIES.items():
            for b in simulate(name, spec, events, start):
                session.add(
                    StrategyBet(
                        strategy=name, prediction_id=b["pred_id"], match_id=b["match_id"],
                        selection_team_id=b["selection"], stake=b["stake"], odds=b["odds"],
                        result="won" if b["won"] else "lost", pnl=b["pnl"],
                        settled_at=b["settled_at"],
                    )
                )
        await session.commit()
    return len(events)


async def strategy_balances() -> list[dict]:
    from app.db.models import StrategyBet
    from app.strategies import STRATEGIES

    start = settings.paper_start_balance
    out = []
    async with SessionLocal() as session:
        for name, spec in STRATEGIES.items():
            bets = list(
                await session.scalars(
                    select(StrategyBet)
                    .where(StrategyBet.strategy == name)
                    .order_by(StrategyBet.settled_at.asc().nullslast(), StrategyBet.created_at.asc())
                )
            )
            bal = peak = start
            max_dd = max_streak = cur_streak = 0
            max_stake = worst_loss = 0.0
            won = staked = 0.0
            wc = 0
            for b in bets:
                stk = float(b.stake)
                pnl_b = float(b.pnl or 0.0)
                bal += pnl_b
                staked += stk
                max_stake = max(max_stake, stk)
                peak = max(peak, bal)
                max_dd = max(max_dd, peak - bal)  # deepest drop from a running high
                if b.result == "won":
                    wc += 1
                    cur_streak = 0
                else:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                    worst_loss = min(worst_loss, pnl_b)
            n = len(bets)
            pnl = bal - start
            out.append(
                {
                    "strategy": name, "desc": spec["desc"], "balance": round(bal, 2),
                    "pnl": round(pnl, 2), "bets": n, "won": wc,
                    "roi": round(pnl / staked * 100, 1) if staked else 0.0,
                    "max_losestreak": max_streak, "cur_streak": cur_streak,
                    "max_stake": round(max_stake, 2), "max_drawdown": round(max_dd, 2),
                    "worst_loss": round(worst_loss, 2),
                }
            )
    out.sort(key=lambda x: x["balance"], reverse=True)
    return out


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
