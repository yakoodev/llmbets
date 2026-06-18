"""Odds providers.

mock = test "polygon": prices the market off PURE Elo (the public line) + a vig.
Our prediction model adds form/news on top of Elo, so value (edge) appears
exactly where our signals diverge from the Elo line. Real bookmakers/odds-APIs
(OddsPapi, PandaScore, Pinnacle…) plug in behind the same capture interface.
"""
from __future__ import annotations

from sqlalchemy import select

from app.config import settings
from app.db.models import Match, OddsSnapshot, TeamRating
from app.prediction.elo import BASE_ELO, expected_score


async def _elo_prob(session, a_id, b_id) -> float:
    ra = await session.get(TeamRating, a_id)
    rb = await session.get(TeamRating, b_id)
    ra = float(ra.elo) if ra else BASE_ELO
    rb = float(rb.elo) if rb else BASE_ELO
    return expected_score(ra, rb)


def _vig_odds(prob: float, margin: float) -> float:
    prob = min(max(prob, 0.01), 0.99)
    return round((1.0 / prob) / (1.0 + margin), 3)


async def capture_odds(session, match: Match) -> dict | None:
    """Quote + persist market odds for a match. Returns {team_id: {odds, implied}}."""
    if settings.odds_provider != "mock":
        return None  # real provider goes here (env-keyed) — see module docstring
    if not (match.team_a_id and match.team_b_id):
        return None
    pa = await _elo_prob(session, match.team_a_id, match.team_b_id)
    margin = settings.odds_margin
    quote = {
        match.team_a_id: {"odds": _vig_odds(pa, margin), "implied": round(pa, 4)},
        match.team_b_id: {"odds": _vig_odds(1 - pa, margin), "implied": round(1 - pa, 4)},
    }
    for team_id, v in quote.items():
        session.add(
            OddsSnapshot(
                match_id=match.id,
                bookmaker=settings.odds_provider,
                selection_team_id=team_id,
                odds_decimal=v["odds"],
                implied_probability=v["implied"],
            )
        )
    return quote


async def latest_odds(session, match_id) -> dict:
    """Most recent odds per selection: {team_id: {'odds': x, 'implied': y}}."""
    rows = list(
        await session.scalars(
            select(OddsSnapshot)
            .where(OddsSnapshot.match_id == match_id)
            .order_by(OddsSnapshot.captured_at.desc())
        )
    )
    out: dict = {}
    for r in rows:
        if r.selection_team_id not in out:
            out[r.selection_team_id] = {
                "odds": float(r.odds_decimal),
                "implied": float(r.implied_probability or 0),
            }
    return out
