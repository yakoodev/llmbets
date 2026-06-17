"""Team Elo — rebuilt from finished matches, used as the prediction baseline.

CLI:  python -m app.prediction.elo rebuild
"""
from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import delete, select

from app.config import settings
from app.db.models import Match, TeamRating
from app.db.session import SessionLocal

log = logging.getLogger("prediction.elo")

BASE_ELO = 1500.0
K = 32.0

# Tier-1 results carry more signal than minor/qualifier games; forfeits
# (free-tier "canceled" walkovers) carry the least.
_TIER_W = {"s": 1.5, "a": 1.25, "b": 1.0, "c": 0.8, "d": 0.6}
_STATUS_W = {"finished": 1.0, "canceled": 0.4}


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _match_k(match) -> float:
    tw = _TIER_W.get((match.tier or "").lower(), 0.6)
    sw = _STATUS_W.get(match.status or "", 0.4)
    return K * tw * sw


async def rebuild_ratings() -> int:
    """Replay all finished matches chronologically; persist final Elo per team."""
    async with SessionLocal() as session:
        matches = list(
            await session.scalars(
                select(Match)
                .where(
                    # any decided result (free-tier data is mostly "canceled"
                    # walkovers that still carry a winner) — see collector note
                    Match.winner_team_id.isnot(None),
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                )
                .order_by(Match.scheduled_at.asc().nullslast())
            )
        )
        ratings: dict = {}
        played: dict = {}
        last_at: dict = {}
        for m in matches:
            a, b = m.team_a_id, m.team_b_id
            ra = ratings.get(a, BASE_ELO)
            rb = ratings.get(b, BASE_ELO)
            ea = expected_score(ra, rb)
            sa = 1.0 if m.winner_team_id == a else 0.0
            k = _match_k(m)
            ratings[a] = ra + k * (sa - ea)
            ratings[b] = rb + k * ((1.0 - sa) - (1.0 - ea))
            for t in (a, b):
                played[t] = played.get(t, 0) + 1
                last_at[t] = m.scheduled_at

        await session.execute(delete(TeamRating))
        for team_id, elo in ratings.items():
            session.add(
                TeamRating(
                    team_id=team_id,
                    elo=round(elo, 1),
                    matches_played=played.get(team_id, 0),
                    last_match_at=last_at.get(team_id),
                )
            )
        await session.commit()
    log.info("rebuild_ratings: rated %d teams from %d matches", len(ratings), len(matches))
    return len(ratings)


async def get_rating(session, team_id) -> float:
    row = await session.get(TeamRating, team_id)
    return float(row.elo) if row else BASE_ELO


async def win_probability(session, team_a_id, team_b_id) -> tuple[float, float]:
    ra = await get_rating(session, team_a_id)
    rb = await get_rating(session, team_b_id)
    pa = expected_score(ra, rb)
    return pa, 1.0 - pa


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    n = await rebuild_ratings()
    print(f"Rated {n} teams.")


if __name__ == "__main__":
    asyncio.run(_main())
