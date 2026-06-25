"""Pinnacle odds — the second, sharpest book (consensus + CLV benchmark).

We were anchored to a single book (1xBet) and bet at its closing line, so CLV
was structurally zero. Pinnacle is the sharpest line in the world; collecting it
gives (a) a consensus market prob and (b) a genuine CLV reference — divergence of
1xBet from Pinnacle is real value. Reached headless via Pinnacle's public guest
API (no account): sport 12 = e-sports, leagues named "CS2 - <event>", moneyline
prices in American odds. Stored as OddsSnapshot(bookmaker='pinnacle').

CLI:  python -m app.collectors.pinnacle
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from curl_cffi import requests as cr
from sqlalchemy import select

from app.config import settings
from app.db.models import Match, OddsSnapshot, Team
from app.db.session import SessionLocal
from app.processing.entities import normalize

log = logging.getLogger("collector.pinnacle")

BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
# Public guest key embedded in pinnacle.com's own site JS — no account needed.
KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"
HEAD = {"x-api-key": KEY, "x-device-uuid": "betsllm", "accept": "application/json"}
ESPORTS = 12


def _am_to_dec(american: float) -> float:
    a = float(american)
    return round((a / 100.0 + 1.0) if a > 0 else (100.0 / (-a) + 1.0), 4)


def _fetch(path: str):
    r = cr.get(f"{BASE}{path}", headers=HEAD, impersonate="chrome124", timeout=25)
    r.raise_for_status()
    return r.json()


async def _get(path: str):
    return await asyncio.to_thread(_fetch, path)


async def collect_pinnacle_odds() -> int:
    """Match Pinnacle CS2 moneylines to our upcoming matches → OddsSnapshot."""
    matchups = await _get(f"/sports/{ESPORTS}/matchups")
    markets = await _get(f"/sports/{ESPORTS}/markets/straight")
    ml = {
        x["matchupId"]: x
        for x in markets
        if x.get("type") == "moneyline" and x.get("period") == 0 and x.get("prices")
    }
    cs = [
        m for m in matchups
        if isinstance(m, dict)
        and "cs2" in str(m.get("league", {}).get("name", "")).lower()
        and m.get("participants") and m.get("id") in ml
    ]
    if not cs:
        log.info("pinnacle: no CS2 matchups with moneylines")
        return 0

    now = datetime.now(timezone.utc)
    saved = 0
    async with SessionLocal() as session:
        ours = list(
            await session.scalars(
                select(Match).where(
                    Match.scheduled_at > now - timedelta(hours=6),
                    Match.winner_team_id.is_(None),
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                )
            )
        )
        index = {}  # frozenset(norm names) -> Match
        names = {}  # match_id -> (norm_a, norm_b)
        for m in ours:
            ta = await session.get(Team, m.team_a_id)
            tb = await session.get(Team, m.team_b_id)
            na, nb = normalize(ta.name if ta else ""), normalize(tb.name if tb else "")
            if na and nb:
                index[frozenset((na, nb))] = m
                names[m.id] = (na, nb)

        for mu in cs:
            parts = {p.get("alignment"): p.get("name") for p in mu["participants"]}
            home, away = parts.get("home"), parts.get("away")
            if not (home and away):
                continue
            nh, naw = normalize(home), normalize(away)
            match = index.get(frozenset((nh, naw)))
            if not match:
                continue
            prices = {p.get("designation"): p.get("price") for p in ml[mu["id"]]["prices"]}
            if prices.get("home") is None or prices.get("away") is None:
                continue
            na, nb = names[match.id]
            # map home/away → our team_a/team_b by normalized name
            for desig, who in (("home", nh), ("away", naw)):
                team_id = match.team_a_id if who == na else (match.team_b_id if who == nb else None)
                if team_id is None:
                    continue
                dec = _am_to_dec(prices[desig])
                session.add(
                    OddsSnapshot(
                        match_id=match.id,
                        bookmaker="pinnacle",
                        selection_team_id=team_id,
                        odds_decimal=dec,
                        implied_probability=round(1.0 / dec, 4),
                    )
                )
                saved += 1
        await session.commit()
    log.info("pinnacle: saved %d odds rows across %d matched matches", saved, saved // 2)
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(f"Pinnacle saved {await collect_pinnacle_odds()} odds rows.")


if __name__ == "__main__":
    asyncio.run(_main())
