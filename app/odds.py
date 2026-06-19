"""Odds providers.

- "mock": test polygon — prices market off pure Elo + vig.
- "oddspapi": real market odds (api.oddspapi.io). CS2 = sportId 17, moneyline =
  market 171 (outcome 171=home/participant1, 172=away/participant2). Prefers the
  Pinnacle line (sharpest). Implied probabilities are de-vigged.

Set ODDS_PROVIDER to switch. capture_odds(match) stores an odds_snapshot;
refresh_odds_for_upcoming() batch-updates all upcoming matches.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from app.config import settings
from app.db.models import Match, OddsSnapshot, Prediction, Team, TeamRating
from app.db.session import SessionLocal
from app.prediction.elo import BASE_ELO, expected_score
from app.processing.entities import normalize

log = logging.getLogger("odds")


# ── mock (Elo + vig) ─────────────────────────────────────────────────


async def _elo_prob(session, a_id, b_id) -> float:
    ra = await session.get(TeamRating, a_id)
    rb = await session.get(TeamRating, b_id)
    ra = float(ra.elo) if ra else BASE_ELO
    rb = float(rb.elo) if rb else BASE_ELO
    return expected_score(ra, rb)


def _vig_odds(prob: float, margin: float) -> float:
    prob = min(max(prob, 0.01), 0.99)
    return round((1.0 / prob) / (1.0 + margin), 3)


async def _capture_mock(session, match: Match) -> dict | None:
    pa = await _elo_prob(session, match.team_a_id, match.team_b_id)
    m = settings.odds_margin
    return _store(session, match, _vig_odds(pa, m), _vig_odds(1 - pa, m), "mock")


# ── OddsPapi (real) ──────────────────────────────────────────────────

_FIX_CACHE: dict = {"at": 0.0, "data": []}


def _nmatch(x: str, y: str) -> bool:
    return bool(x) and bool(y) and (x == y or (len(x) >= 4 and len(y) >= 4 and (x in y or y in x)))


async def _papi_get(client: httpx.AsyncClient, path: str, params: dict):
    for _ in range(4):
        try:
            r = await client.get(path, params={**params, "apiKey": settings.oddspapi_api_key})
            j = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("oddspapi %s failed: %s", path, e)
            await asyncio.sleep(2)
            continue
        if isinstance(j, dict) and (j.get("error") or {}).get("code") == "RATE_LIMITED":
            await asyncio.sleep((j["error"].get("retryMs", 2000) / 1000) + 0.3)
            continue
        return j
    return None


async def _cs2_fixtures(client) -> list:
    if _FIX_CACHE["data"] and time.time() - _FIX_CACHE["at"] < 600:
        return _FIX_CACHE["data"]
    now = datetime.now(timezone.utc)
    j = await _papi_get(
        client,
        "/fixtures",
        {"sportId": 17, "from": now.strftime("%Y-%m-%d"),
         "to": (now + timedelta(days=9)).strftime("%Y-%m-%d")},
    )
    arr = j if isinstance(j, list) else ((j or {}).get("data") or [])
    if arr:
        _FIX_CACHE.update(at=time.time(), data=arr)
    return arr


def _find_fixture(fixtures, a: str, b: str):
    na, nb = normalize(a), normalize(b)
    for fx in fixtures:
        p1, p2 = normalize(fx.get("participant1Name", "")), normalize(fx.get("participant2Name", ""))
        if (_nmatch(p1, na) and _nmatch(p2, nb)) or (_nmatch(p1, nb) and _nmatch(p2, na)):
            return fx
    return None


async def _fixture_prices(client, fixture_id):
    """Return (p1_decimal, p2_decimal) for the moneyline (market 171), preferring
    Pinnacle, else any book."""
    j = await _papi_get(client, "/odds", {"fixtureId": fixture_id, "oddsFormat": "decimal"})
    if not isinstance(j, dict):
        return None
    books = j.get("bookmakerOdds") or {}
    bk = books.get(settings.odds_bookmaker) or next(iter(books.values()), None)
    if not bk:
        return None
    market = (bk.get("markets") or {}).get("171")
    if not market:
        return None
    outs = market.get("outcomes") or {}

    def price(oid):
        try:
            return float(outs[oid]["players"]["0"]["price"])
        except (KeyError, TypeError, ValueError):
            return None

    o1, o2 = price("171"), price("172")
    return (o1, o2) if o1 and o2 else None


async def _capture_oddspapi(session, match: Match) -> dict | None:
    team_a = await session.get(Team, match.team_a_id)
    team_b = await session.get(Team, match.team_b_id)
    async with httpx.AsyncClient(base_url=settings.oddspapi_base_url, timeout=25.0) as client:
        fx = _find_fixture(await _cs2_fixtures(client), team_a.name, team_b.name)
        if not fx:
            return None
        prices = await _fixture_prices(client, fx["fixtureId"])
        if not prices:
            return None
    o1, o2 = prices  # fixture participant1 / participant2
    # map participant1/2 → our team_a/team_b by name
    if _nmatch(normalize(fx.get("participant1Name", "")), normalize(team_a.name)):
        oa, ob = o1, o2
    else:
        oa, ob = o2, o1
    return _store(session, match, oa, ob, settings.odds_bookmaker)


# ── shared store (de-vig implied) ────────────────────────────────────


def _store(session, match: Match, odds_a: float, odds_b: float, bookmaker: str) -> dict:
    ia, ib = 1.0 / odds_a, 1.0 / odds_b
    total = ia + ib  # >1 (vig); de-vig to true market prob
    quote = {
        match.team_a_id: {"odds": round(odds_a, 3), "implied": round(ia / total, 4)},
        match.team_b_id: {"odds": round(odds_b, 3), "implied": round(ib / total, 4)},
    }
    for team_id, v in quote.items():
        session.add(
            OddsSnapshot(
                match_id=match.id,
                bookmaker=bookmaker,
                selection_team_id=team_id,
                odds_decimal=v["odds"],
                implied_probability=v["implied"],
            )
        )
    return quote


# ── 1xBet (real bookmaker, free, via curl_cffi Cloudflare bypass) ────


async def _capture_onexbet(session, match: Match) -> dict | None:
    from curl_cffi.requests import AsyncSession

    from app.collectors.onexbet import cs2_games, find_game, _names_match, winner_odds

    team_a = await session.get(Team, match.team_a_id)
    team_b = await session.get(Team, match.team_b_id)
    async with AsyncSession() as s:
        game = find_game(await cs2_games(s), team_a.name, team_b.name)
        if not game:
            return None
        res = await winner_odds(s, game["id"])
    if not res:
        return None
    o1, o2, n1, n2 = res  # odds + team names from the SAME GetGameZip response
    if _names_match(n1, team_a.name):
        oa, ob = o1, o2
    elif _names_match(n2, team_a.name):
        oa, ob = o2, o1
    else:
        return None  # can't confidently map odds to our team_a — skip, don't guess
    return _store(session, match, oa, ob, "1xbet")


async def capture_odds(session, match: Match) -> dict | None:
    if not (match.team_a_id and match.team_b_id):
        return None
    if settings.odds_provider in ("onexbet", "1xbet"):
        return await _capture_onexbet(session, match)
    if settings.odds_provider == "oddspapi":
        return await _capture_oddspapi(session, match)
    return await _capture_mock(session, match)


async def latest_odds(session, match_id) -> dict:
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


async def refresh_odds_for_upcoming(horizon_days: int = 7) -> int:
    """Pull fresh odds for every upcoming match and update its prediction's
    displayed line. Runs as a scheduler job."""
    horizon = datetime.now(timezone.utc) + timedelta(days=horizon_days)
    n = 0
    async with SessionLocal() as session:
        matches = list(
            await session.scalars(
                select(Match).where(
                    Match.status == "upcoming",
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                    Match.scheduled_at <= horizon,
                )
            )
        )
        for m in matches:
            quote = await capture_odds(session, m)
            if quote:
                pred = await session.scalar(
                    select(Prediction)
                    .where(Prediction.match_id == m.id)
                    .order_by(Prediction.created_at.desc())
                )
                if pred:
                    pred.fair_odds = {
                        "market_team_a": quote[m.team_a_id]["odds"],
                        "market_team_b": quote[m.team_b_id]["odds"],
                    }
                n += 1
            await session.commit()
    log.info("refresh_odds_for_upcoming: odds for %d matches", n)
    return n


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(f"provider={settings.odds_provider} -> refreshed {await refresh_odds_for_upcoming()}")


if __name__ == "__main__":
    asyncio.run(_main())
