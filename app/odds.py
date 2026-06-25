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
from app.db.models import Match, OddsSnapshot, PaperBet, Prediction, Team, TeamRating
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


# ── Pinnacle (sharpest book) — its snapshots are stored by collectors.pinnacle ──

W_PINNACLE = 0.6  # consensus weight on Pinnacle (sharper) vs the betable book


async def _pinnacle_quote(session, match: Match) -> dict | None:
    """Latest stored Pinnacle line for this match, de-vigged to true prob."""
    od = {}
    for tid in (match.team_a_id, match.team_b_id):
        o = await session.scalar(
            select(OddsSnapshot.odds_decimal)
            .where(
                OddsSnapshot.match_id == match.id,
                OddsSnapshot.bookmaker == "pinnacle",
                OddsSnapshot.selection_team_id == tid,
            )
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(1)
        )
        if o:
            od[tid] = float(o)
    if len(od) < 2:
        return None
    ia, ib = 1.0 / od[match.team_a_id], 1.0 / od[match.team_b_id]
    pa = ia / (ia + ib)
    return {
        match.team_a_id: {"odds": round(od[match.team_a_id], 3), "implied": round(pa, 4)},
        match.team_b_id: {"odds": round(od[match.team_b_id], 3), "implied": round(1.0 - pa, 4)},
    }


def _consensus(match: Match, q_book: dict | None, q_pin: dict | None) -> dict | None:
    """Blend two de-vigged lines into a consensus market prob (Pinnacle weighted
    higher, it's sharper). One present → use it (e.g. early Pinnacle before 1xBet
    posts). Odds shown = the betable book when present, else Pinnacle."""
    if q_book and q_pin:
        pa = W_PINNACLE * q_pin[match.team_a_id]["implied"] + (1 - W_PINNACLE) * q_book[match.team_a_id]["implied"]
        return {
            match.team_a_id: {"odds": q_book[match.team_a_id]["odds"], "implied": round(pa, 4)},
            match.team_b_id: {"odds": q_book[match.team_b_id]["odds"], "implied": round(1.0 - pa, 4)},
        }
    return q_book or q_pin


async def capture_odds(session, match: Match) -> dict | None:
    if not (match.team_a_id and match.team_b_id):
        return None
    if settings.odds_provider in ("onexbet", "1xbet"):
        q_book = await _capture_onexbet(session, match)
    elif settings.odds_provider == "oddspapi":
        q_book = await _capture_oddspapi(session, match)
    else:
        q_book = await _capture_mock(session, match)
    q_pin = await _pinnacle_quote(session, match)
    return _consensus(match, q_book, q_pin)


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


def _reblend_market(pred, match, quote, w_market=None, learned=None) -> bool:
    """Re-apply the market to a prediction IN PLACE from freshly captured odds —
    heals predictions made before 1xBet posted a line. With learned weights, the
    whole model is recomputed using the new market component; otherwise the simple
    W_MARKET blend on the stored model prob. No LLM, no new row. Returns True if
    the prediction changed."""
    from app.prediction.engine import (
        W_MARKET, _logit, _sigmoid, feature_x_from_snapshot, learned_logit,
    )

    fd = dict(pred.feature_drivers or {})
    mkt_pa = (quote.get(match.team_a_id) or {}).get("implied")
    if mkt_pa is None or not (0.0 < float(mkt_pa) < 1.0):
        return False
    mkt_pa = float(mkt_pa)
    if learned:
        fd2 = dict(fd)
        fd2["market_prob_a"] = mkt_pa
        pa = round(_sigmoid(learned_logit(learned, feature_x_from_snapshot(fd2))), 4)
    else:
        model_pa = fd.get("model_prob_a")
        if model_pa is None:
            return False
        w = w_market if w_market is not None else W_MARKET
        pa = round(_sigmoid(w * _logit(mkt_pa) + (1.0 - w) * _logit(float(model_pa))), 4)
    pred.team_a_probability = pa
    pred.team_b_probability = round(1.0 - pa, 4)
    pred.predicted_winner_team_id = match.team_a_id if pa >= 0.5 else match.team_b_id
    fd.update(market_prob_a=round(mkt_pa, 4), prob_a=pa, prob_b=round(1.0 - pa, 4))
    pred.feature_drivers = fd
    return True


async def refresh_odds_for_upcoming(horizon_days: int = 7) -> int:
    """Pull fresh odds for every upcoming match and update its prediction's
    displayed line. Runs as a scheduler job."""
    from app.prediction.engine import W_MARKET
    from app.runtime_config import get_config

    horizon = datetime.now(timezone.utc) + timedelta(days=horizon_days)
    _wmv = await get_config("w_market")  # self-calibrated blend weight
    _wm = float(_wmv) if _wmv else W_MARKET
    import json as _json
    _lwv = await get_config("learned_weights")
    _learned = _json.loads(_lwv) if _lwv else None
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
                    _reblend_market(pred, m, quote, _wm, _learned)  # heal/re-fit
                n += 1
            await session.commit()
    log.info("refresh_odds_for_upcoming: odds for %d matches", n)
    return n


async def clv_vs_pinnacle(session) -> dict:
    """Closing-line value of settled paper bets vs Pinnacle's closing line — the
    real edge test. We bet at 1xBet; Pinnacle is the sharpest book, so beating
    its close = genuine value (unlike 1xBet-vs-itself, which was structurally 0).
    Positive avg_clv_pct over many bets ⇒ real edge. Populates as bets settle with
    Pinnacle data (live from 2026-06-25)."""
    bets = list(
        await session.scalars(
            select(PaperBet).where(
                PaperBet.result.isnot(None), PaperBet.selection_team_id.isnot(None)
            )
        )
    )
    clvs = []
    for pb in bets:
        close = await session.scalar(
            select(OddsSnapshot.odds_decimal)
            .where(
                OddsSnapshot.match_id == pb.match_id,
                OddsSnapshot.bookmaker == "pinnacle",
                OddsSnapshot.selection_team_id == pb.selection_team_id,
            )
            .order_by(OddsSnapshot.captured_at.desc())
            .limit(1)
        )
        if close and float(close) > 1 and float(pb.odds) > 1:
            clvs.append(float(pb.odds) / float(close) - 1.0)
    if not clvs:
        return {"n": 0, "avg_clv_pct": None, "beat_close": 0}
    return {
        "n": len(clvs),
        "avg_clv_pct": round(sum(clvs) / len(clvs) * 100, 2),
        "beat_close": sum(1 for c in clvs if c > 0),
    }


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(f"provider={settings.odds_provider} -> refreshed {await refresh_odds_for_upcoming()}")
    async with SessionLocal() as s:
        print("CLV vs Pinnacle:", await clv_vs_pinnacle(s))


if __name__ == "__main__":
    asyncio.run(_main())
