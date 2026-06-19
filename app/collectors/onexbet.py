"""1xBet collector — REAL bookmaker source (odds + betable CS2 matches), free.

1xBet has no public API, and its JSON backend (`/service-api/LineFeed/…`) sits
behind Cloudflare. A normal httpx request gets a 403 "Just a moment…" challenge.
We pass it with **curl_cffi** impersonating Chrome's TLS/JA3 fingerprint — pure
HTTP, headless, no browser, no account, no key. Verified working from the VPS.

CS2 lives under sportId 40 (Esports); each tournament is a "champ" whose name
starts with "CS 2." (e.g. "CS 2. IEM Cologne Major"). Match-winner odds are in
GetGameZip → GE group G==1 (T==1 → team1, T==3 → team2).

Used as the odds provider (ODDS_PROVIDER=onexbet) and to flag which matches are
actually betable on 1xBet.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from curl_cffi.requests import AsyncSession

from app.processing.entities import normalize

log = logging.getLogger("collector.1xbet")

BASE = "https://1xbet.com/service-api/LineFeed/"
ESPORTS_SPORT_ID = 40
IMPERSONATE = "chrome124"
_HEADERS = {
    "Referer": "https://1xbet.com/en/line/",
    "x-requested-with": "XMLHttpRequest",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# cache the CS2 game list (id, teams, start) — refreshed every few minutes
_GAMES_CACHE: dict = {"at": 0.0, "data": []}
_CACHE_TTL = 300.0


def _is_cs2(name: str) -> bool:
    n = (name or "").lower()
    return "cs 2" in n or "cs2" in n or "counter" in n


async def _get(session: AsyncSession, endpoint: str, params: dict) -> Any:
    r = await session.get(
        BASE + endpoint, params=params, headers=_HEADERS,
        impersonate=IMPERSONATE, timeout=30,
    )
    try:
        return r.json()
    except Exception:  # noqa: BLE001 — Cloudflare HTML or transient
        return {}


async def _fetch_cs2_games(session: AsyncSession) -> list[dict]:
    """All upcoming CS2 games on 1xBet: [{id, o1, o2, start, champ}]."""
    champs = (await _get(session, "GetChampsZip", {
        "sport": ESPORTS_SPORT_ID, "lng": "en", "tf": 1000000, "tz": 0, "country": 1,
    })).get("Value") or []
    cs_champs = [c for c in champs if _is_cs2(c.get("L", ""))]
    games: list[dict] = []
    for c in cs_champs:
        champ_id = c.get("LI") or c.get("I")
        if not champ_id:
            continue
        val = (await _get(session, "GetChampZip", {
            "champ": champ_id, "sport": ESPORTS_SPORT_ID, "lng": "en",
            "tf": 3000000, "tz": 0, "country": 1, "afterDays": -1,
        })).get("Value")
        raw = []
        if isinstance(val, dict):
            raw = val.get("G") or []
        elif isinstance(val, list):
            raw = val
        for g in raw:
            if g.get("O1") and g.get("O2") and g.get("I"):
                games.append({
                    "id": g["I"], "o1": g["O1"], "o2": g["O2"],
                    "start": g.get("S"), "champ": c.get("L"),
                })
    return games


async def cs2_games(session: AsyncSession | None = None) -> list[dict]:
    if _GAMES_CACHE["data"] and time.time() - _GAMES_CACHE["at"] < _CACHE_TTL:
        return _GAMES_CACHE["data"]
    own = session is None
    if own:
        session = AsyncSession()
    try:
        games = await _fetch_cs2_games(session)
    finally:
        if own:
            await session.close()
    if games:
        _GAMES_CACHE.update(at=time.time(), data=games)
    return games


def _names_match(x: str, y: str) -> bool:
    x, y = normalize(x), normalize(y)
    if not x or not y:
        return False
    return x == y or (len(x) >= 4 and len(y) >= 4 and (x in y or y in x))


def find_game(games: list[dict], team_a: str, team_b: str) -> dict | None:
    for g in games:
        o1, o2 = g["o1"], g["o2"]
        if (_names_match(o1, team_a) and _names_match(o2, team_b)) or (
            _names_match(o1, team_b) and _names_match(o2, team_a)
        ):
            return g
    return None


async def winner_odds(session: AsyncSession, game_id: int):
    """Match-winner odds + team names FROM THE SAME GetGameZip response.

    Returns (o1, o2, name1, name2) where o1/name1 is T==1/O1 and o2/name2 is
    T==3/O2 — so the caller maps odds→team by name within one response instead
    of trusting cross-endpoint (GetChampZip vs GetGameZip) ordering. None if the
    match-winner market (GE group 1) isn't fully present."""
    val = (await _get(session, "GetGameZip", {
        "id": game_id, "lng": "en", "cfview": 0,
        "isSubGames": "true", "GroupEvents": "true", "countevents": 250,
    })).get("Value") or {}
    n1, n2 = val.get("O1"), val.get("O2")
    o1 = o2 = None
    for grp in val.get("GE") or []:
        if grp.get("G") != 1:  # G==1 is the main match-winner market
            continue
        for sub in grp.get("E") or []:
            # E is sometimes a list of lists, sometimes flat — normalise
            for ev in (sub if isinstance(sub, list) else [sub]):
                if not isinstance(ev, dict):
                    continue
                if ev.get("T") == 1 and ev.get("C"):
                    o1 = float(ev["C"])
                elif ev.get("T") == 3 and ev.get("C"):
                    o2 = float(ev["C"])
    if o1 and o2 and n1 and n2:
        return (o1, o2, n1, n2)
    return None
