"""bo3.gg collector — primary CS2 match/result/history source (HLTV-grade).

Open JSON API at api.bo3.gg (no Cloudflare), fetched through the proxy. Replaces
PandaScore as the match source. Teams are reconciled with any existing rows by
ps_id (bo3 exposes the PandaScore id) or name, else created with a bo3_id.

CLI:
  python -m app.collectors.bo3 upcoming
  python -m app.collectors.bo3 results
  python -m app.collectors.bo3 backfill [pages]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import func, select

from app.config import settings
from app.db.models import Match, Team
from app.db.session import SessionLocal

log = logging.getLogger("collector.bo3")

MATCHES = "/api/v1/matches"
TEAMS = "/api/v1/teams"
TOURNAMENTS = "/api/v1/tournaments"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _status(s: str | None) -> str:
    return {
        "upcoming": "upcoming",
        "scheduled": "upcoming",
        "live": "live",
        "started": "live",
        "current": "live",
        "finished": "finished",
        "defwin": "finished",
        "cancelled": "canceled",
        "canceled": "canceled",
    }.get(s or "", s or "unknown")


class Bo3Client:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.bo3_base_url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (cs2-llm-bot)"},
            proxy=settings.bo3_proxy_url or None,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get(self, path: str, params: dict | None = None) -> dict:
        last: Exception | None = None
        for _ in range(3):  # proxy hiccups → transient timeouts; retry
            try:
                resp = await self._client.get(path, params=params or {})
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last = e
        raise last  # type: ignore[misc]

    async def __aenter__(self) -> "Bo3Client":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


async def _reconcile_team(session, key: str, info: dict, cache: dict) -> Team:
    """Map a bo3 team (id+info) to a Team row, reusing existing by ps_id/name."""
    name = info.get("name") or f"team-{key}"
    ps_id = info.get("ps_id")
    rank = info.get("rank")
    team = None
    if ps_id:
        team = await session.scalar(
            select(Team).where(Team.pandascore_id == str(ps_id))
        )
    if team is None:
        team = await session.scalar(
            select(Team).where(func.lower(Team.name) == name.lower())
        )
    if team is None:
        team = Team(name=name)
        session.add(team)
    team.bo3_id = key
    if ps_id and not team.pandascore_id:
        team.pandascore_id = str(ps_id)
    if rank is not None:
        team.rank = rank
    await session.flush()
    cache[key] = team
    return team


async def _prefetch_teams(session, client, team_ids, cache: dict) -> None:
    """Batch-resolve a set of bo3 team ids (one API call per 50) into cache."""
    need = []
    for tid in {str(t) for t in team_ids if t}:
        if tid in cache:
            continue
        existing = await session.scalar(select(Team).where(Team.bo3_id == tid))
        if existing:
            cache[tid] = existing
        else:
            need.append(tid)
    for i in range(0, len(need), 50):
        chunk = need[i : i + 50]
        info_by_id: dict = {}
        try:
            data = await client.get(
                TEAMS,
                {"filter[teams.id][in]": ",".join(chunk), "page[limit]": 50},
            )
            for t in data.get("results", []):
                info_by_id[str(t["id"])] = t
        except Exception as e:  # noqa: BLE001
            log.warning("bo3 team batch failed: %s", e)
        for tid in chunk:
            await _reconcile_team(session, tid, info_by_id.get(tid, {}), cache)


async def _resolve_team(session, client, team_id, cache: dict) -> Team | None:
    if team_id is None:
        return None
    key = str(team_id)
    if key in cache:
        return cache[key]
    team = await session.scalar(select(Team).where(Team.bo3_id == key))
    if team:
        cache[key] = team
        return team
    try:
        data = await client.get(TEAMS, {"filter[teams.id][eq]": team_id})
        info = (data.get("results") or [{}])[0]
    except Exception as e:  # noqa: BLE001
        log.warning("bo3 team fetch failed %s: %s", team_id, e)
        info = {}
    return await _reconcile_team(session, key, info, cache)


async def _tournament_name(session, client, t_id, cache: dict) -> str | None:
    if t_id is None:
        return None
    key = str(t_id)
    if key in cache:
        return cache[key]
    try:
        data = await client.get(TOURNAMENTS, {"filter[tournaments.id][eq]": t_id})
        info = (data.get("results") or [{}])[0]
        name = info.get("name")
    except Exception:  # noqa: BLE001
        name = None
    cache[key] = name
    return name


async def _upsert_match(session, client, m: dict, tcache: dict, tourcache: dict) -> Match | None:
    if not m.get("team1_id") or not m.get("team2_id"):
        return None  # TBD bracket slot
    ext = str(m["id"])
    team_a = await _resolve_team(session, client, m["team1_id"], tcache)
    team_b = await _resolve_team(session, client, m["team2_id"], tcache)
    if not team_a or not team_b:
        return None

    bo = m.get("bo_type")
    winner = None
    if m.get("winner_team_id"):
        w = str(m["winner_team_id"])
        winner = team_a if team_a.bo3_id == w else (team_b if team_b.bo3_id == w else None)
    elif bo:  # bo3's winner/status can lag — infer from a decisive map score
        need = int(bo) // 2 + 1  # bo3 -> first to 2, bo5 -> 3, bo1 -> 1
        s1, s2 = m.get("team1_score") or 0, m.get("team2_score") or 0
        if s1 >= need:
            winner = team_a
        elif s2 >= need:
            winner = team_b

    match = await session.scalar(
        select(Match).where(Match.external_id == ext, Match.source == "bo3")
    )
    fields = dict(
        source="bo3",
        external_id=ext,
        team_a_id=team_a.id,
        team_b_id=team_b.id,
        tournament_name=await _tournament_name(
            session, client, m.get("tournament_id"), tourcache
        ),
        format=f"bo{bo}" if bo else None,
        scheduled_at=_parse_dt(m.get("start_date")),
        # a decided winner means finished, even if bo3's status field still lags
        status="finished" if winner else _status(m.get("status")),
        winner_team_id=winner.id if winner else None,
        team_a_standin=bool(m.get("team1_new_participant")),
        team_b_standin=bool(m.get("team2_new_participant")),
    )
    if match is None:
        match = Match(**fields)
        session.add(match)
    else:
        for k, v in fields.items():
            setattr(match, k, v)
    await session.flush()
    return match


async def collect_upcoming() -> int:
    horizon = datetime.now(timezone.utc) + timedelta(days=settings.bo3_upcoming_days)
    saved = 0
    async with Bo3Client() as client, SessionLocal() as session:
        tcache: dict = {}
        tourcache: dict = {}
        offset = 0
        while offset < 1000:
            data = await client.get(
                MATCHES,
                {
                    "filter[matches.status][eq]": "upcoming",
                    "sort": "start_date",
                    "page[limit]": 100,
                    "page[offset]": offset,
                },
            )
            results = data.get("results") or []
            if not results:
                break
            await _prefetch_teams(
                session,
                client,
                [t for m in results for t in (m.get("team1_id"), m.get("team2_id"))],
                tcache,
            )
            stop = False
            for m in results:
                sched = _parse_dt(m.get("start_date"))
                if sched and sched > horizon:
                    stop = True
                    break
                if await _upsert_match(session, client, m, tcache, tourcache):
                    saved += 1
            await session.commit()
            if stop or len(results) < 100:
                break
            offset += 100
    log.info("bo3 collect_upcoming: upserted %d matches", saved)
    return saved


async def collect_results() -> int:
    now = datetime.now(timezone.utc)
    updated = 0
    async with Bo3Client() as client, SessionLocal() as session:
        tcache: dict = {}
        tourcache: dict = {}
        overdue = list(
            await session.scalars(
                select(Match).where(
                    Match.source == "bo3",
                    Match.external_id.isnot(None),
                    # re-check ANY overdue match that isn't already terminal —
                    # robust to bo3 statuses we haven't mapped (e.g. "current").
                    Match.status.notin_(["finished", "canceled"]),
                    Match.scheduled_at < now,
                )
            )
        )
        for match in overdue:
            try:
                data = await client.get(
                    MATCHES, {"filter[matches.id][eq]": match.external_id}
                )
                m = (data.get("results") or [None])[0]
            except Exception as e:  # noqa: BLE001
                log.warning("bo3 result fetch failed %s: %s", match.external_id, e)
                continue
            if not m:
                continue
            if await _upsert_match(session, client, m, tcache, tourcache):
                updated += 1
        await session.commit()
    log.info("bo3 collect_results: resolved %d overdue matches", updated)
    return updated


async def collect_past(pages: int = 30) -> int:
    saved = 0
    async with Bo3Client() as client, SessionLocal() as session:
        tcache: dict = {}
        tourcache: dict = {}
        for page in range(pages):
            data = await client.get(
                MATCHES,
                {
                    "filter[matches.status][eq]": "finished",
                    "sort": "-start_date",
                    "page[limit]": 100,
                    "page[offset]": page * 100,
                },
            )
            results = data.get("results") or []
            if not results:
                break
            await _prefetch_teams(
                session,
                client,
                [t for m in results for t in (m.get("team1_id"), m.get("team2_id"))],
                tcache,
            )
            for m in results:
                if not m.get("winner_team_id"):
                    continue
                if await _upsert_match(session, client, m, tcache, tourcache):
                    saved += 1
            await session.commit()
            log.info("bo3 collect_past: page %d (total %d)", page + 1, saved)
    log.info("bo3 collect_past: upserted %d finished matches", saved)
    return saved


async def resolve_placeholders() -> int:
    """Re-fetch names for teams left as 'team-<id>' when a batch lookup missed."""
    fixed = 0
    async with Bo3Client() as client, SessionLocal() as session:
        teams = list(
            await session.scalars(
                select(Team).where(Team.name.like("team-%"), Team.bo3_id.isnot(None))
            )
        )
        ids = [t.bo3_id for t in teams]
        info: dict = {}
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            try:
                data = await client.get(
                    TEAMS, {"filter[teams.id][in]": ",".join(chunk), "page[limit]": 50}
                )
                for t in data.get("results", []):
                    info[str(t["id"])] = t
            except Exception as e:  # noqa: BLE001
                log.warning("placeholder batch failed: %s", e)
        for t in teams:
            row = info.get(t.bo3_id)
            if row and row.get("name"):
                t.name = row["name"]
                if row.get("rank") is not None:
                    t.rank = row["rank"]
                fixed += 1
        await session.commit()
    log.info("resolve_placeholders: fixed %d team names", fixed)
    return fixed


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "upcoming"
    if cmd == "upcoming":
        print(f"Upserted {await collect_upcoming()} upcoming matches.")
    elif cmd == "results":
        print(f"Resolved {await collect_results()} overdue matches.")
    elif cmd == "backfill":
        pages = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        print(f"Backfilled {await collect_past(pages)} finished matches.")
    elif cmd == "fix-teams":
        print(f"Fixed {await resolve_placeholders()} placeholder team names.")
    else:
        print(f"Unknown command: {cmd!r}")


if __name__ == "__main__":
    asyncio.run(_main())
