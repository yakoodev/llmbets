"""PandaScore collector — CS2 matches, teams/rosters and results.

CS2 lives under the `csgo` videogame slug on PandaScore. v1 uses PandaScore for
structured data only (NO odds — paper betting is out of v1 scope).

CLI:
  python -m app.collectors.pandascore upcoming
  python -m app.collectors.pandascore results
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select

from app.config import settings
from app.db.models import Match, Player, Team, TeamRoster
from app.db.session import SessionLocal

log = logging.getLogger("collector.pandascore")

CS2_PATH = "/csgo/matches"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _status(ps_status: str | None) -> str:
    return {
        "not_started": "upcoming",
        "running": "live",
        "finished": "finished",
        "canceled": "canceled",
        "postponed": "upcoming",
    }.get(ps_status or "", ps_status or "unknown")


class PandaScoreClient:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.pandascore_base_url,
            headers={"Authorization": f"Bearer {settings.pandascore_api_key}"},
            timeout=30.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> list[dict]:
        resp = await self._client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    async def __aenter__(self) -> "PandaScoreClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ── Upsert helpers ───────────────────────────────────────────────────


async def _get_or_create_team(session, ps_team: dict) -> Team | None:
    if not ps_team:
        return None
    ps_id = str(ps_team["id"])
    team = await session.scalar(select(Team).where(Team.pandascore_id == ps_id))
    if team is None:
        team = Team(
            name=ps_team.get("name") or ps_team.get("acronym") or f"team-{ps_id}",
            slug=ps_team.get("slug"),
            country=ps_team.get("location"),
            pandascore_id=ps_id,
        )
        session.add(team)
        await session.flush()
    else:
        # keep name fresh
        if ps_team.get("name"):
            team.name = ps_team["name"]
    return team


def _opponents(match: dict) -> tuple[dict, dict]:
    opps = [o.get("opponent") or {} for o in match.get("opponents", [])]
    a = opps[0] if len(opps) > 0 else {}
    b = opps[1] if len(opps) > 1 else {}
    return a, b


async def _upsert_match(session, m: dict, ignore_tier: bool = False) -> Match | None:
    tournament = m.get("tournament") or {}
    tier = (tournament.get("tier") or "").lower() or None
    # tier filter (None tier => keep, can't tell). Backfill ignores it: Elo
    # history benefits from all played matches regardless of tier.
    if not ignore_tier and tier and tier not in settings.tracked_tiers:
        return None

    ext_id = str(m["id"])
    match = await session.scalar(
        select(Match).where(Match.external_id == ext_id, Match.source == "pandascore")
    )
    a, b = _opponents(m)
    team_a = await _get_or_create_team(session, a)
    team_b = await _get_or_create_team(session, b)

    league = m.get("league") or {}
    serie = m.get("serie") or {}
    tournament_name = " — ".join(
        x for x in [league.get("name"), serie.get("full_name")] if x
    ) or tournament.get("name")

    ttype = tournament.get("type")
    is_lan = (ttype == "offline") if ttype else None
    nog = m.get("number_of_games")
    fmt = f"bo{nog}" if nog else None

    winner_team = None
    if m.get("winner_id") and team_a and team_b:
        wid = str(m["winner_id"])
        winner_team = team_a if team_a.pandascore_id == wid else (
            team_b if team_b.pandascore_id == wid else None
        )

    fields = dict(
        source="pandascore",
        external_id=ext_id,
        team_a_id=team_a.id if team_a else None,
        team_b_id=team_b.id if team_b else None,
        tournament_name=tournament_name,
        tier=tier,
        format=fmt,
        is_lan=is_lan,
        scheduled_at=_parse_dt(m.get("begin_at") or m.get("scheduled_at")),
        status=_status(m.get("status")),
        winner_team_id=winner_team.id if winner_team else None,
    )

    if match is None:
        match = Match(**fields)
        session.add(match)
    else:
        for k, v in fields.items():
            setattr(match, k, v)
    await session.flush()
    return match


# ── Collectors ───────────────────────────────────────────────────────


async def collect_upcoming() -> int:
    """Fetch upcoming CS2 matches within the configured window; upsert teams+matches."""
    horizon = datetime.now(timezone.utc) + timedelta(
        days=settings.pandascore_upcoming_days
    )
    saved = 0
    async with PandaScoreClient() as ps, SessionLocal() as session:
        page = 1
        while page <= 10:
            matches = await ps.get(
                f"{CS2_PATH}/upcoming",
                {"per_page": 100, "page": page, "sort": "begin_at"},
            )
            if not matches:
                break
            stop = False
            for m in matches:
                sched = _parse_dt(m.get("begin_at") or m.get("scheduled_at"))
                if sched and sched > horizon:
                    stop = True
                    break
                if await _upsert_match(session, m):
                    saved += 1
            await session.commit()
            if stop or len(matches) < 100:
                break
            page += 1
    log.info("collect_upcoming: upserted %d matches", saved)
    return saved


async def _get_or_create_player(session, ps_player: dict) -> Player:
    ps_id = str(ps_player["id"])
    player = await session.scalar(select(Player).where(Player.pandascore_id == ps_id))
    real_name = " ".join(
        x for x in [ps_player.get("first_name"), ps_player.get("last_name")] if x
    ) or None
    if player is None:
        player = Player(
            nickname=ps_player.get("name") or f"player-{ps_id}",
            real_name=real_name,
            country=ps_player.get("nationality"),
            pandascore_id=ps_id,
        )
        session.add(player)
        await session.flush()
    else:
        player.nickname = ps_player.get("name") or player.nickname
        if real_name:
            player.real_name = real_name
    return player


async def collect_rosters() -> int:
    """For every tracked team, fetch its current players and refresh team_rosters."""
    linked = 0
    async with PandaScoreClient() as ps, SessionLocal() as session:
        teams = list(await session.scalars(select(Team).where(Team.pandascore_id.isnot(None))))
        for team in teams:
            data = await ps.get(f"/teams/{team.pandascore_id}")
            ps_team = data[0] if isinstance(data, list) else data
            for ps_player in (ps_team or {}).get("players", []) or []:
                player = await _get_or_create_player(session, ps_player)
                exists = await session.scalar(
                    select(TeamRoster).where(
                        TeamRoster.team_id == team.id,
                        TeamRoster.player_id == player.id,
                        TeamRoster.active_to.is_(None),
                    )
                )
                if exists is None:
                    session.add(
                        TeamRoster(
                            team_id=team.id,
                            player_id=player.id,
                            role=ps_player.get("role"),
                            source="pandascore",
                        )
                    )
                    linked += 1
            await session.commit()
    log.info("collect_rosters: added %d roster links", linked)
    return linked


async def collect_results() -> int:
    """Resolve outcomes for OUR overdue matches by fetching each directly.

    /past pagination buries our specific matches among masses of others, so we
    query each tracked match that's past its start time but still unresolved via
    /matches/{id} (the /csgo/matches/{id} route is 403 on the free tier)."""
    now = datetime.now(timezone.utc)
    updated = 0
    async with PandaScoreClient() as ps, SessionLocal() as session:
        overdue = list(
            await session.scalars(
                select(Match).where(
                    Match.source == "pandascore",
                    Match.external_id.isnot(None),
                    Match.status.in_(["upcoming", "live"]),
                    Match.scheduled_at < now,
                )
            )
        )
        for match in overdue:
            try:
                data = await ps.get(f"/matches/{match.external_id}")
            except Exception as e:  # noqa: BLE001
                log.warning("result fetch failed for %s: %s", match.external_id, e)
                continue
            m = data[0] if isinstance(data, list) else data
            if not isinstance(m, dict) or "id" not in m:
                continue
            if await _upsert_match(session, m, ignore_tier=True):
                updated += 1
        await session.commit()
    log.info("collect_results: resolved %d overdue matches", updated)
    return updated


async def collect_past(pages: int = 15, per_page: int = 100) -> int:
    """Backfill finished CS2 matches (for Elo history). Ingests teams + results."""
    saved = 0
    async with PandaScoreClient() as ps, SessionLocal() as session:
        for page in range(1, pages + 1):
            matches = await ps.get(
                f"{CS2_PATH}/past",
                {"per_page": per_page, "page": page, "sort": "-end_at"},
            )
            if not matches:
                break
            for m in matches:
                # PandaScore free tier marks most history "canceled" (forfeits/
                # walkovers) yet still carries a winner_id. Use any decided
                # result for Elo volume — noisy but it's what's available.
                if not m.get("winner_id"):
                    continue
                if await _upsert_match(session, m, ignore_tier=True):
                    saved += 1
            await session.commit()
            log.info("collect_past: page %d done (total %d)", page, saved)
    log.info("collect_past: upserted %d finished matches", saved)
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "upcoming"
    if cmd == "upcoming":
        n = await collect_upcoming()
        print(f"Upserted {n} upcoming matches.")
    elif cmd == "rosters":
        n = await collect_rosters()
        print(f"Added {n} roster links.")
    elif cmd == "backfill":
        pages = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        n = await collect_past(pages)
        print(f"Backfilled {n} finished matches.")
    elif cmd == "results":
        n = await collect_results()
        print(f"Settled {n} finished matches.")
    else:
        print(f"Unknown command: {cmd!r}. Use 'upcoming', 'rosters' or 'results'.")


if __name__ == "__main__":
    asyncio.run(_main())
