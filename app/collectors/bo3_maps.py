"""Per-map results for finished matches → match_maps (the map-pool signal).

bo3 `/api/v1/games?filter[games.match_id][eq]=<id>` returns each map of a match:
map_name, number (order), winner_clan_name, winner/loser score. We populate
match_maps so that per-team per-map win-rates can be computed later as a real
prediction feature (the edge that isn't in the bookmaker line). Backfills any
finished bo3 match that has no map rows yet; runs on a schedule.

CLI:  python -m app.collectors.bo3_maps [days]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from app.collectors.bo3 import Bo3Client
from app.config import settings
from app.db.models import Match, MatchMap, Team
from app.db.session import SessionLocal
from app.processing.entities import normalize

log = logging.getLogger("collector.bo3_maps")

GAMES = "/api/v1/games"


def _nm(x: str, y: str) -> bool:
    x, y = normalize(x or ""), normalize(y or "")
    return bool(x) and bool(y) and (x == y or (len(x) >= 4 and len(y) >= 4 and (x in y or y in x)))


async def collect_match_maps(days: int = 21, limit: int = 200) -> int:
    """Populate match_maps for finished bo3 matches lacking them."""
    now = datetime.now(timezone.utc)
    saved = 0
    async with Bo3Client() as client, SessionLocal() as session:
        has_maps = select(MatchMap.match_id).distinct()
        matches = list(
            await session.scalars(
                select(Match)
                .where(
                    Match.source == "bo3",
                    Match.external_id.isnot(None),
                    Match.winner_team_id.isnot(None),
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                    Match.scheduled_at > now - timedelta(days=days),
                    Match.id.notin_(has_maps),
                )
                .order_by(Match.scheduled_at.desc())
                .limit(limit)
            )
        )
        for m in matches:
            ta = await session.get(Team, m.team_a_id)
            tb = await session.get(Team, m.team_b_id)
            try:
                data = await client.get(GAMES, {"filter[games.match_id][eq]": m.external_id})
                games = data.get("results") or []
            except Exception as e:  # noqa: BLE001
                log.warning("bo3 games fetch failed %s: %s", m.external_id, e)
                continue
            for g in games:
                mapn = g.get("map_name")
                if not mapn or g.get("state") not in (None, "done"):
                    continue
                wname = g.get("winner_clan_name") or ""
                ws, ls = g.get("winner_clan_score"), g.get("loser_clan_score")
                winner = ta if _nm(wname, ta.name) else (tb if _nm(wname, tb.name) else None)
                if winner is ta:
                    a_s, b_s = ws, ls
                elif winner is tb:
                    a_s, b_s = ls, ws
                else:
                    a_s = b_s = None
                session.add(
                    MatchMap(
                        match_id=m.id,
                        map_name=mapn,
                        map_order=g.get("number"),
                        team_a_score=a_s,
                        team_b_score=b_s,
                        winner_team_id=winner.id if winner else None,
                    )
                )
                saved += 1
            await session.commit()
    log.info("collect_match_maps: %d map rows from %d matches", saved, len(matches))
    return saved


async def team_map_winrate(session, team_id, map_name: str) -> tuple[float, int]:
    """(win-rate, n) for a team on a map — the feature this all feeds. n=0 → 0.5."""
    total = await session.scalar(
        select(func.count()).select_from(MatchMap).where(
            MatchMap.map_name == map_name,
            ((MatchMap.team_a_score.isnot(None))),
            MatchMap.winner_team_id.isnot(None),
            MatchMap.match_id.in_(
                select(Match.id).where((Match.team_a_id == team_id) | (Match.team_b_id == team_id))
            ),
        )
    ) or 0
    if not total:
        return 0.5, 0
    wins = await session.scalar(
        select(func.count()).select_from(MatchMap).where(
            MatchMap.map_name == map_name,
            MatchMap.winner_team_id == team_id,
        )
    ) or 0
    return round(wins / total, 3), total


_ACTIVE_MAPS = ("de_dust2", "de_mirage", "de_ancient", "de_nuke", "de_overpass", "de_inferno", "de_anubis")
_SHRINK = 2.0  # pull each map win-rate toward 0.5 by this pseudo-count


async def _team_map_stats(session, team_id, before=None) -> dict:
    from sqlalchemy import case
    q = (
        select(
            MatchMap.map_name,
            func.count().label("n"),
            func.sum(case((MatchMap.winner_team_id == team_id, 1), else_=0)).label("w"),
        )
        .join(Match, Match.id == MatchMap.match_id)
        .where(
            MatchMap.winner_team_id.isnot(None),
            (Match.team_a_id == team_id) | (Match.team_b_id == team_id),
        )
        .group_by(MatchMap.map_name)
    )
    if before is not None:
        q = q.where(Match.scheduled_at < before)
    return {r.map_name: (int(r.w or 0), int(r.n or 0)) for r in await session.execute(q)}


async def map_pool_adv(session, team_a_id, team_b_id, before=None) -> float:
    """Team A's map-pool advantage = mean over active maps of (wr_a − wr_b), each
    shrunk toward 0.5. The map-strength edge not priced into the bookmaker line.
    `before` (a match's scheduled_at) makes it leakage-free for backtests."""
    sa = await _team_map_stats(session, team_a_id, before)
    sb = await _team_map_stats(session, team_b_id, before)
    diffs = []
    for m in _ACTIVE_MAPS:
        wa, na = sa.get(m, (0, 0))
        wb, nb = sb.get(m, (0, 0))
        if na + nb == 0:
            continue
        ra = (wa + _SHRINK) / (na + 2 * _SHRINK)
        rb = (wb + _SHRINK) / (nb + 2 * _SHRINK)
        diffs.append(ra - rb)
    return round(sum(diffs) / len(diffs), 4) if diffs else 0.0


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    print(f"Collected {await collect_match_maps(days)} map rows.")


if __name__ == "__main__":
    asyncio.run(_main())
