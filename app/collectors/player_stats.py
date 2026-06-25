"""Team roster strength from bo3 player ratings — the team-strength signal.

A 1300-match no-leakage backtest showed only *team strength* carries signal
(Elo ~60%; form/h2h/map-pool add nothing). Player ratings are a faster, roster-
aware strength estimate than Elo (they update the moment a lineup changes, where
Elo lags). bo3 `/api/v1/players?filter[team_id][eq]=<bo3_id>` returns the current
lineup with `six_month_avg_rating`; we store the mean of the active five on the
team. Used as the `strength` feature.

CLI:  python -m app.collectors.player_stats [limit]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from app.collectors.bo3 import Bo3Client
from app.config import settings
from app.db.models import Match, Team
from app.db.session import SessionLocal

log = logging.getLogger("collector.player_stats")

PLAYERS = "/api/v1/players"


def _roster_strength(players: list) -> float | None:
    """Mean six-month rating of the active five (drop coaches / zero ratings)."""
    rs = []
    for p in players:
        if p.get("is_coach") or (p.get("role") or "").lower() == "coach":
            continue
        r = p.get("six_month_avg_rating")
        if r and float(r) > 0:
            rs.append(float(r))
    if not rs:
        return None
    rs.sort(reverse=True)
    top = rs[:5]
    return round(sum(top) / len(top), 4)


async def collect_player_strength(limit: int = 160) -> int:
    """Refresh Team.strength for ranked + soon-playing teams that have a bo3_id."""
    now = datetime.now(timezone.utc)
    updated = 0
    async with Bo3Client() as client, SessionLocal() as session:
        soon = select(Match.team_a_id).where(
            Match.scheduled_at > now - timedelta(days=2)
        ).union(
            select(Match.team_b_id).where(Match.scheduled_at > now - timedelta(days=2))
        )
        soon_ids = {r for (r,) in await session.execute(soon) if r}
        teams = list(
            await session.scalars(
                select(Team)
                .where(Team.bo3_id.isnot(None))
                .where(or_(Team.rank.isnot(None), Team.id.in_(soon_ids)))
                .order_by(Team.rank.asc().nulls_last())
                .limit(limit)
            )
        )
        for t in teams:
            try:
                data = await client.get(PLAYERS, {"filter[team_id][eq]": t.bo3_id, "page[limit]": "15"})
                players = data.get("results") or []
            except Exception as e:  # noqa: BLE001
                log.warning("bo3 players fetch failed team=%s: %s", t.bo3_id, e)
                continue
            s = _roster_strength(players)
            if s is not None:
                t.strength = s
                t.strength_at = now
                updated += 1
        await session.commit()
    log.info("collect_player_strength: updated %d/%d teams", updated, len(teams))
    return updated


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 160
    print(f"Updated strength for {await collect_player_strength(n)} teams.")


if __name__ == "__main__":
    asyncio.run(_main())
