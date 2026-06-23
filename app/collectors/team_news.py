"""Per-team news for teams in upcoming matches.

The broad "CS2 esports" feeds surface scene-wide noise (map-pool changes, Major
recaps) not tied to the teams actually playing soon. This collector queries
Google News per TEAM that has an upcoming match ("<team>" Counter-Strike), feeds
it into the standard pipeline → classified, embedded and entity-linked to that
team + its upcoming matches — so the news is about who we're predicting.

CLI:  python -m app.collectors.team_news [max_teams]
"""
from __future__ import annotations

import asyncio
import logging
import sys
import urllib.parse
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.collectors.news import _fetch_feed, _hash, _published, _strip_html
from app.config import settings
from app.db.models import Match, NewsItem, NewsSource, Team
from app.db.session import SessionLocal

log = logging.getLogger("collector.team_news")

TEAM_SOURCE_NAME = "Team News (Google)"


def _gn_query(name: str) -> str:
    q = urllib.parse.quote(f'"{name}" Counter-Strike CS2')
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


async def _team_source(session) -> NewsSource:
    src = await session.scalar(
        select(NewsSource).where(NewsSource.name == TEAM_SOURCE_NAME)
    )
    if src is None:
        src = NewsSource(
            name=TEAM_SOURCE_NAME, source_type="team_news", reliability_score=0.45
        )
        session.add(src)
        await session.flush()
    return src


async def collect_team_news(max_teams: int = 30, per_team: int = 8) -> int:
    saved = 0
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=settings.prediction_horizon_hours)
    async with SessionLocal() as session:
        src = await _team_source(session)
        # teams playing an upcoming match within the prediction horizon (soonest first)
        matches = list(
            await session.scalars(
                select(Match)
                .where(
                    Match.status == "upcoming",
                    Match.scheduled_at > now,
                    Match.scheduled_at <= horizon,
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                )
                .order_by(Match.scheduled_at.asc())
            )
        )
        team_ids: list = []
        for m in matches:
            for tid in (m.team_a_id, m.team_b_id):
                if tid not in team_ids:
                    team_ids.append(tid)
        team_ids = team_ids[:max_teams]
        teams = list(await session.scalars(select(Team).where(Team.id.in_(team_ids)))) if team_ids else []
        log.info("collect_team_news: %d teams to scan", len(teams))
        cutoff = now - timedelta(days=settings.news_max_age_days)
        for team in teams:
            name = team.name or ""
            if not name or name.startswith("team-") or len(name) < 2:
                continue
            parsed = await _fetch_feed(_gn_query(name))
            if parsed is None:
                continue
            for e in parsed.entries[:per_team]:
                published = _published(e)
                if published and published < cutoff:
                    continue
                link = e.get("link")
                title = e.get("title")
                summary = e.get("summary") or e.get("description") or ""
                content_hash = _hash(link or "", title or "")
                if await session.scalar(
                    select(NewsItem.id).where(NewsItem.content_hash == content_hash)
                ):
                    continue
                session.add(
                    NewsItem(
                        source_id=src.id,
                        url=link,
                        title=title,
                        raw_text=summary,
                        clean_text=_strip_html(summary),
                        published_at=published,
                        content_hash=content_hash,
                    )
                )
                saved += 1
            await session.commit()
    log.info("collect_team_news: %d new items", saved)
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    saved = await collect_team_news(n)
    print(f"Collected {saved} team news items.")


if __name__ == "__main__":
    asyncio.run(_main())
