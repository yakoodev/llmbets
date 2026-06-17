"""Per-player news/social signal for players in upcoming matches.

Reliable path: a Google News RSS query per active player ("<nick> CS2"), fetched
through the proxy and fed into the standard news pipeline (so it gets classified,
embedded and linked to the player). Direct Twitter/Instagram is unreliable
(nitter instances mostly dead, no free IG API) — kept as a separate best-effort
hook (player_social_accounts) for later.

CLI:  python -m app.collectors.player_news [max_players]
"""
from __future__ import annotations

import asyncio
import logging
import sys
import urllib.parse
from datetime import datetime, timezone

from sqlalchemy import select

from app.collectors.news import _fetch_feed, _hash, _published, _strip_html
from app.config import settings
from app.db.models import Match, NewsItem, NewsSource, Player, TeamRoster
from app.db.session import SessionLocal

log = logging.getLogger("collector.player_news")

PLAYER_SOURCE_NAME = "Player News (Google)"


def _gn_query(nick: str) -> str:
    q = urllib.parse.quote(f'"{nick}" CS2 esports')
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


async def _player_source(session) -> NewsSource:
    src = await session.scalar(
        select(NewsSource).where(NewsSource.name == PLAYER_SOURCE_NAME)
    )
    if src is None:
        src = NewsSource(
            name=PLAYER_SOURCE_NAME, source_type="player_news", reliability_score=0.4
        )
        session.add(src)
        await session.flush()
    return src


async def collect_player_news(max_players: int = 25, per_player: int = 8) -> int:
    saved = 0
    async with SessionLocal() as session:
        src = await _player_source(session)
        # active players on teams that have an upcoming match (soonest first)
        upcoming_team_ids = select(Match.team_a_id).where(
            Match.status == "upcoming", Match.team_a_id.isnot(None)
        ).union(
            select(Match.team_b_id).where(
                Match.status == "upcoming", Match.team_b_id.isnot(None)
            )
        )
        players = list(
            await session.scalars(
                select(Player)
                .join(TeamRoster, TeamRoster.player_id == Player.id)
                .where(
                    TeamRoster.active_to.is_(None),
                    TeamRoster.team_id.in_(upcoming_team_ids),
                )
                .distinct()
                .limit(max_players)
            )
        )
        log.info("collect_player_news: %d players to scan", len(players))
        for player in players:
            parsed = await _fetch_feed(_gn_query(player.nickname))
            if parsed is None:
                continue
            for e in parsed.entries[:per_player]:
                link = e.get("link")
                title = e.get("title")
                summary = e.get("summary") or e.get("description") or ""
                content_hash = _hash(link or "", title or "")
                exists = await session.scalar(
                    select(NewsItem.id).where(NewsItem.content_hash == content_hash)
                )
                if exists:
                    continue
                session.add(
                    NewsItem(
                        source_id=src.id,
                        url=link,
                        title=title,
                        raw_text=summary,
                        clean_text=_strip_html(summary),
                        published_at=_published(e),
                        content_hash=content_hash,
                    )
                )
                saved += 1
            await session.commit()
    log.info("collect_player_news: %d new items", saved)
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    saved = await collect_player_news(n)
    print(f"Collected {saved} player news items.")


if __name__ == "__main__":
    asyncio.run(_main())
