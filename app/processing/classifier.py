"""LLM news classification → events + entity/match links.

For each unprocessed news_item: fast-model classify, store a news_event when
relevant, link mentioned teams/players, and mark relevance to upcoming matches.
Critical events (stand-in/visa/illness/...) flag the item for the fast lane.

CLI:  python -m app.processing.classifier [limit]
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.db.models import (
    Match,
    MatchRelevanceLink,
    NewsEvent,
    NewsItem,
    NewsSource,
    PlayerNewsLink,
    TeamNewsLink,
)
from app.db.session import SessionLocal
from app.llm.client import llm
from app.llm.prompts import load_prompt, render
from app.processing.entities import resolve_player, resolve_team

log = logging.getLogger("processing.classifier")

CRITICAL_EVENTS = {
    "standin",
    "roster_change",
    "injury",
    "illness",
    "visa_issue",
    "travel_issue",
    "coach_change",
    "role_change",
}


def _num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


async def classify_unprocessed(limit: int = 20) -> int:
    prompt = load_prompt("news_classifier")
    template = prompt["template"]
    done = 0
    digest: list[dict] = []
    async with SessionLocal() as session:
        items = list(
            await session.scalars(
                select(NewsItem)
                .join(NewsSource, NewsSource.id == NewsItem.source_id)
                .where(NewsItem.processed.is_(False), NewsSource.enabled.is_(True))
                .order_by(NewsItem.fetched_at.desc())
                .limit(limit)
            )
        )
        for item in items:
            payload = {
                "title": item.title,
                "text": (item.clean_text or item.raw_text or "")[:4000],
                "published_at": str(item.published_at),
            }
            try:
                data = await llm.chat_json(
                    "Верни только валидный JSON по схеме.",
                    render(template, input_json=payload),
                    tier="fast",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("classify failed for %s: %s", item.id, e)
                item.processed = True
                await session.commit()
                continue

            if data.get("is_relevant"):
                event_type = data.get("event_type") or "irrelevant"
                impact = data.get("prediction_impact") or {}
                session.add(
                    NewsEvent(
                        news_item_id=item.id,
                        event_type=event_type,
                        event_subtype=data.get("event_subtype"),
                        summary=data.get("summary"),
                        importance=_num(data.get("importance")),
                        confidence=_num(data.get("confidence")),
                        source_quality=data.get("source_quality"),
                        prediction_impact_direction=impact.get("direction"),
                        prediction_impact_score=_num(impact.get("impact_score")),
                        event_time=datetime.now(timezone.utc),
                    )
                )
                item.is_critical = (
                    data.get("time_sensitivity") == "pre_match_critical"
                    or event_type in CRITICAL_EVENTS
                )

                entities = data.get("entities") or {}
                team_ids: set = set()
                team_names: list[str] = []
                player_names: list[str] = []
                for tname in entities.get("teams", []) or []:
                    team = await resolve_team(session, tname)
                    if team:
                        team_ids.add(team.id)
                        team_names.append(team.name)
                        session.add(
                            TeamNewsLink(news_item_id=item.id, team_id=team.id)
                        )
                for pname in entities.get("players", []) or []:
                    player = await resolve_player(session, pname)
                    if player:
                        player_names.append(player.nickname)
                        session.add(
                            PlayerNewsLink(news_item_id=item.id, player_id=player.id)
                        )

                # relevance to upcoming matches that involve a mentioned team
                matched = 0
                if team_ids:
                    matches = list(
                        await session.scalars(
                            select(Match).where(
                                Match.status == "upcoming",
                                Match.team_a_id.in_(team_ids)
                                | Match.team_b_id.in_(team_ids),
                            )
                        )
                    )
                    matched = len(matches)
                    for m in matches:
                        session.add(
                            MatchRelevanceLink(
                                news_item_id=item.id,
                                match_id=m.id,
                                relevance=_num(data.get("importance"), 0.5),
                            )
                        )

                digest.append(
                    {
                        "event_type": event_type,
                        "summary": data.get("summary"),
                        "teams": team_names,
                        "players": player_names,
                        "matches": matched,
                        "critical": bool(item.is_critical),
                    }
                )

            item.processed = True
            await session.commit()
            done += 1
    log.info("classify_unprocessed: processed %d items, %d relevant", done, len(digest))
    return done, digest


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n, digest = await classify_unprocessed(limit)
    print(f"Classified {n} items, {len(digest)} relevant.")


if __name__ == "__main__":
    asyncio.run(_main())
