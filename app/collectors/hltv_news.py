"""HLTV news — real pro-CS2 scene news (transfers, stand-ins, roster, form).

HLTV is the authoritative pro source and is Cloudflare-protected; hltv-async-api
passes it. `get_last_news()` returns the homepage news groups. We flatten the
headlines into NewsItems and run them through the standard pipeline → classified
+ entity-linked to teams/players/upcoming matches. Far richer than Google News
(which mostly surfaces match previews + betting-tip spam).

CLI:  python -m app.collectors.hltv_news
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.collectors.news import _hash, _strip_html
from app.config import settings
from app.db.models import NewsItem, NewsSource
from app.db.session import SessionLocal

log = logging.getLogger("collector.hltv_news")

HLTV_SOURCE_NAME = "HLTV News"


async def _source(session) -> NewsSource:
    src = await session.scalar(
        select(NewsSource).where(NewsSource.name == HLTV_SOURCE_NAME)
    )
    if src is None:
        src = NewsSource(
            name=HLTV_SOURCE_NAME, source_type="hltv", reliability_score=0.8
        )
        session.add(src)
        await session.flush()
    return src


async def _fetch_news() -> list[dict]:
    from hltv_async_api import Hltv

    hltv = Hltv(timeout=30)
    try:
        groups = (await hltv.get_last_news()) or []
    finally:
        for attr in ("close_session", "close"):
            fn = getattr(hltv, attr, None)
            if fn:
                try:
                    await fn()
                except Exception:  # noqa: BLE001
                    pass
                break
    items: list[dict] = []
    for g in groups:
        for key in ("news", "f_news"):
            for n in g.get(key) or []:
                if n.get("title") and n.get("id"):
                    items.append(n)
    return items


async def collect_hltv_news() -> int:
    """Pull HLTV homepage news headlines into the pipeline. Best-effort."""
    now = datetime.now(timezone.utc)
    try:
        items = await _fetch_news()
    except Exception as e:  # noqa: BLE001
        log.warning("hltv news fetch failed: %s", e)
        return 0
    saved = 0
    async with SessionLocal() as session:
        src = await _source(session)
        for n in items:
            nid = str(n["id"])
            title = (n.get("title") or "").strip()
            if not title:
                continue
            url = f"https://www.hltv.org/news/{nid}/x"
            content_hash = _hash(url, title)
            if await session.scalar(
                select(NewsItem.id).where(NewsItem.content_hash == content_hash)
            ):
                continue
            session.add(
                NewsItem(
                    source_id=src.id,
                    url=url,
                    title=title,
                    raw_text=title,
                    clean_text=_strip_html(title),
                    published_at=now,  # "last news" are fresh; relative "posted" not parsed
                    content_hash=content_hash,
                )
            )
            saved += 1
        await session.commit()
    log.info("collect_hltv_news: %d new items", saved)
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(f"Collected {await collect_hltv_news()} HLTV news items.")


if __name__ == "__main__":
    asyncio.run(_main())
