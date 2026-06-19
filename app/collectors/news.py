"""RSS news collector → news_items (with dedup).

Social posts (Twitter/TG) reuse the same news_items table later (source_type
twitter/telegram), so they flow through the same classify/embed/link pipeline.

HLTV / Escorenews direct RSS sit behind Cloudflare (403) — we lean on a Google
News aggregator query plus feeds that respond. Add more via DB/news_sources.

CLI:  python -m app.collectors.news collect
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sys
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
from sqlalchemy import select

from app.config import settings
from app.db.models import NewsItem, NewsSource
from app.db.session import SessionLocal

log = logging.getLogger("collector.news")

# (name, url, source_type, reliability)
# CS2-targeted feeds only. General gaming feeds (Dexerto/Esports.gg) are too
# noisy — the classifier rejects the off-topic majority anyway. Google News
# search feeds are CS2-scoped and reachable via the proxy.
DEFAULT_FEEDS = [
    (
        "Google News — CS2 (EN)",
        "https://news.google.com/rss/search?q=%22Counter-Strike+2%22+OR+%22CS2%22+esports"
        "&hl=en-US&gl=US&ceid=US:en",
        "rss",
        0.5,
    ),
    (
        "Google News — CS2 (RU)",
        "https://news.google.com/rss/search?q=CS2+%D0%BA%D0%B8%D0%B1%D0%B5%D1%80%D1%81%D0%BF%D0%BE%D1%80%D1%82+OR+%D0%BA%D0%BE%D0%BD%D1%82%D1%80-%D1%81%D1%82%D1%80%D0%B0%D0%B9%D0%BA"
        "&hl=ru&gl=RU&ceid=RU:ru",
        "rss",
        0.5,
    ),
    ("VPEsports", "https://vpesports.com/feed", "rss", 0.5),
]

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _TAG_RE.sub("", text or "").strip()


def _hash(*parts: str) -> str:
    return hashlib.sha256("|".join(p or "" for p in parts).encode()).hexdigest()


_UA = "Mozilla/5.0 (compatible; cs2-llm-bot/0.1; +rss)"


async def _fetch_feed(url: str):
    """Fetch bytes ourselves (robust headers/timeout/redirects), then parse.
    Returns a feedparser result or None on failure. Accept-Encoding: identity
    avoids gzip IncompleteRead errors some feeds trigger."""
    headers = {
        "User-Agent": _UA,
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
        "Accept-Encoding": "identity",
    }
    proxy = settings.news_proxy_url or None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(
                timeout=25.0, follow_redirects=True, headers=headers, proxy=proxy
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return feedparser.parse(resp.content)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "feed fetch failed (try %d) %s: %s: %s",
                attempt,
                url,
                type(e).__name__,
                e,
            )
    return None


def _published(entry) -> datetime | None:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if st:
        return datetime(*st[:6], tzinfo=timezone.utc)
    return None


async def seed_sources() -> None:
    async with SessionLocal() as session:
        for name, url, stype, rel in DEFAULT_FEEDS:
            exists = await session.scalar(
                select(NewsSource).where(NewsSource.url == url)
            )
            if exists is None:
                session.add(
                    NewsSource(
                        name=name, url=url, source_type=stype, reliability_score=rel
                    )
                )
        await session.commit()


async def collect_news(per_feed: int = 50) -> int:
    await seed_sources()
    saved = 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.news_max_age_days)
    async with SessionLocal() as session:
        sources = list(
            await session.scalars(
                select(NewsSource).where(
                    NewsSource.enabled.is_(True), NewsSource.source_type == "rss"
                )
            )
        )
        for src in sources:
            parsed = await _fetch_feed(src.url)
            if parsed is None:
                continue
            entries = parsed.entries[:per_feed]
            log.info("feed %s: %d entries", src.name, len(entries))
            for e in entries:
                published = _published(e)
                if published and published < cutoff:
                    continue  # stale — feeds return old articles by relevance
                link = e.get("link")
                title = e.get("title")
                summary = e.get("summary") or e.get("description") or ""
                clean = _strip_html(summary)
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
                        clean_text=clean,
                        published_at=published,
                        content_hash=content_hash,
                    )
                )
                saved += 1
            await session.commit()
    log.info("collect_news: %d new items", saved)
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    n = await collect_news()
    print(f"Collected {n} new news items.")


if __name__ == "__main__":
    asyncio.run(_main())
