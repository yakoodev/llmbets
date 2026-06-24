"""CS2 Telegram channels via the public web preview (t.me/s/<channel>).

Bots can't read channel history and Telethon needs a phone-code login; but
`t.me/s/<channel>` is the PUBLIC web preview — no auth, no bot — so we fetch and
parse recent posts from it (verified working from the VPS). Fed into the standard
news pipeline → classified + entity-linked. Add channels to CS2_CHANNELS.

CLI:  python -m app.collectors.tg_channels
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from app.collectors.news import _hash, _strip_html
from app.config import settings
from app.db.models import NewsItem, NewsSource
from app.db.session import SessionLocal

log = logging.getLogger("collector.tg_channels")

SOURCE_NAME = "Telegram CS2"
# public, ACTIVE channels (web-preview readable) with CS2 coverage. The classifier
# drops non-CS2 items. Add @usernames here. (escorenews/cs2/csgonews are dead.)
CS2_CHANNELS = ["newcsgo", "counter_strike2", "metaratings"]

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
_POST_RE = re.compile(r'data-post="([^"]+)"')
_TEXT_RE = re.compile(r'tgme_widget_message_text[^>]*>(.*?)</div>', re.S)
_TIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"')


async def _source(session) -> NewsSource:
    src = await session.scalar(select(NewsSource).where(NewsSource.name == SOURCE_NAME))
    if src is None:
        src = NewsSource(name=SOURCE_NAME, source_type="telegram", reliability_score=0.6)
        session.add(src)
        await session.flush()
    return src


async def collect_tg_channels(per_channel: int = 12) -> int:
    saved = 0
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=settings.news_max_age_days)
    async with httpx.AsyncClient(headers=_UA, timeout=25.0, follow_redirects=True) as client, \
            SessionLocal() as session:
        src = await _source(session)
        for ch in CS2_CHANNELS:
            try:
                r = await client.get(f"https://t.me/s/{ch}")
            except Exception as e:  # noqa: BLE001
                log.warning("tg fetch failed for %s: %s", ch, e)
                continue
            # each post carries data-post="channel/<id>"; split on it and take the
            # first message_text after each (that post's own text)
            blocks = r.text.split('data-post="')[1:]
            for block in blocks[-per_channel:]:
                post_id = block[: block.find('"')]
                text = _TEXT_RE.search(block)
                if not post_id or not text:
                    continue
                clean = _strip_html(text.group(1)).strip()
                if len(clean) < 15:
                    continue
                published = None
                tm = _TIME_RE.search(block)
                if tm:
                    try:
                        published = datetime.fromisoformat(tm.group(1))
                    except ValueError:
                        published = None
                if published and published < cutoff:
                    continue
                url = f"https://t.me/{post_id}"
                content_hash = _hash(url, clean[:120])
                if await session.scalar(
                    select(NewsItem.id).where(NewsItem.content_hash == content_hash)
                ):
                    continue
                session.add(
                    NewsItem(
                        source_id=src.id,
                        url=url,
                        title=clean[:140],
                        raw_text=clean,
                        clean_text=clean,
                        published_at=published,
                        content_hash=content_hash,
                    )
                )
                saved += 1
            await session.commit()
    log.info("collect_tg_channels: %d new items from %d channels", saved, len(CS2_CHANNELS))
    return saved


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(f"Collected {await collect_tg_channels()} TG channel items.")


if __name__ == "__main__":
    asyncio.run(_main())
