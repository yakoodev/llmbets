"""Embed news_items into news_embeddings (halfvec) for semantic retrieval.

CLI:  python -m app.processing.embeddings [limit]
"""
from __future__ import annotations

import asyncio
import logging
import sys

from sqlalchemy import select

from app.config import settings
from app.db.models import NewsEmbedding, NewsItem
from app.db.session import SessionLocal
from app.llm.client import llm

log = logging.getLogger("processing.embeddings")

MAX_CHARS = 6000


async def embed_unembedded(limit: int = 50) -> int:
    async with SessionLocal() as session:
        embedded = select(NewsEmbedding.news_item_id).where(
            NewsEmbedding.news_item_id.isnot(None)
        )
        items = list(
            await session.scalars(
                select(NewsItem)
                .where(NewsItem.id.notin_(embedded))
                .order_by(NewsItem.fetched_at.desc())
                .limit(limit)
            )
        )
        if not items:
            return 0
        texts = [
            f"{i.title or ''}\n{(i.clean_text or i.raw_text or '')}"[:MAX_CHARS]
            for i in items
        ]
        vectors = await llm.embed(texts)
        for item, text, vec in zip(items, texts, vectors):
            session.add(
                NewsEmbedding(
                    news_item_id=item.id,
                    embedding=vec,
                    embedding_model=settings.polza_embedding_model,
                    text_chunk=text[:2000],
                )
            )
        await session.commit()
    log.info("embed_unembedded: embedded %d items", len(items))
    return len(items)


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n = await embed_unembedded(limit)
    print(f"Embedded {n} items.")


if __name__ == "__main__":
    asyncio.run(_main())
