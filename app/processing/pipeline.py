"""End-to-end news pipeline: collect → classify → embed.

CLI:  python -m app.processing.pipeline
Used by the scheduler too (see app/scheduler/run.py).
"""
from __future__ import annotations

import asyncio
import logging

from app.collectors.news import collect_news
from app.config import settings
from app.processing.classifier import classify_unprocessed
from app.processing.embeddings import embed_unembedded

log = logging.getLogger("processing.pipeline")


async def run_news_pipeline(
    classify_limit: int = 30, embed_limit: int = 50, notify: bool = True
) -> dict[str, int]:
    from app.telegram.formatters import format_news_digest
    from app.telegram.notify import send_message

    collected = await collect_news()
    classified, digest = await classify_unprocessed(classify_limit)
    embedded = await embed_unembedded(embed_limit)
    # Notify ONLY when there's actually relevant CS2 news. The old "начинаю
    # анализ…" ping fired before classification, so an irrelevant item left a
    # dangling "разбираю…" with no follow-up (looked like it stalled). The digest
    # header already shows собрано/разобрано/релевантных.
    if notify and digest:
        await send_message(format_news_digest(digest, collected, classified))
    result = {
        "collected": collected,
        "classified": classified,
        "relevant": len(digest),
        "embedded": embedded,
    }
    log.info("news pipeline: %s", result)
    return result


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    result = await run_news_pipeline()
    print(result)


if __name__ == "__main__":
    asyncio.run(_main())
