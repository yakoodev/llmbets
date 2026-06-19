"""Redeliver Telegram messages that failed to send (proxy outage).

A scheduler job calls drain() periodically; messages persist in the `outbox`
table until they go through, so a long proxy downtime loses nothing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.db.models import Outbox
from app.db.session import SessionLocal

log = logging.getLogger("telegram.outbox")


async def drain(limit: int = 50) -> int:
    from app.telegram.notify import _raw_send

    target = settings.telegram_chat_id
    if target in ("", "replace_me"):
        return 0
    sent = 0
    async with SessionLocal() as session:
        rows = list(
            await session.scalars(
                select(Outbox)
                .where(Outbox.sent_at.is_(None))
                .order_by(Outbox.created_at.asc())
                .limit(limit)
            )
        )
        MAX_ATTEMPTS = 8
        for row in rows:
            row.attempts += 1
            # truncate to Telegram's hard limit so an over-long message isn't a
            # permanent failure that blocks the queue
            if await _raw_send(row.text[:4096], target, row.parse_mode or "HTML"):
                row.sent_at = datetime.now(timezone.utc)
                sent += 1
            elif row.attempts >= MAX_ATTEMPTS:
                # poison message (e.g. malformed HTML Telegram rejects) — dead-letter
                # it so it stops head-of-line-blocking everything behind it
                row.sent_at = datetime.now(timezone.utc)
                log.warning(
                    "outbox: dead-lettering message %s after %d failed attempts",
                    row.id, row.attempts,
                )
            else:
                break  # likely transient (network) — stop, retry whole batch next cycle
        await session.commit()
    if sent:
        log.info("outbox: redelivered %d messages", sent)
    return sent
