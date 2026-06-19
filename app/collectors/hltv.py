"""HLTV results — authoritative CS2 results (free, headless).

HLTV is the gold-standard CS2 results source and is Cloudflare-protected, but
hltv-async-api passes it (verified on the VPS: returned "G2 1 - Spirit 2" for
IEM Cologne — the exact result bo3 got WRONG). We treat HLTV as the AUTHORITATIVE
results source: when it has a decisive score for one of our recent matches we set
the winner and LOCK it (result_locked → bo3 can't clobber it), re-settling the
prediction if the winner differs from what was stored.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from app.db.models import Match, PaperBet, Postmortem, Prediction, Team
from app.db.session import SessionLocal
from app.processing.entities import normalize

log = logging.getLogger("collector.hltv")


async def fetch_results() -> list[dict]:
    """Recent HLTV results: [{team1, team2, score1, score2, event, ...}]."""
    from hltv_async_api import Hltv

    hltv = Hltv(timeout=30)
    try:
        return (await hltv.get_results()) or []
    finally:
        for attr in ("close_session", "close"):
            fn = getattr(hltv, attr, None)
            if fn:
                try:
                    await fn()
                except Exception:  # noqa: BLE001
                    pass
                break


def _nm(x: str, y: str) -> bool:
    x, y = normalize(x), normalize(y)
    return bool(x) and bool(y) and (
        x == y or (len(x) >= 4 and len(y) >= 4 and (x in y or y in x))
    )


async def apply_results() -> int:
    """Set winners on our recent matches from HLTV (authoritative); lock them.

    Best-effort: never raises into the scheduler — bo3 stays the fallback."""
    try:
        results = await fetch_results()
    except Exception as e:  # noqa: BLE001
        log.warning("hltv fetch failed: %s", e)
        return 0
    if not results:
        return 0

    now = datetime.now(timezone.utc)
    applied = 0
    async with SessionLocal() as session:
        recent = list(
            await session.scalars(
                select(Match).where(
                    Match.scheduled_at > now - timedelta(days=3),
                    Match.team_a_id.isnot(None),
                    Match.team_b_id.isnot(None),
                )
            )
        )
        for m in recent:
            ta = await session.get(Team, m.team_a_id)
            tb = await session.get(Team, m.team_b_id)
            for r in results:
                try:
                    s1, s2 = int(r.get("score1")), int(r.get("score2"))
                except (TypeError, ValueError):
                    continue
                t1, t2 = r.get("team1", ""), r.get("team2", "")
                if _nm(t1, ta.name) and _nm(t2, tb.name):
                    hi, lo = m.team_a_id, m.team_b_id
                elif _nm(t1, tb.name) and _nm(t2, ta.name):
                    hi, lo = m.team_b_id, m.team_a_id
                else:
                    continue
                if s1 == s2:
                    break  # not decisive / draw — skip
                winner = hi if s1 > s2 else lo
                if m.winner_team_id == winner and m.result_locked:
                    break  # already correct + locked
                changed = m.winner_team_id != winner
                m.winner_team_id = winner
                m.status = "finished"
                m.result_locked = True  # HLTV is authoritative
                if changed:
                    for p in list(
                        await session.scalars(
                            select(Prediction).where(Prediction.match_id == m.id)
                        )
                    ):
                        await session.execute(
                            delete(PaperBet).where(PaperBet.prediction_id == p.id)
                        )
                        await session.execute(
                            delete(Postmortem).where(Postmortem.prediction_id == p.id)
                        )
                        p.was_correct = None
                        p.brier_score = None
                        p.settled_at = None
                    log.info(
                        "HLTV corrected winner: %s vs %s -> %s",
                        ta.name, tb.name, ta.name if winner == m.team_a_id else tb.name,
                    )
                applied += 1
                break
        await session.commit()
    log.info("hltv apply_results: %d matches set from HLTV", applied)
    return applied
