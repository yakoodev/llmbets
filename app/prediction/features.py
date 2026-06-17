"""Numeric features that nudge the Elo baseline: recent form + news impact.

These turn signals the LLM used to only *narrate* into actual numbers that move
the probability (in logit space, so Elo stays the anchor).
"""
from __future__ import annotations

from sqlalchemy import select

from app.db.models import (
    Match,
    MatchRelevanceLink,
    NewsEvent,
    TeamNewsLink,
)

# How much to trust a source when converting news into a numeric signal.
_QUALITY_W = {"official": 1.0, "reputable": 0.7, "rumor": 0.3, "unknown": 0.4}
_DIR_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


async def recent_form(session, team_id, n: int = 10) -> tuple[float, int]:
    """Win rate over the team's last n decided matches. (0.5, 0) if no history."""
    matches = list(
        await session.scalars(
            select(Match)
            .where(
                (Match.team_a_id == team_id) | (Match.team_b_id == team_id),
                Match.winner_team_id.isnot(None),
            )
            .order_by(Match.scheduled_at.desc().nullslast())
            .limit(n)
        )
    )
    if not matches:
        return 0.5, 0
    wins = sum(1 for m in matches if m.winner_team_id == team_id)
    return wins / len(matches), len(matches)


async def news_signal(session, match: Match) -> tuple[float, float, list[dict]]:
    """Signed news impact for team_a and team_b, attributed via team mentions.

    Each relevant news event contributes sign(direction) * importance * quality.
    A news item is attributed to whichever of the two match teams it mentions
    (skipped if it mentions both or neither — too ambiguous to take sides)."""
    rows = list(
        await session.execute(
            select(NewsEvent, MatchRelevanceLink.news_item_id)
            .join(MatchRelevanceLink, MatchRelevanceLink.news_item_id == NewsEvent.news_item_id)
            .where(MatchRelevanceLink.match_id == match.id)
        )
    )
    sig_a = sig_b = 0.0
    details: list[dict] = []
    for ev, item_id in rows:
        teams = set(
            await session.scalars(
                select(TeamNewsLink.team_id).where(TeamNewsLink.news_item_id == item_id)
            )
        )
        in_a = match.team_a_id in teams
        in_b = match.team_b_id in teams
        if in_a == in_b:  # both or neither → ambiguous, skip
            continue
        importance = float(ev.importance) if ev.importance is not None else 0.3
        quality = _QUALITY_W.get(ev.source_quality or "unknown", 0.4)
        sign = _DIR_SIGN.get(ev.prediction_impact_direction or "neutral", 0.0)
        magnitude = sign * importance * quality
        if in_a:
            sig_a += magnitude
        else:
            sig_b += magnitude
        details.append(
            {
                "event_type": ev.event_type,
                "team": "a" if in_a else "b",
                "direction": ev.prediction_impact_direction,
                "source_quality": ev.source_quality,
                "magnitude": round(magnitude, 3),
            }
        )
    # bound so a flood of weak rumors can't dominate Elo
    return max(-1.0, min(1.0, sig_a)), max(-1.0, min(1.0, sig_b)), details
