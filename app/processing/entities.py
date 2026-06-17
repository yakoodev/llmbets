"""Minimal entity resolver for v1 — exact/alias/ILIKE name matching.

Good enough for ~hundreds of teams/players; the fuzzy + LLM-arbitration resolver
from the TZ comes later. Logs nothing fancy; unresolved names just return None.
"""
from __future__ import annotations

import re

from sqlalchemy import func, select

from app.db.models import EntityAlias, Player, Team

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize(name: str) -> str:
    return _NORM_RE.sub("", (name or "").lower())


async def resolve_team(session, name: str) -> Team | None:
    if not name or not name.strip():
        return None
    norm = normalize(name)
    teams = list(await session.scalars(select(Team)))
    for t in teams:
        if normalize(t.name) == norm or (t.slug and normalize(t.slug) == norm):
            return t
    # alias table
    alias = await session.scalar(
        select(EntityAlias).where(
            EntityAlias.entity_type == "team",
            func.lower(EntityAlias.alias) == name.strip().lower(),
        )
    )
    if alias:
        return await session.get(Team, alias.entity_id)
    # loose contains as a last resort
    for t in teams:
        if norm and (norm in normalize(t.name) or normalize(t.name) in norm):
            return t
    return None


async def resolve_player(session, nickname: str) -> Player | None:
    if not nickname or not nickname.strip():
        return None
    norm = normalize(nickname)
    players = list(await session.scalars(select(Player)))
    for p in players:
        if normalize(p.nickname) == norm:
            return p
    alias = await session.scalar(
        select(EntityAlias).where(
            EntityAlias.entity_type == "player",
            func.lower(EntityAlias.alias) == nickname.strip().lower(),
        )
    )
    if alias:
        return await session.get(Player, alias.entity_id)
    return None
