"""SQLAlchemy 2.0 models — v1 (pure predictions, NO odds/betting tables).

Entities mirror the TZ §10, trimmed: no odds_snapshots / strategy_configs /
paper_bets. Added: player_social_accounts and news↔entity link tables so social
posts and general news flow through one pipeline.

Embeddings use halfvec(3072) (text-embedding-3-large) — hnsw-indexable.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import HALFVEC
from datetime import date as date_type

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

EMBEDDING_DIM = 3072


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class UpdatedMixin:
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )


# ── Core entities ────────────────────────────────────────────────────


class Team(Base, TimestampMixin, UpdatedMixin):
    __tablename__ = "teams"
    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str | None] = mapped_column(Text, unique=True)
    country: Mapped[str | None] = mapped_column(Text)
    pandascore_id: Mapped[str | None] = mapped_column(Text, index=True)
    bo3_id: Mapped[str | None] = mapped_column(Text, index=True)
    rank: Mapped[int | None] = mapped_column(Integer)  # bo3.gg world rank
    tier: Mapped[str | None] = mapped_column(Text)  # tier1 / tier2 / ...
    # roster strength = mean six-month player rating (bo3) of the active lineup;
    # the team-strength signal — the only feature that proved to carry signal.
    strength: Mapped[float | None] = mapped_column(Numeric)
    strength_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    rosters: Mapped[list["TeamRoster"]] = relationship(back_populates="team")


class Player(Base, TimestampMixin, UpdatedMixin):
    __tablename__ = "players"
    id: Mapped[uuid.UUID] = _pk()
    nickname: Mapped[str] = mapped_column(Text, nullable=False)
    real_name: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(Text)
    pandascore_id: Mapped[str | None] = mapped_column(Text, index=True)
    faceit_id: Mapped[str | None] = mapped_column(Text)
    steam_id: Mapped[str | None] = mapped_column(Text)

    social_accounts: Mapped[list["PlayerSocialAccount"]] = relationship(
        back_populates="player"
    )


class EntityAlias(Base, TimestampMixin):
    __tablename__ = "entity_aliases"
    id: Mapped[uuid.UUID] = _pk()
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)  # team/player/...
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    alias: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Numeric, default=1.0)


class TeamRoster(Base, TimestampMixin):
    __tablename__ = "team_rosters"
    id: Mapped[uuid.UUID] = _pk()
    team_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("teams.id"), nullable=False)
    player_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("players.id"), nullable=False
    )
    role: Mapped[str | None] = mapped_column(Text)
    active_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    active_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Numeric, default=1.0)

    team: Mapped["Team"] = relationship(back_populates="rosters")
    player: Mapped["Player"] = relationship()


class PlayerSocialAccount(Base, TimestampMixin, UpdatedMixin):
    """Which social handles to poll for a player (Twitter/X, Instagram, TG…)."""

    __tablename__ = "player_social_accounts"
    id: Mapped[uuid.UUID] = _pk()
    player_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("players.id"), nullable=False
    )
    platform: Mapped[str] = mapped_column(Text, nullable=False)  # twitter/instagram/tg
    handle: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    player: Mapped["Player"] = relationship(back_populates="social_accounts")


# ── Matches ──────────────────────────────────────────────────────────


class Match(Base, TimestampMixin, UpdatedMixin):
    __tablename__ = "matches"
    id: Mapped[uuid.UUID] = _pk()
    external_id: Mapped[str | None] = mapped_column(Text, index=True)
    source: Mapped[str | None] = mapped_column(Text)
    team_a_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    team_b_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    tournament_name: Mapped[str | None] = mapped_column(Text)
    tier: Mapped[str | None] = mapped_column(Text)
    format: Mapped[str | None] = mapped_column(Text)  # bo1/bo3/bo5
    is_lan: Mapped[bool | None] = mapped_column(Boolean)
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    status: Mapped[str | None] = mapped_column(Text, index=True)  # upcoming/live/finished
    winner_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    # roster integrity (bo3 new_participant): team is playing with a stand-in
    team_a_standin: Mapped[bool | None] = mapped_column(Boolean)
    team_b_standin: Mapped[bool | None] = mapped_column(Boolean)
    # manual result override — collectors must NOT clobber winner/status when set
    # (used when the source served a wrong result we corrected by hand)
    result_locked: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false", nullable=False
    )

    team_a: Mapped["Team"] = relationship(foreign_keys=[team_a_id])
    team_b: Mapped["Team"] = relationship(foreign_keys=[team_b_id])


class TeamRating(Base, UpdatedMixin):
    """Current Elo rating per team (rebuilt from finished matches)."""

    __tablename__ = "team_ratings"
    team_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("teams.id"), primary_key=True
    )
    elo: Mapped[float] = mapped_column(Numeric, nullable=False, default=1500.0)
    matches_played: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_match_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MatchMap(Base, TimestampMixin):
    __tablename__ = "match_maps"
    id: Mapped[uuid.UUID] = _pk()
    match_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matches.id"), nullable=False
    )
    map_name: Mapped[str] = mapped_column(Text, nullable=False)
    map_order: Mapped[int | None] = mapped_column(Integer)
    team_a_score: Mapped[int | None] = mapped_column(Integer)
    team_b_score: Mapped[int | None] = mapped_column(Integer)
    picked_by_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    winner_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))


class PlayerMatchStats(Base, TimestampMixin):
    __tablename__ = "player_match_stats"
    id: Mapped[uuid.UUID] = _pk()
    match_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matches.id"), nullable=False
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("players.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    map_name: Mapped[str | None] = mapped_column(Text)
    kills: Mapped[int | None] = mapped_column(Integer)
    deaths: Mapped[int | None] = mapped_column(Integer)
    assists: Mapped[int | None] = mapped_column(Integer)
    adr: Mapped[float | None] = mapped_column(Numeric)
    kast: Mapped[float | None] = mapped_column(Numeric)
    rating: Mapped[float | None] = mapped_column(Numeric)
    opening_kill_diff: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str | None] = mapped_column(Text)


class PlayerFaceitSnapshot(Base, TimestampMixin):
    __tablename__ = "player_faceit_snapshots"
    id: Mapped[uuid.UUID] = _pk()
    player_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("players.id"), nullable=False
    )
    faceit_id: Mapped[str | None] = mapped_column(Text)
    faceit_elo: Mapped[int | None] = mapped_column(Integer)
    skill_level: Mapped[int | None] = mapped_column(Integer)
    recent_winrate: Mapped[float | None] = mapped_column(Numeric)
    recent_matches: Mapped[int | None] = mapped_column(Integer)
    avg_kd: Mapped[float | None] = mapped_column(Numeric)
    avg_kr: Mapped[float | None] = mapped_column(Numeric)
    raw_json_path: Mapped[str | None] = mapped_column(Text)
    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


# ── News / embeddings ────────────────────────────────────────────────


class NewsSource(Base, TimestampMixin):
    __tablename__ = "news_sources"
    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    source_type: Mapped[str | None] = mapped_column(Text)  # rss/telegram/twitter
    reliability_score: Mapped[float] = mapped_column(Numeric, default=0.5)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class NewsItem(Base, TimestampMixin):
    __tablename__ = "news_items"
    id: Mapped[uuid.UUID] = _pk()
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("news_sources.id"))
    url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    raw_text: Mapped[str | None] = mapped_column(Text)
    clean_text: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    content_hash: Mapped[str | None] = mapped_column(Text, unique=True, index=True)
    dedup_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    raw_path: Mapped[str | None] = mapped_column(Text)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_critical: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class NewsEvent(Base, TimestampMixin):
    __tablename__ = "news_events"
    id: Mapped[uuid.UUID] = _pk()
    news_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_items.id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    event_subtype: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    importance: Mapped[float | None] = mapped_column(Numeric)
    confidence: Mapped[float | None] = mapped_column(Numeric)
    source_quality: Mapped[str | None] = mapped_column(Text)
    prediction_impact_direction: Mapped[str | None] = mapped_column(Text)
    prediction_impact_score: Mapped[float | None] = mapped_column(Numeric)
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class NewsEmbedding(Base, TimestampMixin):
    __tablename__ = "news_embeddings"
    id: Mapped[uuid.UUID] = _pk()
    news_item_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("news_items.id"))
    news_event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("news_events.id")
    )
    embedding: Mapped[list[float]] = mapped_column(HALFVEC(EMBEDDING_DIM))
    embedding_model: Mapped[str | None] = mapped_column(Text)
    text_chunk: Mapped[str | None] = mapped_column(Text)


class TeamNewsLink(Base, TimestampMixin):
    __tablename__ = "team_news_links"
    id: Mapped[uuid.UUID] = _pk()
    news_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_items.id"), nullable=False
    )
    team_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("teams.id"), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric, default=1.0)


class PlayerNewsLink(Base, TimestampMixin):
    __tablename__ = "player_news_links"
    id: Mapped[uuid.UUID] = _pk()
    news_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_items.id"), nullable=False
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("players.id"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Numeric, default=1.0)


class MatchRelevanceLink(Base, TimestampMixin):
    __tablename__ = "match_relevance_links"
    id: Mapped[uuid.UUID] = _pk()
    news_item_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_items.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matches.id"), nullable=False
    )
    relevance: Mapped[float] = mapped_column(Numeric, default=0.5)


# ── Predictions / post-mortems ───────────────────────────────────────


class PredictionSnapshot(Base, TimestampMixin):
    """Immutable record of what was known BEFORE the match (anti-hindsight)."""

    __tablename__ = "prediction_snapshots"
    id: Mapped[uuid.UUID] = _pk()
    match_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matches.id"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_versions: Mapped[dict | None] = mapped_column(JSONB)
    feature_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    retrieved_news_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True))
    )


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"
    id: Mapped[uuid.UUID] = _pk()
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("prediction_snapshots.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matches.id"), nullable=False
    )
    team_a_probability: Mapped[float] = mapped_column(Numeric, nullable=False)
    team_b_probability: Mapped[float] = mapped_column(Numeric, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric)
    risk_level: Mapped[str | None] = mapped_column(Text)
    fair_odds: Mapped[dict | None] = mapped_column(JSONB)
    feature_drivers: Mapped[dict | None] = mapped_column(JSONB)
    explanation: Mapped[dict | None] = mapped_column(JSONB)
    # self-check after the match:
    predicted_winner_team_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("teams.id")
    )
    was_correct: Mapped[bool | None] = mapped_column(Boolean)
    brier_score: Mapped[float | None] = mapped_column(Numeric)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DailyReview(Base, TimestampMixin):
    """Once-a-day reflection over settled results — persistent self-memory.

    conclusions = {"what_worked": [...], "what_failed": [...], "lessons": [...]}.
    Lessons are fed back into future prediction explanations.
    """

    __tablename__ = "daily_reviews"
    id: Mapped[uuid.UUID] = _pk()
    review_date: Mapped[date_type] = mapped_column(Date, unique=True, nullable=False)
    predictions_settled: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    accuracy: Mapped[float | None] = mapped_column(Numeric)
    avg_brier: Mapped[float | None] = mapped_column(Numeric)
    conclusions: Mapped[dict | None] = mapped_column(JSONB)
    raw_llm_output: Mapped[dict | None] = mapped_column(JSONB)


class Postmortem(Base, TimestampMixin):
    __tablename__ = "postmortems"
    id: Mapped[uuid.UUID] = _pk()
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id"), nullable=False
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("matches.id"), nullable=False
    )
    prediction_was_correct: Mapped[bool | None] = mapped_column(Boolean)
    suspected_failure_reasons: Mapped[dict | None] = mapped_column(JSONB)
    data_quality_issues: Mapped[dict | None] = mapped_column(JSONB)
    model_improvement_hypotheses: Mapped[dict | None] = mapped_column(JSONB)
    confidence_in_diagnosis: Mapped[float | None] = mapped_column(Numeric)
    raw_llm_output: Mapped[dict | None] = mapped_column(JSONB)


# ── Scheduler bookkeeping ────────────────────────────────────────────


class OddsSnapshot(Base, TimestampMixin):
    """Market odds for a match selection at a point in time (TZ §10.3).
    bookmaker='mock' for the test polygon; real books plug in via OddsProvider."""

    __tablename__ = "odds_snapshots"
    id: Mapped[uuid.UUID] = _pk()
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(Text, nullable=False)
    selection_team_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("teams.id"), nullable=False
    )
    odds_decimal: Mapped[float] = mapped_column(Numeric, nullable=False)
    implied_probability: Mapped[float | None] = mapped_column(Numeric)
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PaperBet(Base, TimestampMixin):
    """Virtual test bet auto-placed on the predicted winner at the model's fair
    odds (bo3 has no market odds). Balance = start + Σ pnl over settled bets."""

    __tablename__ = "paper_bets"
    id: Mapped[uuid.UUID] = _pk()
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id"), nullable=False, unique=True
    )
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id"), nullable=False)
    selection_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    stake: Mapped[float] = mapped_column(Numeric, nullable=False)
    odds: Mapped[float] = mapped_column(Numeric, nullable=False)
    result: Mapped[str | None] = mapped_column(Text)  # won / lost
    pnl: Mapped[float | None] = mapped_column(Numeric)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class StrategyBet(Base, TimestampMixin):
    """A paper bet under ONE named staking/selection strategy. Every strategy
    replays the same settled-prediction stream with its own rules + bankroll, so
    we can compare tactics head-to-head. Not unique per prediction (each strategy
    may bet it); unique per (strategy, prediction)."""

    __tablename__ = "strategy_bets"
    id: Mapped[uuid.UUID] = _pk()
    strategy: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    prediction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    match_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("matches.id"), nullable=False)
    selection_team_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("teams.id"))
    stake: Mapped[float] = mapped_column(Numeric, nullable=False)
    odds: Mapped[float] = mapped_column(Numeric, nullable=False)
    result: Mapped[str | None] = mapped_column(Text)  # won / lost
    pnl: Mapped[float | None] = mapped_column(Numeric)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("strategy", "prediction_id", name="uq_strategy_pred"),)


class Outbox(Base, TimestampMixin):
    """Durable Telegram queue — messages that failed to send (proxy down) land
    here and a scheduler job resends them, so nothing is lost on an outage."""

    __tablename__ = "outbox"
    id: Mapped[uuid.UUID] = _pk()
    text: Mapped[str] = mapped_column(Text, nullable=False)
    parse_mode: Mapped[str] = mapped_column(Text, default="HTML")
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RuntimeConfig(Base, UpdatedMixin):
    """Cross-process runtime overrides (e.g. model chosen via the /model bot
    command). Read by the LLM client; falls back to .env when unset."""

    __tablename__ = "runtime_config"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class SchedulerLock(Base):
    __tablename__ = "scheduler_locks"
    job_name: Mapped[str] = mapped_column(Text, primary_key=True)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String)
    last_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str | None] = mapped_column(Text)
