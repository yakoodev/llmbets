"""Central app configuration. All secrets come from env only (see .env.example)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_env: str = "dev"
    app_name: str = "cs2-llm-bot"
    log_level: str = "INFO"

    # Infra
    database_url: str = "postgresql+asyncpg://user:pass@postgres:5432/cs2bot"
    redis_url: str = "redis://redis:6379/0"
    object_storage_endpoint: str = "http://minio:9000"
    object_storage_access_key: str = "minio"
    object_storage_secret_key: str = "minio123"
    object_storage_bucket: str = "cs2bot"

    # Polza.ai (OpenAI-compatible)
    polza_api_key: str = "replace_me"
    polza_base_url: str = "https://polza.ai/api/v1"
    polza_chat_model: str = "replace_me"
    polza_fast_model: str = "replace_me"
    polza_embedding_model: str = "replace_me"
    polza_proxy_url: str = ""

    # Telegram
    telegram_bot_token: str = "replace_me"
    telegram_chat_id: str = "replace_me"
    telegram_proxy_url: str = ""

    # bo3.gg — primary match/result source (HLTV-grade coverage, open API)
    bo3_base_url: str = "https://api.bo3.gg"
    bo3_proxy_url: str = ""  # usually the same proxy (egress blocked in-region)
    bo3_upcoming_days: int = 7

    # PandaScore (legacy; kept for optional reference, no longer the match source)
    pandascore_api_key: str = "replace_me"
    pandascore_base_url: str = "https://api.pandascore.co"
    # Which tournament tiers to track (PandaScore: s>a>b>c>d). Tier-1/2 ≈ s,a,b,c.
    pandascore_tiers: str = "s,a,b,c"
    pandascore_upcoming_days: int = 7

    @property
    def tracked_tiers(self) -> set[str]:
        return {t.strip().lower() for t in self.pandascore_tiers.split(",") if t.strip()}

    # News fetching often needs the proxy too (egress blocked in-region).
    news_proxy_url: str = ""

    # ── LLM cost controls ────────────────────────────────────────────
    # Which model tier explains predictions: "fast" (cheap) or "chat" (strong).
    # The probability is computed by Elo+form+news (no LLM), so a cheap model
    # here only makes the prose simpler, not the prediction worse.
    explain_model_tier: str = "fast"
    # Post-mortem tier — defaults cheap too (high volume once tier-d matches
    # settle). Bump to "chat" for deeper error analysis at higher token cost.
    postmortem_model_tier: str = "fast"
    # Only predict matches starting within this many hours (don't burn tokens
    # forecasting the whole week ahead; matches get predicted as they approach).
    prediction_horizon_hours: int = 48
    # Daily review runs once/day — strong model is fine (1 call/day).
    daily_review_model_tier: str = "chat"
    daily_review_hour_utc: int = 21

    # ── Paper betting (test balance) ─────────────────────────────────
    # No market odds from bo3 → bet at the model's fair odds (1/prob) on the
    # predicted winner. Balance = start + Σ pnl. Pure calibration test.
    paper_start_balance: float = 1000.0
    paper_stake: float = 10.0
    # Odds provider: "mock" (test polygon — prices off Elo + vig) or a real one
    # later. odds_margin = bookmaker overround/vig. min_edge = value threshold:
    # only paper-bet when model prob exceeds market implied prob by this much.
    odds_provider: str = "mock"
    odds_margin: float = 0.06
    min_edge: float = 0.03

    # Scheduling (minutes)
    news_collect_interval_minutes: int = 30
    match_schedule_interval_minutes: int = 60
    result_collect_interval_minutes: int = 10
    prediction_interval_minutes: int = 15

    @property
    def is_configured_polza(self) -> bool:
        return self.polza_api_key not in ("", "replace_me")

    @property
    def is_configured_telegram(self) -> bool:
        return self.telegram_bot_token not in ("", "replace_me")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
