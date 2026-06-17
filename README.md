# CS2 LLM Prediction Bot

LLM-assisted analytics agent for pro CS2: collects matches + news, makes match
predictions with explanations, sends them to Telegram, then self-checks accuracy
after each match and keeps a post-mortem memory to improve over time.

> **v1 scope = pure predictions ("guessed right / wrong").** No odds, no betting,
> no bankroll. Odds & paper/real betting come later, only if the experiment proves
> the model is worth it. Full long-term spec lives in the TZ document.

## Stack
- Python 3.12, FastAPI, SQLAlchemy 2.x (async), PostgreSQL 16 + pgvector
- Redis, MinIO (object storage)
- LLM: Polza.ai (OpenAI-compatible) — classify / extract / explain / post-mortem
- Data: PandaScore (matches, rosters, results)
- News: RSS + public Telegram channels
- Telegram bot: aiogram 3.x, **mandatory proxy** via `TELEGRAM_PROXY_URL`
- Scheduling: APScheduler

## Quick start
```bash
cp .env.example .env        # then fill in the secrets
docker compose up -d postgres redis minio
docker compose build
docker compose up -d api bot scheduler
```

### Smoke-test the Polza key/models
```bash
docker compose run --rm api python -m app.llm.smoke_test
```
Prints the models your key can see, runs a chat completion, and an embedding
(with its dimension — we need this to size the pgvector column).

### Check the API
```bash
curl http://localhost:8000/health
```

### Telegram chat_id
Message your bot, send `/start` — it replies with your `chat_id`. Put it into
`TELEGRAM_CHAT_ID`.

## Status
Stage 1 (core skeleton): ✅ scaffolded — compose, config, Polza client + smoke
test, FastAPI `/health`, Telegram bot, scheduler shell.
Next: DB models + migrations, PandaScore collector, news pipeline, Elo + LLM
explanation, prediction → Telegram → self-check → post-mortem.
