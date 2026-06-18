# Deploy to VPS

## ✅ LIVE (since 2026-06-19) — production runs on the VPS, not the local PC
- Host: **95.85.253.249** (Poland, Ubuntu 24.04, 1 vCore / 2 GB / 20 GB), repo at `/root/llmbets`.
- Stack: `docker-compose.prod.yml` (postgres + api + bot + scheduler). **No proxy** —
  Poland reaches Telegram / bo3 / news / Polza directly (verified).
- The **local PC stack is stopped** (`docker compose down`); do NOT run the bot
  locally at the same time (two pollers → Telegram getUpdates conflict).
- **All further changes are deployed to the VPS:** edit locally → `git push` →
  on the VPS `cd /root/llmbets && git pull && docker compose -f docker-compose.prod.yml restart`
  (code is bind-mounted, so `restart` reloads it; plain `up -d` won't restart
  unchanged containers). If requirements.txt changed:
  `docker compose -f docker-compose.prod.yml build && ... up -d`.
- SSH is key-based (no password). Health: `/status` in the bot, or
  `curl localhost:8000/health` on the host.

---


Target: 2 vCPU / 2 GB / 20 GB, EU region (so Telegram/bo3/news work WITHOUT a
proxy). Ubuntu 22.04+.

## First-time setup
```bash
# 1. Docker
curl -fsSL https://get.docker.com | sh

# 2. Code
git clone https://github.com/yakoodev/llmbets.git
cd llmbets
cp .env.example .env
# Edit .env: fill POLZA_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
# PANDASCORE_API_KEY. Leave TELEGRAM_PROXY_URL and NEWS_PROXY_URL EMPTY
# (EU VPS reaches Telegram/news directly). Set PAPER_STAKE_PCT etc. as desired.

# 3. Verify Polza works from this IP (RU service — usually global, but check):
docker compose -f docker-compose.prod.yml build api
docker compose -f docker-compose.prod.yml run --rm api python -m app.llm.smoke_test
#   models + chat + embeddings OK  -> proxy not needed.
#   FAILS (geoblock)               -> set POLZA_PROXY_URL in .env, rerun.

# 4. Schema + history
docker compose -f docker-compose.prod.yml run --rm api python -m app.db.init_db
docker compose -f docker-compose.prod.yml run --rm api python -m app.collectors.bo3 backfill 20
docker compose -f docker-compose.prod.yml run --rm api python -m app.prediction.elo rebuild
docker compose -f docker-compose.prod.yml run --rm api python -m app.collectors.bo3 upcoming

# 5. Go live
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml logs -f scheduler
```

## Updates (every change)
```bash
cd llmbets
git pull
# code only (bind-mounted) → restart reloads it (up -d won't restart unchanged):
docker compose -f docker-compose.prod.yml restart
# if requirements.txt changed → rebuild first:
docker compose -f docker-compose.prod.yml build && docker compose -f docker-compose.prod.yml up -d
```

## Notes
- 2 GB RAM fits the trimmed stack (~0.6 GB) with headroom. If on 1 GB, add swap:
  `fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`
  and add it to /etc/fstab.
- Health: `curl localhost:8000/health`. System state: `/status` in the bot.
- No host ports are exposed publicly (api is localhost-only); only outbound.
