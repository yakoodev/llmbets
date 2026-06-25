"""Bootstrap the schema for v1.

Order matters: the `vector` extension must exist before create_all (the
news_embeddings.embedding column is halfvec). After tables exist we add an
hnsw index for cosine similarity search.

Run:  docker compose run --rm api python -m app.db.init_db
Idempotent — safe to re-run.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.db import models  # noqa: F401  (register all tables on Base.metadata)
from app.db.base import Base
from app.db.session import engine

HNSW_INDEX = """
CREATE INDEX IF NOT EXISTS news_embeddings_embedding_hnsw
ON news_embeddings
USING hnsw (embedding halfvec_cosine_ops)
"""


async def main() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(HNSW_INDEX))
        # create_all doesn't ALTER existing tables — add new columns explicitly.
        await conn.execute(
            text("ALTER TABLE teams ADD COLUMN IF NOT EXISTS bo3_id TEXT")
        )
        await conn.execute(text("ALTER TABLE teams ADD COLUMN IF NOT EXISTS rank INT"))
        await conn.execute(text("ALTER TABLE teams ADD COLUMN IF NOT EXISTS strength NUMERIC"))
        await conn.execute(
            text("ALTER TABLE teams ADD COLUMN IF NOT EXISTS strength_at TIMESTAMPTZ")
        )
        await conn.execute(
            text("ALTER TABLE matches ADD COLUMN IF NOT EXISTS team_a_standin BOOLEAN")
        )
        await conn.execute(
            text("ALTER TABLE matches ADD COLUMN IF NOT EXISTS team_b_standin BOOLEAN")
        )
        await conn.execute(
            text(
                "ALTER TABLE matches ADD COLUMN IF NOT EXISTS "
                "result_locked BOOLEAN NOT NULL DEFAULT FALSE"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS teams_bo3_id_idx ON teams (bo3_id)")
        )
    print("Schema ready: extension + tables + hnsw index.")

    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname='public' ORDER BY tablename"
            )
        )
        tables = [r[0] for r in rows]
    print(f"{len(tables)} tables:")
    for t in tables:
        print(f"   - {t}")


if __name__ == "__main__":
    asyncio.run(main())
