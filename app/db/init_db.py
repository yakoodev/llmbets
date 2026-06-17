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
