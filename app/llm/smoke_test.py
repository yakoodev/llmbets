"""Smoke-test the Polza.ai key & models.

Run inside the stack:   docker compose run --rm api python -m app.llm.smoke_test
Or locally (with .env):  python -m app.llm.smoke_test

Checks, in order:
  1. /models  -> what models the key can actually see
  2. chat     -> a one-line completion on the configured chat model
  3. embed    -> an embedding on the configured embedding model (+ its dimension)
"""
from __future__ import annotations

import asyncio

from app.config import settings
from app.llm.client import llm


async def main() -> None:
    print(f"Base URL: {settings.polza_base_url}")
    print(f"Key set:  {settings.is_configured_polza}\n")

    print("== 1. Listing models ==")
    try:
        models = await llm.list_models()
        for m in sorted(models):
            print(f"   - {m}")
        print(f"   ({len(models)} models)\n")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}\n")

    print(f"== 2. Chat ({settings.polza_chat_model}) ==")
    try:
        out = await llm.chat_text(
            system="You answer in exactly one short sentence.",
            user="Say hello and name yourself.",
            tier="chat",
        )
        print(f"   OK -> {out!r}\n")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}\n")

    print(f"== 3. Embeddings ({settings.polza_embedding_model}) ==")
    try:
        vecs = await llm.embed(["Natus Vincere wins the major"])
        print(f"   OK -> dim={len(vecs[0])}\n")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
