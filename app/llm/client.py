"""Single LLM client wrapper around Polza.ai (OpenAI-compatible).

LLM is NOT the predictor — it classifies/extracts/explains/post-mortems.
Model tiers:
  - chat  (strong): explanations, post-mortems, hard reasoning
  - fast  (cheap):  news classification, relevance, simple extraction
  - embed:          embeddings for semantic memory (pgvector)
"""
from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI

from app.config import settings

Tier = Literal["chat", "fast"]


def _build_http_client() -> httpx.AsyncClient | None:
    if settings.polza_proxy_url:
        return httpx.AsyncClient(proxy=settings.polza_proxy_url, timeout=60.0)
    return None


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.polza_api_key,
            base_url=settings.polza_base_url,
            http_client=_build_http_client(),
        )

    def _model(self, tier: Tier) -> str:
        return settings.polza_chat_model if tier == "chat" else settings.polza_fast_model

    async def chat_text(
        self, system: str, user: str, tier: Tier = "fast", temperature: float = 0.2
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model(tier),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    async def chat_json(
        self, system: str, user: str, tier: Tier = "fast", temperature: float = 0.1
    ) -> dict[str, Any]:
        """Ask for strict JSON. Uses response_format when the provider supports it,
        falls back to parsing the text body otherwise."""
        try:
            resp = await self._client.chat.completions.create(
                model=self._model(tier),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
        except Exception:
            content = await self.chat_text(system, user, tier=tier, temperature=temperature)
        return json.loads(_strip_code_fence(content))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(
            model=settings.polza_embedding_model, input=texts
        )
        return [item.embedding for item in resp.data]

    async def list_models(self) -> list[str]:
        resp = await self._client.models.list()
        return [m.id for m in resp.data]


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


llm = LLMClient()
