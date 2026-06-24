"""Self-calibration — tune the market-blend weight from settled results.

With few settled predictions, fitting ALL model weights would overfit. So we
tune ONE high-leverage knob: how hard to trust the bookmaker line vs our model
(W_MARKET). Grid-search the weight that minimises Brier on settled matches, then
SHRINK it toward the 0.6 prior with a pseudo-count so a small sample can't yank
it. The result is stored in runtime_config('w_market'); predict_match and the
odds re-blend read it. Runs daily — the bot genuinely self-tunes as data grows.

CLI:  python -m app.prediction.calibrate
"""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from app.config import settings
from app.db.models import Match, Prediction
from app.db.session import SessionLocal
from app.prediction.engine import W_MARKET, _logit, _sigmoid
from app.runtime_config import set_config

log = logging.getLogger("prediction.calibrate")

MIN_SAMPLES = 40
PRIOR_STRENGTH = 50  # pseudo-count shrinking the fit toward the W_MARKET prior


async def run_calibration() -> dict:
    async with SessionLocal() as session:
        rows = list(
            await session.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .where(
                    Prediction.was_correct.isnot(None),
                    Match.winner_team_id.isnot(None),
                )
            )
        )
    data = []
    for p, m in rows:
        fd = p.feature_drivers or {}
        mp, md = fd.get("market_prob_a"), fd.get("model_prob_a")
        if mp is None or md is None:
            continue
        if m.winner_team_id not in (m.team_a_id, m.team_b_id):
            continue
        y = 1.0 if m.winner_team_id == m.team_a_id else 0.0
        data.append((float(md), float(mp), y))

    n = len(data)
    if n < MIN_SAMPLES:
        log.info("calibrate: %d usable samples (<%d) — keep W_MARKET=%.2f", n, MIN_SAMPLES, W_MARKET)
        return {"samples": n, "applied": False, "w_market": W_MARKET}

    def brier(w: float) -> float:
        return sum(
            (_sigmoid(w * _logit(mp) + (1.0 - w) * _logit(md)) - y) ** 2
            for md, mp, y in data
        ) / n

    best_w, best_b = W_MARKET, brier(W_MARKET)
    for i in range(0, 101, 5):
        w = i / 100.0
        b = brier(w)
        if b < best_b:
            best_w, best_b = w, b

    # shrink toward the prior so a small sample can't over-commit
    w_final = round((n * best_w + PRIOR_STRENGTH * W_MARKET) / (n + PRIOR_STRENGTH), 3)
    await set_config("w_market", str(w_final))
    out = {
        "samples": n,
        "best_w": round(best_w, 2),
        "w_market": w_final,
        "brier_fit": round(best_b, 4),
        "brier_default": round(brier(W_MARKET), 4),
        "applied": True,
    }
    log.info("calibrate: %s", out)
    return out


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(await run_calibration())


if __name__ == "__main__":
    asyncio.run(_main())
