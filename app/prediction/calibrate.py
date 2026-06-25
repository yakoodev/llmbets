"""Self-learning — fit the model's feature weights on settled results.

This is the real learning loop: replay EVERY settled match (its pre-match feature
snapshot + actual outcome) and fit a logistic regression over the feature
components [elo, form, news, h2h, drift, standin, market], minimising log-loss.
L2-regularised toward PRIOR_WEIGHTS (≈ the hand-tuned blend) with a pseudo-count,
so a small sample stays close to the prior and the weights move toward what the
DATA says as history accumulates — i.e. the model learns how much form / news /
H2H / the market actually predict outcomes. Stored in runtime_config
('learned_weights'); predict_match + the odds re-blend read it.

CLI:  python -m app.prediction.calibrate
"""
from __future__ import annotations

import asyncio
import json
import logging

from sqlalchemy import select

from app.config import settings
from app.db.models import Match, Prediction, PredictionSnapshot
from app.db.session import SessionLocal
from app.prediction.engine import (
    FEATURE_KEYS,
    PRIOR_WEIGHTS,
    _sigmoid,
    feature_x_from_snapshot,
)
from app.runtime_config import get_config, set_config

log = logging.getLogger("prediction.calibrate")

MIN_SAMPLES = 60       # need enough to split train/test meaningfully
TEST_FRAC = 0.25       # most-recent fraction held out for validation
PRIOR_STRENGTH = 40.0  # L2 pull toward PRIOR_WEIGHTS (pseudo-count; bigger = stiffer)
ITERS = 1200
LR = 0.3
ADOPT_MARGIN = 0.0     # learned must beat the live model's Brier on the holdout


def _fit(train: list) -> dict:
    data = train + [({k: -x[k] for k in FEATURE_KEYS}, 1.0 - y) for x, y in train]
    n = len(data)
    w = dict(PRIOR_WEIGHTS)
    lam = PRIOR_STRENGTH / n
    keys = list(FEATURE_KEYS)
    for _ in range(ITERS):
        gb = 0.0
        g = {k: 0.0 for k in keys}
        for x, y in data:
            e = _sigmoid(w["bias"] + sum(w[k] * x[k] for k in keys)) - y
            gb += e
            for k in keys:
                g[k] += e * x[k]
        w["bias"] -= LR * (gb / n)
        for k in keys:
            w[k] -= LR * (g[k] / n + lam * (w[k] - PRIOR_WEIGHTS[k]))
    return {k: round(v, 4) for k, v in w.items()}


def _brier_on(w: dict, data: list) -> float:
    return sum(
        (_sigmoid(w["bias"] + sum(w[k] * x[k] for k in FEATURE_KEYS)) - y) ** 2
        for x, y, *_ in data
    ) / len(data)


async def run_calibration() -> dict:
    async with SessionLocal() as session:
        rows = list(
            await session.execute(
                select(Prediction, Match, PredictionSnapshot)
                .join(Match, Match.id == Prediction.match_id)
                .join(PredictionSnapshot, PredictionSnapshot.id == Prediction.snapshot_id)
                .where(
                    Prediction.was_correct.isnot(None),
                    Match.winner_team_id.isnot(None),
                )
                .order_by(Prediction.created_at.asc())
            )
        )
    samples = []  # (x, y, live_brier) in chronological order
    for p, m, snap in rows:
        fd = (snap.feature_snapshot if snap else None) or p.feature_drivers or {}
        if not fd or m.winner_team_id not in (m.team_a_id, m.team_b_id):
            continue
        y = 1.0 if m.winner_team_id == m.team_a_id else 0.0
        lb = float(p.brier_score) if p.brier_score is not None else None
        samples.append((feature_x_from_snapshot(fd), y, lb))

    n = len(samples)
    if n < MIN_SAMPLES:
        # not enough to trust a fit — stay on the proven hand-tuned model
        await set_config("learned_weights", "")
        log.info("calibrate: %d settled (<%d) — staying on hand-tuned model", n, MIN_SAMPLES)
        return {"samples": n, "adopted": False}

    # walk-forward: fit on the older part, validate on the most-recent held-out
    # part. ADOPT the learned weights ONLY if they beat the LIVE model's Brier on
    # that unseen slice — otherwise keep the hand-tuned model (no overfit takeover).
    cut = int(n * (1 - TEST_FRAC))
    train = [(x, y) for x, y, _ in samples[:cut]]
    test = samples[cut:]
    w = _fit(train)
    brier_learned = _brier_on(w, test)
    live = [b for _, _, b in test if b is not None]
    brier_live = (sum(live) / len(live)) if live else None
    adopt = brier_live is not None and brier_learned < brier_live - ADOPT_MARGIN
    await set_config("learned_weights", json.dumps(w) if adopt else "")
    out = {
        "samples": n, "test": len(test),
        "brier_learned_test": round(brier_learned, 4),
        "brier_live_test": round(brier_live, 4) if brier_live is not None else None,
        "adopted": adopt,
        "weights": {k: round(v, 3) for k, v in w.items()},
    }
    log.info("calibrate: %s", out)
    return out


async def _main() -> None:
    logging.basicConfig(level=settings.log_level)
    print(await run_calibration())


if __name__ == "__main__":
    asyncio.run(_main())
