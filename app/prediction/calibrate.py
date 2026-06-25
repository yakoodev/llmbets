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

MIN_SAMPLES = 60       # need enough for a stable cross-validated estimate
CV_FOLDS = 8           # k-fold CV for the adopt decision (robust on small data)
PRIOR_STRENGTH = 40.0  # L2 pull toward PRIOR_WEIGHTS (pseudo-count; bigger = stiffer)
ITERS = 1200
LR = 0.3
ADOPT_MARGIN = 0.003   # learned must beat the live model's CV-Brier by this margin


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

    # k-fold CV: for each fold, fit on the other folds and score the held-out
    # matches — both the learned model AND the LIVE recorded Brier on the SAME
    # matches. Averaging over all folds (not one lucky split) gives a robust
    # estimate. ADOPT only if learned genuinely beats live across the whole
    # history by a margin — otherwise keep the proven hand-tuned model.
    sl, slive, c = 0.0, 0.0, 0
    for k in range(CV_FOLDS):
        train = [(x, y) for i, (x, y, _) in enumerate(samples) if i % CV_FOLDS != k]
        wf = _fit(train)
        for x, y, lb in (samples[i] for i in range(n) if i % CV_FOLDS == k):
            if lb is None:
                continue
            sl += (_sigmoid(wf["bias"] + sum(wf[kk] * x[kk] for kk in FEATURE_KEYS)) - y) ** 2
            slive += lb
            c += 1
    brier_learned = (sl / c) if c else None
    brier_live = (slive / c) if c else None
    adopt = (
        brier_learned is not None
        and brier_live is not None
        and brier_learned < brier_live - ADOPT_MARGIN
    )
    w = _fit([(x, y) for x, y, _ in samples])  # final weights fit on ALL data
    await set_config("learned_weights", json.dumps(w) if adopt else "")
    out = {
        "samples": n, "cv_scored": c,
        "brier_learned_cv": round(brier_learned, 4) if brier_learned is not None else None,
        "brier_live_cv": round(brier_live, 4) if brier_live is not None else None,
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
