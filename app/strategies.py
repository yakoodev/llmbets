"""Paper-betting strategies — each replays the SAME settled-prediction stream
with its own selection filter + staking, on its own virtual bankroll, so tactics
compete head-to-head. Add/tune freely; `simulate` runs one over an event list.

An "event" (built in app.paper._build_events) is one settled prediction we could
bet: {pred_id, match_id, selection, odds, prob (our prob for the pick), mkt
(market implied for the pick, or None), risk, won, settled_at}. We always bet OUR
predicted winner; strategies differ in WHICH events they take and HOW they stake.
"""
from __future__ import annotations

# filter: bet this event? (default: always). Staking: flat `pct` of the running
# bankroll, or martingale (base_pct of START, doubled per consecutive loss).
STRATEGIES: dict[str, dict] = {
    "flat10": {"desc": "10% банка на каждый прогноз", "pct": 0.10},
    "flat3": {"desc": "3% банка (осторожный)", "pct": 0.03},
    "lowrisk": {
        "desc": "только низкий риск, 10%",
        "pct": 0.10,
        "filter": lambda e: (e["risk"] or "").lower() == "low",
    },
    "value": {
        "desc": "только value (наша вероятн. > рынка +3%), 5%",
        "pct": 0.05,
        "filter": lambda e: e["mkt"] is not None and e["prob"] > e["mkt"] + 0.03,
    },
    "martingale_hi": {
        "desc": "мартингейл на кэф > 2.0",
        "martingale": True,
        "base_pct": 0.03,
        "filter": lambda e: e["odds"] > 2.0,
    },
}


def simulate(name: str, spec: dict, events: list[dict], start: float) -> list[dict]:
    """Replay one strategy over the event stream → its list of placed bets."""
    bal = start
    losses = 0
    filt = spec.get("filter")
    bets: list[dict] = []
    for e in events:
        if filt and not filt(e):
            continue
        if spec.get("martingale"):
            stake = round(min(spec["base_pct"] * start * (2 ** losses), bal), 2)
        else:
            stake = round(spec["pct"] * bal, 2)
        if stake <= 0:
            break  # bankrupt — stop betting
        won = bool(e["won"])
        pnl = round(stake * (float(e["odds"]) - 1.0), 2) if won else -stake
        bal += pnl
        losses = 0 if won else losses + 1
        bets.append(
            {
                "pred_id": e["pred_id"],
                "match_id": e["match_id"],
                "selection": e["selection"],
                "odds": float(e["odds"]),
                "stake": stake,
                "won": won,
                "pnl": pnl,
                "settled_at": e["settled_at"],
            }
        )
    return bets
