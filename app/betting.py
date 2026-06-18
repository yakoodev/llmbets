"""Betting adapter interface (TZ §21).

Paper mode only for now: real bet placement is DISABLED by default and must
never be enabled without an explicit, separate confirmation + a real bookmaker
integration. This is the seam where a real book (when chosen) plugs in.
"""
from __future__ import annotations

from app.config import settings


class BettingAdapter:
    """Real-money betting. Disabled unless ENABLE_REAL_BETTING is true AND a
    concrete bookmaker integration is implemented."""

    name = "disabled"

    async def place_bet(self, match_id, selection_team_id, stake, odds) -> dict:
        raise RuntimeError(
            "Real betting is disabled. Use paper mode (app/paper.py). "
            "Enable only with a real bookmaker integration + explicit confirmation."
        )


def get_adapter() -> BettingAdapter:
    # No real bookmaker wired yet — always the disabled adapter for now.
    _ = settings  # future: branch on settings.betting_provider
    return BettingAdapter()
