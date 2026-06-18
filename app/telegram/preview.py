"""Send the most recent prediction as a formatted forecast (HTML test, no writes).

Run:  python -m app.telegram.preview
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.db.models import Match, Prediction, Team
from app.db.session import SessionLocal
from app.telegram.formatters import format_forecast
from app.telegram.notify import send_message


async def main() -> None:
    async with SessionLocal() as s:
        row = (
            await s.execute(
                select(Prediction, Match)
                .join(Match, Match.id == Prediction.match_id)
                .order_by(Prediction.created_at.desc())
                .limit(1)
            )
        ).first()
        if not row:
            print("No predictions yet.")
            return
        pred, match = row
        team_a = await s.get(Team, match.team_a_id)
        team_b = await s.get(Team, match.team_b_id)
        text = format_forecast(match, team_a, team_b, pred)
    ok = await send_message(text)
    print("sent" if ok else "send failed")


if __name__ == "__main__":
    asyncio.run(main())
