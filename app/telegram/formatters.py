"""Render predictions/results/news into Telegram HTML messages.

Telegram HTML: <b> <i> <code> <a> and <blockquote expandable> (collapsible).
Long reasoning goes into an expandable blockquote so messages stay compact.
All dynamic text is escaped.
"""
from __future__ import annotations

import html
from datetime import datetime

from app.db.models import Match, Prediction, Team


def esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def _bars(pa: float, pb: float) -> str:
    """Tiny visual bar for the favourite side."""
    filled = round(pa / 10)
    return "▓" * filled + "░" * (10 - filled)


def _fmt_when(dt: datetime | None) -> str:
    return dt.strftime("%d.%m %H:%M UTC") if dt else "—"


def _bullets(items, limit: int = 5) -> str:
    out = [f"• {esc(x)}" for x in (items or [])[:limit] if x]
    return "\n".join(out) if out else "• —"


def format_forecast(match: Match, team_a: Team, team_b: Team, pred: Prediction) -> str:
    pa = float(pred.team_a_probability) * 100
    pb = float(pred.team_b_probability) * 100
    a, b = esc(team_a.name), esc(team_b.name)
    fav = a if pred.team_a_probability >= pred.team_b_probability else b
    exp = pred.explanation or {}
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(pred.risk_level, "⚪")

    head = [
        "🎯 <b>CS2 Прогноз</b>",
        "",
        f"<b>{a}</b>  vs  <b>{b}</b>",
        f"🏆 {esc(match.tournament_name or '—')}",
        f"🎮 {esc((match.format or '—').upper())} · 🕒 {_fmt_when(match.scheduled_at)}",
        "",
        f"<code>{a[:14]:<14}</code> <b>{pa:4.0f}%</b>",
        f"<code>{b[:14]:<14}</code> <b>{pb:4.0f}%</b>",
        "",
        f"⭐ Фаворит: <b>{fav}</b>",
        f"{risk_emoji} Confidence <b>{float(pred.confidence):.2f}</b> · Risk <b>{esc(pred.risk_level)}</b>",
    ]
    if exp.get("short_summary"):
        head += ["", f"<i>{esc(exp['short_summary'])}</i>"]

    # long reasoning hidden in an expandable quote
    detail = ["<b>Почему:</b>", _bullets(exp.get("main_reasons")),
              "", "<b>Риски:</b>", _bullets(exp.get("risks"))]
    if exp.get("data_quality_warnings"):
        detail += ["", "<b>⚠️ Качество данных:</b>", _bullets(exp.get("data_quality_warnings"))]
    quote = "<blockquote expandable>" + "\n".join(detail) + "</blockquote>"

    tail = ["", "<i>Аналитический прогноз, не финансовый совет.</i>"]
    return "\n".join(head) + "\n\n" + quote + "\n" + "\n".join(tail)


def format_results_summary(results: list[dict]) -> str:
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = (correct / n * 100) if n else 0
    avg_brier = (sum(r["brier"] for r in results) / n) if n else 0
    lines = [
        "📊 <b>Сверка результатов</b>",
        f"✅ <b>{correct}</b> / ❌ <b>{n - correct}</b> · точность <b>{acc:.0f}%</b> · ср. Brier <b>{avg_brier:.3f}</b>",
    ]
    rows = []
    for r in results:
        mark = "✅" if r["correct"] else "❌"
        rows.append(
            f"{mark} <b>{esc(r['winner'])}</b> "
            f"<i>({esc(r['team_a'])} vs {esc(r['team_b'])})</i> — "
            f"ставили на {esc(r['predicted'])} {r['prob']:.0f}%"
        )
    body = "\n".join(rows)
    if n > 6:
        body = f"<blockquote expandable>{body}</blockquote>"
    return "\n".join(lines) + "\n\n" + body


def format_news_digest(entries: list[dict], collected: int, processed: int) -> str:
    rel = len(entries)
    head = (
        f"🗂 <b>Новости обработаны</b>\n"
        f"собрано {collected} · разобрано {processed} · релевантных <b>{rel}</b>"
    )
    if not entries:
        return head + "\n\n<i>Ничего значимого по CS2 не нашёл.</i>"
    rows = []
    for e in entries[:15]:
        tag = "⚠️ " if e.get("critical") else ""
        where = []
        if e.get("teams"):
            where.append("команды: " + ", ".join(esc(t) for t in e["teams"]))
        if e.get("players"):
            where.append("игроки: " + ", ".join(esc(p) for p in e["players"]))
        if e.get("matches"):
            where.append(f"матчей: {e['matches']}")
        where_str = (" → " + "; ".join(where)) if where else ""
        rows.append(f"{tag}<b>[{esc(e['event_type'])}]</b> {esc(e.get('summary') or '')}{where_str}")
    body = "\n".join(rows)
    return head + "\n\n<blockquote expandable>" + body + "</blockquote>"


# ── compact lines for bot commands ───────────────────────────────────


def prediction_line(team_a: str, team_b: str, pa: float, pb: float, when, risk) -> str:
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
    fav_a = pa >= pb
    a = f"<b>{esc(team_a)}</b>" if fav_a else esc(team_a)
    b = f"<b>{esc(team_b)}</b>" if not fav_a else esc(team_b)
    return (
        f"{risk_emoji} {_fmt_when(when)} · {a} {pa*100:.0f}% — {b} {pb*100:.0f}%"
    )
