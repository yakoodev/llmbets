"""Render predictions/results/news into Telegram HTML messages.

Times shown in MSK (UTC+3, no DST). Favourite (predicted winner) is highlighted
with 🏆 + bold. Long reasoning goes into an expandable blockquote. All dynamic
text is escaped.
"""
from __future__ import annotations

import html
from datetime import datetime, timedelta

from app.db.models import Match, Prediction, Team

MSK = timedelta(hours=3)


def esc(text) -> str:
    return html.escape(str(text)) if text is not None else ""


def team_name(name) -> str:
    """Escape + stop Telegram auto-linking names that look like domains
    ("Virtus.pro" → ".pro" TLD). U+2060 (word joiner) is invisible & no-break."""
    return esc(name).replace(".", ".⁠")


def fmt_when(dt: datetime | None) -> str:
    return (dt + MSK).strftime("%d.%m %H:%M МСК") if dt else "—"


def fmt_day(dt: datetime | None) -> str:
    return (dt + MSK).strftime("%d.%m") if dt else "—"


def fmt_time(dt: datetime | None) -> str:
    return (dt + MSK).strftime("%H:%M") if dt else "--:--"


def _risk_emoji(risk) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")


def _bullets(items, limit: int = 5) -> str:
    out = [f"• {esc(x)}" for x in (items or [])[:limit] if x]
    return "\n".join(out) if out else "• —"


def _side(name: str, pct: float, fav: bool) -> str:
    if fav:  # predicted winner — emphasised, no emoji
        return f"<b><u>{team_name(name)} {pct:.0f}%</u></b>"
    return f"{team_name(name)} {pct:.0f}%"


def prediction_line(team_a, team_b, pa, pb, when, risk, settled=None) -> str:
    fav_a = pa >= pb
    a = _side(team_a, pa * 100, fav_a)
    b = _side(team_b, pb * 100, not fav_a)
    mark = ""
    if settled is True:
        mark = "  ✅"
    elif settled is False:
        mark = "  ❌"
    return f"{_risk_emoji(risk)} <code>{fmt_time(when)}</code>  {a}  —  {b}{mark}"


def format_prediction_list(title: str, items: list[dict], empty: str) -> str:
    """items: dicts with a,b,pa,pb,when,risk[,settled]. Grouped by MSK day."""
    if not items:
        return f"<b>{esc(title)}</b>\n\n{esc(empty)}"
    by_day: dict[str, list[dict]] = {}
    for it in items:
        by_day.setdefault(fmt_day(it["when"]), []).append(it)
    out = [f"<b>{esc(title)}</b>"]
    for day in by_day:
        out.append(f"\n📅 <b>{esc(day)}</b>")
        for it in by_day[day]:
            out.append(
                prediction_line(
                    it["a"], it["b"], it["pa"], it["pb"], it["when"],
                    it.get("risk"), it.get("settled"),
                )
            )
    return "\n".join(out)


def format_forecast(match: Match, team_a: Team, team_b: Team, pred: Prediction) -> str:
    pa = float(pred.team_a_probability) * 100
    pb = float(pred.team_b_probability) * 100
    fav_a = pred.team_a_probability >= pred.team_b_probability
    exp = pred.explanation or {}

    head = [
        "🎯 <b>CS2 Прогноз</b>",
        "",
        f"{_side(team_a.name, pa, fav_a)}",
        f"{_side(team_b.name, pb, not fav_a)}",
        "",
        f"🏟 <b>{esc(match.tournament_name or '—')}</b>",
        f"🎮 {esc((match.format or '—').upper())}  ·  🕒 {fmt_when(match.scheduled_at)}",
        f"{_risk_emoji(pred.risk_level)} Confidence <b>{float(pred.confidence):.2f}</b>"
        f"  ·  Risk <b>{esc(pred.risk_level)}</b>",
    ]
    if exp.get("short_summary"):
        head += ["", f"<i>{esc(exp['short_summary'])}</i>"]

    detail = ["<b>Почему:</b>", _bullets(exp.get("main_reasons")),
              "", "<b>Риски:</b>", _bullets(exp.get("risks"))]
    if exp.get("data_quality_warnings"):
        detail += ["", "<b>⚠️ Качество данных:</b>", _bullets(exp.get("data_quality_warnings"))]
    quote = "<blockquote expandable>" + "\n".join(detail) + "</blockquote>"

    return "\n".join(head) + "\n\n" + quote + "\n\n<i>Не финансовый совет.</i>"


def format_results_summary(results: list[dict]) -> str:
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = (correct / n * 100) if n else 0
    avg_brier = (sum(r["brier"] for r in results) / n) if n else 0
    lines = [
        "📊 <b>Сверка результатов</b>",
        f"✅ <b>{correct}</b> / ❌ <b>{n - correct}</b>  ·  точность <b>{acc:.0f}%</b>"
        f"  ·  ср. Brier <b>{avg_brier:.3f}</b>",
        "",
    ]
    rows = []
    for r in results:
        mark = "✅" if r["correct"] else "❌"
        rows.append(
            f"{mark} <b><u>{team_name(r['winner'])}</u></b> "
            f"<i>({team_name(r['team_a'])} vs {team_name(r['team_b'])})</i> — "
            f"ставили на {team_name(r['predicted'])} {r['prob']:.0f}%"
        )
    body = "\n".join(rows)
    if n > 6:
        body = f"<blockquote expandable>{body}</blockquote>"
    return "\n".join(lines) + body


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
            where.append("команды: " + ", ".join(team_name(t) for t in e["teams"]))
        if e.get("players"):
            where.append("игроки: " + ", ".join(team_name(p) for p in e["players"]))
        if e.get("matches"):
            where.append(f"матчей: {e['matches']}")
        where_str = (" → " + "; ".join(where)) if where else ""
        rows.append(f"{tag}<b>[{esc(e['event_type'])}]</b> {esc(e.get('summary') or '')}{where_str}")
    return head + "\n\n<blockquote expandable>" + "\n".join(rows) + "</blockquote>"


def format_postmortem(team_a, team_b, winner, pred, data: dict) -> str:
    """Per-match LLM error analysis — its reasoning, in an expandable quote."""
    on_a = pred.predicted_winner_team_id == team_a.id
    predicted = team_a.name if on_a else team_b.name
    prob = float(pred.team_a_probability if on_a else pred.team_b_probability) * 100
    mark = "✅" if pred.was_correct else "❌"
    head = [
        f"🧠 <b>Разбор: {team_name(team_a.name)} vs {team_name(team_b.name)}</b>",
        f"{mark} ставили на <b><u>{team_name(predicted)}</u></b> {prob:.0f}%  ·  "
        f"победил {team_name(winner.name)}",
    ]
    detail = []
    if data.get("suspected_failure_reasons"):
        detail += ["<b>🔍 Возможные причины:</b>", _bullets(data["suspected_failure_reasons"])]
    if data.get("data_quality_issues"):
        detail += ["", "<b>⚠️ Проблемы данных:</b>", _bullets(data["data_quality_issues"])]
    if data.get("model_improvement_hypotheses"):
        detail += ["", "<b>💡 Гипотезы улучшения:</b>", _bullets(data["model_improvement_hypotheses"])]
    conf = data.get("confidence_in_diagnosis")
    try:
        if conf is not None:
            detail += ["", f"<i>Уверенность в диагнозе: {float(conf):.2f}</i>"]
    except (TypeError, ValueError):
        pass
    body = "<blockquote expandable>" + "\n".join(detail) + "</blockquote>" if detail else ""
    return "\n".join(head) + ("\n\n" + body if body else "")


def format_daily_review(r: dict) -> str:
    c = r.get("conclusions") or {}
    acc = (r.get("accuracy") or 0) * 100
    head = [
        "🧠 <b>Дневной разбор</b>",
        f"📅 {esc(r.get('date'))}",
        f"Сверено <b>{r.get('settled', 0)}</b> · верных <b>{r.get('correct', 0)}</b>"
        f"  ·  точность <b>{acc:.0f}%</b>  ·  ср. Brier <b>{r.get('avg_brier', 0):.3f}</b>",
    ]
    detail = []
    if c.get("what_worked"):
        detail += ["<b>✅ Что зашло:</b>", _bullets(c["what_worked"])]
    if c.get("what_failed"):
        detail += ["", "<b>❌ Что не зашло:</b>", _bullets(c["what_failed"])]
    if c.get("why"):
        detail += ["", "<b>🤔 Почему:</b>", _bullets(c["why"])]
    if c.get("lessons"):
        detail += ["", "<b>📌 Уроки:</b>", _bullets(c["lessons"])]
    body = "<blockquote expandable>" + "\n".join(detail) + "</blockquote>" if detail else ""
    return "\n".join(head) + ("\n\n" + body if body else "")
