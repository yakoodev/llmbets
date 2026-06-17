"""Render predictions into Telegram messages (v1: no odds / no paper pick)."""
from __future__ import annotations

from app.db.models import Match, Prediction, Team


def _bullets(items, limit: int = 4) -> str:
    out = [f"• {x}" for x in (items or [])[:limit] if x]
    return "\n".join(out) if out else "—"


def format_forecast(match: Match, team_a: Team, team_b: Team, pred: Prediction) -> str:
    pa = float(pred.team_a_probability) * 100
    pb = float(pred.team_b_probability) * 100
    winner = team_a.name if pred.team_a_probability >= pred.team_b_probability else team_b.name
    exp = pred.explanation or {}
    when = (
        match.scheduled_at.strftime("%Y-%m-%d %H:%M UTC")
        if match.scheduled_at
        else "—"
    )

    lines = [
        "🎯 CS2 Прогноз",
        "",
        f"Матч: {team_a.name} vs {team_b.name}",
        f"Турнир: {match.tournament_name or '—'}",
        f"Формат: {(match.format or '—').upper()}",
        f"Старт: {when}",
        "",
        "Вероятности модели:",
        f"{team_a.name} — {pa:.0f}%",
        f"{team_b.name} — {pb:.0f}%",
        "",
        f"Фаворит: {winner}",
        f"Confidence: {float(pred.confidence):.2f} · Risk: {pred.risk_level}",
    ]
    if exp.get("short_summary"):
        lines += ["", exp["short_summary"]]
    lines += ["", "Почему:", _bullets(exp.get("main_reasons"))]
    lines += ["", "Риски:", _bullets(exp.get("risks"))]
    if exp.get("data_quality_warnings"):
        lines += ["", "⚠️ Качество данных:", _bullets(exp.get("data_quality_warnings"))]
    lines += ["", "Это аналитический прогноз, не финансовый совет."]
    return "\n".join(lines)


def format_news_digest(entries: list[dict], collected: int, processed: int) -> str:
    """What was filed where (per TZ §11 routing), for transparency."""
    rel = len(entries)
    lines = [
        f"🗂 Новости обработаны: собрано {collected}, разобрано {processed}, релевантных {rel}",
    ]
    if not entries:
        lines.append("Ничего значимого по CS2 не нашёл.")
        return "\n".join(lines)
    lines.append("")
    for e in entries[:10]:
        tag = "⚠️ " if e.get("critical") else ""
        where = []
        if e.get("teams"):
            where.append("команды: " + ", ".join(e["teams"]))
        if e.get("players"):
            where.append("игроки: " + ", ".join(e["players"]))
        if e.get("matches"):
            where.append(f"матчей затронуто: {e['matches']}")
        where_str = (" → " + "; ".join(where)) if where else " → без привязки"
        lines.append(f"{tag}[{e['event_type']}] {e.get('summary') or ''}{where_str}")
    if rel > 10:
        lines.append(f"…и ещё {rel - 10}")
    return "\n".join(lines)


def format_results_summary(results: list[dict]) -> str:
    """Scorecard of settled predictions (✅/❌) — the 'сверка' table."""
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = (correct / n * 100) if n else 0
    avg_brier = (sum(r["brier"] for r in results) / n) if n else 0
    lines = [
        f"📊 Сверка результатов: {n} матч(ей)",
        f"✅ {correct} / ❌ {n - correct} · точность {acc:.0f}% · ср. Brier {avg_brier:.3f}",
        "",
    ]
    for r in results:
        mark = "✅" if r["correct"] else "❌"
        lines.append(
            f"{mark} {r['winner']} победил ({r['team_a']} vs {r['team_b']}) — "
            f"ставили на {r['predicted']} {r['prob']:.0f}%"
        )
    return "\n".join(lines)
