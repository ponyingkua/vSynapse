#!/usr/bin/env python3
"""belenggu.py - JSON-only dark dashboard builder for Synaptic Futures Journey.
Baca scan_results/<timestamp>/synaptic_candidates.json, render satu file HTML
statis, self-contained. Link chart mengasumsikan belenggu.html sejajar dengan
folder --results-dir."""

import argparse
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# DESIGN TOKENS

BG = "#000000"          # hitam pekat, bukan kecoklatan
PANEL = "#101112"
PANEL_SOFT = "#1a1b1d"
BORDER = "#2b2e31"
TEXT = "#eaecef"        # putih Binance
TEXT_SOFT = "#848e9c"   # abu-abu Binance
MUTED = "#4a4e54"

ACCENT = "#F0B90B"      # kuning Binance asli - aksen utama
ACCENT_DEEP = "#FCD535" # kuning terang Binance - aksen sekunder
UP = "#0ECB81"          # long (hijau Binance)
DOWN = "#F6465D"        # short (merah Binance)
LINK = ACCENT           # chart link & tren ikut kuning
REFERENCE = ACCENT_DEEP # breakout/breakdown
READY = UP              # entry ready = sinyal "go", pakai warna sama dgn long
WAITING = ACCENT
GOLD = ACCENT           # top symbols
CYAN = TEXT_SOFT        # total run - dinetralkan, bukan metrik "sinyal"


SETUP_COLORS = {
    "BREAKOUT": REFERENCE,
    "BREAKDOWN": REFERENCE,
    "PULLBACK": LINK,
    "CONTINUATION": UP,
    "EXTENDED": WAITING,
    "NO_SETUP": MUTED,
}

ENTRY_STATE_COLORS = {
    "ENTRY_READY": READY,
    "WAITING_RETEST": WAITING,
    "WAITING_PULLBACK": WAITING,
    "NO_SETUP": MUTED,
}

SIDE_COLORS = {
    "LONG": UP,
    "SHORT": DOWN,
}


# NUMBER FORMATTING

def fmt_num(value, decimals=None):

    if value is None:
        return "N/A"

    try:
        value = float(value)

        if decimals is not None:
            return f"{value:.{int(decimals)}f}"

        text = f"{value:.12f}".rstrip("0").rstrip(".")

        return text if text else "0"

    except (TypeError, ValueError):
        return html.escape(str(value))


def esc(value):
    return html.escape(str(value)) if value is not None else "N/A"


# WIN-RATE STATS (optional - produced by tracker.py)

def load_winrate_stats(path):
    """Reads the stats file tracker.py writes. Returns None if it doesn't
    exist yet or is unreadable, so the dashboard degrades gracefully for
    anyone not running the tracker."""

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skip winrate stats ({path}): {exc}")
        return None


def _render_breakdown_rows(groups, label_map=None):
    """Shared renderer for any {key: {wins, losses, win_rate_pct}} breakdown
    - used for by_setup_style and by_htf_alignment so both look consistent."""

    label_map = label_map or {}

    return "".join(
        f'<div class="setup-row">'
        f'<span class="setup-name">{esc(label_map.get(key, key))}</span>'
        f'<span class="setup-record">{v["wins"]}W / {v["losses"]}L</span>'
        f'<span class="setup-rate" style="color:{UP if (v["win_rate_pct"] or 0) >= 50 else DOWN}">'
        f'{v["win_rate_pct"]}%</span>'
        f'</div>'
        for key, v in sorted(
            groups.items(),
            key=lambda kv: kv[1]["wins"] + kv[1]["losses"],
            reverse=True,
        )
    )


HTF_ALIGNMENT_LABELS = {
    "ALIGNED": "Aligned with daily trend",
    "COUNTER": "Against daily trend",
    "UNKNOWN_OR_NEUTRAL": "Unknown / neutral bias",
}


def render_winrate_card(stats):

    if stats is None or not stats.get("resolved_trades"):
        return ""

    win_rate = stats.get("win_rate_pct")
    win_rate_html = f"{win_rate}%" if win_rate is not None else "&mdash;"
    win_color = UP if (win_rate or 0) >= 50 else DOWN

    avg_r = stats.get("avg_r_multiple")
    avg_r_html = f"{avg_r:+.2f}R" if avg_r is not None else "&mdash;"

    setup_rows = _render_breakdown_rows(stats.get("by_setup_style", {}))
    htf_rows = _render_breakdown_rows(stats.get("by_htf_alignment", {}), HTF_ALIGNMENT_LABELS)

    return f"""
    <div class="card wide" style="--card-accent:{win_color}">
      <h3>Win Rate (Resolved Trades)</h3>
      <div class="winrate-summary">
        <div class="winrate-main">
          <span class="value" style="color:{win_color}">{win_rate_html}</span>
          <span class="sub">{stats['wins']}W / {stats['losses']}L &middot; {stats['resolved_trades']} resolved</span>
        </div>
        <div class="winrate-secondary">
          <div><span class="muted">Avg R multiple</span><br><strong>{avg_r_html}</strong></div>
          <div><span class="muted">Open</span><br><strong>{stats.get('currently_open', 0)}</strong></div>
          <div><span class="muted">Pending entry</span><br><strong>{stats.get('currently_pending', 0)}</strong></div>
          <div><span class="muted">Expired / never triggered</span><br>
            <strong>{stats.get('expired_unresolved', 0)} / {stats.get('never_triggered', 0)}</strong></div>
        </div>
      </div>
      {f'<div class="setup-breakdown"><div class="breakdown-label">By setup style</div>{setup_rows}</div>' if setup_rows else ""}
      {f'<div class="setup-breakdown"><div class="breakdown-label">By daily (HTF) trend alignment</div>{htf_rows}</div>' if htf_rows else ""}
    </div>
    """


# LOAD RUNS

def _iter_run_dirs(results_dir):

    if not results_dir.exists():
        return []

    dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    dirs.sort(key=lambda d: d.name, reverse=True)

    return dirs


def _load_run(run_dir):

    json_path = run_dir / "synaptic_candidates.json"

    if not json_path.exists():
        return None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skip {run_dir.name}: {exc}")
        return None

    if not isinstance(data, dict):
        return None

    return {"run_id": run_dir.name, "dir": run_dir, "data": data}


def load_runs(results_dir, max_runs=None):

    run_dirs = _iter_run_dirs(results_dir)

    if max_runs is not None:
        run_dirs = run_dirs[:max_runs]

    runs = []

    for run_dir in run_dirs:
        run = _load_run(run_dir)
        if run is not None:
            runs.append(run)

    return runs


# CHART LOOKUP: f"{symbol}_{side}_{execution_tf}_chart.png"

def find_chart(run, candidate):

    symbol = candidate.get("symbol")
    side = candidate.get("side")
    tf = candidate.get("execution_tf")

    if not (symbol and side and tf):
        return None

    chart_path = run["dir"] / "charts" / f"{symbol}_{side}_{tf}_chart.png"

    if chart_path.exists():
        return chart_path

    return None


# BADGE HELPER

def badge(text, color, extra_class=""):
    return (
        f'<span class="badge {extra_class}" '
        f'style="color:{color};border-color:{color}55;background:{color}1f">'
        f'{esc(text)}</span>'
    )


# SCORE METER

def render_score_meter(score, segments=10):

    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return f'<span class="num muted">{esc(score)}</span>'

    filled = round(max(min(score_f, segments), 0))

    bars = "".join(
        f'<span class="meter-seg{" on" if i < filled else ""}"></span>'
        for i in range(segments)
    )

    return (
        f'<div class="score-wrap">'
        f'<span class="score-val">{score_f:.1f}</span>'
        f'<div class="meter">{bars}</div>'
        f'</div>'
    )


# AGGREGATE STATS

def build_aggregate(runs):

    total_candidates = 0
    symbol_counter = Counter()
    side_counter = Counter()
    state_counter = Counter()
    setup_counter = Counter()
    trend_points = []
    scores = []

    for run in runs:

        candidates = run["data"].get("candidates", [])
        total_candidates += len(candidates)

        for c in candidates:
            symbol_counter[c.get("symbol", "?")] += 1
            side_counter[c.get("side", "?")] += 1
            state_counter[c.get("entry_state", "?")] += 1
            setup_counter[c.get("setup_style", "?")] += 1
            sc = c.get("score")
            if sc is not None:
                try:
                    scores.append(float(sc))
                except (TypeError, ValueError):
                    pass

        stats = run["data"].get("scan_stats", {})
        trend_points.append(
            (run["run_id"], stats.get("final_candidates", len(candidates)))
        )

    trend_points.reverse()

    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    return {
        "total_runs": len(runs),
        "total_candidates": total_candidates,
        "top_symbols": symbol_counter.most_common(8),
        "side_counter": side_counter,
        "state_counter": state_counter,
        "setup_counter": setup_counter,
        "avg_score": avg_score,
        "trend_points": trend_points,
    }


# SVG SPARKLINE

def render_trend_svg(trend_points, width=680, height=110):

    if len(trend_points) < 2:
        return (
            '<p class="muted small">'
            "Not enough data for a trend chart "
            "(minimum 2 scan runs)."
            "</p>"
        )

    values = [v for _, v in trend_points]

    v_min = min(values)
    v_max = max(values)
    v_range = (v_max - v_min) or 1

    pad = 14
    usable_w = width - pad * 2
    usable_h = height - pad * 2
    n = len(values)

    def x_at(i):
        return pad + (usable_w * i / (n - 1))

    def y_at(v):
        return pad + usable_h * (1 - (v - v_min) / v_range)

    points = " ".join(
        f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values)
    )

    area_points = (
        f"{x_at(0):.1f},{height - pad} "
        + points
        + f" {x_at(n - 1):.1f},{height - pad}"
    )

    dots = "".join(
        f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" '
        f'r="3.5" fill="{LINK}" stroke="{BG}" stroke-width="1.5">'
        f"<title>{esc(trend_points[i][0])}: {v}</title>"
        f"</circle>"
        for i, v in enumerate(values)
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'class="trend-svg" preserveAspectRatio="none">'
        f"<defs>"
        f'<linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{LINK}" stop-opacity="0.3"/>'
        f'<stop offset="100%" stop-color="{LINK}" stop-opacity="0"/>'
        f"</linearGradient>"
        f'<filter id="glow" x="-20%" y="-50%" width="140%" height="200%">'
        f'<feGaussianBlur stdDeviation="2.4" result="blur"/>'
        f'<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f"</filter>"
        f"</defs>"
        f'<polygon points="{area_points}" fill="url(#trendGrad)" />'
        f'<polyline points="{points}" fill="none" stroke="{LINK}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'filter="url(#glow)" />'
        f"{dots}"
        f"</svg>"
    )


# CANDIDATE ROW

def render_candidate_row(run, candidate):

    symbol = candidate.get("symbol", "N/A")
    side = candidate.get("side", "N/A")
    setup_style = candidate.get("setup_style", "N/A")
    entry_state = candidate.get("entry_state", "N/A")
    score = candidate.get("score", "N/A")
    tf_agreement = candidate.get("tf_agreement", "-")
    execution_tf = candidate.get("execution_tf", "-")
    decimals = candidate.get("decimals")
    funding_alert = candidate.get("funding_alert", False)
    htf_bias = candidate.get("htf_bias")

    entry = fmt_num(candidate.get("entry"), decimals)
    sl = fmt_num(candidate.get("sl"), decimals)

    tps = candidate.get("tp", []) or []
    tp_text = ", ".join(fmt_num(t, decimals) for t in tps)

    chart_path = find_chart(run, candidate)
    chart_cell = '<span class="muted">&mdash;</span>'

    if chart_path is not None:
        rel = chart_path.relative_to(run["dir"].parent.parent).as_posix()
        chart_cell = (
            f'<img class="chart-thumb" src="{rel}" alt="{esc(symbol)} chart" '
            f'loading="lazy" onclick="openLightbox(\'{rel}\')">'
        )

    funding_cell = (
        '<span class="warn-dot" title="Funding rate crowded">&#9888;</span>'
        if funding_alert
        else '<span class="muted">&mdash;</span>'
    )

    bias_cell = esc(htf_bias) if htf_bias else '<span class="muted">&mdash;</span>'

    score_cell = render_score_meter(score)

    side_class = (
        "side-long" if side == "LONG" else "side-short" if side == "SHORT" else ""
    )

    return (
        f'<tr class="{side_class}">'
        f'<td class="sym">{esc(symbol)}</td>'
        f"<td>{badge(side, SIDE_COLORS.get(side, MUTED))}</td>"
        f"<td>{badge(setup_style, SETUP_COLORS.get(setup_style, MUTED))}</td>"
        f"<td>{badge(entry_state, ENTRY_STATE_COLORS.get(entry_state, MUTED))}</td>"
        f"<td>{score_cell}</td>"
        f'<td class="tf">{esc(execution_tf)} '
        f'<span class="muted">({esc(tf_agreement)}/3)</span></td>'
        f'<td class="num">{entry}</td>'
        f'<td class="num">{sl}</td>'
        f'<td class="num tp">{tp_text}</td>'
        f"<td>{bias_cell}</td>"
        f'<td class="center">{funding_cell}</td>'
        f'<td class="center">{chart_cell}</td>'
        "</tr>"
    )


# RUN SECTION

def render_run_section(run, open_by_default=False):

    data = run["data"]
    candidates = data.get("candidates", [])
    stats = data.get("scan_stats", {})
    generated_at = data.get("generated_at", "N/A")
    selection_mode = data.get("selection_mode", "N/A")

    open_attr = " open" if open_by_default else ""

    rows_html = (
        "".join(render_candidate_row(run, c) for c in candidates)
        if candidates
        else (
            '<tr><td colspan="12" class="empty-row">'
            "No candidates found in this scan."
            "</td></tr>"
        )
    )

    funnel_parts = [
        ("UNIVERSE", stats.get("universe", "-")),
        ("STAGE 1", stats.get("stage1_selected", "-")),
        ("MTF VALID", stats.get("mtf_valid", "-")),
        ("FINAL", stats.get("final_candidates", len(candidates))),
    ]

    funnel_html = '<span class="funnel-arrow">&rarr;</span>'.join(
        f'<span class="funnel-chip"><b>{label}</b> {val}</span>'
        for label, val in funnel_parts
    )

    elapsed = stats.get("elapsed_seconds", "-")
    if isinstance(elapsed, (int, float)):
        elapsed = f"{elapsed:.1f}s"

    return f"""
    <details class="run-card"{open_attr}>
      <summary>
        <div class="run-summary">
          <div class="run-left">
            <span class="run-time">{esc(generated_at)}</span>
            <span class="run-id muted">{esc(run['run_id'])}</span>
            {badge(selection_mode, MUTED, "mode-badge")}
          </div>
          <div class="run-right">
            <span class="run-count"><strong>{len(candidates)}</strong> candidates</span>
            <span class="run-elapsed muted">{elapsed}</span>
          </div>
        </div>
      </summary>
      <div class="run-body">
        <div class="funnel-line">{funnel_html}</div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Setup</th>
                <th>Entry State</th>
                <th>Score</th>
                <th>Exec TF</th>
                <th>Entry</th>
                <th>SL</th>
                <th>TP</th>
                <th>HTF Bias</th>
                <th></th>
                <th>Chart</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </details>
    """


# BUILD HTML

def build_html(runs, aggregate, winrate_stats=None):

    generated_at = datetime.now(timezone.utc).isoformat()

    def _short_symbol(sym):
        sym = str(sym)
        for suffix in ("USDT", "USDC", "BUSD"):
            if sym.endswith(suffix):
                sym = sym[: -len(suffix)]
                break
        return f"${sym}"

    top_symbols_html = "".join(
        f'<div class="symbol-chip">'
        f'<span class="sym-name">{esc(_short_symbol(sym))}</span>'
        f'<span class="sym-count">{count}&times;</span></div>'
        for sym, count in aggregate["top_symbols"]
    ) or '<span class="muted">No data yet.</span>'

    trend_svg = render_trend_svg(aggregate["trend_points"])

    long_c = aggregate["side_counter"].get("LONG", 0)
    short_c = aggregate["side_counter"].get("SHORT", 0)
    total_side = long_c + short_c or 1
    long_pct = round(long_c / total_side * 100)
    short_pct = 100 - long_pct

    ready_c = aggregate["state_counter"].get("ENTRY_READY", 0)
    waiting_c = aggregate["state_counter"].get(
        "WAITING_RETEST", 0
    ) + aggregate["state_counter"].get("WAITING_PULLBACK", 0)

    avg_score_html = (
        f"{aggregate['avg_score']}" if aggregate["avg_score"] is not None else "&mdash;"
    )

    if runs:
        run_sections = "".join(
            render_run_section(run, open_by_default=(i == 0))
            for i, run in enumerate(runs)
        )
    else:
        run_sections = (
            '<div class="empty-state">'
            "<p>No scan_results found yet.</p>"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synaptic // Futures Journey</title>
<style>
  :root {{
    --bg: {BG};
    --panel: {PANEL};
    --panel-soft: {PANEL_SOFT};
    --border: {BORDER};
    --text: {TEXT};
    --text-soft: {TEXT_SOFT};
    --muted: {MUTED};
    --accent: {ACCENT};
    --up: {UP};
    --down: {DOWN};
    --link: {LINK};
    --ready: {READY};
    --waiting: {WAITING};
    --radius: 8px;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
  }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    line-height: 1.5;
    padding: 32px 20px 56px;
    max-width: 1280px;
    margin: 0 auto;
  }}

  /* Header */
  .header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 30px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-family: var(--sans);
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    text-transform: uppercase;
    color: var(--accent);
  }}
  .header h1 .slash {{ color: var(--text); }}
  .header .subtitle {{
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-soft);
    font-family: var(--mono);
    font-size: 0.78rem;
  }}
  .live-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--ready);
    box-shadow: 0 0 0 0 rgba(14,203,129,0.6);
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(14,203,129,0.5); }}
    70%  {{ box-shadow: 0 0 0 6px rgba(14,203,129,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(14,203,129,0); }}
  }}

  /* Section titles */
  .section-title {{
    font-family: var(--mono);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-soft);
    margin: 34px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-title::before {{ content: "//"; color: var(--accent); }}
  .section-title::after {{
    content: "";
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  /* Cards grid */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--card-accent, var(--border));
    border-radius: var(--radius);
    padding: 16px 18px;
  }}
  .card.wide {{
    grid-column: 1 / -1;
  }}
  .card h3 {{
    font-family: var(--mono);
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-soft);
    margin-bottom: 10px;
  }}
  .card .value {{
    font-family: var(--mono);
    font-size: 1.7rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.2;
    color: var(--text);
  }}
  .card .sub {{
    font-size: 0.78rem;
    color: var(--text-soft);
    margin-top: 4px;
  }}

  /* Side split bar */
  .side-bar {{
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 12px;
    background: var(--border);
  }}
  .side-bar .long {{ background: var(--up); }}
  .side-bar .short {{ background: var(--down); }}
  .side-legend {{
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.72rem;
    margin-top: 8px;
    color: var(--text-soft);
  }}

  /* Top symbols grid (wide card) */
  .symbol-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }}
  .symbol-chip {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: var(--panel-soft);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 9px 12px;
    font-size: 0.85rem;
  }}
  .sym-name {{ font-weight: 600; }}
  .sym-count {{
    font-family: var(--mono);
    color: var(--text-soft);
    font-size: 0.76rem;
    flex-shrink: 0;
  }}

  @media (max-width: 640px) {{
    .symbol-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  .trend-svg {{ display: block; margin-top: 2px; }}

  /* Run cards */
  .run-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 10px;
    overflow: hidden;
  }}
  .run-card summary {{
    cursor: pointer;
    list-style: none;
    padding: 13px 16px;
    user-select: none;
  }}
  .run-card summary::-webkit-details-marker {{ display: none; }}
  .run-card summary::before {{
    content: "\\25B8";
    color: var(--accent);
    margin-right: 10px;
    font-size: 0.78rem;
    display: inline-block;
  }}
  .run-card[open] summary::before {{ content: "\\25BE"; }}
  .run-summary {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .run-left {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .run-right {{ display: flex; align-items: center; gap: 12px; }}
  .run-time {{ font-family: var(--mono); font-weight: 600; font-size: 0.88rem; }}
  .run-id {{ font-family: var(--mono); font-size: 0.72rem; }}
  .run-count {{ font-size: 0.84rem; }}
  .run-elapsed {{ font-family: var(--mono); font-size: 0.76rem; }}
  .mode-badge {{ font-size: 0.62rem !important; }}

  .run-body {{ padding: 0 16px 16px; border-top: 1px solid var(--border); }}

  /* Funnel */
  .funnel-line {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    font-family: var(--mono);
    font-size: 0.74rem;
    color: var(--text-soft);
    padding: 12px 0 10px;
  }}
  .funnel-chip {{
    background: var(--panel-soft);
    padding: 3px 9px;
    border-radius: 4px;
    white-space: nowrap;
  }}
  .funnel-chip b {{ color: var(--text); font-weight: 600; }}
  .funnel-arrow {{ color: var(--accent); margin: 0 6px; }}

  /* Table */
  .table-wrap {{
    overflow-x: auto;
    border-radius: 6px;
    border: 1px solid var(--border);
  }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; min-width: 900px; }}
  th, td {{ padding: 9px 12px; text-align: left; white-space: nowrap; }}
  th {{
    background: var(--panel-soft);
    color: var(--text-soft);
    font-family: var(--mono);
    font-weight: 600;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 1px solid var(--border);
  }}
  td {{ border-bottom: 1px solid var(--panel-soft); }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--panel-soft); }}
  tr.side-long:hover td {{ background: #0c211b; }}
  tr.side-short:hover td {{ background: #2b1418; }}

  td.sym {{ font-family: var(--mono); font-weight: 600; }}
  td.num {{ font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: 0.78rem; }}
  td.tp {{ max-width: 200px; white-space: normal; line-height: 1.35; }}
  td.center {{ text-align: center; }}
  td.tf {{ font-size: 0.78rem; }}

  .empty-row {{ text-align: center; color: var(--text-soft); padding: 24px !important; }}

  /* Score meter (signature element) */
  .score-wrap {{ display: flex; align-items: center; gap: 8px; min-width: 110px; }}
  .score-val {{
    font-family: var(--mono);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 0.8rem;
    min-width: 26px;
    color: var(--accent);
  }}
  .meter {{ display: flex; gap: 2px; align-items: flex-end; }}
  .meter-seg {{
    width: 5px;
    height: 12px;
    background: var(--border);
    border-radius: 1px;
  }}
  .meter-seg.on {{
    background: var(--accent);
    box-shadow: 0 0 4px {ACCENT}99;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid;
    font-family: var(--mono);
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    line-height: 1.5;
  }}

  .chart-link {{ color: var(--link); text-decoration: none; font-weight: 500; font-size: 0.78rem; }}
  .chart-link:hover {{ text-decoration: underline; }}

  .chart-thumb {{
    width: 84px;
    height: 48px;
    object-fit: cover;
    object-position: top;
    border-radius: 4px;
    border: 1px solid var(--border);
    cursor: zoom-in;
    display: block;
    background: var(--panel-soft);
    transition: border-color 0.15s ease;
  }}
  .chart-thumb:hover {{ border-color: var(--accent); }}

  .lightbox-overlay {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.9);
    z-index: 999;
    align-items: center;
    justify-content: center;
    padding: 30px;
    cursor: zoom-out;
  }}
  .lightbox-overlay.open {{ display: flex; }}
  .lightbox-overlay img {{
    max-width: 95vw;
    max-height: 95vh;
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
  }}

  .warn-dot {{ color: var(--waiting); font-size: 0.95rem; }}

  .muted {{ color: var(--text-soft); }}
  .small {{ font-size: 0.8rem; }}

  .empty-state {{
    background: var(--panel);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 40px;
    text-align: center;
    color: var(--text-soft);
  }}

  /* Win-rate card */
  .winrate-summary {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
  }}
  .winrate-main {{ display: flex; flex-direction: column; gap: 4px; }}
  .winrate-main .value {{ font-family: var(--mono); font-size: 2rem; font-weight: 700; }}
  .winrate-secondary {{
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    font-size: 0.82rem;
  }}
  .winrate-secondary strong {{ font-family: var(--mono); }}
  .setup-breakdown {{
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .breakdown-label {{
    font-family: var(--mono);
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-soft);
    margin-bottom: 4px;
  }}
  .setup-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.8rem;
  }}
  .setup-name {{
    font-family: var(--mono);
    font-weight: 600;
    flex: 1;
    min-width: 130px;
  }}
  .setup-record {{ color: var(--text-soft); width: 90px; }}
  .setup-rate {{ font-family: var(--mono); font-weight: 600; width: 50px; text-align: right; }}

  footer {{
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--text-soft);
    font-family: var(--mono);
    font-size: 0.7rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }}

  @media (max-width: 640px) {{
    body {{ padding: 20px 12px 40px; }}
    .header h1 {{ font-size: 1.05rem; }}
    .cards {{ grid-template-columns: 1fr 1fr; }}
    .run-summary {{ flex-direction: column; align-items: flex-start; }}
  }}
</style>
</head>
<body>

  <header class="header">
    <h1>SYNAPTIC<span class="slash">//</span>FUTURES JOURNEY</h1>
    <div class="subtitle">
      <span class="live-dot"></span>
      generated {esc(generated_at)} UTC
    </div>
  </header>

  <h2 class="section-title">Aggregate Summary</h2>

  <div class="cards">
    {render_winrate_card(winrate_stats)}

    <div class="card wide" style="--card-accent:{GOLD}">
      <h3>Most Frequent Symbols</h3>
      <div class="symbol-grid">{top_symbols_html}</div>
    </div>

    <div class="card" style="--card-accent:{CYAN}">
      <h3>Total Scan Runs</h3>
      <div class="value" style="color:{CYAN}">{aggregate['total_runs']}</div>
      <div class="sub">recent runs processed</div>
    </div>

    <div class="card" style="--card-accent:{ACCENT}">
      <h3>Total Candidates</h3>
      <div class="value" style="color:{ACCENT}">{aggregate['total_candidates']}</div>
      <div class="sub">across all runs combined</div>
    </div>

    <div class="card" style="--card-accent:{REFERENCE}">
      <h3>Avg Score</h3>
      <div class="value" style="color:{REFERENCE}">{avg_score_html}</div>
      <div class="sub">average across all candidates</div>
    </div>

    <div class="card" style="--card-accent:{READY}">
      <h3>Entry State</h3>
      <div class="value" style="font-size:1.3rem">
        <span style="color:var(--ready)">{ready_c}</span>
        <span class="muted" style="font-weight:400;font-size:0.85rem"> ready</span>
      </div>
      <div class="sub">{waiting_c} waiting</div>
    </div>

    <div class="card" style="--card-accent:{UP}">
      <h3>Side Distribution</h3>
      <div class="side-bar">
        <div class="long" style="width:{long_pct}%"></div>
        <div class="short" style="width:{short_pct}%"></div>
      </div>
      <div class="side-legend">
        <span style="color:var(--up)">LONG {long_c}</span>
        <span style="color:var(--down)">SHORT {short_c}</span>
      </div>
    </div>

    <div class="card" style="--card-accent:{LINK}">
      <h3>Final Candidates / Run Trend</h3>
      {trend_svg}
    </div>
  </div>

  <h2 class="section-title">Scan Runs (most recent first)</h2>

  {run_sections}

  <footer>
    <span>Synaptic Futures Journey &middot; not financial advice</span>
    <span>vSynapse</span>
  </footer>

  <div class="lightbox-overlay" id="lightboxOverlay" onclick="closeLightbox()">
    <img id="lightboxImg" src="" alt="Chart preview">
  </div>

  <script>
    function openLightbox(src) {{
      var overlay = document.getElementById('lightboxOverlay');
      var img = document.getElementById('lightboxImg');
      img.src = src;
      overlay.classList.add('open');
    }}
    function closeLightbox() {{
      document.getElementById('lightboxOverlay').classList.remove('open');
      document.getElementById('lightboxImg').src = '';
    }}
    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') closeLightbox();
    }});
  </script>

</body>
</html>
"""


# MAIN

def main():

    parser = argparse.ArgumentParser(
        description="belenggu - JSON-only static dashboard for Synaptic Futures Journey"
    )

    parser.add_argument(
        "--results-dir",
        default="scan_results",
        help="Folder containing timestamped scan-result subfolders.",
    )

    parser.add_argument(
        "--out",
        default="belenggu.html",
        help="Output HTML path.",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help=(
            "Limit on how many recent runs are processed "
            "(so old/large JSON files don't slow down the "
            "dashboard build). Default 50."
        ),
    )

    parser.add_argument(
        "--stats-file",
        default="winrate_stats.json",
        help="Optional win-rate stats produced by tracker.py. Skipped if not found.",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = load_runs(results_dir, max_runs=args.max_runs)

    print(f"Found {len(runs)} scan run(s) in '{results_dir}'.")

    aggregate = build_aggregate(runs)
    winrate_stats = load_winrate_stats(Path(args.stats_file))
    output_html = build_html(runs, aggregate, winrate_stats)

    output_path = Path(args.out)
    output_path.write_text(output_html, encoding="utf-8")

    print(f"Dashboard saved: {output_path}")


if __name__ == "__main__":
    main()
