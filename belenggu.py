#!/usr/bin/env python3
"""
belenggu.py - JSON-only static dashboard builder for Synaptic Futures Journey

COMPATIBLE WITH:
    Synaptic.py + vSch.py output layout, i.e. a repo tree like:

        scan_results/
            2026-08-29_15-41-03/
                synaptic_candidates.json
                summary.txt
                charts/
                    HYPEUSDT_LONG_15m_chart.png
                    ...
            2026-08-29_18-35-45/
                synaptic_candidates.json
                ...

IMPORTANT:
- Tidak melakukan Binance API call.
- Tidak menghitung ulang setup/entry/score.
- Semua angka (entry/sl/tp) ditampilkan memakai field "decimals"
  yang sudah dikirim Synaptic.py -- konsisten dengan chart (vSch.py)
  dan summary.txt.
- Hanya membaca file JSON yang sudah ada di disk dan merender
  jadi satu halaman HTML statis, self-contained (CSS inline,
  tanpa dependency eksternal / CDN).
- Link chart di dashboard mengasumsikan belenggu.html diletakkan
  SEJAJAR dengan folder --results-dir (default: root repo),
  supaya path relatif ke charts/ tetap valid saat dibuka di
  GitHub Pages atau langsung dari file lokal.
"""

import argparse
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# STYLE
#
# Palet warna sengaja SAMA PERSIS dengan vSch.py, supaya
# dashboard dan chart terasa satu keluarga visual.
# ============================================================

UP = "#26a69a"
DOWN = "#ef5350"
EMA = "#1565c0"
ST_UP = "#2e7d32"
ST_DOWN = "#c62828"
REFERENCE = "#7b1fa2"
WAITING = "#ef6c00"
READY = "#1565c0"
MUTED = "#607d8b"
BG = "#f5f7fa"
PANEL = "#ffffff"
BORDER = "#e2e8f0"
TEXT = "#1e293b"
TEXT_SOFT = "#64748b"


SETUP_COLORS = {
    "BREAKOUT": REFERENCE,
    "BREAKDOWN": REFERENCE,
    "PULLBACK": EMA,
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
    "LONG": ST_UP,
    "SHORT": ST_DOWN,
}


# ============================================================
# NUMBER FORMATTING
#
# Sama persis dengan fmt_num() di vSynapse.yml -- kalau field
# "decimals" tersedia, pakai fixed-point. Kalau tidak (JSON
# lama), fallback ke pembersihan floating-point noise.
# ============================================================

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


# ============================================================
# LOAD RUNS
# ============================================================

def _iter_run_dirs(results_dir):

    if not results_dir.exists():
        return []

    dirs = [
        d for d in results_dir.iterdir()
        if d.is_dir()
    ]

    # Nama folder dari workflow berformat YYYY-MM-DD_HH-MM-SS,
    # jadi sort leksikografis == sort kronologis.
    dirs.sort(
        key=lambda d: d.name,
        reverse=True,
    )

    return dirs


def _load_run(run_dir):

    json_path = run_dir / "synaptic_candidates.json"

    if not json_path.exists():
        return None

    try:
        data = json.loads(
            json_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skip {run_dir.name}: {exc}")
        return None

    if not isinstance(data, dict):
        return None

    return {
        "run_id": run_dir.name,
        "dir": run_dir,
        "data": data,
    }


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


# ============================================================
# CHART LOOKUP
#
# Nama file chart mengikuti persis konvensi vSch.py:
#   f"{symbol}_{side}_{execution_tf}_chart.png"
# ============================================================

def find_chart(run, candidate):

    symbol = candidate.get("symbol")
    side = candidate.get("side")
    tf = candidate.get("execution_tf")

    if not (symbol and side and tf):
        return None

    chart_path = (
        run["dir"] / "charts" /
        f"{symbol}_{side}_{tf}_chart.png"
    )

    if chart_path.exists():
        return chart_path

    return None


# ============================================================
# BADGE HELPER
# ============================================================

def badge(text, color, extra_class=""):

    return (
        f'<span class="badge {extra_class}" '
        f'style="background:{color}">'
        f'{esc(text)}</span>'
    )


# ============================================================
# AGGREGATE STATS
# ============================================================

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
            (
                run["run_id"],
                stats.get("final_candidates", len(candidates)),
            )
        )

    # Trend chart butuh urutan kronologis (lama -> baru),
    # sedangkan `runs` sudah newest-first untuk listing.
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


# ============================================================
# SVG SPARKLINE (tanpa library eksternal)
# ============================================================

def render_trend_svg(trend_points, width=680, height=110):

    if len(trend_points) < 2:
        return (
            '<p class="muted small">'
            'Belum cukup data untuk grafik tren '
            '(minimal 2 scan run).'
            '</p>'
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
        f"{x_at(i):.1f},{y_at(v):.1f}"
        for i, v in enumerate(values)
    )

    # Area fill di bawah line
    area_points = (
        f"{x_at(0):.1f},{height - pad} " +
        points +
        f" {x_at(n-1):.1f},{height - pad}"
    )

    dots = "".join(
        f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" '
        f'r="3.5" fill="{EMA}" stroke="#fff" stroke-width="1.5">'
        f'<title>{esc(trend_points[i][0])}: {v}</title>'
        f'</circle>'
        for i, v in enumerate(values)
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}" '
        f'class="trend-svg" preserveAspectRatio="none">'
        f'<defs>'
        f'<linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0%" stop-color="{EMA}" stop-opacity="0.25"/>'
        f'<stop offset="100%" stop-color="{EMA}" stop-opacity="0.02"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<polygon points="{area_points}" fill="url(#trendGrad)" />'
        f'<polyline points="{points}" fill="none" '
        f'stroke="{EMA}" stroke-width="2.2" stroke-linecap="round" '
        f'stroke-linejoin="round" />'
        f'{dots}'
        f'</svg>'
    )


# ============================================================
# CANDIDATE ROW
# ============================================================

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
    tp_text = ", ".join(
        fmt_num(t, decimals) for t in tps
    )

    chart_path = find_chart(run, candidate)

    chart_cell = '<span class="muted">—</span>'

    if chart_path is not None:

        rel = chart_path.relative_to(run["dir"].parent.parent)

        chart_cell = (
            f'<a class="chart-link" href="{rel.as_posix()}" target="_blank" '
            f'rel="noopener">Chart ↗</a>'
        )

    funding_cell = (
        '<span class="warn-dot" title="Funding rate crowded">'
        '⚠</span>'
        if funding_alert else
        '<span class="muted">—</span>'
    )

    bias_cell = esc(htf_bias) if htf_bias else '<span class="muted">—</span>'

    # Score visual bar (0-10 scale assumed)
    try:
        score_f = float(score)
        bar_w = min(max(score_f / 10 * 100, 4), 100)
        score_bar = (
            f'<div class="score-wrap">'
            f'<span class="score-val">{esc(score)}</span>'
            f'<div class="score-bar"><div class="score-fill" style="width:{bar_w:.0f}%"></div></div>'
            f'</div>'
        )
    except (TypeError, ValueError):
        score_bar = f'<span class="num">{esc(score)}</span>'

    side_class = "side-long" if side == "LONG" else "side-short" if side == "SHORT" else ""

    return (
        f'<tr class="{side_class}">'
        f'<td class="sym">{esc(symbol)}</td>'
        f'<td>{badge(side, SIDE_COLORS.get(side, MUTED))}</td>'
        f'<td>{badge(setup_style, SETUP_COLORS.get(setup_style, MUTED))}</td>'
        f'<td>{badge(entry_state, ENTRY_STATE_COLORS.get(entry_state, MUTED))}</td>'
        f'<td>{score_bar}</td>'
        f'<td class="tf">{esc(execution_tf)} <span class="muted">({esc(tf_agreement)}/3)</span></td>'
        f'<td class="num">{entry}</td>'
        f'<td class="num">{sl}</td>'
        f'<td class="num tp">{tp_text}</td>'
        f'<td>{bias_cell}</td>'
        f'<td class="center">{funding_cell}</td>'
        f'<td class="center">{chart_cell}</td>'
        '</tr>'
    )


# ============================================================
# RUN SECTION
# ============================================================

def render_run_section(run, open_by_default=False):

    data = run["data"]
    candidates = data.get("candidates", [])
    stats = data.get("scan_stats", {})
    generated_at = data.get("generated_at", "N/A")
    selection_mode = data.get("selection_mode", "N/A")

    open_attr = " open" if open_by_default else ""

    rows_html = (
        "".join(
            render_candidate_row(run, c)
            for c in candidates
        )
        if candidates
        else (
            '<tr><td colspan="12" class="empty-row">'
            "Tidak ada kandidat pada scan ini."
            "</td></tr>"
        )
    )

    # Funnel chips
    funnel_parts = [
        ("Universe", stats.get("universe", "-")),
        ("Stage 1", stats.get("stage1_selected", "-")),
        ("MTF Valid", stats.get("mtf_valid", "-")),
        ("Final", stats.get("final_candidates", len(candidates))),
    ]

    funnel_html = " → ".join(
        f'<span class="funnel-chip"><b>{label}</b> {val}</span>'
        for label, val in funnel_parts
    )

    elapsed = stats.get("elapsed_seconds", "-")
    if isinstance(elapsed, (int, float)):
        elapsed = f"{elapsed:.1f}s"

    return f"""
    **Summary:**

        <div class="run-summary">
          <div class="run-left">
            <span class="run-time">{esc(generated_at)}</span>
            <span class="run-id muted">{esc(run['run_id'])}</span>
            {badge(selection_mode, MUTED, "mode-badge")}
          </div>
          <div class="run-right">
            <span class="run-count"><strong>{len(candidates)}</strong> kandidat</span>
            <span class="run-elapsed muted">{elapsed}</span>
          </div>
        </div>
      
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
    """


# ============================================================
# BUILD HTML
# ============================================================

def build_html(runs, aggregate):

    generated_at = datetime.now(timezone.utc).isoformat()

    top_symbols_html = "".join(
        f'<li><span class="sym-name">{esc(sym)}</span>'
        f'<span class="sym-count">{count}×</span></li>'
        for sym, count in aggregate["top_symbols"]
    ) or '<li class="muted">Belum ada data.</li>'

    trend_svg = render_trend_svg(aggregate["trend_points"])

    # Side distribution
    long_c = aggregate["side_counter"].get("LONG", 0)
    short_c = aggregate["side_counter"].get("SHORT", 0)
    total_side = long_c + short_c or 1
    long_pct = round(long_c / total_side * 100)
    short_pct = 100 - long_pct

    # Entry state
    ready_c = aggregate["state_counter"].get("ENTRY_READY", 0)
    waiting_c = (
        aggregate["state_counter"].get("WAITING_RETEST", 0) +
        aggregate["state_counter"].get("WAITING_PULLBACK", 0)
    )

    avg_score_html = (
        f"{aggregate['avg_score']}"
        if aggregate["avg_score"] is not None
        else "—"
    )

    if runs:
        run_sections = "".join(
            render_run_section(run, open_by_default=(i == 0))
            for i, run in enumerate(runs)
        )
    else:
        run_sections = (
            '<div class="empty-state">'
            '<p>Belum ada scan_results yang ditemukan.</p>'
            '</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synaptic Futures Journey</title>
<style>
  :root {{
    --up: {UP};
    --down: {DOWN};
    --ema: {EMA};
    --st-up: {ST_UP};
    --st-down: {ST_DOWN};
    --border: {BORDER};
    --bg: {BG};
    --panel: {PANEL};
    --text: {TEXT};
    --text-soft: {TEXT_SOFT};
    --muted: {MUTED};
    --ready: {READY};
    --waiting: {WAITING};
    --radius: 12px;
    --shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.5;
    padding: 28px 20px 48px;
    max-width: 1280px;
    margin: 0 auto;
  }}

  /* Header */
  .header {{
    margin-bottom: 28px;
  }}
  .header h1 {{
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
  }}
  .header .subtitle {{
    color: var(--text-soft);
    font-size: 0.875rem;
  }}
  .header .subtitle code {{
    background: #e2e8f0;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.8rem;
  }}

  /* Section titles */
  .section-title {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-soft);
    margin: 32px 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
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
    gap: 14px;
    margin-bottom: 8px;
  }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    box-shadow: var(--shadow);
  }}
  .card h3 {{
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-soft);
    margin-bottom: 8px;
  }}
  .card .value {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.2;
  }}
  .card .sub {{
    font-size: 0.8rem;
    color: var(--text-soft);
    margin-top: 4px;
  }}

  /* Side split bar */
  .side-bar {{
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 10px;
    background: #e2e8f0;
  }}
  .side-bar .long {{ background: var(--st-up); }}
  .side-bar .short {{ background: var(--st-down); }}
  .side-legend {{
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    margin-top: 6px;
    color: var(--text-soft);
  }}

  /* Top symbols list */
  .card ul {{
    list-style: none;
    font-size: 0.85rem;
  }}
  .card li {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid #f1f5f9;
  }}
  .card li:last-child {{ border-bottom: none; }}
  .sym-name {{ font-weight: 500; }}
  .sym-count {{
    color: var(--text-soft);
    font-variant-numeric: tabular-nums;
    font-size: 0.8rem;
  }}

  /* Trend SVG */
  .trend-svg {{ display: block; margin-top: 4px; }}

  /* Run cards */
  .run-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }}
  .run-card summary {{
    cursor: pointer;
    list-style: none;
    padding: 14px 18px;
    user-select: none;
  }}
  .run-card summary::-webkit-details-marker {{ display: none; }}
  .run-card summary::before {{
    content: "▸";
    color: var(--text-soft);
    margin-right: 10px;
    font-size: 0.85rem;
    transition: transform 0.15s ease;
  }}
  .run-card[open] summary::before {{
    content: "▾";
  }}
  .run-summary {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .run-left {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .run-right {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .run-time {{
    font-weight: 600;
    font-size: 0.95rem;
  }}
  .run-id {{
    font-size: 0.78rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }}
  .run-count {{
    font-size: 0.85rem;
  }}
  .run-elapsed {{
    font-size: 0.8rem;
  }}
  .mode-badge {{
    font-size: 0.65rem !important;
  }}

  .run-body {{
    padding: 0 18px 16px;
    border-top: 1px solid var(--border);
  }}

  /* Funnel */
  .funnel-line {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    font-size: 0.78rem;
    color: var(--text-soft);
    padding: 12px 0 10px;
  }}
  .funnel-chip {{
    background: #f1f5f9;
    padding: 3px 8px;
    border-radius: 6px;
    white-space: nowrap;
  }}
  .funnel-chip b {{
    color: var(--text);
    font-weight: 600;
  }}

  /* Table */
  .table-wrap {{
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid var(--border);
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.82rem;
    min-width: 900px;
  }}
  th, td {{
    padding: 9px 12px;
    text-align: left;
    white-space: nowrap;
  }}
  th {{
    background: #f8fafc;
    color: var(--text-soft);
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
  }}
  td {{
    border-bottom: 1px solid #f1f5f9;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
  tr.side-long:hover td {{ background: #f0fdf4; }}
  tr.side-short:hover td {{ background: #fef2f2; }}

  td.sym {{
    font-weight: 600;
    letter-spacing: -0.01em;
  }}
  td.num {{
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.8rem;
  }}
  td.tp {{
    max-width: 200px;
    white-space: normal;
    line-height: 1.35;
  }}
  td.center {{ text-align: center; }}
  td.tf {{ font-size: 0.8rem; }}

  .empty-row {{
    text-align: center;
    color: var(--text-soft);
    padding: 24px !important;
  }}

  /* Score bar */
  .score-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 90px;
  }}
  .score-val {{
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 0.85rem;
    min-width: 28px;
  }}
  .score-bar {{
    flex: 1;
    height: 5px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
  }}
  .score-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--ema), #42a5f5);
    border-radius: 3px;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    color: #fff;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    line-height: 1.4;
  }}

  /* Chart link */
  .chart-link {{
    color: var(--ema);
    text-decoration: none;
    font-weight: 500;
    font-size: 0.8rem;
  }}
  .chart-link:hover {{
    text-decoration: underline;
  }}

  .warn-dot {{
    color: var(--waiting);
    font-size: 1rem;
  }}

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

  footer {{
    margin-top: 36px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    color: var(--text-soft);
    font-size: 0.75rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }}

  /* Responsive */
  @media (max-width: 640px) {{
    body {{ padding: 16px 12px 32px; }}
    .header h1 {{ font-size: 1.3rem; }}
    .cards {{ grid-template-columns: 1fr 1fr; }}
    .run-summary {{ flex-direction: column; align-items: flex-start; }}
  }}
</style>
</head>
<body>

  <header class="header">
    <h1>Synaptic Futures Journey</h1>
    <div class="subtitle">
      Dibuat (UTC): {esc(generated_at)}
      · JSON-only, tidak menghitung ulang sinyal
    </div>
  </header>

  <h2 class="section-title">Ringkasan Agregat</h2>

  <div class="cards">
    <div class="card">
      <h3>Total Scan Run</h3>
      <div class="value">{aggregate['total_runs']}</div>
      <div class="sub">run terbaru diproses</div>
    </div>

    <div class="card">
      <h3>Total Kandidat</h3>
      <div class="value">{aggregate['total_candidates']}</div>
      <div class="sub">semua run digabung</div>
    </div>

    <div class="card">
      <h3>Avg Score</h3>
      <div class="value">{avg_score_html}</div>
      <div class="sub">rata-rata semua kandidat</div>
    </div>

    <div class="card">
      <h3>Entry State</h3>
      <div class="value" style="font-size:1.35rem">
        <span style="color:var(--ready)">{ready_c}</span>
        <span class="muted" style="font-weight:400;font-size:0.9rem"> ready</span>
      </div>
      <div class="sub">{waiting_c} waiting</div>
    </div>

    <div class="card">
      <h3>Side Distribution</h3>
      <div class="side-bar">
        <div class="long" style="width:{long_pct}%"></div>
        <div class="short" style="width:{short_pct}%"></div>
      </div>
      <div class="side-legend">
        <span style="color:var(--st-up)">LONG {long_c}</span>
        <span style="color:var(--st-down)">SHORT {short_c}</span>
      </div>
    </div>

    <div class="card">
      <h3>Tren Final / Run</h3>
      {trend_svg}
    </div>

    <div class="card">
      <h3>Symbol Paling Sering</h3>
      <ul>{top_symbols_html}</ul>
    </div>
  </div>

  <h2 class="section-title">Scan Runs (terbaru dulu)</h2>

  {run_sections}

  <footer>
    <span>belenggu.py · not financial advice</span>
    <span>Palet warna selaras dengan vSch.py charts</span>
  </footer>

</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "belenggu - JSON-only static dashboard "
            "for Synaptic Futures Journey"
        )
    )

    parser.add_argument(
        "--results-dir",
        default="scan_results",
        help="Folder berisi sub-folder timestamped hasil scan.",
    )

    parser.add_argument(
        "--out",
        default="belenggu.html",
        help="Path output HTML.",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=50,
        help=(
            "Batas jumlah run terbaru yang diproses "
            "(supaya file JSON lama/besar tidak memperlambat "
            "build dashboard). Default 50."
        ),
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    runs = load_runs(results_dir, max_runs=args.max_runs)

    print(f"Ditemukan {len(runs)} scan run di '{results_dir}'.")

    aggregate = build_aggregate(runs)

    output_html = build_html(runs, aggregate)

    output_path = Path(args.out)

    output_path.write_text(output_html, encoding="utf-8")

    print(f"Dashboard tersimpan: {output_path}")


if __name__ == "__main__":
    main()