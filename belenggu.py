#!/usr/bin/env python3
"""
belenggu.py - JSON-only static dashboard builder for Synaptic scan history

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
BG = "#fafafa"
PANEL = "#ffffff"
BORDER = "#e0e0e0"
TEXT = "#212121"


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

def badge(text, color):

    return (
        f'<span class="badge" '
        f'style="background:{color}">'
        f'{esc(text)}</span>'
    )


# ============================================================
# AGGREGATE STATS
# ============================================================

def build_aggregate(runs):

    total_candidates = 0
    symbol_counter = Counter()
    trend_points = []

    for run in runs:

        candidates = run["data"].get("candidates", [])

        total_candidates += len(candidates)

        for c in candidates:
            symbol_counter[c.get("symbol", "?")] += 1

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

    return {
        "total_runs": len(runs),
        "total_candidates": total_candidates,
        "top_symbols": symbol_counter.most_common(8),
        "trend_points": trend_points,
    }


# ============================================================
# SVG SPARKLINE (tanpa library eksternal)
# ============================================================

def render_trend_svg(trend_points, width=680, height=120):

    if len(trend_points) < 2:
        return (
            '<p class="muted">'
            'Belum cukup data untuk grafik tren '
            '(minimal 2 scan run).'
            '</p>'
        )

    values = [v for _, v in trend_points]

    v_min = min(values)
    v_max = max(values)

    v_range = (v_max - v_min) or 1

    pad = 12

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

    dots = "".join(
        f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" '
        f'r="3" fill="{EMA}">'
        f'<title>{esc(trend_points[i][0])}: {v}</title>'
        f'</circle>'
        for i, v in enumerate(values)
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}" '
        f'class="trend-svg">'
        f'<polyline points="{points}" fill="none" '
        f'stroke="{EMA}" stroke-width="2" />'
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

    chart_cell = "-"

    if chart_path is not None:

        rel = chart_path.relative_to(run["dir"].parent.parent)

        chart_cell = (
            f'<a href="{rel.as_posix()}" target="_blank">'
            f'chart</a>'
        )

    funding_cell = (
        '<span class="warn-dot" title="Funding rate crowded">'
        '&#9888;</span>'
        if funding_alert else ""
    )

    bias_cell = esc(htf_bias) if htf_bias else "-"

    return (
        "<tr>"
        f"<td>{esc(symbol)}</td>"
        f"<td>{badge(side, SIDE_COLORS.get(side, MUTED))}</td>"
        f"<td>{badge(setup_style, SETUP_COLORS.get(setup_style, MUTED))}</td>"
        f"<td>{badge(entry_state, ENTRY_STATE_COLORS.get(entry_state, MUTED))}</td>"
        f"<td class=\"num\">{esc(score)}</td>"
        f"<td>{esc(execution_tf)} ({esc(tf_agreement)}/3)</td>"
        f"<td class=\"num\">{entry}</td>"
        f"<td class=\"num\">{sl}</td>"
        f"<td class=\"num\">{tp_text}</td>"
        f"<td>{bias_cell}</td>"
        f"<td>{funding_cell}</td>"
        f"<td>{chart_cell}</td>"
        "</tr>"
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
            '<tr><td colspan="12" class="muted">'
            "Tidak ada kandidat pada scan ini."
            "</td></tr>"
        )
    )

    stats_line = (
        f"universe {stats.get('universe', '-')}"
        f" &rarr; stage1 {stats.get('stage1_selected', '-')}"
        f" &rarr; mtf-valid {stats.get('mtf_valid', '-')}"
        f" &rarr; final {stats.get('final_candidates', len(candidates))}"
        f" &middot; {stats.get('elapsed_seconds', '-')}s"
    )

    return f"""
    <details class="run-card"{open_attr}>
      <summary>
        <span class="run-time">{esc(generated_at)}</span>
        <span class="run-id muted">({esc(run['run_id'])})</span>
        {badge(selection_mode, MUTED)}
        <span class="run-count">{len(candidates)} kandidat</span>
      </summary>
      <p class="funnel muted">{stats_line}</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Symbol</th><th>Side</th><th>Setup</th>
              <th>Entry State</th><th>Score</th><th>Exec TF</th>
              <th>Entry</th><th>SL</th><th>TP</th>
              <th>HTF Bias</th><th></th><th>Chart</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
    </details>
    """


# ============================================================
# BUILD HTML
# ============================================================

def build_html(runs, aggregate):

    generated_at = datetime.now(timezone.utc).isoformat()

    top_symbols_html = "".join(
        f'<li><span>{esc(sym)}</span>'
        f'<span class="muted">{count}x</span></li>'
        for sym, count in aggregate["top_symbols"]
    ) or '<li class="muted">Belum ada data.</li>'

    trend_svg = render_trend_svg(aggregate["trend_points"])

    if runs:
        run_sections = "".join(
            render_run_section(run, open_by_default=(i == 0))
            for i, run in enumerate(runs)
        )
    else:
        run_sections = (
            '<p class="muted">'
            "Belum ada scan_results yang ditemukan."
            "</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>belenggu</title>
<style>
  :root {{
    --up: {UP}; --down: {DOWN}; --ema: {EMA};
    --border: {BORDER}; --bg: {BG}; --panel: {PANEL};
    --text: {TEXT}; --muted: {MUTED};
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  h1 {{ font-size: 1.4rem; margin: 0 0 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .section-title {{
    font-size: 0.85rem; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted);
    margin: 28px 0 12px; border-top: 1px solid var(--border);
    padding-top: 18px;
  }}
  .muted {{ color: var(--muted); }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px; margin-bottom: 28px;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }}
  .card h3 {{ margin: 0 0 8px; font-size: 0.8rem; color: var(--muted); font-weight: 600; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; }}
  .card ul {{ list-style: none; margin: 0; padding: 0; font-size: 0.85rem; }}
  .card li {{ display: flex; justify-content: space-between; padding: 3px 0; }}
  .trend-svg {{ display: block; }}
  .run-card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 12px; padding: 12px 16px;
  }}
  .run-card summary {{
    cursor: pointer; display: flex; align-items: center; gap: 10px;
    font-weight: 600; list-style: none;
  }}
  .run-card summary::-webkit-details-marker {{ display: none; }}
  .run-card summary::before {{ content: "\\25B8"; color: var(--muted); }}
  .run-card[open] summary::before {{ content: "\\25BE"; }}
  .run-time {{ font-size: 0.95rem; }}
  .run-count {{ margin-left: auto; font-size: 0.8rem; color: var(--muted); }}
  .funnel {{ font-size: 0.78rem; margin: 8px 0 12px; }}
  .table-wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82rem; }}
  th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }}
  th {{ color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  tr:hover td {{ background: #f5f5f5; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    color: #fff; font-size: 0.7rem; font-weight: 600;
  }}
  .warn-dot {{ color: {WAITING}; font-weight: 700; }}
  a {{ color: {EMA}; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  footer {{ margin-top: 24px; color: var(--muted); font-size: 0.75rem; }}
</style>
</head>
<body>

  <h1>Synaptic Crypto Journey</h1>
  <div class="subtitle">
    Dibuat (UTC): {esc(generated_at)}
    &middot; JSON-only, tidak menghitung ulang sinyal.
  </div>

  {run_sections}

  <h2 class="section-title">Ringkasan Agregat</h2>

  <div class="cards">
    <div class="card">
      <h3>Total scan run</h3>
      <div class="value">{aggregate['total_runs']}</div>
    </div>
    <div class="card">
      <h3>Total kandidat (semua run)</h3>
      <div class="value">{aggregate['total_candidates']}</div>
    </div>
    <div class="card">
      <h3>Tren kandidat final / run</h3>
      {trend_svg}
    </div>
    <div class="card">
      <h3>Symbol paling sering muncul</h3>
      <ul>{top_symbols_html}</ul>
    </div>
  </div>

  <footer>
    belenggu.py &middot; not financial advice
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
            "for Synaptic scan history"
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
