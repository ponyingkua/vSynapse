"""
dashboard.py - JSON-only static HTML dashboard for Synaptic.py.

- Never fetch Binance data, never touch Synaptic.py or vSch.py.
- Reads the same candidates JSON that vSch.py reads.
- Reads chart PNGs that vSch.py already rendered (does not render
  its own charts, does not recompute any indicator/setup logic).
- Output is a single self-contained dashboard.html file that can
  be opened directly in a browser (no server required).
- If a chart PNG is missing for a candidate, the dashboard shows
  a placeholder instead of failing.
"""

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# STYLE - matches vSch.py's palette for visual consistency
# ============================================================
BG = "#ffffff"
PANEL = "#f7f7f8"
BORDER = "#e0e0e0"
TEXT = "#212121"
MUTED = "#607080"

UP = "#26a69a"      # LONG
DOWN = "#ef5350"    # SHORT
ENTRY = "#1565c0"
TP_COLOR = "#00897b"
SL_COLOR = "#c62828"

SETUP_COLORS = {
    "BREAKOUT": "#6a1b9a",
    "PULLBACK": "#1565c0",
    "CONTINUATION": "#2e7d32",
    "EXTENDED": "#ef6c00",
}
SETUP_DEFAULT_COLOR = "#607080"


# ============================================================
# LOAD
# ============================================================
def load_payload(input_path):
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))

    if "candidates" not in raw:
        raise ValueError(f"'candidates' key not found in {input_path}")

    return raw


def resolve_chart_path(charts_dir, output_dir, candidate):
    """
    Mereplikasi konvensi nama file PNG dari vSch.py:
    {SYMBOL}_{SIDE}_{execution_tf}_chart.png

    Tidak mereplikasi logic pemilihan timeframe vSch.py (itu
    tanggung jawab vSch.py). Kalau file tidak ditemukan dengan
    nama ini, dashboard cukup menampilkan placeholder.
    """
    symbol = str(candidate.get("symbol", "")).upper()
    side = str(candidate.get("side", "")).upper()
    tf = str(candidate.get("execution_tf", ""))

    filename = f"{symbol}_{side}_{tf}_chart.png"
    chart_path = charts_dir / filename

    if not chart_path.exists():
        return None

    # Relative path from the dashboard.html location, so the
    # file can be opened from anywhere on disk.
    return os.path.relpath(chart_path, start=output_dir)


# ============================================================
# HTML BUILDING
# ============================================================
def fmt_price(value):
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def setup_badge(setup_style):
    color = SETUP_COLORS.get(setup_style, SETUP_DEFAULT_COLOR)
    label = html.escape(str(setup_style or "-"))
    return (
        f'<span class="badge" style="background:{color}">{label}</span>'
    )


def side_badge(side):
    color = UP if side == "LONG" else DOWN
    return f'<span class="badge" style="background:{color}">{side}</span>'


def build_card(candidate, charts_dir, output_dir):
    symbol = html.escape(str(candidate.get("symbol", "?")))
    side = str(candidate.get("side", "?")).upper()
    setup_style = candidate.get("setup_style")
    score = candidate.get("score", "-")
    tf_agreement = candidate.get("tf_agreement", "-")
    execution_tf = html.escape(str(candidate.get("execution_tf", "-")))
    change24h = candidate.get("change24h", "-")
    quote_volume24h = candidate.get("quote_volume24h", "-")
    entry = candidate.get("entry")
    sl = candidate.get("sl")
    tp = candidate.get("tp") or []
    risk_pct = candidate.get("risk_pct", "-")
    key_points = candidate.get("key_points") or []

    tp_html = " / ".join(fmt_price(v) for v in tp) if tp else "-"

    key_points_html = "".join(
        f"<li>{html.escape(str(point))}</li>" for point in key_points
    )

    chart_rel_path = resolve_chart_path(charts_dir, output_dir, candidate)

    if chart_rel_path:
        chart_html = (
            f'<a href="{html.escape(chart_rel_path)}" target="_blank">'
            f'<img class="chart-thumb" '
            f'src="{html.escape(chart_rel_path)}" '
            f'alt="{symbol} chart"></a>'
        )
    else:
        chart_html = (
            '<div class="chart-missing">Chart belum tersedia '
            "(jalankan vSch.py dulu)</div>"
        )

    return f"""
    <div class="card">
      <div class="card-header">
        <span class="symbol">{symbol}</span>
        {side_badge(side)}
        {setup_badge(setup_style)}
        <span class="tf">{execution_tf}</span>
      </div>

      <div class="card-body">
        <div class="chart-col">
          {chart_html}
        </div>

        <div class="stats-col">
          <div class="stat-row">
            <span class="stat-label">Score</span>
            <span class="stat-value">{score}</span>
            <span class="stat-label">MTF</span>
            <span class="stat-value">{tf_agreement}/3</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">24h</span>
            <span class="stat-value">{change24h}%</span>
            <span class="stat-label">Vol24h</span>
            <span class="stat-value">{quote_volume24h}</span>
          </div>
          <div class="level-row">
            <span class="level-tag" style="color:{ENTRY}">ENTRY</span>
            <span>{fmt_price(entry)}</span>
          </div>
          <div class="level-row">
            <span class="level-tag" style="color:{SL_COLOR}">SL</span>
            <span>{fmt_price(sl)}</span>
          </div>
          <div class="level-row">
            <span class="level-tag" style="color:{TP_COLOR}">TP</span>
            <span>{tp_html}</span>
          </div>
          <div class="level-row">
            <span class="level-tag">RISK</span>
            <span>{risk_pct}%</span>
          </div>
          <ul class="key-points">{key_points_html}</ul>
        </div>
      </div>
    </div>
    """


def build_stat_pill(label, value):
    return (
        '<div class="pill">'
        f'<div class="pill-value">{value}</div>'
        f'<div class="pill-label">{html.escape(label)}</div>'
        "</div>"
    )


def build_html(payload, charts_dir, output_dir):
    candidates = payload.get("candidates", [])
    scan_stats = payload.get("scan_stats", {})
    generated_at = html.escape(str(payload.get("generated_at", "-")))
    selection_mode = html.escape(str(payload.get("selection_mode", "-")))
    dashboard_built_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    cards_html = "".join(
        build_card(candidate, charts_dir, output_dir)
        for candidate in candidates
    )

    pills = [
        build_stat_pill("Universe", scan_stats.get("universe", "-")),
        build_stat_pill(
            "Stage1 Selected", scan_stats.get("stage1_selected", "-")
        ),
        build_stat_pill("MTF Valid", scan_stats.get("mtf_valid", "-")),
        build_stat_pill(
            "Final Candidates", scan_stats.get("final_candidates", "-")
        ),
        build_stat_pill(
            "Elapsed (s)", scan_stats.get("elapsed_seconds", "-")
        ),
    ]
    pills_html = "".join(pills)

    empty_state = (
        '<div class="empty">Tidak ada kandidat pada scan ini.</div>'
        if not candidates
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synaptic Dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: {BG};
    color: {TEXT};
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
  }}
  header {{
    padding: 20px 16px 12px;
    border-bottom: 1px solid {BORDER};
  }}
  h1 {{
    margin: 0 0 4px;
    font-size: 20px;
  }}
  .meta {{
    color: {MUTED};
    font-size: 13px;
  }}
  .pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    padding: 14px 16px;
  }}
  .pill {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 8px 14px;
    min-width: 84px;
    text-align: center;
  }}
  .pill-value {{
    font-size: 18px;
    font-weight: 700;
  }}
  .pill-label {{
    font-size: 11px;
    color: {MUTED};
    margin-top: 2px;
  }}
  .cards {{
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 6px 16px 30px;
  }}
  .card {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
    background: {PANEL};
  }}
  .card-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: {BG};
    border-bottom: 1px solid {BORDER};
  }}
  .symbol {{
    font-weight: 700;
    font-size: 15px;
    margin-right: 4px;
  }}
  .tf {{
    margin-left: auto;
    color: {MUTED};
    font-size: 12px;
  }}
  .badge {{
    color: #fff;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 999px;
    letter-spacing: 0.02em;
  }}
  .card-body {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 12px 14px;
  }}
  .chart-col {{
    flex: 1 1 260px;
  }}
  .chart-thumb {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid {BORDER};
    display: block;
  }}
  .chart-missing {{
    width: 100%;
    min-height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: {MUTED};
    font-size: 12px;
    border: 1px dashed {BORDER};
    border-radius: 8px;
    text-align: center;
    padding: 10px;
  }}
  .stats-col {{
    flex: 1 1 220px;
    font-size: 13px;
  }}
  .stat-row {{
    display: grid;
    grid-template-columns: auto auto auto 1fr;
    gap: 4px 8px;
    margin-bottom: 6px;
  }}
  .stat-label {{
    color: {MUTED};
    font-size: 11px;
  }}
  .stat-value {{
    font-weight: 600;
  }}
  .level-row {{
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px dashed {BORDER};
  }}
  .level-tag {{
    font-weight: 700;
    font-size: 11px;
    color: {MUTED};
  }}
  .key-points {{
    margin: 8px 0 0;
    padding-left: 16px;
    color: {MUTED};
    font-size: 12px;
  }}
  .empty {{
    padding: 30px 16px;
    text-align: center;
    color: {MUTED};
  }}
</style>
</head>
<body>

<header>
  <h1>Synaptic Dashboard</h1>
  <div class="meta">
    Scan generated: {generated_at} &middot;
    Mode: {selection_mode} &middot;
    Dashboard dibuat: {dashboard_built_at}
  </div>
</header>

<div class="pills">{pills_html}</div>

{empty_state}

<div class="cards">
{cards_html}
</div>

</body>
</html>
"""


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="dashboard - static HTML viewer for Synaptic JSON"
    )
    parser.add_argument("--input", default="synaptic_candidates.json")
    parser.add_argument("--charts-dir", default="charts")
    parser.add_argument("--output", default="dashboard.html")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Synaptic JSON not found: {input_path}")

    charts_dir = Path(args.charts_dir)
    output_path = Path(args.output)
    output_dir = output_path.parent if output_path.parent != Path("") else Path(".")

    payload = load_payload(input_path)
    candidates = payload.get("candidates", [])

    doc = build_html(payload, charts_dir, output_dir)
    output_path.write_text(doc, encoding="utf-8")

    print(f"Dashboard: {output_path} ({len(candidates)} candidates)")


if __name__ == "__main__":
    main()
