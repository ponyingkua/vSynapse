#!/usr/bin/env python3
"""
vSch.py - JSON-only visual renderer for Synaptic.py.

Rules:
- Never fetch Binance data.
- The chart timeframe is exactly candidate['execution_tf'].
- Candle/indicator data comes from candidate['chart_data'][execution_tf].
- Setup levels (Entry/SL/TP) come from Synaptic JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


BG = "#050608"
PANEL = "#080b0f"
GRID = "#26303a"
TEXT = "#e6edf3"
MUTED = "#8b98a7"
UP = "#19c3a3"
DOWN = "#ff5c66"
EMA = "#ffd21f"                 # yellow, as requested
ST_UP = "#31c46b"
ST_DOWN = "#ff5b67"
ENTRY = "#2f9bff"
TP = "#18d6bd"
SL = "#ff4f5e"

VISIBLE_DEFAULTS = {"15m": 60, "1h": 60, "4h": 50}


def format_price(value, decimals):
    return f"{float(value):.{int(decimals)}f}"


def decimals_from_price(price):
    p = abs(float(price))
    if p < 0.0001:
        return 8
    if p < 0.001:
        return 7
    if p < 0.01:
        return 6
    if p < 0.1:
        return 5
    if p < 1:
        return 5
    if p < 10:
        return 4
    if p < 100:
        return 3
    return 2


def calculate_supertrend(df, period=10, multiplier=2.5):
    """Fallback only. Normally Synaptic's stored Supertrend is used."""
    high, low, close = df["high"], df["low"], df["close"]
    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(1, index=df.index, dtype=int)
    st = pd.Series(np.nan, index=df.index, dtype=float)

    for i in range(1, len(df)):
        if upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1

        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    if len(df):
        st.iloc[0] = final_lower.iloc[0]

    return st, direction


def _load_candidates(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON field 'candidates' must be a list")
    return candidates


def _build_dataframe(candidate, tf):
    chart_data = candidate.get("chart_data")
    if not isinstance(chart_data, dict):
        raise ValueError("candidate is missing chart_data")

    candles = chart_data.get(tf)
    if candles is None:
        for key, value in chart_data.items():
            if str(key).lower() == str(tf).lower():
                candles = value
                tf = key
                break
    if not candles:
        raise ValueError(f"no chart_data for timeframe {tf}")

    df = pd.DataFrame(candles)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("missing candle fields: " + ", ".join(missing))

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    time_col = "time" if "time" in df.columns else "timestamp"
    if time_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("not enough candles in JSON")

    # Use the exact indicators Synaptic serialized. Local calculation is only
    # a defensive fallback for old JSON files.
    if "ema200" in df:
        df["EMA200"] = pd.to_numeric(df["ema200"], errors="coerce")
    else:
        df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    if "supertrend" in df and "st_dir" in df:
        df["ST"] = pd.to_numeric(df["supertrend"], errors="coerce")
        df["ST_DIR"] = pd.to_numeric(df["st_dir"], errors="coerce")
    else:
        df["ST"], df["ST_DIR"] = calculate_supertrend(df, 10, 2.5)

    if "volume_ma" in df:
        df["VOL_MA"] = pd.to_numeric(df["volume_ma"], errors="coerce")
    else:
        df["VOL_MA"] = df["volume"].rolling(20, min_periods=1).mean()

    return df, str(tf)


def _resolve_visible_count(candidate, tf, cli_value):
    if cli_value is not None:
        return max(20, int(cli_value))
    chart = candidate.get("chart", {})
    visible = chart.get("visible_candles", {}) if isinstance(chart, dict) else {}
    return max(20, int(visible.get(tf, VISIBLE_DEFAULTS.get(tf, 60))))


def _level_label(level, color, text):
    return {
        "level": float(level),
        "color": color,
        "text": text,
    }


def _place_level_labels(ax, levels, label_x, y_min, y_max):
    """Keep bright level boxes separated while preserving their price order."""
    span = max(y_max - y_min, 1e-12)
    min_gap = span * 0.075
    edge = span * 0.025

    ordered = sorted(levels, key=lambda item: item["level"])
    positions = [item["level"] for item in ordered]

    # Forward pass: minimum vertical distance.
    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap

    # Pull the stack back inside the plot if needed.
    upper = y_max - edge
    if positions[-1] > upper:
        shift = positions[-1] - upper
        positions = [p - shift for p in positions]

    lower = y_min + edge
    if positions[0] < lower:
        shift = lower - positions[0]
        positions = [p + shift for p in positions]

    for item, y in zip(ordered, positions):
        ax.text(
            label_x, y, f" {item['text']} ",
            transform=ax.transData,
            ha="left", va="center",
            color="#ffffff", fontsize=8.0, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.38",
                facecolor=item["color"],
                edgecolor="#ffffff",
                linewidth=0.45,
                alpha=0.84,
            ),
            clip_on=False,
            zorder=20,
        )


def draw_visual_chart(df, setup, output_path, visible_count):
    symbol = str(setup.get("symbol", "UNKNOWN")).upper()
    side = str(setup.get("side", "LONG")).upper()
    tf = str(setup.get("execution_tf", "15m"))
    change_24h = float(setup.get("change24h", 0.0))
    q_vol = float(setup.get("quote_volume24h", 0.0))
    entry = float(setup["entry"])
    sl = float(setup["sl"])
    tps = [float(x) for x in setup.get("tp", [])[:3]]
    dec = int(setup.get("decimals", decimals_from_price(entry)))

    # Keep enough history for indicator context, but show an analysis-friendly
    # number of recent candles. Synaptic stores 240 candles for calculation.
    df = df.iloc[-min(visible_count, len(df)):].copy().reset_index(drop=True)
    x = np.arange(len(df), dtype=float)
    last_x = float(x[-1])

    fig = plt.figure(figsize=(15.0, 8.4), facecolor=BG)
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[4.7, 1.0],
        hspace=0.035,
        left=0.055, right=0.87, top=0.86, bottom=0.105,
    )
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    ax.set_facecolor(PANEL)
    axv.set_facecolor(PANEL)

    # Price range includes setup levels but avoids excessive whitespace.
    level_values = [entry, sl] + tps
    y_low = min(float(df["low"].min()), *level_values)
    y_high = max(float(df["high"].max()), *level_values)
    span = max(y_high - y_low, abs(y_high) * 0.01, 1e-12)
    pad = span * 0.10
    y_min, y_max = y_low - pad, y_high + pad

    # Candles + volume.
    for i, row in df.iterrows():
        o, h, l, c = map(float, (row.open, row.high, row.low, row.close))
        up = c >= o
        body_color = UP if up else DOWN
        ax.plot([i, i], [l, h], color=body_color, linewidth=1.15, alpha=0.95, zorder=4)
        body_low = min(o, c)
        body_h = max(abs(c - o), (h - l) * 0.012, 1e-12)
        ax.add_patch(Rectangle(
            (i - 0.36, body_low), 0.72, body_h,
            facecolor=body_color, edgecolor=body_color,
            linewidth=0.3, alpha=0.95, zorder=5,
        ))
        axv.bar(i, float(row.volume), width=0.72, color=body_color, alpha=0.32, linewidth=0)

    # EMA 200: yellow.
    ax.plot(x, df["EMA200"], color=EMA, linewidth=1.65, label="EMA 200", zorder=8)

    # Supertrend: exact Synaptic values, plus translucent fill from ST to the
    # candle low/high so the trend zone visually occupies the candle area.
    st_up = df["ST"].where(df["ST_DIR"] > 0)
    st_down = df["ST"].where(df["ST_DIR"] < 0)
    ax.plot(x, st_up, color=ST_UP, linewidth=1.8, label="Supertrend 10 / 2.5", zorder=7)
    ax.plot(x, st_down, color=ST_DOWN, linewidth=1.8, zorder=7)

    # The requested transparent Supertrend zone.
    bull = df["ST_DIR"] > 0
    bear = df["ST_DIR"] < 0
    ax.fill_between(x, df["ST"].astype(float), df["low"].astype(float),
                    where=bull, color=ST_UP, alpha=0.075, interpolate=True, zorder=1)
    ax.fill_between(x, df["high"].astype(float), df["ST"].astype(float),
                    where=bear, color=ST_DOWN, alpha=0.075, interpolate=True, zorder=1)

    # Market structure labels.
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    swing = 2
    raw_highs, raw_lows = [], []
    for i in range(swing, len(df) - swing):
        if highs[i] >= highs[i-swing:i+swing+1].max():
            raw_highs.append((i, highs[i]))
        if lows[i] <= lows[i-swing:i+swing+1].min():
            raw_lows.append((i, lows[i]))

    last_high = None
    for i, price in raw_highs[-4:]:
        label = "HH" if last_high is None or price > last_high else "LH"
        last_high = price
        col = ST_UP if label == "HH" else DOWN
        ax.text(i, price + span * 0.018, label, color=col, fontsize=8,
                fontweight="bold", ha="center", va="bottom", zorder=10)

    last_low = None
    for i, price in raw_lows[-4:]:
        label = "HL" if last_low is None or price > last_low else "LL"
        last_low = price
        col = UP if label == "HL" else DOWN
        ax.text(i, price - span * 0.018, label, color=col, fontsize=8,
                fontweight="bold", ha="center", va="top", zorder=10)

    # Entry/TP/SL lines.
    levels = [_level_label(entry, ENTRY, f"ENTRY  {format_price(entry, dec)}")]
    for i, value in enumerate(tps, 1):
        levels.append(_level_label(value, TP, f"TP{i}  {format_price(value, dec)}"))
    levels.append(_level_label(sl, SL, f"SL  {format_price(sl, dec)}"))

    for item in levels:
        ax.axhline(item["level"], color=item["color"], linestyle=(0, (5, 4)),
                   linewidth=1.05, alpha=0.58, zorder=2)

    # Give labels their own clean right-side lane.
    label_x = last_x + max(2.4, len(df) * 0.025)
    ax.set_xlim(-1.0, label_x + max(9.0, len(df) * 0.08))
    _place_level_labels(ax, levels, label_x, y_min, y_max)

    ax.set_ylim(y_min, y_max)

    # Volume MA, also from Synaptic JSON.
    axv.plot(x, df["VOL_MA"], color="#ff9b4a", linewidth=1.15, alpha=0.80)

    # Dark UI styling.
    ax.grid(True, color=GRID, alpha=0.55, linewidth=0.7)
    axv.grid(True, axis="y", color=GRID, alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)
    axv.set_axisbelow(True)
    for a in (ax, axv):
        a.tick_params(colors=MUTED, labelsize=8, length=3)
        for spine in a.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.7)
    ax.tick_params(labelbottom=False)

    # Time axis: 6 clean reference points.
    tick_count = min(6, len(df))
    tick_idx = np.linspace(0, len(df) - 1, tick_count, dtype=int)
    axv.set_xticks(tick_idx)
    labels = []
    for i in tick_idx:
        ts = df["timestamp"].iloc[i]
        labels.append(ts.strftime("%d %b  %H:%M") if pd.notna(ts) else str(i))
    axv.set_xticklabels(labels, color=MUTED, fontsize=8)

    # Compact black UI legend.
    legend = ax.legend(
        loc="upper left", ncol=2, fontsize=8,
        frameon=True, facecolor="#0b0f14", edgecolor=GRID,
        framealpha=0.92, labelcolor=TEXT, borderpad=0.55,
    )
    legend.get_frame().set_linewidth(0.7)

    # Header: explicitly tells the viewer the setup TF and the visual TF.
    current_price = float(df["close"].iloc[-1])
    header = (
        f"{symbol}  •  SETUP {tf.upper()}  •  VISUAL {tf.upper()}  •  "
        f"BIAS {side}"
    )
    subheader = (
        f"PRICE {format_price(current_price, dec)}    "
        f"24H {change_24h:+.2f}%    "
        f"VOL ${q_vol:,.0f}"
    )

    fig.text(
        0.055, 0.925, header,
        color="#ffffff", fontsize=14.5, fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.48", facecolor="#000000",
                  edgecolor="#20262d", linewidth=0.8, alpha=0.98),
    )
    fig.text(
        0.055, 0.887, subheader,
        color=MUTED, fontsize=8.5, fontweight="bold",
        ha="left", va="center",
    )
    fig.text(
        0.87, 0.925, f"SCORE {float(setup.get('score', 0)):.2f}  •  MTF {setup.get('tf_agreement', '-')}/3",
        color=TEXT, fontsize=8.5, fontweight="bold", ha="right", va="center",
    )

    fig.text(
        0.055, 0.045,
        f"EMA 200  •  Supertrend 10 / 2.5  •  {visible_count} candles shown  •  setup levels from Synaptic JSON",
        color=MUTED, fontsize=7.5, ha="left", va="center",
    )

    fig.savefig(output_path, dpi=160, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="vSch - JSON-only Synaptic chart renderer")
    parser.add_argument("--input", default="synaptic_candidates.json")
    parser.add_argument("--output-dir", default="charts")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--chart-candles", type=int, default=None,
                        help="Override visible candle count; otherwise use Synaptic chart settings")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Synaptic JSON not found: {input_path}")

    candidates = _load_candidates(input_path)
    if not candidates:
        print("No candidates found in Synaptic JSON. Nothing to render.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for candidate in candidates:
        symbol = str(candidate.get("symbol", "")).upper()
        if args.symbol and symbol != args.symbol.upper():
            continue

        tf = str(candidate.get("execution_tf", "15m"))
        side = str(candidate.get("side", "LONG")).upper()
        try:
            for field in ("entry", "sl", "tp"):
                if field not in candidate:
                    raise ValueError(f"missing top-level field '{field}'")

            df, actual_tf = _build_dataframe(candidate, tf)
            visible = _resolve_visible_count(candidate, actual_tf, args.chart_candles)
            output_file = output_dir / f"{symbol}_{side}_{actual_tf}_chart.png"

            print(f"Rendering {symbol} | setup={actual_tf} | visual={actual_tf} | candles={visible}")
            draw_visual_chart(df, candidate, output_file, visible)
            print(f"Chart: {output_file}")
            rendered += 1
        except Exception as exc:
            print(f"Skipping {symbol or 'UNKNOWN'}: {exc}")

    print(f"Chart generation completed: {rendered} chart(s).")


if __name__ == "__main__":
    main()
