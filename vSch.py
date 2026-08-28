#!/usr/bin/env python3
"""
vSch.py - JSON-only visual renderer for Synaptic.py.

Rules:
- Never fetch Binance data.
- Chart timeframe is exactly candidate['execution_tf'].
- Candle/indicator data comes from candidate['chart_data'][execution_tf].
- Setup levels (Entry/SL/TP) come from Synaptic JSON.
- Output optimized for 1:1 mobile viewing.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


# ============================================================
# MOBILE / BINANCE-LIKE LIGHT UI
# ============================================================

BG = "#ffffff"
PANEL = "#ffffff"

GRID = "#d9dde3"
TEXT = "#111111"
MUTED = "#626b75"

UP = "#16b89f"
DOWN = "#e84b5b"

EMA = "#f0c400"

ST_UP = "#22b965"
ST_DOWN = "#ed5363"

ENTRY = "#1687e8"
TP = "#14b8a6"
SL = "#e84b5b"

RSI_COLOR = "#7873ff"


# ============================================================
# VISIBLE CANDLES
# ============================================================

VISIBLE_DEFAULTS = {
    "15m": 44,
    "1h": 33,
    "4h": 36,
}


# ============================================================
# PRICE FORMATTING
# ============================================================

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


# ============================================================
# SYMBOL DISPLAY
# ============================================================

def _display_symbol(symbol):
    value = str(symbol).upper().strip()
    if value.endswith("USDT"):
        value = value[:-4]
    return value


# ============================================================
# TIMEFRAME NORMALIZATION
# ============================================================

def _normalize_tf(tf):
    value = str(tf).strip().lower()
    aliases = {
        "1h": "1h", "1hr": "1h", "1hour": "1h",
        "15m": "15m", "15min": "15m", "15minute": "15m",
        "4h": "4h", "4hr": "4h", "4hour": "4h",
    }
    return aliases.get(value, value)


# ============================================================
# JSON VALUE HELPERS
# ============================================================

def _first_numeric(data, keys, default=None):
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key not in data:
            continue
        value = data.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _get_mark_price(setup, fallback):
    value = _first_numeric(
        setup,
        ["mark_price", "markPrice", "current_price", "currentPrice", "price"],
        None,
    )
    if value is None:
        return float(fallback)
    return value


# ============================================================
# RSI CALCULATOR
# ============================================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


# ============================================================
# SUPERTREND FALLBACK
# ============================================================

def calculate_supertrend(df, period=10, multiplier=2.5):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

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


# ============================================================
# JSON & DATAFRAME
# ============================================================

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

    # EMA 200
    if "ema200" in df:
        df["EMA200"] = pd.to_numeric(df["ema200"], errors="coerce")
    else:
        df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    # SUPERTREND
    if "supertrend" in df and "st_dir" in df:
        df["ST"] = pd.to_numeric(df["supertrend"], errors="coerce")
        df["ST_DIR"] = pd.to_numeric(df["st_dir"], errors="coerce")
    else:
        df["ST"], df["ST_DIR"] = calculate_supertrend(df, 10, 2.5)

    # VOLUME MA
    if "volume_ma" in df:
        df["VOL_MA"] = pd.to_numeric(df["volume_ma"], errors="coerce")
    else:
        df["VOL_MA"] = df["volume"].rolling(20, min_periods=1).mean()

    # RSI
    if "rsi" in df:
        df["RSI"] = pd.to_numeric(df["rsi"], errors="coerce")
    else:
        df["RSI"] = calculate_rsi(df["close"], 14)

    return df, str(tf)


def _resolve_visible_count(candidate, tf, cli_value):
    normalized_tf = _normalize_tf(tf)
    if cli_value is not None:
        return max(20, int(cli_value))
    chart = candidate.get("chart", {})
    visible = chart.get("visible_candles", {}) if isinstance(chart, dict) else {}
    return max(20, int(visible.get(tf, VISIBLE_DEFAULTS.get(normalized_tf, 40))))


# ============================================================
# LEVEL HELPERS
# ============================================================

def _level_label(level, color, text):
    return {"level": float(level), "color": color, "text": text}


def _place_level_labels(ax, levels, label_x, y_min, y_max):
    span = max(y_max - y_min, 1e-12)
    min_gap = span * 0.038
    edge = span * 0.015

    ordered = sorted(levels, key=lambda item: item["level"])
    positions = [item["level"] for item in ordered]

    for i in range(1, len(positions)):
        required = positions[i - 1] + min_gap
        if positions[i] < required:
            positions[i] = required

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
            label_x, y,
            f" {item['text']} ",
            transform=ax.transData,
            ha="left", va="center",
            color=TEXT,
            fontsize=7.6,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=item["color"],
                edgecolor=item["color"],
                linewidth=0.7,
                alpha=0.15,
            ),
            clip_on=False,
            zorder=30,
        )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def _find_swing_points(df):
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    swing = 3
    raw_highs, raw_lows = [], []

    for i in range(swing, n - swing):
        local_high = highs[i - swing : i + swing + 1]
        local_low = lows[i - swing : i + swing + 1]
        if highs[i] >= local_high.max():
            raw_highs.append((i, highs[i]))
        if lows[i] <= local_low.min():
            raw_lows.append((i, lows[i]))

    return raw_highs, raw_lows


def _classify_structure(points, bullish=True):
    classified, previous = [], None
    for i, price in points:
        if previous is None:
            label = "H" if bullish else "L"
        else:
            if bullish:
                label = "HH" if price > previous else "LH"
            else:
                label = "HL" if price > previous else "LL"
        classified.append((i, price, label))
        previous = price
    return classified


def _select_structure_points(raw_highs, raw_lows, max_points=3):
    highs = _classify_structure(raw_highs, bullish=True)
    lows = _classify_structure(raw_lows, bullish=False)
    return highs[-max_points:], lows[-max_points:]


def _draw_structure(ax, df, span):
    raw_highs, raw_lows = _find_swing_points(df)
    highs, lows = _select_structure_points(raw_highs, raw_lows, max_points=3)
    
    candle_ranges = df["high"] - df["low"]
    median_range = float(candle_ranges.rolling(5, min_periods=1).median().iloc[-1])
    offset = max(span * 0.010, median_range * 0.42)

    for i, price, label in highs:
        if label == "H":
            continue
        col = ST_UP if label == "HH" else ST_DOWN
        ax.text(
            i, price + offset, label,
            color=col, fontsize=7.2, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", facecolor=BG, edgecolor="none", alpha=0.82),
            zorder=18,
        )

    for i, price, label in lows:
        if label == "L":
            continue
        col = ST_UP if label == "HL" else ST_DOWN
        ax.text(
            i, price - offset, label,
            color=col, fontsize=7.2, fontweight="bold",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.16", facecolor=BG, edgecolor="none", alpha=0.82),
            zorder=18,
        )


def _draw_target_arrow(ax, df, side, entry, tps, span):
    if not tps or len(df) < 5:
        return
    last_x = len(df) - 1
    current_price = float(df["close"].iloc[-1])
    target = float(tps[0])

    start_x = last_x - 1.2
    end_x = last_x + 2.0
    start_y = current_price
    target_y = current_price + (target - current_price) * 0.80
    rad = 0.20 if side == "LONG" else -0.20

    arrow = FancyArrowPatch(
        (start_x, start_y), (end_x, target_y),
        arrowstyle="-|>", mutation_scale=11, linewidth=1.55,
        color=ST_UP if side == "LONG" else ST_DOWN,
        alpha=0.68, connectionstyle=f"arc3,rad={rad}",
        zorder=16,
    )
    ax.add_patch(arrow)


# ============================================================
# MAIN CHART
# ============================================================

def draw_visual_chart(df, setup, output_path, visible_count):
    raw_symbol = str(setup.get("symbol", "UNKNOWN")).upper()
    display_symbol = _display_symbol(raw_symbol)
    side = str(setup.get("side", "LONG")).upper()
    tf = str(setup.get("execution_tf", "15m"))
    normalized_tf = _normalize_tf(tf)
    change_24h = float(setup.get("change24h", 0.0))

    entry = float(setup["entry"])
    sl = float(setup["sl"])
    tps = [float(x) for x in setup.get("tp", [])[:3]]
    dec = int(setup.get("decimals", decimals_from_price(entry)))

    visible_count = min(visible_count, len(df))
    df = df.iloc[-visible_count:].copy().reset_index(drop=True)
    x = np.arange(len(df), dtype=float)
    last_x = float(x[-1])

    # FIGURE & SUBPLOTS (Grid 3 row: Chart, Volume, RSI)
    fig = plt.figure(figsize=(10.0, 10.5), facecolor=BG)
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[5.2, 0.70, 0.70],
        hspace=0.040,
        left=0.060,
        right=0.850,
        top=0.790,
        bottom=0.080,
    )

    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    axrsi = fig.add_subplot(gs[2], sharex=ax)

    ax.set_facecolor(PANEL)
    axv.set_facecolor(PANEL)
    axrsi.set_facecolor(PANEL)

    # PRICE RANGE
    level_values = [entry, sl, *tps]
    y_low = min(float(df["low"].min()), *level_values)
    y_high = max(float(df["high"].max()), *level_values)
    span = max(y_high - y_low, abs(y_high) * 0.01, 1e-12)
    pad = span * 0.075
    y_min = y_low - pad
    y_max = y_high + pad

    candle_width = 0.82

    # CANDLES & VOLUME
    for i, row in df.iterrows():
        o = float(row.open)
        h = float(row.high)
        l = float(row.low)
        c = float(row.close)
        up = c >= o
        body_color = UP if up else DOWN

        ax.plot([i, i], [l, h], color=body_color, linewidth=1.30, alpha=0.92, zorder=5)
        body_low = min(o, c)
        body_height = max(abs(c - o), (h - l) * 0.015, 1e-12)
        ax.add_patch(
            Rectangle(
                (i - candle_width / 2, body_low),
                candle_width, body_height,
                facecolor=body_color, edgecolor=body_color,
                linewidth=0.30, alpha=0.98, zorder=6,
            )
        )

        axv.bar(i, float(row.volume), width=candle_width, color=body_color, alpha=0.26, linewidth=0, zorder=2)

    # EMA 200 & SUPERTREND
    ax.plot(x, df["EMA200"], color=EMA, linewidth=1.85, label="EMA 200", zorder=9)
    st_up = df["ST"].where(df["ST_DIR"] > 0)
    st_down = df["ST"].where(df["ST_DIR"] < 0)
    ax.plot(x, st_up, color=ST_UP, linewidth=1.75, label="Supertrend 10 / 2.5", zorder=8)
    ax.plot(x, st_down, color=ST_DOWN, linewidth=1.75, zorder=8)

    bull = df["ST_DIR"] > 0
    bear = df["ST_DIR"] < 0
    ax.fill_between(x, df["ST"].astype(float), df["low"].astype(float), where=bull, color=ST_UP, alpha=0.085, interpolate=True, zorder=1)
    ax.fill_between(x, df["high"].astype(float), df["ST"].astype(float), where=bear, color=ST_DOWN, alpha=0.085, interpolate=True, zorder=1)

    _draw_structure(ax, df, span)

    levels = [_level_label(entry, ENTRY, f"ENTRY  {format_price(entry, dec)}")]
    for i, value in enumerate(tps, 1):
        levels.append(_level_label(value, TP, f"TP{i}  {format_price(value, dec)}"))
    levels.append(_level_label(sl, SL, f"SL  {format_price(sl, dec)}"))

    for item in levels:
        ax.axhline(item["level"], color=item["color"], linestyle=(0, (5, 4)), linewidth=0.95, alpha=0.48, zorder=3)

    right_space = max(8.0, len(df) * 0.18)
    ax.set_xlim(-0.8, last_x + right_space)
    _place_level_labels(ax, levels, last_x + 2.65, y_min, y_max)
    _draw_target_arrow(ax, df, side, entry, tps, span)
    ax.set_ylim(y_min, y_max)

    # VOLUME MA
    volume = df["volume"].astype(float)
    volume_cap = float(volume.quantile(0.95))
    display_volume = volume.clip(upper=volume_cap) if volume_cap > 0 else volume

    axv.clear()
    axv.set_facecolor(PANEL)
    for i, row in df.iterrows():
        o = float(row.open)
        c = float(row.close)
        body_color = UP if c >= o else DOWN
        axv.bar(i, float(display_volume.iloc[i]), width=candle_width, color=body_color, alpha=0.27, linewidth=0, zorder=2)

    vol_ma = df["VOL_MA"].astype(float)
    vol_ma_display = vol_ma.clip(upper=volume_cap) if volume_cap > 0 else vol_ma
    axv.plot(x, vol_ma_display, color="#b87333", linewidth=1.05, alpha=0.70, zorder=3)

    # RSI SUBPLOT
    axrsi.clear()
    axrsi.set_facecolor(PANEL)
    axrsi.plot(x, df["RSI"], color=RSI_COLOR, linewidth=1.2, label="RSI", zorder=3)
    axrsi.axhline(70, color=DOWN, linestyle="--", linewidth=0.7, alpha=0.5)
    axrsi.axhline(30, color=UP, linestyle="--", linewidth=0.7, alpha=0.5)
    axrsi.set_ylim(0, 100)
    axrsi.set_yticks([30, 70])

    # GRIDS
    ax.grid(True, color=GRID, alpha=0.62, linewidth=0.60)
    axv.grid(True, axis="y", color=GRID, alpha=0.50, linewidth=0.60)
    axrsi.grid(True, axis="y", color=GRID, alpha=0.50, linewidth=0.60)
    ax.set_axisbelow(True)
    axv.set_axisbelow(True)
    axrsi.set_axisbelow(True)

    # RIGHT PRICE AXIS
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", colors=TEXT, labelsize=7.6, length=0, pad=5)
    axv.tick_params(labelbottom=False, axis="y", colors=MUTED, labelsize=6.8, length=0, pad=4)
    axrsi.tick_params(axis="y", colors=MUTED, labelsize=6.8, length=0, pad=4)

    # SPINES
    for a in (ax, axv, axrsi):
        for spine in a.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(0.60)

    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    axv.spines["left"].set_visible(False)
    axv.spines["top"].set_visible(False)
    axrsi.spines["left"].set_visible(False)
    axrsi.spines["top"].set_visible(False)

    ax.tick_params(labelbottom=False)
    axv.tick_params(labelbottom=False)

    # X TICKS
    tick_count = min(5, len(df))
    tick_idx = np.linspace(0, len(df) - 1, tick_count, dtype=int)
    axrsi.set_xticks(tick_idx)

    labels = []
    for i in tick_idx:
        ts = df["timestamp"].iloc[i]
        if pd.notna(ts):
            labels.append(ts.strftime("%d %b\n%H:%M"))
        else:
            labels.append(str(i))

    axrsi.set_xticklabels(labels, color=MUTED, fontsize=7.0)

    # LEGEND
    legend = ax.legend(
        loc="upper left", ncol=2, fontsize=6.9, frameon=True,
        facecolor="#ffffff", edgecolor=GRID, framealpha=0.90,
        labelcolor=TEXT, borderpad=0.38, handlelength=1.8, columnspacing=0.8,
    )
    legend.get_frame().set_linewidth(0.60)

    # ========================================================
    # HEADER DATA & LAYOUT PENYESUAIAN
    # ========================================================

    candle_close = float(df["close"].iloc[-1])
    current_price = _get_mark_price(setup, candle_close)

    header_left = 0.065

    # 1. Nama Token
    fig.text(
        header_left, 0.955,
        f"${display_symbol}/USDT",
        color=TEXT, fontsize=19.0, fontweight="bold",
        ha="left", va="center",
    )

    # 2. Harga & Persen Kenaikan disamping harga
    price_change_color = ST_UP if change_24h >= 0 else ST_DOWN
    price_y = 0.908

    fig.text(
        header_left, price_y,
        f"${format_price(current_price, dec)}",
        color=TEXT, fontsize=10.5, fontweight="bold",
        ha="left", va="center",
    )

    # Menggunakan offset proporsional untuk persen di samping harga
    fig.text(
        header_left + 0.170, price_y,
        f"{change_24h:+.2f}%",
        color=price_change_color, fontsize=9.5, fontweight="bold",
        ha="left", va="center",
    )

    # 3. Skor dan MTF di bawah nama token, di atas harga
    score_val = float(setup.get('score', 0))
    mtf_val = setup.get('tf_agreement', '-')
    fig.text(
        header_left, 0.865,
        f"SCORE {score_val:.2f}  •  MTF {mtf_val}/3",
        color=MUTED, fontsize=7.8, fontweight="bold",
        ha="left", va="center",
    )

    # 4. Setup & Bias
    fig.text(
        header_left, 0.825,
        f"SETUP {normalized_tf.upper()}  —  BIAS {side}",
        color="#f0b900", fontsize=9.0, fontweight="bold",
        ha="left", va="center",
    )

    # FOOTER
    footer = "Setup levels from Synaptic JSON. Output optimized for 1:1 mobile viewing."
    fig.text(header_left, 0.025, footer, color=MUTED, fontsize=6.8, ha="left", va="center")

    # SAVE
    fig.savefig(
        output_path, dpi=180, facecolor=BG, edgecolor=BG,
        bbox_inches="tight", pad_inches=0.06,
    )
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="vSch - JSON-only Synaptic chart renderer")
    parser.add_argument("--input", default="synaptic_candidates.json")
    parser.add_argument("--output-dir", default="charts")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--chart-candles", type=int, default=None)
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

            if args.chart_candles is None and _normalize_tf(actual_tf) == "1h":
                visible = min(33, len(df))

            output_file = output_dir / f"{symbol}_{side}_{actual_tf}_chart.png"
            print(f"Rendering {symbol} | setup={actual_tf} | candles={visible} | frame=1:1")

            draw_visual_chart(df, candidate, output_file, visible)
            print(f"Chart: {output_file}")
            rendered += 1

        except Exception as exc:
            print(f"Skipping {symbol or 'UNKNOWN'}: {exc}")

    print(f"Chart generation completed: {rendered} chart(s).")


if __name__ == "__main__":
    main()
