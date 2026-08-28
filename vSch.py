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
from matplotlib.patches import Rectangle, FancyArrowPatch


# ============================================================
# UI COLORS
# ============================================================

BG = "#020304"
PANEL = "#030507"
GRID = "#171d24"

TEXT = "#edf2f7"
MUTED = "#7d8996"

UP = "#19c7a5"
DOWN = "#f05260"

EMA = "#ffd21f"

ST_UP = "#32c875"
ST_DOWN = "#f45b6b"

ENTRY = "#3b9cff"
TP = "#20cdb5"
SL = "#f15a67"

VOL_MA = "#c8874b"


VISIBLE_DEFAULTS = {
    "15m": 60,
    "1h": 60,
    "4h": 50,
}


# ============================================================
# PRICE HELPERS
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
# SUPERTREND FALLBACK
# ============================================================

def calculate_supertrend(df, period=10, multiplier=2.5):
    """Fallback only. Normally Synaptic's stored Supertrend is used."""

    high = df["high"]
    low = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()

    direction = pd.Series(
        1,
        index=df.index,
        dtype=int
    )

    st = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    for i in range(1, len(df)):

        if (
            upper.iloc[i] < final_upper.iloc[i - 1]
            or close.iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == 1:
            direction.iloc[i] = (
                -1
                if close.iloc[i] < final_lower.iloc[i]
                else 1
            )
        else:
            direction.iloc[i] = (
                1
                if close.iloc[i] > final_upper.iloc[i]
                else -1
            )

        st.iloc[i] = (
            final_lower.iloc[i]
            if direction.iloc[i] == 1
            else final_upper.iloc[i]
        )

    if len(df):
        st.iloc[0] = final_lower.iloc[0]

    return st, direction


# ============================================================
# JSON
# ============================================================

def _load_candidates(path):

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            "JSON root must be an object"
        )

    candidates = data.get(
        "candidates",
        []
    )

    if not isinstance(candidates, list):
        raise ValueError(
            "JSON field 'candidates' must be a list"
        )

    return candidates


# ============================================================
# DATAFRAME
# ============================================================

def _build_dataframe(candidate, tf):

    chart_data = candidate.get("chart_data")

    if not isinstance(chart_data, dict):
        raise ValueError(
            "candidate is missing chart_data"
        )

    candles = chart_data.get(tf)

    if candles is None:

        for key, value in chart_data.items():

            if str(key).lower() == str(tf).lower():

                candles = value
                tf = key
                break

    if not candles:
        raise ValueError(
            f"no chart_data for timeframe {tf}"
        )

    df = pd.DataFrame(candles)

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "missing candle fields: "
            + ", ".join(missing)
        )

    for c in required:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    time_col = (
        "time"
        if "time" in df.columns
        else "timestamp"
    )

    if time_col in df.columns:

        df["timestamp"] = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce"
        )

    else:

        df["timestamp"] = pd.NaT

    df = (
        df
        .dropna(subset=required)
        .reset_index(drop=True)
    )

    if len(df) < 10:
        raise ValueError(
            "not enough candles in JSON"
        )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    if "ema200" in df:

        df["EMA200"] = pd.to_numeric(
            df["ema200"],
            errors="coerce"
        )

    else:

        df["EMA200"] = (
            df["close"]
            .ewm(
                span=200,
                adjust=False
            )
            .mean()
        )

    # --------------------------------------------------------
    # SUPERTREND
    # --------------------------------------------------------

    if (
        "supertrend" in df.columns
        and "st_dir" in df.columns
    ):

        df["ST"] = pd.to_numeric(
            df["supertrend"],
            errors="coerce"
        )

        df["ST_DIR"] = pd.to_numeric(
            df["st_dir"],
            errors="coerce"
        )

    else:

        (
            df["ST"],
            df["ST_DIR"]
        ) = calculate_supertrend(
            df,
            10,
            2.5
        )

    # --------------------------------------------------------
    # VOLUME MA
    # --------------------------------------------------------

    if "volume_ma" in df:

        df["VOL_MA"] = pd.to_numeric(
            df["volume_ma"],
            errors="coerce"
        )

    else:

        df["VOL_MA"] = (
            df["volume"]
            .rolling(
                20,
                min_periods=1
            )
            .mean()
        )

    return df, str(tf)


# ============================================================
# VISIBLE CANDLE COUNT
# ============================================================

def _resolve_visible_count(
    candidate,
    tf,
    cli_value
):

    if cli_value is not None:

        return max(
            20,
            int(cli_value)
        )

    chart = candidate.get(
        "chart",
        {}
    )

    visible = (
        chart.get(
            "visible_candles",
            {}
        )
        if isinstance(chart, dict)
        else {}
    )

    return max(
        20,
        int(
            visible.get(
                tf,
                VISIBLE_DEFAULTS.get(
                    tf,
                    60
                )
            )
        )
    )


# ============================================================
# LEVEL LABELS
# ============================================================

def _level_label(
    level,
    color,
    text
):

    return {
        "level": float(level),
        "color": color,
        "text": text,
    }


def _place_level_labels(
    ax,
    levels,
    label_x,
    y_min,
    y_max
):

    """
    Places Entry / TP / SL labels with
    dynamic vertical spacing.
    """

    span = max(
        y_max - y_min,
        1e-12
    )

    # Slightly tighter than previous version,
    # but still prevents collisions.
    min_gap = span * 0.065
    edge = span * 0.022

    ordered = sorted(
        levels,
        key=lambda item: item["level"]
    )

    positions = [
        item["level"]
        for item in ordered
    ]

    # Forward pass.
    for i in range(
        1,
        len(positions)
    ):

        if (
            positions[i]
            - positions[i - 1]
            < min_gap
        ):

            positions[i] = (
                positions[i - 1]
                + min_gap
            )

    # Keep inside chart.
    upper = y_max - edge

    if positions[-1] > upper:

        shift = (
            positions[-1]
            - upper
        )

        positions = [
            p - shift
            for p in positions
        ]

    lower = y_min + edge

    if positions[0] < lower:

        shift = (
            lower
            - positions[0]
        )

        positions = [
            p + shift
            for p in positions
        ]

    for item, y in zip(
        ordered,
        positions
    ):

        ax.text(
            label_x,
            y,
            f" {item['text']} ",
            transform=ax.transData,
            ha="left",
            va="center",
            color="#ffffff",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor=item["color"],
                edgecolor=item["color"],
                linewidth=0.65,
                alpha=0.76,
            ),
            clip_on=False,
            zorder=30,
        )


# ============================================================
# IMPORTANT MARKET STRUCTURE
# ============================================================

def _find_market_structure(df):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swing = 2

    raw_highs = []
    raw_lows = []

    for i in range(
        swing,
        len(df) - swing
    ):

        local_high = highs[
            i - swing:i + swing + 1
        ]

        local_low = lows[
            i - swing:i + swing + 1
        ]

        if highs[i] >= local_high.max():
            raw_highs.append(
                (i, highs[i])
            )

        if lows[i] <= local_low.min():
            raw_lows.append(
                (i, lows[i])
            )

    # --------------------------------------------------------
    # Only keep the most relevant recent structures.
    # --------------------------------------------------------

    important_highs = raw_highs[-2:]
    important_lows = raw_lows[-2:]

    high_labels = []

    previous = None

    for i, price in important_highs:

        label = (
            "HH"
            if previous is None
            or price > previous
            else "LH"
        )

        high_labels.append(
            (i, price, label)
        )

        previous = price

    low_labels = []

    previous = None

    for i, price in important_lows:

        label = (
            "HL"
            if previous is None
            or price > previous
            else "LL"
        )

        low_labels.append(
            (i, price, label)
        )

        previous = price

    return (
        high_labels,
        low_labels
    )


# ============================================================
# CURVED TARGET DIRECTION
# ============================================================

def _draw_target_direction(
    ax,
    last_x,
    current_price,
    targets,
    side,
    candle_span
):

    """
    Draws a curved continuation arrow in the
    empty lane to the right of the last candle.

    It does not alter setup prices.
    """

    if not targets:
        return

    side = str(side).upper()

    target = (
        targets[0]
        if side == "LONG"
        else targets[0]
    )

    # Directional vertical distance.
    direction = (
        1
        if target > current_price
        else -1
    )

    distance = abs(
        target - current_price
    )

    # Keep annotation visually contained.
    distance = max(
        distance,
        candle_span * 0.22
    )

    start_x = (
        last_x
        + max(1.4, len(ax.lines) * 0.0)
    )

    end_x = start_x + 3.8

    start_y = current_price

    end_y = (
        current_price
        + direction * distance * 0.72
    )

    # Arrow color follows direction.
    arrow_color = (
        TP
        if direction > 0
        else SL
    )

    # Main curved arrow.
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.5,
        color=arrow_color,
        alpha=0.78,
        connectionstyle="arc3,rad=-0.24"
        if direction > 0
        else "arc3,rad=0.24",
        zorder=25,
    )

    ax.add_patch(arrow)

    # Small annotation text.
    text_y = (
        end_y
        + direction * candle_span * 0.035
    )

    ax.text(
        end_x + 0.25,
        text_y,
        "NEXT TARGET",
        color=arrow_color,
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="center",
        alpha=0.88,
        zorder=26,
    )


# ============================================================
# MAIN DRAW
# ============================================================

def draw_visual_chart(
    df,
    setup,
    output_path,
    visible_count
):

    symbol = str(
        setup.get(
            "symbol",
            "UNKNOWN"
        )
    ).upper()

    side = str(
        setup.get(
            "side",
            "LONG"
        )
    ).upper()

    tf = str(
        setup.get(
            "execution_tf",
            "15m"
        )
    )

    change_24h = float(
        setup.get(
            "change24h",
            0.0
        )
    )

    q_vol = float(
        setup.get(
            "quote_volume24h",
            0.0
        )
    )

    entry = float(
        setup["entry"]
    )

    sl = float(
        setup["sl"]
    )

    tps = [
        float(x)
        for x in setup.get(
            "tp",
            []
        )[:3]
    ]

    dec = int(
        setup.get(
            "decimals",
            decimals_from_price(entry)
        )
    )

    # --------------------------------------------------------
    # Recent candles only.
    # --------------------------------------------------------

    df = (
        df.iloc[
            -min(
                visible_count,
                len(df)
            ):
        ]
        .copy()
        .reset_index(drop=True)
    )

    x = np.arange(
        len(df),
        dtype=float
    )

    last_x = float(
        x[-1]
    )

    # --------------------------------------------------------
    # Extra right-side annotation lane.
    # --------------------------------------------------------

    right_gap = max(
        5.5,
        len(df) * 0.12
    )

    label_gap = max(
        5.0,
        len(df) * 0.075
    )

    chart_right = (
        last_x
        + right_gap
    )

    label_x = (
        last_x
        + label_gap
    )

    # --------------------------------------------------------
    # Figure.
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(15.0, 8.4),
        facecolor=BG
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            4.9,
            1.0
        ],
        hspace=0.035,
        left=0.055,
        right=0.87,
        top=0.86,
        bottom=0.105,
    )

    ax = fig.add_subplot(
        gs[0]
    )

    axv = fig.add_subplot(
        gs[1],
        sharex=ax
    )

    ax.set_facecolor(PANEL)
    axv.set_facecolor(PANEL)

    # --------------------------------------------------------
    # Price range.
    # --------------------------------------------------------

    level_values = [
        entry,
        sl,
        *tps
    ]

    y_low = min(
        float(df["low"].min()),
        *level_values
    )

    y_high = max(
        float(df["high"].max()),
        *level_values
    )

    span = max(
        y_high - y_low,
        abs(y_high) * 0.01,
        1e-12
    )

    pad = span * 0.09

    y_min = y_low - pad
    y_max = y_high + pad

    # --------------------------------------------------------
    # Candles.
    # --------------------------------------------------------

    candle_width = 0.78

    for i, row in df.iterrows():

        o = float(row.open)
        h = float(row.high)
        l = float(row.low)
        c = float(row.close)

        up = c >= o

        body_color = (
            UP
            if up
            else DOWN
        )

        # Wick.
        ax.plot(
            [i, i],
            [l, h],
            color=body_color,
            linewidth=1.45,
            alpha=0.98,
            solid_capstyle="round",
            zorder=6,
        )

        # Body.
        body_low = min(
            o,
            c
        )

        real_body = abs(
            c - o
        )

        # Keep tiny candles visible without
        # artificially changing their price geometry.
        body_h = max(
            real_body,
            span * 0.0025
        )

        ax.add_patch(
            Rectangle(
                (
                    i
                    - candle_width / 2,
                    body_low
                ),
                candle_width,
                body_h,
                facecolor=body_color,
                edgecolor=body_color,
                linewidth=0.35,
                alpha=0.98,
                zorder=7,
            )
        )

        # Volume.
        axv.bar(
            i,
            float(row.volume),
            width=candle_width,
            color=body_color,
            alpha=0.30,
            linewidth=0,
        )

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    ema_values = df[
        "EMA200"
    ].astype(float)

    # Subtle glow.
    ax.plot(
        x,
        ema_values,
        color=EMA,
        linewidth=4.2,
        alpha=0.09,
        zorder=7,
    )

    # Main line.
    ax.plot(
        x,
        ema_values,
        color=EMA,
        linewidth=1.85,
        alpha=0.96,
        label="EMA 200",
        zorder=9,
    )

    # --------------------------------------------------------
    # SUPERTREND
    # --------------------------------------------------------

    st_values = df[
        "ST"
    ].astype(float)

    st_up = st_values.where(
        df["ST_DIR"] > 0
    )

    st_down = st_values.where(
        df["ST_DIR"] < 0
    )

    # Softer line glow.
    ax.plot(
        x,
        st_up,
        color=ST_UP,
        linewidth=4.0,
        alpha=0.055,
        zorder=6,
    )

    ax.plot(
        x,
        st_down,
        color=ST_DOWN,
        linewidth=4.0,
        alpha=0.055,
        zorder=6,
    )

    # Main Supertrend lines.
    ax.plot(
        x,
        st_up,
        color=ST_UP,
        linewidth=1.65,
        alpha=0.76,
        label="Supertrend 10 / 2.5",
        zorder=8,
    )

    ax.plot(
        x,
        st_down,
        color=ST_DOWN,
        linewidth=1.65,
        alpha=0.76,
        zorder=8,
    )

    # --------------------------------------------------------
    # Transparent ST zones.
    #
    # Bull:
    # ST -> candle LOW
    #
    # Bear:
    # candle HIGH -> ST
    # --------------------------------------------------------

    bull = (
        df["ST_DIR"] > 0
    )

    bear = (
        df["ST_DIR"] < 0
    )

    ax.fill_between(
        x,
        st_values,
        df["low"].astype(float),
        where=bull,
        color=ST_UP,
        alpha=0.105,
        interpolate=True,
        zorder=1,
    )

    ax.fill_between(
        x,
        df["high"].astype(float),
        st_values,
        where=bear,
        color=ST_DOWN,
        alpha=0.105,
        interpolate=True,
        zorder=1,
    )

    # --------------------------------------------------------
    # Market structure - only important points.
    # --------------------------------------------------------

    (
        high_labels,
        low_labels
    ) = _find_market_structure(df)

    structure_offset = (
        span * 0.018
    )

    for i, price, label in high_labels:

        col = (
            ST_UP
            if label == "HH"
            else DOWN
        )

        ax.text(
            i,
            price + structure_offset,
            label,
            color=col,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=12,
        )

    for i, price, label in low_labels:

        col = (
            UP
            if label == "HL"
            else DOWN
        )

        ax.text(
            i,
            price - structure_offset,
            label,
            color=col,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=12,
        )

    # --------------------------------------------------------
    # ENTRY / TP / SL
    # --------------------------------------------------------

    levels = [
        _level_label(
            entry,
            ENTRY,
            f"ENTRY  {format_price(entry, dec)}"
        )
    ]

    for i, value in enumerate(
        tps,
        1
    ):

        levels.append(
            _level_label(
                value,
                TP,
                f"TP{i}  {format_price(value, dec)}"
            )
        )

    levels.append(
        _level_label(
            sl,
            SL,
            f"SL  {format_price(sl, dec)}"
        )
    )

    for item in levels:

        ax.axhline(
            item["level"],
            color=item["color"],
            linestyle=(0, (5, 4)),
            linewidth=1.0,
            alpha=0.48,
            zorder=2,
        )

    # --------------------------------------------------------
    # Right-side label lane.
    # --------------------------------------------------------

    ax.set_xlim(
        -1.0,
        chart_right
    )

    _place_level_labels(
        ax,
        levels,
        label_x,
        y_min,
        y_max
    )

    # --------------------------------------------------------
    # Curved next-target annotation.
    # --------------------------------------------------------

    current_price = float(
        df["close"].iloc[-1]
    )

    if tps:

        # For both directions the first TP is the
        # nearest planned target in the JSON.
        next_target = tps[0]

        _draw_target_direction(
            ax,
            last_x,
            current_price,
            [next_target],
            side,
            span
        )

    ax.set_ylim(
        y_min,
        y_max
    )

    # --------------------------------------------------------
    # Volume MA.
    # --------------------------------------------------------

    axv.plot(
        x,
        df["VOL_MA"],
        color=VOL_MA,
        linewidth=1.2,
        alpha=0.72,
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    ax.grid(
        True,
        color=GRID,
        alpha=0.48,
        linewidth=0.65,
    )

    axv.grid(
        True,
        axis="y",
        color=GRID,
        alpha=0.38,
        linewidth=0.65,
    )

    ax.set_axisbelow(True)
    axv.set_axisbelow(True)

    for a in (
        ax,
        axv
    ):

        a.tick_params(
            colors=MUTED,
            labelsize=8,
            length=3
        )

        for spine in a.spines.values():

            spine.set_color(
                GRID
            )

            spine.set_linewidth(
                0.6
            )

    ax.tick_params(
        labelbottom=False
    )

    # --------------------------------------------------------
    # TIME AXIS
    # --------------------------------------------------------

    tick_count = min(
        6,
        len(df)
    )

    tick_idx = np.linspace(
        0,
        len(df) - 1,
        tick_count,
        dtype=int
    )

    axv.set_xticks(
        tick_idx
    )

    labels = []

    for i in tick_idx:

        ts = df[
            "timestamp"
        ].iloc[i]

        if pd.notna(ts):

            labels.append(
                ts.strftime(
                    "%d %b  %H:%M"
                )
            )

        else:

            labels.append(
                str(i)
            )

    axv.set_xticklabels(
        labels,
        color=MUTED,
        fontsize=8
    )

    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend = ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=8,
        frameon=True,
        facecolor="#05070a",
        edgecolor="#161b20",
        framealpha=0.78,
        labelcolor=TEXT,
        borderpad=0.5,
    )

    legend.get_frame().set_linewidth(
        0.55
    )

    # --------------------------------------------------------
    # HEADER
    #
    # NO BORDER
    # NO "VISUAL 1H / VISUAL 4H"
    # --------------------------------------------------------

    current_price = float(
        df["close"].iloc[-1]
    )

    header = (
        f"{symbol}"
        f"  •  SETUP {tf.upper()}"
        f"  •  BIAS {side}"
    )

    subheader = (
        f"PRICE "
        f"{format_price(current_price, dec)}"
        f"    24H "
        f"{change_24h:+.2f}%"
        f"    VOL "
        f"${q_vol:,.0f}"
    )

    fig.text(
        0.055,
        0.925,
        header,
        color="#ffffff",
        fontsize=14.5,
        fontweight="bold",
        ha="left",
        va="center",
    )

    fig.text(
        0.055,
        0.887,
        subheader,
        color=MUTED,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
    )

    fig.text(
        0.87,
        0.925,
        (
            f"SCORE "
            f"{float(setup.get('score', 0)):.2f}"
            f"  •  MTF "
            f"{setup.get('tf_agreement', '-')}/3"
        ),
        color=TEXT,
        fontsize=8.5,
        fontweight="bold",
        ha="right",
        va="center",
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    fig.text(
        0.055,
        0.045,
        (
            f"EMA 200"
            f"  •  Supertrend 10 / 2.5"
            f"  •  {visible_count} candles shown"
            f"  •  setup levels from Synaptic JSON"
        ),
        color=MUTED,
        fontsize=7.5,
        ha="left",
        va="center",
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fig.savefig(
        output_path,
        dpi=160,
        facecolor=BG,
        bbox_inches="tight",
        pad_inches=0.12
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "vSch - JSON-only "
            "Synaptic chart renderer"
        )
    )

    parser.add_argument(
        "--input",
        default="synaptic_candidates.json"
    )

    parser.add_argument(
        "--output-dir",
        default="charts"
    )

    parser.add_argument(
        "--symbol",
        default=None
    )

    parser.add_argument(
        "--chart-candles",
        type=int,
        default=None,
        help=(
            "Override visible candle count; "
            "otherwise use Synaptic chart settings"
        )
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Synaptic JSON not found: "
            f"{input_path}"
        )

    candidates = _load_candidates(
        input_path
    )

    if not candidates:

        print(
            "No candidates found in "
            "Synaptic JSON. Nothing to render."
        )

        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rendered = 0

    for candidate in candidates:

        symbol = str(
            candidate.get(
                "symbol",
                ""
            )
        ).upper()

        if (
            args.symbol
            and symbol != args.symbol.upper()
        ):
            continue

        # ----------------------------------------------------
        # IMPORTANT:
        # visual TF ALWAYS equals execution_tf.
        # ----------------------------------------------------

        tf = str(
            candidate.get(
                "execution_tf",
                "15m"
            )
        )

        side = str(
            candidate.get(
                "side",
                "LONG"
            )
        ).upper()

        try:

            for field in (
                "entry",
                "sl",
                "tp"
            ):

                if field not in candidate:

                    raise ValueError(
                        f"missing top-level field "
                        f"'{field}'"
                    )

            df, actual_tf = _build_dataframe(
                candidate,
                tf
            )

            visible = _resolve_visible_count(
                candidate,
                actual_tf,
                args.chart_candles
            )

            output_file = (
                output_dir
                / f"{symbol}_{side}_{actual_tf}_chart.png"
            )

            print(
                f"Rendering {symbol}"
                f" | setup={actual_tf}"
                f" | visual={actual_tf}"
                f" | candles={visible}"
            )

            draw_visual_chart(
                df,
                candidate,
                output_file,
                visible
            )

            print(
                f"Chart: {output_file}"
            )

            rendered += 1

        except Exception as exc:

            print(
                f"Skipping "
                f"{symbol or 'UNKNOWN'}: "
                f"{exc}"
            )

    print(
        f"Chart generation completed: "
        f"{rendered} chart(s)."
    )


if __name__ == "__main__":
    main()