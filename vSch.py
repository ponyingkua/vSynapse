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
# WHITE UI PALETTE
# ============================================================

BG = "#ffffff"
PANEL = "#ffffff"

GRID = "#dfe3e8"
GRID_MINOR = "#edf0f3"

TEXT = "#111111"
MUTED = "#68717c"

UP = "#16b89a"
DOWN = "#ef5965"

EMA = "#f2c900"

ST_UP = "#21b866"
ST_DOWN = "#ef5965"

# Soft / bright colors for white UI
ENTRY = "#93c5fd"
TP = "#7dd3c7"
SL = "#fca5a5"

# Volume
VOL_MA = "#c47b32"


# ============================================================
# IDEAL VISIBLE CANDLE COUNTS
#
# Fewer candles = clearer price movement.
# Enough history remains because Synaptic already stores
# the indicator calculation history separately.
# ============================================================

VISIBLE_DEFAULTS = {
    "15m": 48,
    "1h": 48,
    "4h": 42,
}


# ============================================================
# PRICE FORMAT
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
    """
    Fallback only.
    Normally Synaptic's stored Supertrend is used.
    """

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
# LOAD CANDIDATES
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
# BUILD DATAFRAME
# ============================================================

def _build_dataframe(candidate, tf):

    chart_data = candidate.get(
        "chart_data"
    )

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

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError(
            "not enough candles in JSON"
        )

    # --------------------------------------------------------
    # EMA 200
    # Prefer exact Synaptic JSON value.
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
        "supertrend" in df
        and "st_dir" in df
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
# RESOLVE CANDLE COUNT
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
                    48
                )
            )
        )
    )


# ============================================================
# LEVEL HELPERS
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
    Place Entry / TP / SL labels in a clean right-side lane.
    """

    span = max(
        y_max - y_min,
        1e-12
    )

    # More compact than previous version,
    # but still prevents labels from touching.
    min_gap = span * 0.060
    edge = span * 0.025

    ordered = sorted(
        levels,
        key=lambda item: item["level"]
    )

    positions = [
        item["level"]
        for item in ordered
    ]

    # --------------------------------------------------------
    # Forward spacing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Keep labels inside visible price area
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

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
            color="#111111",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.34",
                facecolor=item["color"],
                edgecolor="#ffffff",
                linewidth=0.55,
                alpha=0.90,
            ),
            clip_on=False,
            zorder=30,
        )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def _find_significant_structure(df):
    """
    Select only important HH / LH / HL / LL points.

    The goal is to avoid filling the chart with labels.
    """

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    n = len(df)

    # Wider swing detection than before.
    swing = 3

    raw_highs = []
    raw_lows = []

    for i in range(
        swing,
        n - swing
    ):

        left_highs = highs[
            i - swing:i
        ]

        right_highs = highs[
            i + 1:i + swing + 1
        ]

        left_lows = lows[
            i - swing:i
        ]

        right_lows = lows[
            i + 1:i + swing + 1
        ]

        if (
            highs[i] >= max(
                highs[i - swing:i + swing + 1]
            )
        ):
            raw_highs.append(
                (i, highs[i])
            )

        if (
            lows[i] <= min(
                lows[i - swing:i + swing + 1]
            )
        ):
            raw_lows.append(
                (i, lows[i])
            )

    # --------------------------------------------------------
    # Remove points that are too close together.
    # --------------------------------------------------------

    def reduce_points(points, min_distance=5):

        if not points:
            return []

        selected = []

        for point in points:

            if not selected:

                selected.append(point)
                continue

            previous_i = selected[-1][0]

            if (
                point[0]
                - previous_i
                >= min_distance
            ):

                selected.append(point)

            else:

                # Keep the stronger extreme.
                if (
                    abs(point[1])
                    > abs(selected[-1][1])
                ):

                    selected[-1] = point

        return selected

    raw_highs = reduce_points(
        raw_highs,
        min_distance=5
    )

    raw_lows = reduce_points(
        raw_lows,
        min_distance=5
    )

    # --------------------------------------------------------
    # Keep only recent important structures.
    # Maximum two highs and two lows.
    # --------------------------------------------------------

    highs_selected = raw_highs[-2:]
    lows_selected = raw_lows[-2:]

    return (
        highs_selected,
        lows_selected
    )


# ============================================================
# CURVED TARGET ANNOTATION
# ============================================================

def _draw_target_curve(
    ax,
    last_x,
    current_price,
    target,
    side,
    x_end
):
    """
    Draw a clean curved directional arrow toward TP.
    No text is added.
    """

    if target is None:
        return

    target = float(target)

    # Keep arrow short enough to remain an annotation,
    # rather than becoming a giant diagonal line.
    distance = abs(
        target - current_price
    )

    if distance <= 0:
        return

    # Start slightly away from the final candle.
    start_x = last_x + 0.45

    # End before the level-label lane.
    end_x = min(
        x_end,
        start_x + 5.5
    )

    # Direction
    if side == "LONG":

        start_y = current_price
        end_y = (
            current_price
            + distance * 0.68
        )

        rad = 0.20

    else:

        start_y = current_price
        end_y = (
            current_price
            - distance * 0.68
        )

        rad = -0.20

    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.35,
        linestyle="-",
        color=TP,
        alpha=0.62,
        connectionstyle=(
            f"arc3,rad={rad}"
        ),
        zorder=12,
    )

    ax.add_patch(arrow)


# ============================================================
# MAIN DRAW FUNCTION
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
            decimals_from_price(
                entry
            )
        )
    )

    # ========================================================
    # VISIBLE DATA
    # ========================================================

    visible_count = min(
        visible_count,
        len(df)
    )

    df = (
        df.iloc[-visible_count:]
        .copy()
        .reset_index(drop=True)
    )

    n = len(df)

    # --------------------------------------------------------
    # Compress x positions slightly.
    #
    # This gives more visual breathing room on the right
    # without stretching the chart too much.
    # --------------------------------------------------------

    x = np.arange(
        n,
        dtype=float
    )

    last_x = float(
        x[-1]
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(15.0, 8.4),
        facecolor=BG
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            5.25,
            1.0
        ],
        hspace=0.035,
        left=0.055,
        right=0.87,
        top=0.865,
        bottom=0.105,
    )

    ax = fig.add_subplot(
        gs[0]
    )

    axv = fig.add_subplot(
        gs[1],
        sharex=ax
    )

    ax.set_facecolor(
        PANEL
    )

    axv.set_facecolor(
        PANEL
    )

    # ========================================================
    # PRICE RANGE
    # ========================================================

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

    pad = span * 0.085

    y_min = (
        y_low
        - pad
    )

    y_max = (
        y_high
        + pad
    )

    # ========================================================
    # CANDLE WIDTH
    # ========================================================

    candle_width = 0.68

    # ========================================================
    # CANDLES
    # ========================================================

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

        # ----------------------------------------------------
        # Wick
        # ----------------------------------------------------

        ax.plot(
            [i, i],
            [l, h],
            color=body_color,
            linewidth=1.20,
            alpha=0.92,
            zorder=5
        )

        # ----------------------------------------------------
        # Candle body
        # ----------------------------------------------------

        body_low = min(
            o,
            c
        )

        # Slightly stronger visual body.
        body_h = max(
            abs(c - o),
            (h - l) * 0.018,
            1e-12
        )

        ax.add_patch(
            Rectangle(
                (
                    i - candle_width / 2,
                    body_low
                ),
                candle_width,
                body_h,
                facecolor=body_color,
                edgecolor=body_color,
                linewidth=0.35,
                alpha=0.96,
                zorder=6,
            )
        )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        axv.bar(
            i,
            float(row.volume),
            width=candle_width,
            color=body_color,
            alpha=0.28,
            linewidth=0,
            zorder=2
        )

    # ========================================================
    # EMA 200
    # ========================================================

    ax.plot(
        x,
        df["EMA200"],
        color=EMA,
        linewidth=1.85,
        label="EMA 200",
        zorder=9
    )

    # ========================================================
    # SUPERTREND
    # ========================================================

    st_up = df["ST"].where(
        df["ST_DIR"] > 0
    )

    st_down = df["ST"].where(
        df["ST_DIR"] < 0
    )

    ax.plot(
        x,
        st_up,
        color=ST_UP,
        linewidth=1.85,
        label="Supertrend 10 / 2.5",
        zorder=8
    )

    ax.plot(
        x,
        st_down,
        color=ST_DOWN,
        linewidth=1.85,
        zorder=8
    )

    # ========================================================
    # SUPERTREND ZONE
    #
    # Stronger than previous alpha=0.075.
    #
    # Bull:
    # ST -> candle low
    #
    # Bear:
    # ST -> candle high
    #
    # This makes the trend region visibly extend across
    # the candle area.
    # ========================================================

    bull = (
        df["ST_DIR"] > 0
    )

    bear = (
        df["ST_DIR"] < 0
    )

    ax.fill_between(
        x,
        df["ST"].astype(float),
        df["low"].astype(float),
        where=bull,
        color=ST_UP,
        alpha=0.14,
        interpolate=True,
        zorder=1,
    )

    ax.fill_between(
        x,
        df["high"].astype(float),
        df["ST"].astype(float),
        where=bear,
        color=ST_DOWN,
        alpha=0.14,
        interpolate=True,
        zorder=1,
    )

    # ========================================================
    # MARKET STRUCTURE
    # ========================================================

    structure_highs, structure_lows = (
        _find_significant_structure(df)
    )

    # --------------------------------------------------------
    # HIGH STRUCTURE
    # --------------------------------------------------------

    last_high = None

    for i, price in structure_highs:

        if last_high is None:
            label = "HH"
        else:
            label = (
                "HH"
                if price > last_high
                else "LH"
            )

        last_high = price

        col = (
            ST_UP
            if label == "HH"
            else DOWN
        )

        ax.text(
            i,
            price + span * 0.018,
            label,
            color=col,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=15,
        )

    # --------------------------------------------------------
    # LOW STRUCTURE
    # --------------------------------------------------------

    last_low = None

    for i, price in structure_lows:

        if last_low is None:
            label = "HL"
        else:
            label = (
                "HL"
                if price > last_low
                else "LL"
            )

        last_low = price

        col = (
            UP
            if label == "HL"
            else DOWN
        )

        ax.text(
            i,
            price - span * 0.018,
            label,
            color=col,
            fontsize=8.0,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=15,
        )

    # ========================================================
    # ENTRY / TP / SL LEVELS
    # ========================================================

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

    # --------------------------------------------------------
    # Horizontal level lines
    # --------------------------------------------------------

    for item in levels:

        ax.axhline(
            item["level"],
            color=item["color"],
            linestyle=(0, (5, 4)),
            linewidth=1.05,
            alpha=0.58,
            zorder=3,
        )

    # ========================================================
    # RIGHT-SIDE GAP
    #
    # Larger blank area after the final candle.
    # Labels stay in the same right-side lane concept.
    # ========================================================

    label_x = (
        last_x
        + max(
            3.8,
            len(df) * 0.030
        )
    )

    right_limit = (
        label_x
        + max(
            5.5,
            len(df) * 0.065
        )
    )

    ax.set_xlim(
        -1.0,
        right_limit
    )

    # ========================================================
    # PRICE LABELS
    # ========================================================

    _place_level_labels(
        ax,
        levels,
        label_x,
        y_min,
        y_max
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    # ========================================================
    # CURVED TARGET DIRECTION
    #
    # No text.
    # Uses TP1 as primary destination.
    # ========================================================

    if tps:

        target = tps[0]

        _draw_target_curve(
            ax,
            last_x,
            float(df["close"].iloc[-1]),
            target,
            side,
            label_x - 0.8
        )

    # ========================================================
    # VOLUME MA
    # ========================================================

    axv.plot(
        x,
        df["VOL_MA"],
        color=VOL_MA,
        linewidth=1.15,
        alpha=0.80,
        zorder=4
    )

    # ========================================================
    # WHITE UI GRID
    # ========================================================

    ax.grid(
        True,
        color=GRID,
        alpha=0.72,
        linewidth=0.65
    )

    axv.grid(
        True,
        axis="y",
        color=GRID,
        alpha=0.65,
        linewidth=0.65
    )

    ax.set_axisbelow(
        True
    )

    axv.set_axisbelow(
        True
    )

    # ========================================================
    # AXIS STYLING
    # ========================================================

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
                0.65
            )

    ax.tick_params(
        labelbottom=False
    )

    # ========================================================
    # TIME AXIS
    # ========================================================

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

    # ========================================================
    # LEGEND
    # ========================================================

    legend = ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=8,
        frameon=True,
        facecolor="#ffffff",
        edgecolor=GRID,
        framealpha=0.94,
        labelcolor=TEXT,
        borderpad=0.50,
    )

    legend.get_frame().set_linewidth(
        0.65
    )

    # ========================================================
    # HEADER
    #
    # No border.
    # White background.
    # Black text.
    #
    # VISUAL timeframe removed.
    # ========================================================

    current_price = float(
        df["close"].iloc[-1]
    )

    header = (
        f"{symbol}"
        f"  •  SETUP {tf.upper()}"
        f"  •  BIAS {side}"
    )

    subheader = (
        f"PRICE {format_price(current_price, dec)}"
        f"    24H {change_24h:+.2f}%"
        f"    VOL ${q_vol:,.0f}"
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    fig.text(
        0.055,
        0.925,
        header,
        color=TEXT,
        fontsize=14.5,
        fontweight="bold",
        ha="left",
        va="center",
    )

    # --------------------------------------------------------
    # Header metadata
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Score / MTF
    # --------------------------------------------------------

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

    # ========================================================
    # FOOTER
    # ========================================================

    fig.text(
        0.055,
        0.045,
        (
            f"EMA 200  •  "
            f"Supertrend 10 / 2.5  •  "
            f"{visible_count} candles shown  •  "
            f"setup levels from Synaptic JSON"
        ),
        color=MUTED,
        fontsize=7.5,
        ha="left",
        va="center",
    )

    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        output_path,
        dpi=160,
        facecolor=BG,
        edgecolor=BG,
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

        # ====================================================
        # IMPORTANT:
        # execution_tf remains the single source of truth.
        # ====================================================

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

            # ------------------------------------------------
            # Build chart using EXACT execution timeframe.
            # ------------------------------------------------

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
                f"Rendering "
                f"{symbol} | "
                f"setup={actual_tf} | "
                f"visual={actual_tf} | "
                f"candles={visible}"
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