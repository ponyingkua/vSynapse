#!/usr/bin/env python3
"""
vSch.py - JSON-only visual renderer for Synaptic.py.

Rules:
- Never fetch Binance data.
- The chart timeframe is exactly candidate['execution_tf'].
- Candle/indicator data comes from candidate['chart_data'][execution_tf].
- Setup levels (Entry/SL/TP) come from Synaptic JSON.
- Output is optimized for 1:1 mobile viewing.
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
    """
    Normalize symbol for visual display.

    Example:
        1000000BOBUSDT -> 1000000BOB
        BOBUSDT        -> BOB
        BOB            -> BOB
    """

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
        "1h": "1h",
        "1hr": "1h",
        "1hour": "1h",

        "15m": "15m",
        "15min": "15m",
        "15minute": "15m",

        "4h": "4h",
        "4hr": "4h",
        "4hour": "4h",
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
        [
            "mark_price",
            "markPrice",
            "current_price",
            "currentPrice",
            "price",
        ],
        None,
    )

    if value is None:
        return float(fallback)

    return value


# ============================================================
# SUPERTREND FALLBACK
# ============================================================

def calculate_supertrend(
    df,
    period=10,
    multiplier=2.5
):

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

            final_upper.iloc[i] = (
                final_upper.iloc[i - 1]
            )

        if (
            lower.iloc[i] > final_lower.iloc[i - 1]
            or close.iloc[i - 1] < final_lower.iloc[i - 1]
        ):

            final_lower.iloc[i] = lower.iloc[i]

        else:

            final_lower.iloc[i] = (
                final_lower.iloc[i - 1]
            )

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
        path.read_text(
            encoding="utf-8"
        )
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

def _build_dataframe(
    candidate,
    tf
):

    chart_data = candidate.get(
        "chart_data"
    )

    if not isinstance(
        chart_data,
        dict
    ):

        raise ValueError(
            "candidate is missing chart_data"
        )

    candles = chart_data.get(tf)

    if candles is None:

        for key, value in chart_data.items():

            if (
                str(key).lower()
                == str(tf).lower()
            ):

                candles = value
                tf = key

                break

    if not candles:

        raise ValueError(
            f"no chart_data for timeframe {tf}"
        )

    df = pd.DataFrame(
        candles
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c
        for c in required
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
        .dropna(
            subset=required
        )
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

        df["ST"], df["ST_DIR"] = (
            calculate_supertrend(
                df,
                10,
                2.5
            )
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
# VISIBLE CANDLES
# ============================================================

def _resolve_visible_count(
    candidate,
    tf,
    cli_value
):

    normalized_tf = _normalize_tf(
        tf
    )

    if cli_value is not None:

        return max(
            20,
            int(cli_value)
        )

    if normalized_tf == "1h":

        return min(
            33,
            len(
                candidate
                .get(
                    "chart_data",
                    {}
                )
                .get(
                    tf,
                    []
                )
            )
            or 33
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
        if isinstance(
            chart,
            dict
        )
        else {}
    )

    return max(
        20,
        int(
            visible.get(
                tf,
                VISIBLE_DEFAULTS.get(
                    normalized_tf,
                    40
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
    Entry / TP / SL labels are intentionally placed
    inside the dedicated right-side level lane.

    Candle x positions are NOT modified.
    """

    span = max(
        y_max - y_min,
        1e-12
    )

    min_gap = span * 0.052
    edge = span * 0.020

    ordered = sorted(
        levels,
        key=lambda item: item["level"]
    )

    positions = [
        item["level"]
        for item in ordered
    ]

    # --------------------------------------------------------
    # Vertical spacing
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
    # Keep labels inside visible price range
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
    # Draw labels
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

            color=TEXT,

            fontsize=8.1,

            fontweight="bold",

            bbox=dict(
                boxstyle="round,pad=0.36",

                facecolor=item["color"],

                edgecolor=item["color"],

                linewidth=1.0,

                alpha=0.18,
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

    raw_highs = []
    raw_lows = []

    for i in range(
        swing,
        n - swing
    ):

        local_high = highs[
            i - swing:
            i + swing + 1
        ]

        local_low = lows[
            i - swing:
            i + swing + 1
        ]

        if highs[i] >= local_high.max():

            raw_highs.append(
                (
                    i,
                    highs[i]
                )
            )

        if lows[i] <= local_low.min():

            raw_lows.append(
                (
                    i,
                    lows[i]
                )
            )

    return raw_highs, raw_lows


def _select_important_structure(
    raw_highs,
    raw_lows,
    df
):

    max_points = 4

    selected_highs = (
        raw_highs[-max_points:]
    )

    selected_lows = (
        raw_lows[-max_points:]
    )

    return (
        selected_highs,
        selected_lows
    )


def _draw_structure(
    ax,
    df,
    span
):

    raw_highs, raw_lows = (
        _find_swing_points(df)
    )

    highs, lows = (
        _select_important_structure(
            raw_highs,
            raw_lows,
            df
        )
    )

    # --------------------------------------------------------
    # High structure
    # --------------------------------------------------------

    previous_high = None

    for i, price in highs:

        if previous_high is None:

            label = "HH"

        else:

            label = (
                "HH"
                if price > previous_high
                else "LH"
            )

        previous_high = price

        col = (
            ST_UP
            if label == "HH"
            else ST_DOWN
        )

        ax.text(
            i,
            price + span * 0.014,

            label,

            color=col,

            fontsize=7.8,

            fontweight="bold",

            ha="center",
            va="bottom",

            zorder=15,
        )

    # --------------------------------------------------------
    # Low structure
    # --------------------------------------------------------

    previous_low = None

    for i, price in lows:

        if previous_low is None:

            label = "HL"

        else:

            label = (
                "HL"
                if price > previous_low
                else "LL"
            )

        previous_low = price

        col = (
            ST_UP
            if label == "HL"
            else ST_DOWN
        )

        ax.text(
            i,
            price - span * 0.014,

            label,

            color=col,

            fontsize=7.8,

            fontweight="bold",

            ha="center",
            va="top",

            zorder=15,
        )


# ============================================================
# CURVED TARGET ARROW
# ============================================================

def _draw_target_arrow(
    ax,
    df,
    side,
    entry,
    tps,
    span
):

    if not tps:
        return

    if len(df) < 5:
        return

    last_x = len(df) - 1

    current_price = float(
        df["close"].iloc[-1]
    )

    target = (
        tps[1]
        if len(tps) >= 2
        else tps[0]
    )

    start_x = last_x - 2.5
    end_x = last_x + 8.0

    start_y = current_price

    target_y = (
        current_price
        + (target - current_price) * 0.72
    )

    rad = (
        0.22
        if side == "LONG"
        else -0.22
    )

    arrow = FancyArrowPatch(
        (
            start_x,
            start_y
        ),

        (
            end_x,
            target_y
        ),

        arrowstyle="-|>",

        mutation_scale=13,

        linewidth=1.7,

        color=(
            ST_UP
            if side == "LONG"
            else ST_DOWN
        ),

        alpha=0.75,

        connectionstyle=(
            f"arc3,rad={rad}"
        ),

        zorder=16,
    )

    ax.add_patch(
        arrow
    )


# ============================================================
# MAIN CHART
# ============================================================

def draw_visual_chart(
    df,
    setup,
    output_path,
    visible_count
):

    raw_symbol = str(
        setup.get(
            "symbol",
            "UNKNOWN"
        )
    ).upper()

    display_symbol = _display_symbol(
        raw_symbol
    )

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

    normalized_tf = _normalize_tf(
        tf
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

    # --------------------------------------------------------
    # Base volume
    # --------------------------------------------------------

    base_vol = _first_numeric(
        setup,
        [
            "volume24h",
            "base_volume24h",
            "baseVolume24h",
            "volume_24h",
        ],
        None,
    )

    if base_vol is None:

        base_vol = float(
            df["volume"].sum()
        )

    # --------------------------------------------------------
    # Setup levels
    # --------------------------------------------------------

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
        df
        .iloc[-visible_count:]
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

    # ========================================================
    # 1:1 MOBILE FRAME
    # ========================================================

    fig = plt.figure(
        figsize=(10.0, 10.0),
        facecolor=BG
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Price chart receives substantially more vertical space.
    # This makes candles visually larger without changing
    # their x positions.
    # --------------------------------------------------------

    gs = fig.add_gridspec(
        2,
        1,

        height_ratios=[
            5.65,
            0.82
        ],

        hspace=0.035,

        left=0.065,

        # More room on the right for ENTRY / TP / SL lane.
        right=0.875,

        top=0.790,

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

    y_min = y_low - pad
    y_max = y_high + pad

    # ========================================================
    # CANDLE WIDTH
    # ========================================================

    # Previous: 0.68
    # New: substantially wider.
    candle_width = 0.92

    # ========================================================
    # CANDLES
    # ========================================================

    for i, row in df.iterrows():

        o = float(
            row.open
        )

        h = float(
            row.high
        )

        l = float(
            row.low
        )

        c = float(
            row.close
        )

        up = c >= o

        body_color = (
            UP
            if up
            else DOWN
        )

        # ----------------------------------------------------
        # Larger wick
        # ----------------------------------------------------

        ax.plot(
            [i, i],
            [l, h],

            color=body_color,

            linewidth=1.45,

            alpha=0.95,

            zorder=5,
        )

        # ----------------------------------------------------
        # Larger body
        # ----------------------------------------------------

        body_low = min(
            o,
            c
        )

        body_height = max(
            abs(c - o),

            # Slightly larger minimum visual body.
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

                body_height,

                facecolor=body_color,

                edgecolor=body_color,

                linewidth=0.35,

                alpha=0.98,

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

            zorder=2,
        )

    # ========================================================
    # EMA 200
    # ========================================================

    ax.plot(
        x,
        df["EMA200"],

        color=EMA,

        linewidth=2.0,

        label="EMA 200",

        zorder=9,
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

        linewidth=1.9,

        label="Supertrend 10 / 2.5",

        zorder=8,
    )

    ax.plot(
        x,
        st_down,

        color=ST_DOWN,

        linewidth=1.9,

        zorder=8,
    )

    # ========================================================
    # SUPERTREND ZONE
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

    _draw_structure(
        ax,
        df,
        span
    )

    # ========================================================
    # ENTRY / TP / SL
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

    # ========================================================
    # LEVEL LINES
    # ========================================================

    for item in levels:

        ax.axhline(
            item["level"],

            color=item["color"],

            linestyle=(
                0,
                (5, 4)
            ),

            linewidth=1.0,

            alpha=0.55,

            zorder=3,
        )

    # ========================================================
    # RIGHT-SIDE LEVEL LANE
    # ========================================================
    #
    # Candle positions stay EXACTLY where they are.
    #
    # Only the labels are moved further right.
    #
    # This creates the visual area corresponding to the
    # black box marked in the reference image.
    # ========================================================

    level_label_x = (
        last_x + 3.65
    )

    # Wider dedicated lane.
    right_space = max(
        12.0,
        len(df) * 0.30
    )

    ax.set_xlim(
        -1.0,
        last_x + right_space
    )

    _place_level_labels(
        ax,
        levels,
        level_label_x,
        y_min,
        y_max
    )

    # ========================================================
    # CURVED DIRECTION / TARGET ARROW
    # ========================================================

    _draw_target_arrow(
        ax,
        df,
        side,
        entry,
        tps,
        span
    )

    ax.set_ylim(
        y_min,
        y_max
    )

    # ========================================================
    # VOLUME MA
    # ========================================================

    axv.plot(
        x,
        df["VOL_MA"],

        color="#b87333",

        linewidth=1.15,

        alpha=0.75,

        zorder=3,
    )

    # ========================================================
    # LIGHT BINANCE-LIKE GRID
    # ========================================================

    ax.grid(
        True,

        color=GRID,

        alpha=0.65,

        linewidth=0.65
    )

    axv.grid(
        True,

        axis="y",

        color=GRID,

        alpha=0.55,

        linewidth=0.65
    )

    ax.set_axisbelow(
        True
    )

    axv.set_axisbelow(
        True
    )

    # ========================================================
    # RIGHT PRICE AXIS
    # ========================================================

    ax.yaxis.tick_right()

    ax.yaxis.set_label_position(
        "right"
    )

    ax.tick_params(
        axis="y",

        colors=TEXT,

        labelsize=7.8,

        length=0,

        pad=5,
    )

    axv.tick_params(
        axis="x",

        colors=MUTED,

        labelsize=7.2,

        length=3,

        pad=4,
    )

    axv.tick_params(
        axis="y",

        colors=MUTED,

        labelsize=7.0,

        length=0,

        pad=4,
    )

    # ========================================================
    # SPINES
    # ========================================================

    for a in (
        ax,
        axv
    ):

        for spine in a.spines.values():

            spine.set_color(
                GRID
            )

            spine.set_linewidth(
                0.65
            )

    ax.spines["left"].set_visible(
        False
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["bottom"].set_color(
        GRID
    )

    axv.spines["left"].set_visible(
        False
    )

    axv.spines["top"].set_visible(
        False
    )

    # ========================================================
    # X AXIS
    # ========================================================

    ax.tick_params(
        labelbottom=False
    )

    tick_count = min(
        5,
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
                    "%d %b\n%H:%M"
                )
            )

        else:

            labels.append(
                str(i)
            )

    axv.set_xticklabels(
        labels,

        color=MUTED,

        fontsize=7.0
    )

    # ========================================================
    # LEGEND
    # ========================================================

    legend = ax.legend(
        loc="upper left",

        ncol=2,

        fontsize=7.2,

        frameon=True,

        facecolor="#ffffff",

        edgecolor="#d9dde3",

        framealpha=0.92,

        labelcolor=TEXT,

        borderpad=0.45,

        handlelength=2.0,

        columnspacing=0.9,
    )

    legend.get_frame().set_linewidth(
        0.65
    )

    # ========================================================
    # HEADER
    # ========================================================

    candle_close = float(
        df["close"].iloc[-1]
    )

    current_price = _get_mark_price(
        setup,
        candle_close
    )

    # --------------------------------------------------------
    # 24H values
    # --------------------------------------------------------

    high_24h = _first_numeric(
        setup,
        [
            "high24h",
            "high_24h",
            "24h_high",
            "priceChangeHigh",
        ],
        None,
    )

    low_24h = _first_numeric(
        setup,
        [
            "low24h",
            "low_24h",
            "24h_low",
            "priceChangeLow",
        ],
        None,
    )

    if high_24h is None:

        high_24h = float(
            df["high"].max()
        )

    if low_24h is None:

        low_24h = float(
            df["low"].min()
        )

    # ========================================================
    # HEADER GEOMETRY
    # ========================================================
    #
    # Left and right header now use fixed aligned columns.
    # ========================================================

    header_left = 0.065
    header_right = 0.875

    # ========================================================
    # LEFT HEADER
    # ========================================================

    fig.text(
        header_left,

        0.947,

        f"${display_symbol}/USDT",

        color=TEXT,

        fontsize=21.5,

        fontweight="bold",

        ha="left",

        va="center",
    )

    price_change_color = (
        ST_UP
        if change_24h >= 0
        else ST_DOWN
    )

    price_y = 0.897

    fig.text(
        header_left,

        price_y,

        f"${format_price(current_price, dec)}",

        color=TEXT,

        fontsize=11.2,

        fontweight="bold",

        ha="left",

        va="center",
    )

    # Percentage is anchored relative to the left price
    # but moved slightly closer for cleaner alignment.
    fig.text(
        header_left + 0.225,

        price_y,

        f"{change_24h:+.2f}%",

        color=price_change_color,

        fontsize=10.2,

        fontweight="bold",

        ha="left",

        va="center",
    )

    # --------------------------------------------------------
    # Setup line
    # --------------------------------------------------------

    fig.text(
        header_left,

        0.851,

        (
            f"SETUP "
            f"{normalized_tf.upper()}  —  "
            f"BIAS {side}"
        ),

        color="#f0b900",

        fontsize=9.9,

        fontweight="bold",

        ha="left",

        va="center",
    )

    # ========================================================
    # RIGHT HEADER
    # ========================================================
    #
    # Four clean rows.
    #
    # Label column and value column are deliberately separated
    # so long symbols / volumes cannot collide.
    # ========================================================

    right_x_label = 0.635
    right_x_value = 0.875

    right_rows = [
        (
            0.947,
            "24h High",
            format_price(
                high_24h,
                dec
            ),
        ),

        (
            0.908,
            "24h Low",
            format_price(
                low_24h,
                dec
            ),
        ),

        (
            0.869,
            f"24h Vol({display_symbol})",
            f"{base_vol:,.2f}",
        ),

        (
            0.830,
            "24h Vol(USDT)",
            f"{q_vol:,.2f}",
        ),
    ]

    for y, label, value in right_rows:

        fig.text(
            right_x_label,

            y,

            label,

            color=MUTED,

            fontsize=7.9,

            ha="left",

            va="center",
        )

        fig.text(
            right_x_value,

            y,

            value,

            color=TEXT,

            fontsize=8.5,

            fontweight="bold",

            ha="right",

            va="center",
        )

    # ========================================================
    # SCORE / MTF
    # ========================================================

    fig.text(
        header_right,

        0.792,

        (
            f"SCORE "
            f"{float(setup.get('score', 0)):.2f}"
            f"  •  MTF "
            f"{setup.get('tf_agreement', '-')}/3"
        ),

        color=TEXT,

        fontsize=7.6,

        fontweight="bold",

        ha="right",

        va="center",
    )

    # ========================================================
    # FOOTER
    # ========================================================

    footer = (
        "Setup levels from Synaptic JSON. "
        "Output optimized for 1:1 mobile viewing."
    )

    fig.text(
        0.065,

        0.040,

        footer,

        color=MUTED,

        fontsize=7.0,

        ha="left",

        va="center",
    )

    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        output_path,

        dpi=180,

        facecolor=BG,

        edgecolor=BG,

        bbox_inches="tight",

        pad_inches=0.08,
    )

    plt.close(
        fig
    )


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
            "otherwise use Synaptic chart settings. "
            "1H defaults to exactly 33 candles."
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
        # execution_tf determines both setup and visual TF.
        # ====================================================

        tf = str(
            candidate.get(
                "execution_tf",
                "15m"
            )
        )

        normalized_tf = _normalize_tf(
            tf
        )

        side = str(
            candidate.get(
                "side",
                "LONG"
            )
        ).upper()

        try:

            # ------------------------------------------------
            # Validate setup levels
            # ------------------------------------------------

            for field in (
                "entry",
                "sl",
                "tp"
            ):

                if field not in candidate:

                    raise ValueError(
                        f"missing top-level "
                        f"field '{field}'"
                    )

            # ------------------------------------------------
            # Exact execution timeframe
            # ------------------------------------------------

            df, actual_tf = (
                _build_dataframe(
                    candidate,
                    tf
                )
            )

            # ------------------------------------------------
            # Visible candle count
            # ------------------------------------------------

            visible = (
                _resolve_visible_count(
                    candidate,
                    actual_tf,
                    args.chart_candles
                )
            )

            # ------------------------------------------------
            # Force 1H to 33 candles unless CLI overrides it.
            # ------------------------------------------------

            if (
                args.chart_candles is None
                and _normalize_tf(actual_tf) == "1h"
            ):

                visible = min(
                    33,
                    len(df)
                )

            # ------------------------------------------------
            # Output
            # ------------------------------------------------

            output_file = (
                output_dir
                /
                (
                    f"{symbol}_"
                    f"{side}_"
                    f"{actual_tf}_"
                    f"chart.png"
                )
            )

            print(
                f"Rendering "
                f"{symbol} | "
                f"setup={actual_tf} | "
                f"visual={actual_tf} | "
                f"candles={visible} | "
                f"frame=1:1"
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()