#!/usr/bin/env python3
"""
vSch.py - JSON-only visual renderer for Synaptic-1.py

COMPATIBLE WITH:
    Synaptic-1.py

IMPORTANT:
- Tidak melakukan Binance API call.
- Tidak menghitung ulang setup/entry.
- setup_style berasal dari Synaptic.
- entry_state berasal dari Synaptic.
- reference_level berasal dari Synaptic.
- entry/sl/tp berasal dari Synaptic.
- vSch hanya merender data JSON menjadi chart.

Supported setup styles:
    BREAKOUT
    BREAKDOWN
    PULLBACK
    CONTINUATION
    EXTENDED
    NO_SETUP

Supported entry states:
    ENTRY_READY
    WAITING_RETEST
    WAITING_PULLBACK
    NO_SETUP
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


# ============================================================
# STYLE
# ============================================================

BG = "#ffffff"
PANEL = "#ffffff"
GRID = "#9e9e9e"
TEXT = "#212121"
MUTED = "#455a64"
AXIS = "#555555"
SPINE = "#9e9e9e"

UP = "#26a69a"
DOWN = "#ef5350"

EMA = "#1565c0"

ST_UP = "#2e7d32"
ST_DOWN = "#c62828"

ENTRY = "#1565c0"
TP1 = "#00897b"
TP2 = "#00695c"
SL = "#c62828"

VOLUME_MA = "#e65100"

# Reference level:
# dipakai terutama untuk breakout/breakdown.
REFERENCE = "#7b1fa2"

# State:
# hanya dipakai untuk header.
WAITING = "#ef6c00"
READY = "#1565c0"

# ------------------------------------------------------------
# Visual markers
# ------------------------------------------------------------

# Sangat tipis/transparan agar zona retest tidak mengganggu
# candle dan struktur chart.
RETEST_ZONE_ALPHA = 0.055

# Marker breakout "B"
BREAKOUT_MARKER_SIZE = 8

# Arrow ENTRY READY
ENTRY_ARROW_SIZE = 10

VISIBLE_DEFAULTS = {
    "15m": 60,
    "1h": 48,
    "4h": 50,
}


# ============================================================
# HELPERS
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


def _display_symbol(symbol):
    value = str(symbol).upper().strip()

    if value.endswith("USDT"):
        return value[:-4]

    return value


def _normalize_tf(tf):
    value = str(tf).strip().lower()

    return {
        "1h": "1h",
        "1hr": "1h",
        "1hour": "1h",

        "15m": "15m",
        "15min": "15m",
        "15minute": "15m",

        "4h": "4h",
        "4hr": "4h",
        "4hour": "4h",
    }.get(value, value)


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
            value = float(value)

            if np.isfinite(value):
                return value

        except (TypeError, ValueError):
            pass

    return default


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

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()

    direction = pd.Series(
        1,
        index=df.index,
        dtype=int,
    )

    st = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for i in range(1, len(df)):

        if (
            upper.iloc[i] < final_upper.iloc[i - 1]
            or
            close.iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upper.iloc[i]

        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lower.iloc[i] > final_lower.iloc[i - 1]
            or
            close.iloc[i - 1] < final_lower.iloc[i - 1]
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
# LOAD JSON
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
        [],
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

    chart_data = candidate.get(
        "chart_data"
    )

    if not isinstance(chart_data, dict):
        raise ValueError(
            "candidate is missing chart_data"
        )

    candles = chart_data.get(tf)

    actual_tf = tf

    if candles is None:

        for key, value in chart_data.items():

            if str(key).lower() == str(tf).lower():

                candles = value
                actual_tf = key
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
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "missing candle fields: "
            + ", ".join(missing)
        )

    for col in required:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
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
            errors="coerce",
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
    # EMA200
    # --------------------------------------------------------

    if "ema200" in df.columns:

        df["EMA200"] = pd.to_numeric(
            df["ema200"],
            errors="coerce",
        )

    else:

        df["EMA200"] = (
            df["close"]
            .ewm(
                span=200,
                adjust=False,
            )
            .mean()
        )

    # --------------------------------------------------------
    # Supertrend
    # --------------------------------------------------------

    if (
        "supertrend" in df.columns
        and
        "st_dir" in df.columns
    ):

        df["ST"] = pd.to_numeric(
            df["supertrend"],
            errors="coerce",
        )

        df["ST_DIR"] = pd.to_numeric(
            df["st_dir"],
            errors="coerce",
        )

    else:

        (
            df["ST"],
            df["ST_DIR"],
        ) = calculate_supertrend(
            df,
            period=10,
            multiplier=2.5,
        )

    # --------------------------------------------------------
    # Volume MA
    # --------------------------------------------------------

    if "volume_ma" in df.columns:

        df["VOL_MA"] = pd.to_numeric(
            df["volume_ma"],
            errors="coerce",
        )

    else:

        df["VOL_MA"] = (
            df["volume"]
            .rolling(
                20,
                min_periods=1,
            )
            .mean()
        )

    return df, str(actual_tf)


# ============================================================
# VISIBLE CANDLE COUNT
# ============================================================

def _resolve_visible_count(
    candidate,
    tf,
    cli_value,
):

    normalized_tf = _normalize_tf(tf)

    if cli_value is not None:

        return max(
            20,
            int(cli_value),
        )

    chart = candidate.get(
        "chart",
        {},
    )

    if not isinstance(chart, dict):
        chart = {}

    visible = chart.get(
        "visible_candles",
        {},
    )

    if not isinstance(visible, dict):
        visible = {}

    value = visible.get(
        tf,
        VISIBLE_DEFAULTS.get(
            normalized_tf,
            40,
        ),
    )

    return max(
        20,
        int(value),
    )


# ============================================================
# MARKET STRUCTURE
#
# Hanya visual.
# TIDAK dipakai menentukan setup/entry.
# ============================================================

def _find_swing_points(df):

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    n = len(df)

    swing_w = 3

    raw_highs = []
    raw_lows = []

    for i in range(
        swing_w,
        n - swing_w,
    ):

        local_high = highs[
            i - swing_w:
            i + swing_w + 1
        ]

        local_low = lows[
            i - swing_w:
            i + swing_w + 1
        ]

        if (
            highs[i] == local_high.max()
            and
            highs[i] > highs[i - 1]
            and
            highs[i] > highs[i + 1]
        ):

            raw_highs.append(
                (
                    i,
                    float(highs[i]),
                )
            )

        if (
            lows[i] == local_low.min()
            and
            lows[i] < lows[i - 1]
            and
            lows[i] < lows[i + 1]
        ):

            raw_lows.append(
                (
                    i,
                    float(lows[i]),
                )
            )

    return raw_highs, raw_lows


def _classify_structure(
    points,
    bullish=True,
):

    result = []

    previous = None

    for idx, price in points:

        if previous is None:

            label = (
                "HH"
                if bullish
                else "HL"
            )

        elif bullish:

            label = (
                "HH"
                if price > previous
                else "LH"
            )

        else:

            label = (
                "HL"
                if price > previous
                else "LL"
            )

        result.append(
            (
                idx,
                price,
                label,
            )
        )

        previous = price

    return result


def _draw_structure(
    ax,
    df,
    y_span,
):

    raw_highs, raw_lows = (
        _find_swing_points(df)
    )

    highs = _classify_structure(
        raw_highs,
        True,
    )[-2:]

    lows = _classify_structure(
        raw_lows,
        False,
    )[-2:]

    offset = y_span * 0.028

    for idx, price, label in highs:

        ax.text(
            idx,
            price + offset,
            label,
            color=(
                ST_UP
                if label == "HH"
                else ST_DOWN
            ),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=7,
        )

    for idx, price, label in lows:

        ax.text(
            idx,
            price - offset,
            label,
            color=(
                ST_UP
                if label == "HL"
                else ST_DOWN
            ),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=7,
        )


# ============================================================
# LEVEL LABELS
# ============================================================

def _place_level_labels(
    ax,
    levels,
    label_x,
):

    for item in levels:

        ax.text(
            label_x,
            item["level"],
            f" {item['text']} ",
            color="#ffffff",
            bbox=dict(
                facecolor=item["color"],
                edgecolor="none",
                boxstyle="round,pad=0.32",
                alpha=0.95,
            ),
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=7.5,
            zorder=8,
            clip_on=False,
        )


# ============================================================
# ENTRY STATE
# ============================================================

def _entry_state_text(state):

    state = str(
        state or "NO_SETUP"
    ).upper()

    mapping = {

        "ENTRY_READY":
            "ENTRY READY",

        "WAITING_RETEST":
            "WAITING RETEST",

        "WAITING_PULLBACK":
            "WAITING PULLBACK",

        "NO_SETUP":
            "NO SETUP",
    }

    return mapping.get(
        state,
        state.replace(
            "_",
            " ",
        ),
    )


def _entry_state_color(state):

    state = str(
        state or "NO_SETUP"
    ).upper()

    if state == "ENTRY_READY":
        return READY

    if state in (
        "WAITING_RETEST",
        "WAITING_PULLBACK",
    ):
        return WAITING

    return SL


# ============================================================
# RETEST ZONE
#
# Hanya visual.
# Tidak mengubah reference_level dari Synaptic.
#
# Zona dibuat sangat tipis/transparan.
# ============================================================

def _calculate_retest_zone(
    reference_level,
    setup,
    y_span,
):

    # Jika Synaptic menyediakan batas zona secara eksplisit,
    # renderer akan menggunakannya.
    zone_low = _first_numeric(
        setup,
        [
            "retest_zone_low",
            "retest_low",
        ],
        None,
    )

    zone_high = _first_numeric(
        setup,
        [
            "retest_zone_high",
            "retest_high",
        ],
        None,
    )

    if (
        zone_low is not None
        and
        zone_high is not None
        and
        zone_high > zone_low
    ):
        return (
            float(zone_low),
            float(zone_high),
        )

    # --------------------------------------------------------
    # Fallback visual.
    #
    # Sangat tipis: sekitar 0.15% dari reference level.
    # Dibatasi supaya tidak terlalu lebar atau terlalu kecil.
    # --------------------------------------------------------

    reference_abs = abs(
        float(reference_level)
    )

    percentage_width = (
        reference_abs * 0.0015
    )

    minimum_width = (
        y_span * 0.008
    )

    zone_half_width = max(
        percentage_width,
        minimum_width,
    )

    # Jangan biarkan zona menjadi terlalu tebal.
    maximum_half_width = (
        y_span * 0.025
    )

    zone_half_width = min(
        zone_half_width,
        maximum_half_width,
    )

    return (
        float(reference_level - zone_half_width),
        float(reference_level + zone_half_width),
    )


def _draw_retest_zone(
    ax,
    setup,
    reference_level,
    y_span,
    x_start,
    x_end,
):

    setup_style = str(
        setup.get(
            "setup_style",
            "",
        )
    ).upper()

    entry_state = str(
        setup.get(
            "entry_state",
            "",
        )
    ).upper()

    if setup_style not in (
        "BREAKOUT",
        "BREAKDOWN",
    ):
        return

    if entry_state != "WAITING_RETEST":
        return

    if reference_level is None:
        return

    zone_low, zone_high = (
        _calculate_retest_zone(
            reference_level,
            setup,
            y_span,
        )
    )

    height = (
        zone_high
        - zone_low
    )

    if height <= 0:
        return

    ax.add_patch(
        Rectangle(
            (
                x_start,
                zone_low,
            ),
            x_end - x_start,
            height,
            facecolor=REFERENCE,
            edgecolor="none",
            alpha=RETEST_ZONE_ALPHA,
            linewidth=0,
            zorder=1,
        )
    )


# ============================================================
# BREAKOUT MARKER
#
# BREAKOUT:
#     marker "B" pada candle pertama yang menembus
#     reference_level.
#
# Hanya visual.
# Tidak menentukan setup baru.
# ============================================================

def _find_breakout_candle(
    df,
    reference_level,
    setup_style,
):

    if (
        reference_level is None
        or len(df) < 2
    ):
        return None

    if setup_style == "BREAKOUT":

        for i in range(
            1,
            len(df),
        ):

            previous_close = float(
                df["close"].iloc[i - 1]
            )

            current_close = float(
                df["close"].iloc[i]
            )

            current_high = float(
                df["high"].iloc[i]
            )

            if (
                previous_close
                <= reference_level
                and
                (
                    current_close
                    > reference_level
                    or
                    current_high
                    > reference_level
                )
            ):
                return i

    elif setup_style == "BREAKDOWN":

        for i in range(
            1,
            len(df),
        ):

            previous_close = float(
                df["close"].iloc[i - 1]
            )

            current_close = float(
                df["close"].iloc[i]
            )

            current_low = float(
                df["low"].iloc[i]
            )

            if (
                previous_close
                >= reference_level
                and
                (
                    current_close
                    < reference_level
                    or
                    current_low
                    < reference_level
                )
            ):
                return i

    return None


def _draw_breakout_marker(
    ax,
    df,
    setup,
    y_span,
):

    setup_style = str(
        setup.get(
            "setup_style",
            "",
        )
    ).upper()

    if setup_style not in (
        "BREAKOUT",
        "BREAKDOWN",
    ):
        return

    reference_level = _first_numeric(
        setup,
        [
            "reference_level",
        ],
        None,
    )

    if reference_level is None:
        return

    idx = _find_breakout_candle(
        df,
        reference_level,
        setup_style,
    )

    if idx is None:
        return

    high_p = float(
        df["high"].iloc[idx]
    )

    low_p = float(
        df["low"].iloc[idx]
    )

    offset = (
        y_span * 0.018
    )

    if setup_style == "BREAKOUT":

        marker_y = (
            high_p
            + offset
        )

        va = "bottom"

    else:

        marker_y = (
            low_p
            - offset
        )

        va = "top"

    ax.text(
        idx,
        marker_y,
        "B",
        color=REFERENCE,
        fontsize=BREAKOUT_MARKER_SIZE,
        fontweight="bold",
        ha="center",
        va=va,
        zorder=10,
        clip_on=False,
    )


# ============================================================
# ENTRY READY MARKER
#
# ENTRY_READY:
#     panah kecil ↑ pada candle terakhir.
#
# Tidak lagi menggunakan label text di sisi kanan.
# ============================================================

def _draw_entry_ready_marker(
    ax,
    df,
    setup,
    y_span,
):

    entry_state = str(
        setup.get(
            "entry_state",
            "",
        )
    ).upper()

    if entry_state != "ENTRY_READY":
        return

    if len(df) == 0:
        return

    idx = len(df) - 1

    high_p = float(
        df["high"].iloc[idx]
    )

    low_p = float(
        df["low"].iloc[idx]
    )

    offset = (
        y_span * 0.022
    )

    side = str(
        setup.get(
            "side",
            "LONG",
        )
    ).upper()

    if side == "LONG":

        marker_y = (
            low_p
            - offset
        )

        ax.annotate(
            "",
            xy=(
                idx,
                marker_y + offset * 0.65,
            ),
            xytext=(
                idx,
                marker_y + offset * 0.05,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=READY,
                lw=1.25,
                mutation_scale=ENTRY_ARROW_SIZE,
            ),
            zorder=10,
            annotation_clip=False,
        )

    else:

        marker_y = (
            high_p
            + offset
        )

        ax.annotate(
            "",
            xy=(
                idx,
                marker_y - offset * 0.65,
            ),
            xytext=(
                idx,
                marker_y - offset * 0.05,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=READY,
                lw=1.25,
                mutation_scale=ENTRY_ARROW_SIZE,
            ),
            zorder=10,
            annotation_clip=False,
        )


# ============================================================
# DRAW REFERENCE LEVEL
# ============================================================

def _draw_reference_level(
    ax,
    setup,
    dec,
    label_x,
):

    setup_style = str(
        setup.get(
            "setup_style",
            "",
        )
    ).upper()

    reference_level = _first_numeric(
        setup,
        [
            "reference_level",
        ],
        None,
    )

    if (
        reference_level is None
        or not np.isfinite(
            reference_level
        )
    ):
        return

    # --------------------------------------------------------
    # BREAKOUT / BREAKDOWN
    #
    # reference_level = level yang ditembus.
    #
    # Level tetap ditampilkan karena penting sebagai
    # referensi retest.
    # --------------------------------------------------------

    if setup_style in (
        "BREAKOUT",
        "BREAKDOWN",
    ):

        label = "RETEST LEVEL"

        ax.axhline(
            y=reference_level,
            color=REFERENCE,
            linestyle=":",
            linewidth=1.1,
            alpha=0.80,
            zorder=2,
        )

        ax.text(
            label_x,
            reference_level,
            (
                f" {label}  "
                f"{format_price(reference_level, dec)} "
            ),
            color="#ffffff",
            bbox=dict(
                facecolor=REFERENCE,
                edgecolor="none",
                boxstyle="round,pad=0.28",
                alpha=0.90,
            ),
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=7,
            zorder=8,
            clip_on=False,
        )


# ============================================================
# MAIN CHART
# ============================================================

def draw_visual_chart(
    df,
    setup,
    output_path,
    visible_count,
):

    raw_symbol = str(
        setup.get(
            "symbol",
            "UNKNOWN",
        )
    ).upper()

    display_symbol = _display_symbol(
        raw_symbol
    )

    side = str(
        setup.get(
            "side",
            "LONG",
        )
    ).upper()

    tf = str(
        setup.get(
            "execution_tf",
            "15m",
        )
    )

    normalized_tf = _normalize_tf(
        tf
    )

    setup_style = str(
        setup.get(
            "setup_style",
            "NO_SETUP",
        )
    ).upper()

    entry_state = str(
        setup.get(
            "entry_state",
            "NO_SETUP",
        )
    ).upper()

    structure = str(
        setup.get(
            "structure",
            "neutral",
        )
    ).upper()

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
            [],
        )[:3]
    ]

    dec = int(
        setup.get(
            "decimals",
            decimals_from_price(
                entry
            ),
        )
    )

    score = float(
        setup.get(
            "score",
            0,
        )
    )

    tf_agreement = setup.get(
        "tf_agreement",
        "-",
    )

    visible_count = min(
        visible_count,
        len(df),
    )

    df = (
        df.iloc[-visible_count:]
        .copy()
        .reset_index(drop=True)
    )

    x = np.arange(
        len(df)
    )

    last_x = int(
        x[-1]
    )

    # ========================================================
    # FIGURE
    # ========================================================

    fig, (
        ax1,
        ax2,
    ) = plt.subplots(
        2,
        1,
        figsize=(12, 6.8),
        gridspec_kw={
            "height_ratios": [
                4.2,
                0.75,
            ],
            "hspace": 0.06,
        },
        sharex=True,
    )

    fig.patch.set_facecolor(
        BG
    )

    ax1.set_facecolor(
        PANEL
    )

    ax2.set_facecolor(
        PANEL
    )

    # ========================================================
    # PRICE RANGE
    # ========================================================

    level_values = [
        entry,
        sl,
        *tps,
    ]

    reference_level = _first_numeric(
        setup,
        [
            "reference_level",
        ],
        None,
    )

    if (
        reference_level is not None
        and np.isfinite(
            reference_level
        )
    ):
        level_values.append(
            reference_level
        )

    y_low = min(
        float(df["low"].min()),
        *level_values,
    )

    y_high = max(
        float(df["high"].max()),
        *level_values,
    )

    y_span = max(
        y_high - y_low,
        (
            abs(y_low) * 0.01
            if y_low != 0
            else 0.01
        ),
    )

    y_padding = (
        y_span * 0.16
    )

    ax1.set_ylim(
        y_low - y_padding,
        y_high + y_padding,
    )

    # ========================================================
    # CANDLES
    # ========================================================

    candle_width = 0.72

    for i in range(
        len(df)
    ):

        open_p = float(
            df["open"].iloc[i]
        )

        close_p = float(
            df["close"].iloc[i]
        )

        high_p = float(
            df["high"].iloc[i]
        )

        low_p = float(
            df["low"].iloc[i]
        )

        color = (
            UP
            if close_p >= open_p
            else DOWN
        )

        # Wick

        ax1.plot(
            [i, i],
            [low_p, high_p],
            color=color,
            linewidth=1.3,
            solid_capstyle="round",
            zorder=5,
        )

        # Body

        body_bottom = min(
            open_p,
            close_p,
        )

        body_height = max(
            abs(
                close_p - open_p
            ),
            (
                high_p - low_p
            ) * 0.012,
        )

        ax1.add_patch(
            Rectangle(
                (
                    i
                    - candle_width / 2,
                    body_bottom,
                ),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                alpha=0.92,
                linewidth=0,
                zorder=6,
            )
        )

        # Volume

        ax2.bar(
            i,
            float(
                df["volume"].iloc[i]
            ),
            color=color,
            alpha=0.30,
            width=candle_width,
            linewidth=0,
            zorder=2,
        )

    # ========================================================
    # EMA200
    # ========================================================

    ax1.plot(
        x,
        df["EMA200"],
        color=EMA,
        linewidth=1.4,
        label="EMA 200",
        zorder=3,
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

    ax1.plot(
        x,
        st_up,
        color=ST_UP,
        linewidth=1.4,
        label="Supertrend 10 / 2.5",
        zorder=3,
    )

    ax1.plot(
        x,
        st_down,
        color=ST_DOWN,
        linewidth=1.4,
        zorder=3,
    )

    # ========================================================
    # STRUCTURE
    #
    # Visual only.
    # Tidak menggantikan structure logic Synaptic.
    # ========================================================

    _draw_structure(
        ax1,
        df,
        y_span,
    )

    # ========================================================
    # LABEL AREA
    # ========================================================

    gap_from_candle = 4.0

    label_width_est = 13.0

    gap_from_edge = 1.8

    extra_margin = (
        gap_from_candle
        + label_width_est
        + gap_from_edge
    )

    label_x = (
        last_x
        + gap_from_candle
    )

    ax1.set_xlim(
        -0.6,
        last_x + extra_margin,
    )

    ax2.set_xlim(
        -0.6,
        last_x + extra_margin,
    )

    # ========================================================
    # RETEST ZONE
    #
    # WAITING RETEST:
    #     hanya tampilkan area sangat tipis/transparan.
    #
    # Tidak ada label teks WAITING RETEST di chart.
    # ========================================================

    _draw_retest_zone(
        ax1,
        setup,
        reference_level,
        y_span,
        -0.6,
        last_x + extra_margin,
    )

    # ========================================================
    # PRICE LEVELS
    # ========================================================

    levels = [
        {
            "level": entry,
            "color": ENTRY,
            "text": (
                f"ENTRY  "
                f"{format_price(entry, dec)}"
            ),
        }
    ]

    for i, value in enumerate(
        tps,
        1,
    ):

        levels.append(
            {
                "level": value,
                "color": (
                    TP1
                    if i == 1
                    else TP2
                ),
                "text": (
                    f"TP{i}  "
                    f"{format_price(value, dec)}"
                ),
            }
        )

    levels.append(
        {
            "level": sl,
            "color": SL,
            "text": (
                f"SL  "
                f"{format_price(sl, dec)}"
            ),
        }
    )

    for item in levels:

        ax1.axhline(
            y=item["level"],
            color=item["color"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.70,
            zorder=2,
        )

    _place_level_labels(
        ax1,
        levels,
        label_x,
    )

    # ========================================================
    # REFERENCE LEVEL
    # ========================================================

    _draw_reference_level(
        ax1,
        setup,
        dec,
        label_x,
    )

    # ========================================================
    # BREAKOUT MARKER
    #
    # BREAKOUT:
    #     B kecil pada candle yang menembus reference level.
    # ========================================================

    _draw_breakout_marker(
        ax1,
        df,
        setup,
        y_span,
    )

    # ========================================================
    # ENTRY READY MARKER
    #
    # ENTRY_READY:
    #     panah kecil pada candle terakhir.
    #
    # Tidak ada lagi box ENTRY READY di sisi kanan.
    # ========================================================

    _draw_entry_ready_marker(
        ax1,
        df,
        setup,
        y_span,
    )

    # ========================================================
    # TARGET ARROW
    #
    # ENTRY_READY:
    #     tetap menunjukkan jalur entry -> TP1.
    #
    # WAITING_*:
    #     jangan membuat chart terlihat seolah harga sekarang
    #     sudah masuk.
    #
    # Arrow tetap dimulai dari entry ideal, bukan candle terakhir.
    # ========================================================

    if tps:

        arrow_color = (
            TP1
            if side == "LONG"
            else SL
        )

        rad = (
            -0.28
            if side == "LONG"
            else 0.28
        )

        ax1.annotate(
            "",
            xy=(
                last_x
                + gap_from_candle
                * 0.70,
                tps[0],
            ),
            xytext=(
                last_x,
                entry,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=1.35,
                linestyle="--",
                connectionstyle=(
                    f"arc3,rad={rad}"
                ),
                mutation_scale=12,
                alpha=(
                    0.55
                    if entry_state
                    != "ENTRY_READY"
                    else 0.80
                ),
            ),
        )

    # ========================================================
    # VOLUME MA
    # ========================================================

    ax2.plot(
        x,
        df["VOL_MA"],
        color=VOLUME_MA,
        linewidth=1.2,
        alpha=0.70,
    )

    # ========================================================
    # GRID
    # ========================================================

    ax1.grid(
        True,
        linestyle="-",
        alpha=0.35,
        color=GRID,
    )

    ax2.grid(
        True,
        linestyle="-",
        alpha=0.35,
        color=GRID,
    )

    ax1.set_axisbelow(
        True
    )

    ax2.set_axisbelow(
        True
    )

    # ========================================================
    # LEGEND
    # ========================================================

    legend = ax1.legend(
        loc="upper left",
        fontsize=7.5,
        framealpha=0.95,
        facecolor=BG,
        edgecolor="#bdbdbd",
        labelcolor="#333333",
        borderpad=0.4,
    )

    legend.get_frame().set_linewidth(
        0.7
    )

    # ========================================================
    # AXES
    # ========================================================

    ax1.set_ylabel(
        "Price (USDT)",
        fontsize=8.5,
        color=AXIS,
        labelpad=5,
    )

    ax2.set_ylabel(
        "Vol",
        fontsize=8,
        color=AXIS,
        labelpad=5,
    )

    ax1.tick_params(
        colors=AXIS,
        labelcolor=AXIS,
        labelsize=7.5,
    )

    ax2.tick_params(
        colors=AXIS,
        labelcolor=AXIS,
        labelsize=7.5,
    )

    ax1.tick_params(
        labelbottom=False
    )

    for ax in (
        ax1,
        ax2,
    ):

        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

        ax.spines[
            "left"
        ].set_color(SPINE)

        ax.spines[
            "bottom"
        ].set_color(SPINE)

        ax.spines[
            "left"
        ].set_linewidth(0.8)

        ax.spines[
            "bottom"
        ].set_linewidth(0.8)

    # ========================================================
    # X TICKS
    # ========================================================

    tick_count = min(
        6,
        len(df),
    )

    ticks_idx = np.linspace(
        0,
        len(df) - 1,
        tick_count,
        dtype=int,
    )

    ax2.set_xticks(
        ticks_idx
    )

    labels = []

    for t in ticks_idx:

        ts = df[
            "timestamp"
        ].iloc[t]

        if pd.notna(ts):

            labels.append(
                ts.strftime(
                    "%d %b  %H:%M"
                )
            )

        else:

            labels.append(
                str(t)
            )

    ax2.set_xticklabels(
        labels,
        fontsize=7.5,
        color=AXIS,
    )

    # ========================================================
    # HEADER
    #
    # Status tetap ditampilkan di header agar informasi
    # ENTRY_READY / WAITING RETEST tidak hilang.
    #
    # Visual chart tidak lagi memakai box status di kanan.
    # ========================================================

    state_text = _entry_state_text(
        entry_state
    )

    subheader = (
        f"{structure}"
        f"  ·  "
        f"{setup_style}"
        f"  ·  "
        f"{state_text}"
        f"  ·  "
        f"Score {score:.2f}"
        f"  ·  "
        f"MTF {tf_agreement}/3"
    )

    fig.text(
        0.08,
        0.965,
        (
            f"${display_symbol}/USDT "
            f"{normalized_tf.upper()} "
            f"- {side} SETUP"
        ),
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )

    fig.text(
        0.08,
        0.932,
        subheader,
        fontsize=8.5,
        color=(
            WAITING
            if entry_state
            != "ENTRY_READY"
            else MUTED
        ),
        ha="left",
        va="top",
        fontweight="bold"
        if entry_state != "ENTRY_READY"
        else "medium",
    )

    # ========================================================
    # FOOTER
    # ========================================================

    fig.text(
        0.08,
        0.012,
        (
            f"BINANCE FUTURES  ·  "
            f"${display_symbol}/USDT  ·  "
            f"{normalized_tf.upper()}"
        ),
        fontsize=7,
        color="#555555",
        ha="left",
        va="bottom",
    )

    fig.text(
        0.92,
        0.012,
        "Not financial advice",
        fontsize=7,
        color="#555555",
        ha="right",
        va="bottom",
    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.subplots_adjust(
        left=0.08,
        right=0.96,
        top=0.88,
        bottom=0.07,
    )

    plt.savefig(
        output_path,
        dpi=140,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.18,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "vSch - JSON-only "
            "Synaptic-1 chart renderer"
        )
    )

    parser.add_argument(
        "--input",
        default="synaptic_candidates.json",
    )

    parser.add_argument(
        "--output-dir",
        default="charts",
    )

    parser.add_argument(
        "--symbol",
        default=None,
    )

    parser.add_argument(
        "--chart-candles",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            "Synaptic JSON not found: "
            f"{input_path}"
        )

    candidates = _load_candidates(
        input_path
    )

    if not candidates:

        print(
            "No candidates found "
            "in Synaptic JSON. "
            "Nothing to render."
        )

        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered = 0

    for candidate in candidates:

        symbol = str(
            candidate.get(
                "symbol",
                "",
            )
        ).upper()

        if (
            args.symbol
            and
            symbol
            != args.symbol.upper()
        ):
            continue

        tf = str(
            candidate.get(
                "execution_tf",
                "15m",
            )
        )

        side = str(
            candidate.get(
                "side",
                "LONG",
            )
        ).upper()

        try:

            # ------------------------------------------------
            # Required Synaptic fields
            # ------------------------------------------------

            for field in (
                "entry",
                "sl",
                "tp",
            ):

                if field not in candidate:

                    raise ValueError(
                        "missing top-level "
                        f"field '{field}'"
                    )

            # ------------------------------------------------
            # New Synaptic-1 fields
            # ------------------------------------------------

            setup_style = str(
                candidate.get(
                    "setup_style",
                    "NO_SETUP",
                )
            ).upper()

            entry_state = str(
                candidate.get(
                    "entry_state",
                    "NO_SETUP",
                )
            ).upper()

            reference_level = candidate.get(
                "reference_level"
            )

            # Jangan mengubah nilainya.
            # Hanya validasi bahwa jika ada,
            # nilainya numeric.

            if reference_level is not None:

                try:

                    reference_level = float(
                        reference_level
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    reference_level = None

            # ------------------------------------------------
            # Data
            # ------------------------------------------------

            df, actual_tf = (
                _build_dataframe(
                    candidate,
                    tf,
                )
            )

            visible = (
                _resolve_visible_count(
                    candidate,
                    actual_tf,
                    args.chart_candles,
                )
            )

            output_file = (
                output_dir
                /
                f"{symbol}_{side}_"
                f"{actual_tf}_chart.png"
            )

            print(
                f"Rendering {symbol} | "
                f"setup={actual_tf} | "
                f"style={setup_style} | "
                f"state={entry_state} | "
                f"candles={visible}"
            )

            draw_visual_chart(
                df,
                candidate,
                output_file,
                visible,
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