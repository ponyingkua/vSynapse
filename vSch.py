#!/usr/bin/env python3
"""
vSch.py - Optimized JSON-only visual renderer for Synaptic.py (v2.0)

- No Binance API calls.
- No hardcoded market data.
- Chart timeframe is candidate['execution_tf'].
- Candle/indicator data comes from candidate['chart_data'][execution_tf].
- Setup levels come from Synaptic JSON.
- RSI completely removed.
- Layout/size/visual style aligned with Synaptic v2.0.
- Output file naming uses setup_style + execution_tf.
- Optional CLI candle limit respects VISIBLE_DEFAULTS.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# STYLE — 100% aligned with Synaptic v2.0
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


VISIBLE_DEFAULTS = {
    "15m": 60,
    "1h": 48,
    "4h": 50,
}


# ============================================================
# HELPERS
# ============================================================

def format_price(value: float, decimals: int) -> str:
    return f"{float(value):.{int(decimals)}f}"


def decimals_from_price(price: float) -> int:
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


def _display_symbol(symbol: str) -> str:
    value = str(symbol).upper().strip()
    return value[:-4] if value.endswith("USDT") else value


def _normalize_tf(tf: str) -> str:
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


def _first_numeric(
    data: Optional[Dict],
    keys: List[str],
    default: Optional[Any] = None,
) -> Optional[float]:

    if not isinstance(data, dict):
        return default

    for key in keys:
        if key not in data or data.get(key) is None:
            continue

        try:
            return float(data[key])
        except (TypeError, ValueError):
            pass

    return default


def _get_mark_price(setup: Dict, fallback: float) -> float:
    return _first_numeric(
        setup,
        [
            "mark_price",
            "markPrice",
            "current_price",
            "currentPrice",
            "price",
        ],
        fallback,
    )


# ============================================================
# SUPERTREND — 10 / 2.5
# ============================================================

def calculate_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n = len(high)

    if n == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=int),
            np.array([], dtype=float),
        )

    if n < period + 1:
        return (
            np.full(n, np.nan),
            np.full(n, 0, dtype=int),
            np.full(n, np.nan),
        )

    # --------------------------------------------------------
    # FIX:
    # Previous close must have the same length as high/low.
    # The old version used close[:-1], producing N-1 values
    # against N values and causing a broadcasting error.
    # --------------------------------------------------------

    previous_close = np.empty(n, dtype=float)
    previous_close[0] = close[0]
    previous_close[1:] = close[:-1]

    tr0 = high - low
    tr1 = np.abs(high - previous_close)
    tr2 = np.abs(low - previous_close)

    tr = np.maximum.reduce([tr0, tr1, tr2])

    atr = (
        pd.Series(tr)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        .to_numpy()
    )

    # Basic bands
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)

    direction = np.full(n, 1, dtype=int)
    st_line = np.full(n, np.nan)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]

    st_line[0] = basic_lower[0]

    # --------------------------------------------------------
    # Final bands + direction
    # --------------------------------------------------------

    for i in range(1, n):

        if (
            basic_upper[i] < final_upper[i - 1]
            or close[i - 1] > final_upper[i - 1]
        ):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if (
            basic_lower[i] > final_lower[i - 1]
            or close[i - 1] < final_lower[i - 1]
        ):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if (
            direction[i - 1] == 1
            and close[i] < final_lower[i]
            and close[i] < final_lower[i - 1]
        ):
            direction[i] = -1

        elif (
            direction[i - 1] == -1
            and close[i] > final_upper[i]
            and close[i] > final_upper[i - 1]
        ):
            direction[i] = 1

        else:
            direction[i] = direction[i - 1]

        st_line[i] = (
            final_lower[i]
            if direction[i] > 0
            else final_upper[i]
        )

    return st_line, direction, atr


# ============================================================
# JSON / DATAFRAME
# ============================================================

def _load_candidates(path: Path) -> List[Dict[str, Any]]:

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")

    candidates = data.get("candidates", [])

    if not isinstance(candidates, list):
        raise ValueError(
            "JSON field 'candidates' must be a list"
        )

    return candidates


def _build_dataframe(
    candidate: Dict[str, Any],
    tf: str,
) -> Tuple[pd.DataFrame, str]:

    chart_data = candidate.get("chart_data")

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
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "missing candle fields: "
            + ", ".join(missing)
        )

    # Numeric conversion
    for c in required:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # Timestamp
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

    # Remove invalid candles
    df = (
        df
        .dropna(subset=required)
        .reset_index(drop=True)
    )

    if len(df) < 10:
        raise ValueError(
            "not enough candles in JSON"
        )

    # ========================================================
    # Indicators
    # ========================================================

    x = df.copy()

    # EMA 200
    x["ema200"] = (
        x["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    # MACD
    fast = (
        x["close"]
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    slow = (
        x["close"]
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    x["macd"] = fast - slow

    x["macd_signal"] = (
        x["macd"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    x["macd_hist"] = (
        x["macd"]
        - x["macd_signal"]
    )

    # ATR
    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                - previous_close
            ).abs(),
            (
                x["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr"] = (
        true_range
        .ewm(
            alpha=1 / 14,
            adjust=False,
        )
        .mean()
    )

    # Volume
    x["volume_ma"] = (
        x["volume"]
        .rolling(
            20,
            min_periods=1,
        )
        .mean()
    )

    x["volume_ratio"] = (
        x["volume"]
        / x["volume_ma"].replace(
            0,
            np.nan,
        )
    )

    # ========================================================
    # Supertrend 10 / 2.5
    # ========================================================

    high = x["high"].to_numpy()
    low = x["low"].to_numpy()
    close = x["close"].to_numpy()

    st_line, st_dir, _ = calculate_supertrend(
        high,
        low,
        close,
        10,
        2.5,
    )

    x["supertrend"] = st_line
    x["st_dir"] = st_dir

    return x, str(actual_tf)


def _resolve_visible_count(
    candidate: Dict[str, Any],
    tf: str,
    cli_value: Optional[int],
) -> int:

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

    visible = (
        chart.get(
            "visible_candles",
            {},
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
                    normalized_tf,
                    40,
                ),
            )
        ),
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def _find_swing_points(
    df: pd.DataFrame,
) -> Tuple[
    List[Tuple[int, float]],
    List[Tuple[int, float]],
]:

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
            and highs[i] > highs[i - 1]
            and highs[i] > highs[i + 1]
        ):
            raw_highs.append(
                (
                    i,
                    float(highs[i]),
                )
            )

        if (
            lows[i] == local_low.min()
            and lows[i] < lows[i - 1]
            and lows[i] < lows[i + 1]
        ):
            raw_lows.append(
                (
                    i,
                    float(lows[i]),
                )
            )

    return raw_highs, raw_lows


def _classify_structure(
    points: List[Tuple[int, float]],
    bullish: bool = True,
) -> List[Tuple[int, float, str]]:

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
    ax: Any,
    df: pd.DataFrame,
    y_span: float,
) -> None:

    raw_highs, raw_lows = (
        _find_swing_points(df)
    )

    highs = (
        _classify_structure(
            raw_highs,
            True,
        )[-2:]
    )

    lows = (
        _classify_structure(
            raw_lows,
            False,
        )[-2:]
    )

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
# LEVEL LABEL PLACEMENT
# ============================================================

def _place_level_labels(
    ax: Any,
    levels: List[Dict],
    label_x: float,
) -> None:

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
# MAIN CHART
# ============================================================

def draw_visual_chart(
    df: pd.DataFrame,
    setup: Dict[str, Any],
    output_path: Path,
    visible_count: int,
) -> None:

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

    normalized_tf = _normalize_tf(tf)

    change_24h = float(
        setup.get(
            "change24h",
            0.0,
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
            [],
        )[:3]
    ]

    dec = int(
        setup.get(
            "decimals",
            decimals_from_price(entry),
        )
    )

    visible_count = min(
        visible_count,
        len(df),
    )

    df = (
        df
        .iloc[-visible_count:]
        .copy()
        .reset_index(drop=True)
    )

    x = np.arange(len(df))

    last_x = int(x[-1])

    # ========================================================
    # FIGURE
    # ========================================================

    fig, (ax1, ax2) = plt.subplots(
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

    fig.patch.set_facecolor(BG)

    ax1.set_facecolor(PANEL)
    ax2.set_facecolor(PANEL)

    # ========================================================
    # PRICE RANGE
    # ========================================================

    level_values = [
        entry,
        sl,
        *tps,
    ]

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
        abs(y_low) * 0.01
        if y_low != 0
        else 0.01,
    )

    y_padding = y_span * 0.16

    ax1.set_ylim(
        y_low - y_padding,
        y_high + y_padding,
    )

    # ========================================================
    # CANDLES
    # ========================================================

    candle_width = 0.72

    for i in range(len(df)):

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

        body_bottom = min(
            open_p,
            close_p,
        )

        body_height = max(
            abs(close_p - open_p),
            (high_p - low_p) * 0.012,
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
    # EMA 200
    # ========================================================

    ax1.plot(
        x,
        df["ema200"],
        color=EMA,
        linewidth=1.4,
        label="EMA 200",
        zorder=3,
    )

    # ========================================================
    # SUPERTREND 10 / 2.5
    # ========================================================

    st_up = df["supertrend"].where(
        df["st_dir"] > 0
    )

    st_down = df["supertrend"].where(
        df["st_dir"] < 0
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
    # MARKET STRUCTURE
    # ========================================================

    _draw_structure(
        ax1,
        df,
        y_span,
    )

    # ========================================================
    # LABEL SPACING
    # ========================================================

    gap_from_candle = 4.0
    label_width_est = 9.5
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
    # TARGET ARROW
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
                + gap_from_candle * 0.70,
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
                alpha=0.80,
            ),
        )

    # ========================================================
    # VOLUME MA
    # ========================================================

    ax2.plot(
        x,
        df["volume_ma"],
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

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

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

        labels.append(
            ts.strftime(
                "%d %b  %H:%M"
            )
            if pd.notna(ts)
            else str(t)
        )

    ax2.set_xticklabels(
        labels,
        fontsize=7.5,
        color=AXIS,
    )

    # ========================================================
    # HEADER
    # ========================================================

    structure_label = str(
        setup.get(
            "structure",
            "neutral",
        )
    ).upper()

    style_label = str(
        setup.get(
            "setup_style",
            "continuation",
        )
    ).upper()

    confidence = setup.get(
        "confidence",
        None,
    )

    if confidence is not None:

        try:

            confidence_text = (
                f"{float(confidence):.0f}%"
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence_text = str(
                confidence
            )

        subheader = (
            f"{structure_label}"
            f"  ·  "
            f"{style_label}"
            f"  ·  "
            f"Conf {confidence_text}"
        )

    else:

        score_val = float(
            setup.get(
                "score",
                0,
            )
        )

        mtf_val = setup.get(
            "tf_agreement",
            "-",
        )

        subheader = (
            f"{structure_label}"
            f"  ·  "
            f"{style_label}"
            f"  ·  "
            f"Score {score_val:.2f}"
            f"  ·  "
            f"MTF {mtf_val}/3"
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
        fontsize=9,
        color=MUTED,
        ha="left",
        va="top",
        fontweight="medium",
    )

    # ========================================================
    # FOOTER
    # ========================================================

    fig.text(
        0.08,
        0.012,
        (
            f"BINANCE FUTURES"
            f"  ·  "
            f"${display_symbol}/USDT"
            f"  ·  "
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

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "vSch - Optimized JSON-only "
            "Synaptic chart renderer"
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
            and symbol
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
            # Required setup fields
            # ------------------------------------------------

            for field in (
                "entry",
                "sl",
                "tp",
            ):

                if field not in candidate:

                    raise ValueError(
                        f"missing top-level "
                        f"field '{field}'"
                    )

            # ------------------------------------------------
            # Build dataframe
            # ------------------------------------------------

            df, actual_tf = (
                _build_dataframe(
                    candidate,
                    tf,
                )
            )

            # ------------------------------------------------
            # Visible candles
            # ------------------------------------------------

            visible = (
                _resolve_visible_count(
                    candidate,
                    actual_tf,
                    args.chart_candles,
                )
            )

            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            style = candidate.get(
                "setup_style",
                "continuation",
            )

            output_file = (
                output_dir
                / (
                    f"{symbol}_"
                    f"{side}_"
                    f"{actual_tf}_"
                    f"{style}_"
                    f"chart.png"
                )
            )

            print(
                f"Rendering {symbol} | "
                f"setup={actual_tf} | "
                f"candles={visible} | "
                f"frame=12x6.8 | "
                f"indicators="
                f"EMA200+ST+MACD+Vol"
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


if __name__ == "__main__":
    main()