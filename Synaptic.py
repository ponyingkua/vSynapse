#!/usr/bin/env python3

import argparse
import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("Synaptic")


# ============================================================
# TIMEFRAMES
# ============================================================

TFS = ["15m", "1h", "4h"]


# ============================================================
# SYMBOL EXCLUSIONS
# ============================================================

IGNORED_SYMBOLS = {
    "USDCUSDT",
    "BUSDUSDT",
    "DAIUSDT",
    "TUSDUSDT",
    "FDUSDUSDT",
    "USDPUSDT",
    "EURUSDT",
    "USTCUSDT",
    "PAXGUSDT",
}


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # UNIVERSE
    # --------------------------------------------------------
    "min_quote_volume_24h": 500_000,
    "universe_size": 0,                  # 0 = seluruh universe
    "momentum_pool": 60,

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------
    "klines": 240,

    # --------------------------------------------------------
    # THREADS
    # --------------------------------------------------------
    "workers_stage1": 8,
    "workers_stage2": 6,

    # --------------------------------------------------------
    # FINAL SELECTION
    # --------------------------------------------------------
    "min_score": 6.0,
    "min_candidates": 2,
    "max_results": 5,

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------
    "ema_period": 200,

    # EMA slope lookback.
    # Digunakan untuk membedakan EMA yang benar-benar naik/turun
    # dengan EMA yang sekadar dilewati harga.
    "ema_slope_bars": 8,

    # Minimum slope relatif (%) untuk dianggap meaningful.
    "ema_slope_min_pct": 0.03,

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------
    "volume_ma_period": 20,
    "volume_ratio_min": 1.30,
    "volume_ratio_strong": 2.00,

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    # --------------------------------------------------------
    # SUPERTREND
    # --------------------------------------------------------
    "supertrend_period": 10,
    "supertrend_multiplier": 2.50,

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------
    "atr_period": 14,

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------
    "breakout_window": 20,

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------
    "momentum_fast_bars": 4,
    "momentum_slow_bars": 16,

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------
    "swing_window": 5,
    "structure_lookback": 40,

    # --------------------------------------------------------
    # SETUP / RISK
    # --------------------------------------------------------
    "risk_reward": [1.5, 2.25, 3.0],

    "atr_stop_multiplier": 1.25,
    "max_risk_pct": 8.0,

    # --------------------------------------------------------
    # EXTENSION / CHASING CONTROL
    # --------------------------------------------------------

    # Harga yang terlalu jauh dari EMA200 bisa tetap bullish,
    # tetapi kualitas entry menjadi lebih rendah.
    "max_ema_extension_atr": 4.0,

    # Stage 1 movement minimum.
    "stage1_min_fast_move": 0.35,

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------
    "visible_candles": {
        "15m": 60,
        "1h": 48,
        "4h": 50,
    },
}


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://www.binance.com",
]


# requests.Session dibuat per-thread.
# Ini lebih aman dibanding satu Session global yang dipakai
# bersamaan oleh banyak worker.
_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session

    return _thread_local.session


# ============================================================
# BINANCE API
# ============================================================

def api(path, params=None, timeout=15):

    session = get_session()
    last_error = None

    for base_url in BASE_URLS:

        for attempt in range(3):

            try:
                response = session.get(
                    base_url + path,
                    params=params,
                    timeout=timeout,
                )

                status = response.status_code

                if status == 451:
                    last_error = "HTTP 451"
                    break

                if status in (418, 429):

                    last_error = f"HTTP {status}"

                    # exponential-ish backoff
                    time.sleep(
                        min(
                            3.0,
                            0.8 * (attempt + 1)
                        )
                    )

                    continue

                if status != 200:

                    last_error = f"HTTP {status}"

                    time.sleep(0.5)

                    continue

                data = response.json()

                if (
                    isinstance(data, dict)
                    and "code" in data
                    and "msg" in data
                ):
                    last_error = (
                        f"{data.get('code')}: "
                        f"{data.get('msg')}"
                    )

                    time.sleep(0.5)

                    continue

                return data

            except requests.RequestException as exc:

                last_error = str(exc)

                time.sleep(0.5)

    raise RuntimeError(
        f"All Binance endpoints failed: {last_error}"
    )


def exchange_info():
    return api(
        "/fapi/v1/exchangeInfo",
        timeout=20,
    )


def ticker_24h():
    return api(
        "/fapi/v1/ticker/24hr",
        timeout=20,
    )


# ============================================================
# GLOBAL UNIVERSE
# ============================================================

def universe():

    info = exchange_info()
    tickers = ticker_24h()

    ticker_map = {
        str(item.get("symbol", "")): item
        for item in tickers
        if isinstance(item, dict)
    }

    rows = []

    for item in info.get("symbols", []):

        if not isinstance(item, dict):
            continue

        symbol = str(item.get("symbol", ""))

        if not symbol:
            continue

        if item.get("contractType") != "PERPETUAL":
            continue

        if item.get("quoteAsset") != "USDT":
            continue

        if item.get("status") != "TRADING":
            continue

        if symbol in IGNORED_SYMBOLS:
            continue

        ticker = ticker_map.get(symbol)

        if not ticker:
            continue

        try:

            quote_volume = float(
                ticker.get("quoteVolume", 0)
            )

            change_24h = float(
                ticker.get("priceChangePercent", 0)
            )

            last_price = float(
                ticker.get("lastPrice", 0)
            )

        except (TypeError, ValueError):

            continue

        if quote_volume < CONFIG["min_quote_volume_24h"]:
            continue

        if last_price <= 0:
            continue

        # IMPORTANT:
        #
        # Tidak ada filter berdasarkan 24h change.
        #
        # Universe tetap global.
        #
        # Movement + technical selection dilakukan setelah
        # candle diambil.

        rows.append(
            (
                symbol,
                change_24h,
                quote_volume,
            )
        )

    if CONFIG["universe_size"] > 0:

        rows = rows[:CONFIG["universe_size"]]

    logger.info(
        f"Universe matched "
        f"{len(rows)} active USDT-M perpetual symbols globally."
    )

    return rows


# ============================================================
# KLINES
# ============================================================

def klines(symbol, interval):

    raw = api(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": CONFIG["klines"],
        },
        timeout=15,
    )

    columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(
        raw,
        columns=columns,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    return df.reset_index(drop=True)


# ============================================================
# SUPERTREND
# ============================================================

def calculate_supertrend(
    df,
    period=None,
    multiplier=None,
):

    if period is None:
        period = CONFIG["supertrend_period"]

    if multiplier is None:
        multiplier = CONFIG["supertrend_multiplier"]

    high = df["high"]
    low = df["low"]
    close = df["close"]

    hl2 = (high + low) / 2.0

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
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

    supertrend = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
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

            if close.iloc[i] < final_lower.iloc[i - 1]:
                direction.iloc[i] = -1

            else:
                direction.iloc[i] = 1

        else:

            if close.iloc[i] > final_upper.iloc[i - 1]:
                direction.iloc[i] = 1

            else:
                direction.iloc[i] = -1

        if direction.iloc[i] > 0:
            supertrend.iloc[i] = final_lower.iloc[i]

        else:
            supertrend.iloc[i] = final_upper.iloc[i]

    if len(df):

        supertrend.iloc[0] = final_lower.iloc[0]

    return supertrend, direction


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    x = df.copy()

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    x["ema200"] = x["close"].ewm(
        span=CONFIG["ema_period"],
        adjust=False,
    ).mean()

    slope_bars = CONFIG["ema_slope_bars"]

    x["ema200_slope_pct"] = (
        (
            x["ema200"]
            / x["ema200"].shift(slope_bars)
        ) - 1.0
    ) * 100.0

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    fast = x["close"].ewm(
        span=CONFIG["macd_fast"],
        adjust=False,
    ).mean()

    slow = x["close"].ewm(
        span=CONFIG["macd_slow"],
        adjust=False,
    ).mean()

    x["macd"] = fast - slow

    x["macd_signal"] = x["macd"].ewm(
        span=CONFIG["macd_signal"],
        adjust=False,
    ).mean()

    x["macd_hist"] = (
        x["macd"]
        - x["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - previous_close).abs(),
            (x["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr"] = true_range.ewm(
        alpha=1 / CONFIG["atr_period"],
        adjust=False,
    ).mean()

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    x["volume_ma"] = x["volume"].rolling(
        CONFIG["volume_ma_period"],
        min_periods=CONFIG["volume_ma_period"],
    ).mean()

    x["volume_ratio"] = (
        x["volume"]
        / x["volume_ma"]
    )

    # --------------------------------------------------------
    # SUPERTREND
    # --------------------------------------------------------

    (
        x["supertrend"],
        x["st_dir"],
    ) = calculate_supertrend(
        x,
        CONFIG["supertrend_period"],
        CONFIG["supertrend_multiplier"],
    )

    # --------------------------------------------------------
    # CANDLE BODY / RANGE
    # --------------------------------------------------------

    x["candle_range"] = (
        x["high"] - x["low"]
    )

    x["body"] = (
        x["close"] - x["open"]
    ).abs()

    x["body_ratio"] = np.where(
        x["candle_range"] > 0,
        x["body"] / x["candle_range"],
        0.0,
    )

    # --------------------------------------------------------
    # EMA DISTANCE
    # --------------------------------------------------------

    x["ema_distance_atr"] = np.where(
        x["atr"] > 0,
        (x["close"] - x["ema200"])
        / x["atr"],
        0.0,
    )

    return x


# ============================================================
# STRUCTURE
# ============================================================

def find_swing_points(
    df,
    window=None,
):

    if window is None:
        window = CONFIG["swing_window"]

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    n = len(df)

    swing_highs = []
    swing_lows = []

    for i in range(
        window,
        n - window,
    ):

        local_high = highs[
            i - window:i + window + 1
        ]

        local_low = lows[
            i - window:i + window + 1
        ]

        if (
            highs[i] == local_high.max()
            and highs[i] > highs[i - 1]
            and highs[i] > highs[i + 1]
        ):
            swing_highs.append(
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
            swing_lows.append(
                (
                    i,
                    float(lows[i]),
                )
            )

    return swing_highs, swing_lows


def structure_state(df):

    lookback = CONFIG["structure_lookback"]

    if len(df) > lookback:
        x = df.iloc[-lookback:].copy()
    else:
        x = df.copy()

    highs, lows = find_swing_points(x)

    result = {
        "high_state": "neutral",
        "low_state": "neutral",
        "bullish": False,
        "bearish": False,
        "last_swing_high": None,
        "last_swing_low": None,
    }

    if len(highs) >= 2:

        prev_high = highs[-2][1]
        last_high = highs[-1][1]

        result["last_swing_high"] = last_high

        if last_high > prev_high:
            result["high_state"] = "HH"

        elif last_high < prev_high:
            result["high_state"] = "LH"

    elif len(highs) == 1:

        result["last_swing_high"] = highs[-1][1]

    if len(lows) >= 2:

        prev_low = lows[-2][1]
        last_low = lows[-1][1]

        result["last_swing_low"] = last_low

        if last_low > prev_low:
            result["low_state"] = "HL"

        elif last_low < prev_low:
            result["low_state"] = "LL"

    elif len(lows) == 1:

        result["last_swing_low"] = lows[-1][1]

    result["bullish"] = (
        result["high_state"] == "HH"
        and result["low_state"] == "HL"
    )

    result["bearish"] = (
        result["high_state"] == "LH"
        and result["low_state"] == "LL"
    )

    return result


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):

        pass

    return default


def clamp(value, low, high):

    return max(
        low,
        min(high, value),
    )


# ============================================================
# STAGE 1 — MOVEMENT / MOMENTUM
# ============================================================

def movement_score(df):

    if len(df) < 60:
        return -1.0, None

    x = add_indicators(df)

    last = x.iloc[-1]

    close = safe_float(
        last["close"]
    )

    atr = safe_float(
        last["atr"]
    )

    if close <= 0 or atr <= 0:
        return -1.0, None

    fast_n = CONFIG["momentum_fast_bars"]
    slow_n = CONFIG["momentum_slow_bars"]

    if len(x) <= slow_n + 2:
        return -1.0, None

    fast_ref = safe_float(
        x["close"].iloc[
            -1 - fast_n
        ]
    )

    slow_ref = safe_float(
        x["close"].iloc[
            -1 - slow_n
        ]
    )

    if fast_ref <= 0 or slow_ref <= 0:
        return -1.0, None

    fast_return = (
        close / fast_ref - 1.0
    ) * 100.0

    slow_return = (
        close / slow_ref - 1.0
    ) * 100.0

    direction = (
        1
        if fast_return > 0
        else -1
        if fast_return < 0
        else 0
    )

    abs_fast = abs(fast_return)
    abs_slow = abs(slow_return)

    atr_move = (
        abs(close - fast_ref)
        / atr
    )

    volume_ratio = safe_float(
        last["volume_ratio"],
        1.0,
    )

    # --------------------------------------------------------
    # Movement base
    # --------------------------------------------------------

    score = 0.0

    score += min(
        abs_fast * 2.5,
        5.0,
    )

    score += min(
        abs_slow * 0.75,
        2.5,
    )

    score += min(
        atr_move * 0.8,
        3.0,
    )

    # --------------------------------------------------------
    # Volume participation
    # --------------------------------------------------------

    if volume_ratio >= 1.0:

        score += min(
            (volume_ratio - 1.0) * 1.2,
            2.5,
        )

    # --------------------------------------------------------
    # Directional technical confirmation
    # --------------------------------------------------------

    ema = safe_float(
        last["ema200"]
    )

    st_dir = int(
        safe_float(
            last["st_dir"],
            0,
        )
    )

    macd = safe_float(
        last["macd"]
    )

    macd_signal = safe_float(
        last["macd_signal"]
    )

    if direction > 0:

        if close > ema:
            score += 2.0

        if st_dir > 0:
            score += 2.0

        if macd > macd_signal:
            score += 1.0

    elif direction < 0:

        if close < ema:
            score += 2.0

        if st_dir < 0:
            score += 2.0

        if macd < macd_signal:
            score += 1.0

    # --------------------------------------------------------
    # Breakout impulse
    # --------------------------------------------------------

    window = CONFIG["breakout_window"]

    if len(x) > window + 2:

        previous_high = float(
            x["high"].iloc[
                -window - 1:-1
            ].max()
        )

        previous_low = float(
            x["low"].iloc[
                -window - 1:-1
            ].min()
        )

        if direction > 0 and close > previous_high:
            score += 2.0

        elif direction < 0 and close < previous_low:
            score += 2.0

    return float(score), {
        "df": x,
        "direction": direction,
        "fast_return": fast_return,
        "slow_return": slow_return,
        "volume_ratio": volume_ratio,
        "atr_move": atr_move,
    }


# ============================================================
# TIMEFRAME SCORING
# ============================================================

def score_tf(df):

    if len(df) < 210:
        return None

    x = add_indicators(df)

    last = x.iloc[-1]
    previous = x.iloc[-2]

    close = safe_float(
        last["close"]
    )

    ema = safe_float(
        last["ema200"]
    )

    atr = safe_float(
        last["atr"]
    )

    if close <= 0 or ema <= 0 or atr <= 0:
        return None

    volume_ratio = safe_float(
        last["volume_ratio"],
        1.0,
    )

    ema_slope = safe_float(
        last["ema200_slope_pct"]
    )

    st_dir = int(
        safe_float(
            last["st_dir"],
            0,
        )
    )

    macd = safe_float(
        last["macd"]
    )

    macd_signal = safe_float(
        last["macd_signal"]
    )

    hist_now = safe_float(
        last["macd_hist"]
    )

    hist_previous = safe_float(
        previous["macd_hist"]
    )

    structure = structure_state(x)

    # ========================================================
    # LONG SCORE
    # ========================================================

    long_score = 0.0
    long_reasons = []

    # --------------------------------------------------------
    # EMA200 position — 2 points
    # --------------------------------------------------------

    if close > ema:

        long_score += 2.0

        long_reasons.append(
            "above EMA200"
        )

    # --------------------------------------------------------
    # EMA200 slope — 1 point
    # --------------------------------------------------------

    if ema_slope >= CONFIG["ema_slope_min_pct"]:

        long_score += 1.0

        long_reasons.append(
            f"EMA200 rising {ema_slope:.2f}%"
        )

    # --------------------------------------------------------
    # Supertrend — 2 points
    # --------------------------------------------------------

    if st_dir > 0:

        long_score += 2.0

        long_reasons.append(
            "Supertrend bullish"
        )

    # --------------------------------------------------------
    # MACD position — 1 point
    # --------------------------------------------------------

    if macd > macd_signal:

        long_score += 1.0

        long_reasons.append(
            "MACD bullish"
        )

    # --------------------------------------------------------
    # MACD histogram momentum — 0.5
    # --------------------------------------------------------

    if (
        macd > macd_signal
        and hist_now > hist_previous
    ):

        long_score += 0.5

        long_reasons.append(
            "MACD histogram rising"
        )

    # --------------------------------------------------------
    # Volume — 1.5 points
    # --------------------------------------------------------

    if volume_ratio >= CONFIG["volume_ratio_min"]:

        if close > float(last["open"]):

            volume_points = (
                1.0
                if volume_ratio < CONFIG["volume_ratio_strong"]
                else 1.5
            )

            long_score += volume_points

            long_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

    # --------------------------------------------------------
    # Breakout — 1.0
    # --------------------------------------------------------

    window = CONFIG["breakout_window"]

    previous_high = float(
        x["high"].iloc[
            -window - 1:-1
        ].max()
    )

    if close > previous_high:

        long_score += 1.0

        long_reasons.append(
            "20-bar breakout"
        )

    # --------------------------------------------------------
    # Structure — 1.0
    # --------------------------------------------------------

    if structure["bullish"]:

        long_score += 1.0

        long_reasons.append(
            "HH-HL structure"
        )

    elif (
        structure["high_state"] == "HH"
        or structure["low_state"] == "HL"
    ):

        long_score += 0.5

        long_reasons.append(
            "bullish structure"
        )

    # ========================================================
    # SHORT SCORE
    # ========================================================

    short_score = 0.0
    short_reasons = []

    # --------------------------------------------------------
    # EMA200 position
    # --------------------------------------------------------

    if close < ema:

        short_score += 2.0

        short_reasons.append(
            "below EMA200"
        )

    # --------------------------------------------------------
    # EMA200 slope
    # --------------------------------------------------------

    if ema_slope <= -CONFIG["ema_slope_min_pct"]:

        short_score += 1.0

        short_reasons.append(
            f"EMA200 falling {abs(ema_slope):.2f}%"
        )

    # --------------------------------------------------------
    # Supertrend
    # --------------------------------------------------------

    if st_dir < 0:

        short_score += 2.0

        short_reasons.append(
            "Supertrend bearish"
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if macd < macd_signal:

        short_score += 1.0

        short_reasons.append(
            "MACD bearish"
        )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    if (
        macd < macd_signal
        and hist_now < hist_previous
    ):

        short_score += 0.5

        short_reasons.append(
            "MACD histogram falling"
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if volume_ratio >= CONFIG["volume_ratio_min"]:

        if close < float(last["open"]):

            volume_points = (
                1.0
                if volume_ratio < CONFIG["volume_ratio_strong"]
                else 1.5
            )

            short_score += volume_points

            short_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

    # --------------------------------------------------------
    # Breakdown
    # --------------------------------------------------------

    previous_low = float(
        x["low"].iloc[
            -window - 1:-1
        ].min()
    )

    if close < previous_low:

        short_score += 1.0

        short_reasons.append(
            "20-bar breakdown"
        )

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if structure["bearish"]:

        short_score += 1.0

        short_reasons.append(
            "LH-LL structure"
        )

    elif (
        structure["high_state"] == "LH"
        or structure["low_state"] == "LL"
    ):

        short_score += 0.5

        short_reasons.append(
            "bearish structure"
        )

    # ========================================================
    # EXTENSION
    # ========================================================

    ema_distance_atr = safe_float(
        last["ema_distance_atr"]
    )

    # Harga yang terlalu jauh dari EMA200
    # tidak langsung dibuang di sini.
    #
    # Informasi ini diteruskan supaya analyze_symbol()
    # bisa membedakan momentum sehat dan kondisi terlalu extended.

    return {

        "long": round(
            clamp(long_score, 0.0, 10.0),
            3,
        ),

        "short": round(
            clamp(short_score, 0.0, 10.0),
            3,
        ),

        "long_reasons": long_reasons,
        "short_reasons": short_reasons,

        "close": close,
        "ema200": ema,
        "ema_slope_pct": ema_slope,

        "atr": atr,
        "volume_ratio": volume_ratio,

        "st_dir": st_dir,

        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": hist_now,

        "ema_distance_atr": ema_distance_atr,

        "structure": structure,

        "structure_high": structure["high_state"],
        "structure_low": structure["low_state"],
    }


# ============================================================
# CHART SERIALIZATION
# ============================================================

def serialize_chart_data(df):

    required_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema200",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "volume_ma",
        "volume_ratio",
        "supertrend",
        "st_dir",
        "ema200_slope_pct",
        "ema_distance_atr",
    ]

    available = [
        col
        for col in required_columns
        if col in df.columns
    ]

    records = []

    for _, row in df.iterrows():

        item = {}

        for col in available:

            value = row[col]

            if col == "time":

                item[col] = (
                    None
                    if pd.isna(value)
                    else pd.Timestamp(value).isoformat()
                )

                continue

            if pd.isna(value):

                item[col] = None

                continue

            if col == "st_dir":

                item[col] = int(value)

            else:

                item[col] = float(value)

        records.append(item)

    return records


# ============================================================
# EXECUTION TIMEFRAME
# ============================================================

def choose_execution_tf(
    data,
    side,
):

    tf_rank = {
        "4h": 3,
        "1h": 2,
        "15m": 1,
    }

    candidates = []

    for tf in TFS:

        scored = data[tf]["score"]

        if side == "LONG":

            direction_score = float(
                scored["long"]
            )

        else:

            direction_score = float(
                scored["short"]
            )

        candidates.append(
            (
                direction_score,
                tf_rank[tf],
                tf,
            )
        )

    candidates.sort(
        reverse=True
    )

    return candidates[0][2]


# ============================================================
# SETUP QUALITY
# ============================================================

def calculate_setup_quality(
    execution_score,
    structure,
    ema_distance_atr,
    volume_ratio,
    momentum_15m,
):

    quality = 0.0

    # Strong MTF execution score
    quality += min(
        execution_score,
        10.0,
    ) * 0.25

    # Structure
    if structure["bullish"] or structure["bearish"]:
        quality += 1.5

    elif (
        structure["high_state"] != "neutral"
        or structure["low_state"] != "neutral"
    ):
        quality += 0.75

    # Volume
    if volume_ratio >= 2.0:
        quality += 1.5

    elif volume_ratio >= 1.3:
        quality += 0.75

    # Momentum
    quality += min(
        abs(momentum_15m),
        2.0,
    ) * 0.5

    # Extension penalty
    extension = abs(
        ema_distance_atr
    )

    if extension > 5.0:

        quality -= 2.0

    elif extension > CONFIG["max_ema_extension_atr"]:

        quality -= 1.0

    return round(
        max(0.0, quality),
        3,
    )


# ============================================================
# MTF ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    change_24h,
    quote_volume_24h,
    stage1_score,
    stage1_meta,
):

    data = {}

    # ========================================================
    # 15M
    # ========================================================

    try:

        scored_15m = score_tf(
            stage1_meta["df"]
        )

        if scored_15m:

            data["15m"] = {
                "score": scored_15m,
                "df": stage1_meta["df"],
            }

    except Exception as exc:

        logger.debug(
            f"MTF {symbol} 15m error: {exc}"
        )

    # ========================================================
    # 1H + 4H
    # ========================================================

    for tf in ["1h", "4h"]:

        try:

            candles = klines(
                symbol,
                tf,
            )

            scored = score_tf(
                candles
            )

            if scored:

                data[tf] = {
                    "score": scored,
                    "df": add_indicators(
                        candles
                    ),
                }

        except Exception as exc:

            logger.debug(
                f"MTF {symbol} {tf} error: {exc}"
            )

    # Need all three TF.
    if set(data.keys()) != set(TFS):

        return None

    # ========================================================
    # MTF WEIGHTS
    # ========================================================

    weights = {
        "15m": 0.25,
        "1h": 0.35,
        "4h": 0.40,
    }

    long_total = sum(
        weights[tf]
        * data[tf]["score"]["long"]
        for tf in TFS
    )

    short_total = sum(
        weights[tf]
        * data[tf]["score"]["short"]
        for tf in TFS
    )

    # ========================================================
    # SIDE
    # ========================================================

    if long_total > short_total:

        side = "LONG"
        raw_score = long_total
        wanted_direction = 1

    elif short_total > long_total:

        side = "SHORT"
        raw_score = short_total
        wanted_direction = -1

    else:

        return None

    # ========================================================
    # MTF AGREEMENT
    # ========================================================

    votes = []

    for tf in TFS:

        tf_score = data[tf]["score"]

        if (
            tf_score["long"]
            == tf_score["short"]
        ):

            votes.append(0)

        elif (
            tf_score["long"]
            > tf_score["short"]
        ):

            votes.append(1)

        else:

            votes.append(-1)

    agreement = sum(
        vote == wanted_direction
        for vote in votes
    )

    # Minimum 2/3.
    if agreement < 2:

        return None

    # ========================================================
    # 15M MOMENTUM DIRECTION
    # ========================================================

    df15 = data["15m"]["df"]

    current_close = safe_float(
        df15.iloc[-1]["close"]
    )

    reference_close = safe_float(
        df15.iloc[
            -1 - CONFIG["momentum_fast_bars"]
        ]["close"]
    )

    if reference_close <= 0:

        return None

    move_15 = (
        current_close
        / reference_close
        - 1.0
    ) * 100.0

    # Direction must agree with selected side.
    if side == "LONG" and move_15 <= 0:
        return None

    if side == "SHORT" and move_15 >= 0:
        return None

    # Avoid completely dead 15m momentum.
    if abs(move_15) < CONFIG["stage1_min_fast_move"]:
        return None

    # ========================================================
    # EXECUTION TIMEFRAME
    # ========================================================

    execution_tf = choose_execution_tf(
        data,
        side,
    )

    exec_df = data[
        execution_tf
    ]["df"]

    exec_score = data[
        execution_tf
    ]["score"]

    price = safe_float(
        exec_df.iloc[-1]["close"]
    )

    atr_value = safe_float(
        exec_df.iloc[-1]["atr"]
    )

    if price <= 0 or atr_value <= 0:

        return None

    # ========================================================
    # EXECUTION DIRECTION SCORE
    # ========================================================

    execution_direction_score = (
        exec_score["long"]
        if side == "LONG"
        else exec_score["short"]
    )

    # If strongest TF itself is very weak,
    # don't create a setup just because the weighted
    # MTF score happened to pass.
    if execution_direction_score < 5.0:

        return None

    # ========================================================
    # STRUCTURE
    # ========================================================

    structure = exec_score[
        "structure"
    ]

    # ========================================================
    # EXTENSION CONTROL
    # ========================================================

    ema_distance_atr = safe_float(
        exec_score["ema_distance_atr"]
    )

    # Extreme extension is rejected.
    #
    # We don't want the scanner selecting a coin simply because
    # it has already travelled very far from EMA200.

    if abs(ema_distance_atr) > 6.0:

        return None

    # ========================================================
    # SWING / STOP
    # ========================================================

    swing_n = CONFIG["swing_window"]

    recent_low = float(
        exec_df["low"].iloc[
            -swing_n:
        ].min()
    )

    recent_high = float(
        exec_df["high"].iloc[
            -swing_n:
        ].max()
    )

    entry = price

    if side == "LONG":

        sl = min(
            recent_low,
            entry
            - CONFIG["atr_stop_multiplier"]
            * atr_value,
        )

        risk = entry - sl

        invalidation = (
            f"Close below "
            f"{sl:.8g} / loss of recent "
            f"{execution_tf} swing low"
        )

    else:

        sl = max(
            recent_high,
            entry
            + CONFIG["atr_stop_multiplier"]
            * atr_value,
        )

        risk = sl - entry

        invalidation = (
            f"Close above "
            f"{sl:.8g} / reclaim of recent "
            f"{execution_tf} swing high"
        )

    if risk <= 0:

        return None

    risk_pct = (
        risk / entry
    ) * 100.0

    if risk_pct > CONFIG["max_risk_pct"]:

        return None

    # ========================================================
    # TP
    # ========================================================

    tp = []

    for rr in CONFIG["risk_reward"]:

        if side == "LONG":

            target = (
                entry
                + risk * rr
            )

        else:

            target = (
                entry
                - risk * rr
            )

        tp.append(target)

    # ========================================================
    # MOMENTUM BONUS
    # ========================================================

    momentum_bonus = min(
        max(
            abs(stage1_score)
            / 20.0,
            0.0,
        ),
        1.0,
    )

    # ========================================================
    # SETUP QUALITY
    # ========================================================

    quality_bonus = calculate_setup_quality(
        execution_direction_score,
        structure,
        ema_distance_atr,
        safe_float(
            exec_score["volume_ratio"],
            1.0,
        ),
        move_15,
    )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    #
    # raw_score sudah 0-10.
    #
    # Bonus dibatasi agar technical MTF tetap menjadi
    # faktor utama dan Stage 1 tidak bisa "membajak" ranking.
    #

    score = (
        raw_score
        + momentum_bonus
        + min(
            quality_bonus * 0.35,
            1.0,
        )
    )

    score = round(
        min(score, 10.0),
        2,
    )

    # ========================================================
    # REASONS
    # ========================================================

    reasons = (
        exec_score["long_reasons"]
        if side == "LONG"
        else exec_score["short_reasons"]
    )

    # Tambahkan informasi extension bila relevan.
    if abs(ema_distance_atr) <= 2.0:

        reasons = list(reasons)

        reasons.append(
            "price near EMA200 zone"
        )

    elif abs(ema_distance_atr) <= 4.0:

        reasons = list(reasons)

        reasons.append(
            f"EMA distance {abs(ema_distance_atr):.1f} ATR"
        )

    # ========================================================
    # CHART DATA
    # ========================================================

    chart_data = {
        tf: serialize_chart_data(
            data[tf]["df"]
        )
        for tf in TFS
    }

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "symbol": symbol,

        "side": side,

        "score": score,

        "change24h": round(
            change_24h,
            2,
        ),

        "quote_volume24h": round(
            quote_volume_24h,
            2,
        ),

        "execution_tf": execution_tf,

        "timeframes": {
            tf: data[tf]["score"]
            for tf in TFS
        },

        "momentum_15m": round(
            move_15,
            3,
        ),

        "stage1_momentum_score": round(
            stage1_score,
            3,
        ),

        "entry": entry,

        "tp": tp,

        "sl": sl,

        "risk_pct": round(
            risk_pct,
            3,
        ),

        "invalidation": invalidation,

        "key_points": reasons[:6],

        "tf_agreement": agreement,

        "structure": {
            "high": structure[
                "high_state"
            ],
            "low": structure[
                "low_state"
            ],
            "bullish": structure[
                "bullish"
            ],
            "bearish": structure[
                "bearish"
            ],
        },

        "quality": round(
            quality_bonus,
            3,
        ),

        "ema200_distance_atr": round(
            ema_distance_atr,
            3,
        ),

        "ema200_slope_pct": round(
            safe_float(
                exec_score["ema_slope_pct"]
            ),
            4,
        ),

        "volume_ratio": round(
            safe_float(
                exec_score["volume_ratio"],
                1.0,
            ),
            3,
        ),

        "chart": {
            "execution_tf": execution_tf,

            "available_timeframes": TFS,

            "analysis_candles": CONFIG[
                "klines"
            ],

            "visible_candles": CONFIG[
                "visible_candles"
            ],

            "show_ema200": True,

            "show_supertrend": True,

            "show_volume": True,

            "show_entry": True,

            "show_sl": True,

            "show_tp": True,
        },

        "chart_levels": {
            "entry": entry,
            "sl": sl,
            "tp": tp,
        },

        "chart_data": chart_data,
    }


# ============================================================
# FINAL RANKING
# ============================================================

def rank_candidates(results):

    if not results:
        return []

    # Ranking tidak hanya score mentah.
    #
    # Score tetap faktor utama.
    # Agreement, quality, dan momentum digunakan sebagai
    # tie-breaker.

    return sorted(
        results,
        key=lambda item: (
            float(item.get("score", 0)),
            int(item.get("tf_agreement", 0)),
            float(item.get("quality", 0)),
            abs(float(item.get("momentum_15m", 0))),
        ),
        reverse=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Synaptic - "
            "Global Binance Futures "
            "Multi-Timeframe Scanner"
        )
    )

    parser.add_argument(
        "--out",
        default="synaptic_candidates.json",
    )

    args = parser.parse_args()

    started = time.time()

    # ========================================================
    # UNIVERSE
    # ========================================================

    try:

        universe_rows = universe()

    except Exception as exc:

        logger.error(
            f"Cannot build universe: {exc}"
        )

        Path(args.out).write_text(
            json.dumps(
                {
                    "candidates": [],
                    "error": str(exc),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        raise

    # ========================================================
    # STAGE 1
    # ========================================================

    momentum = []

    logger.info(
        f"Scanning "
        f"{len(universe_rows)} symbols "
        f"on 15m (Stage 1)..."
    )

    with ThreadPoolExecutor(
        max_workers=CONFIG[
            "workers_stage1"
        ]
    ) as pool:

        jobs = {
            pool.submit(
                klines,
                symbol,
                "15m",
            ): (
                symbol,
                change_24h,
                quote_volume,
            )
            for (
                symbol,
                change_24h,
                quote_volume,
            ) in universe_rows
        }

        for future in as_completed(jobs):

            (
                symbol,
                change_24h,
                quote_volume,
            ) = jobs[future]

            try:

                candles = future.result()

                score, meta = movement_score(
                    candles
                )

                if (
                    score > 0
                    and meta is not None
                ):

                    # Dead / tiny movement tidak masuk
                    # momentum pool.

                    if (
                        abs(
                            meta["fast_return"]
                        )
                        >= CONFIG[
                            "stage1_min_fast_move"
                        ]
                    ):

                        momentum.append(
                            (
                                score,
                                symbol,
                                change_24h,
                                quote_volume,
                                meta,
                            )
                        )

            except Exception as exc:

                logger.debug(
                    f"15m scan error "
                    f"on {symbol}: {exc}"
                )

    # ========================================================
    # STAGE 1 RANKING
    # ========================================================

    momentum.sort(
        key=lambda row: row[0],
        reverse=True,
    )

    selected = momentum[
        :CONFIG["momentum_pool"]
    ]

    logger.info(
        f"Stage 1 movement candidates: "
        f"{len(momentum)}"
    )

    logger.info(
        f"Selected "
        f"{len(selected)} "
        f"symbols for Stage 2 "
        f"multi-timeframe validation."
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    results = []
    mtf_valid = []

    with ThreadPoolExecutor(
        max_workers=CONFIG[
            "workers_stage2"
        ]
    ) as pool:

        jobs = {
            pool.submit(
                analyze_symbol,
                symbol,
                change_24h,
                quote_volume,
                stage1_score,
                stage1_meta,
            ): symbol
            for (
                stage1_score,
                symbol,
                change_24h,
                quote_volume,
                stage1_meta,
            ) in selected
        }

        for future in as_completed(jobs):

            symbol = jobs[future]

            try:

                result = future.result()

                if result is None:
                    continue

                mtf_valid.append(
                    result
                )

                if (
                    result["score"]
                    >= CONFIG["min_score"]
                ):

                    results.append(
                        result
                    )

            except Exception as exc:

                logger.debug(
                    f"MTF scan error "
                    f"on {symbol}: {exc}"
                )

    # ========================================================
    # RANKING
    # ========================================================

    results = rank_candidates(
        results
    )

    mtf_valid = rank_candidates(
        mtf_valid
    )

    # ========================================================
    # FINAL SELECTION
    # ========================================================

    final_results = results[
        :CONFIG["max_results"]
    ]

    selection_mode = "min_score"

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    #
    # Jika kandidat >= 2 yang mencapai min_score tidak tersedia,
    # ambil MTF-valid terbaik.
    #
    # Tidak mengarang sinyal.
    #

    if (
        len(final_results)
        < CONFIG["min_candidates"]
    ):

        final_results = mtf_valid[
            :CONFIG["max_results"]
        ]

        selection_mode = "mtf_fallback"

    # ========================================================
    # FINAL SORT
    # ========================================================

    final_results = rank_candidates(
        final_results
    )[
        :CONFIG["max_results"]
    ]

    # ========================================================
    # STATS
    # ========================================================

    elapsed = round(
        time.time() - started,
        2,
    )

    logger.info(
        f"Stage 2 MTF-valid: "
        f"{len(mtf_valid)}"
    )

    logger.info(
        f"Min-score valid: "
        f"{len(results)}"
    )

    logger.info(
        f"Selection: "
        f"{selection_mode}"
    )

    logger.info(
        f"Scan completed. "
        f"Found "
        f"{len(final_results)} "
        f"valid candidates."
    )

    # ========================================================
    # PAYLOAD
    # ========================================================

    payload = {

        "generated_at":
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat(),

        "scanner":
            "Synaptic",

        "version":
            "2.0",

        "selection_mode":
            selection_mode,

        "configuration": {
            "timeframes": TFS,
            "ema200": CONFIG[
                "ema_period"
            ],
            "ema_slope_bars":
                CONFIG[
                    "ema_slope_bars"
                ],
            "macd": [
                CONFIG["macd_fast"],
                CONFIG["macd_slow"],
                CONFIG["macd_signal"],
            ],
            "supertrend": [
                CONFIG[
                    "supertrend_period"
                ],
                CONFIG[
                    "supertrend_multiplier"
                ],
            ],
            "volume_ma":
                CONFIG[
                    "volume_ma_period"
                ],
            "volume_ratio_min":
                CONFIG[
                    "volume_ratio_min"
                ],
            "breakout_window":
                CONFIG[
                    "breakout_window"
                ],
            "risk_reward":
                CONFIG[
                    "risk_reward"
                ],
        },

        "scan_stats": {

            "universe":
                len(universe_rows),

            "stage1_momentum":
                len(momentum),

            "stage1_selected":
                len(selected),

            "mtf_valid":
                len(mtf_valid),

            "min_score_valid":
                len(results),

            "final_candidates":
                len(final_results),

            "elapsed_seconds":
                elapsed,
        },

        "candidates":
            final_results,
    }

    # ========================================================
    # SAVE
    # ========================================================

    Path(args.out).write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("=" * 78)

    for rank, item in enumerate(
        final_results,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{item['symbol']} "
            f"{item['side']} | "
            f"Score {item['score']:.2f} | "
            f"TF {item['tf_agreement']}/3 | "
            f"Exec {item['execution_tf']} | "
            f"Momentum "
            f"{item['momentum_15m']:+.2f}% | "
            f"Entry {item['entry']:.8g} | "
            f"SL {item['sl']:.8g}"
        )

    print("=" * 78)

    logger.info(
        f"Output successfully saved to: "
        f"{args.out}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()