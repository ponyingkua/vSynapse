#!/usr/bin/env python3

import argparse
import json
import logging
import random
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
# SYMBOL FILTER
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
    # Universe
    # --------------------------------------------------------

    "min_quote_volume_24h": 500_000,

    # 0 = seluruh active USDT perpetual
    "universe_size": 0,

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    "momentum_pool": 60,
    "klines": 240,
    "workers_stage1": 16,

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    "workers_stage2": 12,

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    "min_score": 6.0,
    "min_candidates": 2,
    "max_results": 5,

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    "ema_period": 200,

    "volume_ma_period": 20,
    "volume_ratio_min": 1.30,

    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    "supertrend_period": 10,
    "supertrend_multiplier": 2.50,

    "atr_period": 14,

    "breakout_window": 20,

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    "momentum_fast_bars": 4,
    "momentum_slow_bars": 16,

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    "swing_window": 8,

    # --------------------------------------------------------
    # Risk / TP
    # --------------------------------------------------------

    "risk_reward": [
        1.5,
        2.25,
        3.0,
    ],

    "max_risk_pct": 8.0,

    # --------------------------------------------------------
    # Setup Engine
    # --------------------------------------------------------

    # EMA pullback area.
    #
    # LONG:
    #   EMA - 0.50 ATR  -> EMA + 1.00 ATR
    #
    # SHORT:
    #   EMA - 1.00 ATR  -> EMA + 0.50 ATR
    #
    "pullback_ema_atr_lower": 0.50,
    "pullback_ema_atr_upper": 1.00,

    # Breakout/retest tolerance.
    "retest_atr": 0.35,

    # Minimum impulse sebelum pullback dianggap valid.
    "minimum_impulse_atr": 1.50,

    # Minimum retracement.
    "minimum_retrace_atr": 0.40,

    # Maximum distance dari EMA200 sebelum dianggap extended.
    "extended_atr": 2.50,

    # Maximum extension dari breakout level untuk entry langsung.
    "breakout_entry_extension_atr": 0.80,

    # Buffer SL di luar structure.
    "sl_structure_buffer_atr": 0.20,

    # ATR fallback execution.
    "sl_execution_atr": 0.90,

    # --------------------------------------------------------
    # 15M execution
    # --------------------------------------------------------

    # Untuk confirmation biasa.
    "execution_volume_min": 1.00,

    # Untuk breakout/breakdown.
    "execution_breakout_volume_min": 1.30,

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    "api_timeout": 10,
    "api_retries": 2,
    "retry_base_delay": 0.35,

    # --------------------------------------------------------
    # Chart defaults
    # --------------------------------------------------------

    "visible_candles": {
        "15m": 60,
        "1h": 48,
        "4h": 50,
    },
}


# ============================================================
# BINANCE ENDPOINTS
# ============================================================

BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://www.binance.com",
]


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


# ============================================================
# THREAD LOCAL SESSION
# ============================================================

_thread_local = threading.local()


def get_session():
    session = getattr(
        _thread_local,
        "session",
        None,
    )

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session

    return session


# ============================================================
# API ENGINE
# ============================================================

def _retry_delay(attempt, response=None):

    if response is not None:

        retry_after = response.headers.get(
            "Retry-After"
        )

        if retry_after:

            try:

                value = float(
                    retry_after
                )

                return min(
                    max(value, 0.2),
                    5.0,
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

    base = CONFIG[
        "retry_base_delay"
    ]

    delay = (
        base
        * (2 ** attempt)
    )

    delay += random.uniform(
        0.05,
        0.20,
    )

    return min(
        delay,
        3.0,
    )


def _parse_response(response):

    try:
        return response.json()

    except ValueError:

        return None


def api(
    path,
    params=None,
    timeout=None,
):

    if timeout is None:
        timeout = CONFIG[
            "api_timeout"
        ]

    session = get_session()

    last_error = None

    for base_url in BASE_URLS:

        url = (
            base_url
            + path
        )

        for attempt in range(
            CONFIG["api_retries"] + 1
        ):

            try:

                response = session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )

                status = response.status_code

                # ------------------------------------------------
                # 200
                # ------------------------------------------------

                if status == 200:

                    data = _parse_response(
                        response
                    )

                    if data is None:

                        last_error = (
                            f"{base_url} "
                            "HTTP 200 but invalid JSON"
                        )

                        break

                    if (
                        isinstance(data, dict)
                        and "code" in data
                        and "msg" in data
                    ):

                        last_error = (
                            f"{data.get('code')}: "
                            f"{data.get('msg')}"
                        )

                        if (
                            attempt
                            < CONFIG["api_retries"]
                        ):

                            time.sleep(
                                _retry_delay(
                                    attempt,
                                    response,
                                )
                            )

                            continue

                        break

                    return data

                # ------------------------------------------------
                # 202
                # ------------------------------------------------

                if status == 202:

                    data = _parse_response(
                        response
                    )

                    if isinstance(
                        data,
                        (dict, list),
                    ):

                        if not (
                            isinstance(
                                data,
                                dict,
                            )
                            and "code" in data
                            and "msg" in data
                        ):

                            return data

                    last_error = (
                        f"{base_url} HTTP 202"
                    )

                    if (
                        attempt
                        < CONFIG["api_retries"]
                    ):

                        time.sleep(
                            _retry_delay(
                                attempt,
                                response,
                            )
                        )

                        continue

                    break

                # ------------------------------------------------
                # Rate limit
                # ------------------------------------------------

                if status in (
                    418,
                    429,
                ):

                    last_error = (
                        f"{base_url} HTTP {status}"
                    )

                    if (
                        attempt
                        < CONFIG["api_retries"]
                    ):

                        time.sleep(
                            _retry_delay(
                                attempt,
                                response,
                            )
                        )

                        continue

                    break

                # ------------------------------------------------
                # 451
                # ------------------------------------------------

                if status == 451:

                    last_error = (
                        f"{base_url} HTTP 451"
                    )

                    break

                # ------------------------------------------------
                # Other
                # ------------------------------------------------

                last_error = (
                    f"{base_url} HTTP {status}"
                )

                if (
                    attempt
                    < CONFIG["api_retries"]
                ):

                    time.sleep(
                        _retry_delay(
                            attempt,
                            response,
                        )
                    )

                    continue

                break

            except requests.Timeout:

                last_error = (
                    f"{base_url} timeout"
                )

                if (
                    attempt
                    < CONFIG["api_retries"]
                ):

                    time.sleep(
                        _retry_delay(attempt)
                    )

                    continue

                break

            except requests.RequestException as exc:

                last_error = (
                    f"{base_url}: {exc}"
                )

                if (
                    attempt
                    < CONFIG["api_retries"]
                ):

                    time.sleep(
                        _retry_delay(attempt)
                    )

                    continue

                break

    raise RuntimeError(
        "All Binance endpoints failed: "
        f"{last_error}"
    )


# ============================================================
# BASIC API
# ============================================================

def exchange_info():

    return api(
        "/fapi/v1/exchangeInfo",
        timeout=15,
    )


def ticker_24h():

    return api(
        "/fapi/v1/ticker/24hr",
        timeout=15,
    )


# ============================================================
# UNIVERSE
# ============================================================

def universe():

    started = time.time()

    info = exchange_info()
    tickers = ticker_24h()

    ticker_map = {
        str(item.get("symbol", "")): item
        for item in tickers
        if isinstance(item, dict)
    }

    rows = []

    for item in info.get(
        "symbols",
        [],
    ):

        if not isinstance(
            item,
            dict,
        ):
            continue

        symbol = str(
            item.get(
                "symbol",
                "",
            )
        )

        if not symbol:
            continue

        if item.get(
            "contractType"
        ) != "PERPETUAL":
            continue

        if item.get(
            "quoteAsset"
        ) != "USDT":
            continue

        if item.get(
            "status"
        ) != "TRADING":
            continue

        if symbol in IGNORED_SYMBOLS:
            continue

        ticker = ticker_map.get(
            symbol
        )

        if not ticker:
            continue

        try:

            quote_volume = float(
                ticker.get(
                    "quoteVolume",
                    0,
                )
            )

            change_24h = float(
                ticker.get(
                    "priceChangePercent",
                    0,
                )
            )

            last_price = float(
                ticker.get(
                    "lastPrice",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        if (
            quote_volume
            < CONFIG[
                "min_quote_volume_24h"
            ]
        ):
            continue

        if last_price <= 0:
            continue

        rows.append(
            (
                symbol,
                change_24h,
                quote_volume,
            )
        )

    # Universe tetap global.
    #
    # Jangan memotong berdasarkan 24h change.
    # Stage 1 yang menentukan momentum.
    if CONFIG["universe_size"] > 0:

        rows = rows[
            :CONFIG["universe_size"]
        ]

    elapsed = (
        time.time()
        - started
    )

    logger.info(
        "Universe matched %d active "
        "USDT-M perpetual symbols globally "
        "(%.2fs).",
        len(rows),
        elapsed,
    )

    return rows


# ============================================================
# KLINES
# ============================================================

def klines(
    symbol,
    interval,
):

    raw = api(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": CONFIG["klines"],
        },
        timeout=CONFIG["api_timeout"],
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
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    x = df.copy()

    # --------------------------------------------------------
    # EMA200
    # --------------------------------------------------------

    x["ema200"] = (
        x["close"]
        .ewm(
            span=CONFIG["ema_period"],
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    fast = (
        x["close"]
        .ewm(
            span=CONFIG["macd_fast"],
            adjust=False,
        )
        .mean()
    )

    slow = (
        x["close"]
        .ewm(
            span=CONFIG["macd_slow"],
            adjust=False,
        )
        .mean()
    )

    x["macd"] = (
        fast - slow
    )

    x["macd_signal"] = (
        x["macd"]
        .ewm(
            span=CONFIG["macd_signal"],
            adjust=False,
        )
        .mean()
    )

    x["macd_hist"] = (
        x["macd"]
        - x["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = (
        x["close"].shift(1)
    )

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
            alpha=1 / CONFIG["atr_period"],
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    x["volume_ma"] = (
        x["volume"]
        .rolling(
            CONFIG["volume_ma_period"],
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

    # --------------------------------------------------------
    # Supertrend 10 / 2.5
    # --------------------------------------------------------

    multiplier = CONFIG[
        "supertrend_multiplier"
    ]

    hl2 = (
        x["high"]
        + x["low"]
    ) / 2.0

    basic_upper = (
        hl2
        + multiplier * x["atr"]
    )

    basic_lower = (
        hl2
        - multiplier * x["atr"]
    )

    final_upper = (
        basic_upper.copy()
    )

    final_lower = (
        basic_lower.copy()
    )

    direction = pd.Series(
        1,
        index=x.index,
        dtype=int,
    )

    supertrend = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    for i in range(
        1,
        len(x),
    ):

        if (
            basic_upper.iloc[i]
            < final_upper.iloc[i - 1]
            or
            x["close"].iloc[i - 1]
            > final_upper.iloc[i - 1]
        ):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

        else:

            final_upper.iloc[i] = (
                final_upper.iloc[i - 1]
            )

        if (
            basic_lower.iloc[i]
            > final_lower.iloc[i - 1]
            or
            x["close"].iloc[i - 1]
            < final_lower.iloc[i - 1]
        ):

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

        else:

            final_lower.iloc[i] = (
                final_lower.iloc[i - 1]
            )

        if (
            direction.iloc[i - 1] == -1
            and
            x["close"].iloc[i]
            > final_upper.iloc[i - 1]
        ):

            direction.iloc[i] = 1

        elif (
            direction.iloc[i - 1] == 1
            and
            x["close"].iloc[i]
            < final_lower.iloc[i - 1]
        ):

            direction.iloc[i] = -1

        else:

            direction.iloc[i] = (
                direction.iloc[i - 1]
            )

        if direction.iloc[i] > 0:

            supertrend.iloc[i] = (
                final_lower.iloc[i]
            )

        else:

            supertrend.iloc[i] = (
                final_upper.iloc[i]
            )

    if len(x):

        supertrend.iloc[0] = (
            final_lower.iloc[0]
        )

    x["supertrend"] = supertrend
    x["st_dir"] = direction

    return x


# ============================================================
# CLOSED CANDLE
# ============================================================

def closed_candles(df):

    if len(df) <= 1:
        return df.copy()

    return (
        df.iloc[:-1]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# UTILITY
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):

        pass

    return None


def candle_time(df, index):

    if (
        df is None
        or len(df) == 0
        or index < 0
        or index >= len(df)
    ):

        return None

    value = df.iloc[index]["time"]

    if pd.isna(value):
        return None

    return pd.Timestamp(
        value
    ).isoformat()


def make_setup_result(
    setup,
    status,
    entry_zone,
    reason,
    signal_index=None,
    trigger_level=None,
    pullback_zone=None,
):

    return {

        "setup": setup,

        "status": status,

        "entry_zone": (
            [
                float(entry_zone[0]),
                float(entry_zone[1]),
            ]
            if entry_zone is not None
            else None
        ),

        "reason": reason,

        "signal_index": (
            int(signal_index)
            if signal_index is not None
            else None
        ),

        "trigger_level": (
            float(trigger_level)
            if trigger_level is not None
            else None
        ),

        "pullback_zone": (
            [
                float(pullback_zone[0]),
                float(pullback_zone[1]),
            ]
            if pullback_zone is not None
            else None
        ),
    }


# ============================================================
# PULLBACK ZONE
# ============================================================

def calculate_pullback_zone(
    df,
    side,
    reference_level=None,
):

    last = df.iloc[-1]

    ema = float(
        last["ema200"]
    )

    atr = float(
        last["atr"]
    )

    lower = CONFIG[
        "pullback_ema_atr_lower"
    ]

    upper = CONFIG[
        "pullback_ema_atr_upper"
    ]

    # --------------------------------------------------------
    # EMA zone
    # --------------------------------------------------------

    if side == "LONG":

        ema_low = (
            ema - lower * atr
        )

        ema_high = (
            ema + upper * atr
        )

    else:

        ema_low = (
            ema - upper * atr
        )

        ema_high = (
            ema + lower * atr
        )

    # --------------------------------------------------------
    # Breakout retest zone
    # --------------------------------------------------------

    if reference_level is None:

        return (
            float(ema_low),
            float(ema_high),
        )

    tolerance = (
        CONFIG["retest_atr"]
        * atr
    )

    retest_low = (
        reference_level
        - tolerance
    )

    retest_high = (
        reference_level
        + tolerance
    )

    # Jika breakout level masih dekat EMA,
    # gabungkan kedua area.
    #
    # Jika breakout sudah terlalu jauh dari EMA,
    # retest breakout menjadi area yang lebih relevan.
    overlap_low = max(
        ema_low,
        retest_low,
    )

    overlap_high = min(
        ema_high,
        retest_high,
    )

    if overlap_low <= overlap_high:

        return (
            float(overlap_low),
            float(overlap_high),
        )

    # Tidak ada overlap.
    # Gunakan retest level karena setup berasal
    # dari breakout/reclaim tersebut.
    return (
        float(retest_low),
        float(retest_high),
    )


# ============================================================
# RECENT STRUCTURE
# ============================================================

def recent_structure(
    df,
    window=None,
):

    if window is None:
        window = CONFIG[
            "swing_window"
        ]

    if len(df) <= window + 1:

        return None

    recent = df.iloc[
        -window - 1:-1
    ]

    high_idx_local = (
        recent["high"]
        .idxmax()
    )

    low_idx_local = (
        recent["low"]
        .idxmin()
    )

    return {

        "high": float(
            recent["high"].max()
        ),

        "low": float(
            recent["low"].min()
        ),

        "high_index": int(
            high_idx_local
        ),

        "low_index": int(
            low_idx_local
        ),
    }


# ============================================================
# RECENT BREAKOUT
# ============================================================

def find_recent_breakout(
    df,
    side,
    lookback_bars=4,
):

    window = CONFIG[
        "breakout_window"
    ]

    if len(df) < (
        window
        + lookback_bars
        + 2
    ):

        return None

    start = max(
        window,
        len(df)
        - lookback_bars,
    )

    for idx in range(
        len(df) - 1,
        start - 1,
        -1,
    ):

        if idx - window < 0:
            continue

        reference_high = float(
            df["high"]
            .iloc[
                idx - window:idx
            ]
            .max()
        )

        reference_low = float(
            df["low"]
            .iloc[
                idx - window:idx
            ]
            .min()
        )

        close = float(
            df["close"].iloc[idx]
        )

        previous_close = float(
            df["close"].iloc[idx - 1]
        )

        if side == "LONG":

            crossed = (
                close > reference_high
                and
                previous_close
                <= reference_high
            )

            if crossed:

                return {

                    "index": idx,

                    "level":
                        reference_high,

                    "type":
                        "BREAKOUT",
                }

        else:

            crossed = (
                close < reference_low
                and
                previous_close
                >= reference_low
            )

            if crossed:

                return {

                    "index": idx,

                    "level":
                        reference_low,

                    "type":
                        "BREAKDOWN",
                }

    return None


# ============================================================
# SETUP ENGINE
# ============================================================

def determine_setup(
    df,
    side,
):

    """
    1H Setup Engine.

    Prioritas:

        1. Validasi
        2. Trend alignment
        3. Recent breakout / breakdown
        4. Pullback / retest
        5. Continuation
        6. Extended
        7. NO_SETUP

    Prinsip penting:

        - 1H menentukan setup.
        - 15m tidak boleh mengubah WAITING menjadi READY.
        - READY membutuhkan trigger yang benar-benar terjadi.
        - WAITING PULLBACK berarti entry belum terjadi.
        - Entry zone berasal dari struktur, EMA200,
          atau breakout retest.
    """

    if (
        df is None
        or len(df) < 210
    ):

        return make_setup_result(
            "NO_SETUP",
            "NO_SETUP",
            None,
            "insufficient candles",
        )

    x = df.copy()

    last = x.iloc[-1]
    previous = x.iloc[-2]

    close = safe_float(
        last["close"]
    )

    open_price = safe_float(
        last["open"]
    )

    high = safe_float(
        last["high"]
    )

    low = safe_float(
        last["low"]
    )

    ema = safe_float(
        last["ema200"]
    )

    atr = safe_float(
        last["atr"]
    )

    st_dir = safe_float(
        last["st_dir"]
    )

    macd = safe_float(
        last["macd"]
    )

    macd_signal = safe_float(
        last["macd_signal"]
    )

    macd_hist = safe_float(
        last["macd_hist"]
    )

    previous_hist = safe_float(
        previous["macd_hist"]
    )

    volume_ratio = safe_float(
        last["volume_ratio"]
    )

    previous_close = safe_float(
        previous["close"]
    )

    previous_open = safe_float(
        previous["open"]
    )

    values = [
        close,
        open_price,
        high,
        low,
        ema,
        atr,
        st_dir,
        macd,
        macd_signal,
        macd_hist,
        previous_hist,
        volume_ratio,
        previous_close,
        previous_open,
    ]

    if not all(
        value is not None
        and np.isfinite(value)
        for value in values
    ):

        return make_setup_result(
            "NO_SETUP",
            "NO_SETUP",
            None,
            "invalid indicator data",
        )

    if (
        close <= 0
        or ema <= 0
        or atr <= 0
    ):

        return make_setup_result(
            "NO_SETUP",
            "NO_SETUP",
            None,
            "invalid price or ATR",
        )

    # ========================================================
    # TREND ALIGNMENT
    # ========================================================

    if side == "LONG":

        trend_aligned = (
            close > ema
            and st_dir > 0
            and macd > macd_signal
        )

    else:

        trend_aligned = (
            close < ema
            and st_dir < 0
            and macd < macd_signal
        )

    if not trend_aligned:

        return make_setup_result(
            "NO_SETUP",
            "NO_SETUP",
            None,
            "EMA200 / Supertrend / MACD alignment failed",
        )

    # ========================================================
    # DISTANCE FROM EMA
    # ========================================================

    distance_atr = (
        abs(close - ema)
        / atr
    )

    # ========================================================
    # RECENT STRUCTURE
    # ========================================================

    structure = recent_structure(
        x,
        CONFIG["swing_window"],
    )

    if structure is None:

        return make_setup_result(
            "NO_SETUP",
            "NO_SETUP",
            None,
            "insufficient structure history",
        )

    # ========================================================
    # RECENT BREAKOUT
    # ========================================================

    recent_breakout = (
        find_recent_breakout(
            x,
            side,
            lookback_bars=4,
        )
    )

    breakout_level = None

    if recent_breakout:

        breakout_level = float(
            recent_breakout["level"]
        )

    # ========================================================
    # EXTENSION
    # ========================================================

    if distance_atr > CONFIG[
        "extended_atr"
    ]:

        zone = calculate_pullback_zone(
            x,
            side,
            breakout_level,
        )

        return make_setup_result(
            "EXTENDED",
            "WAITING PULLBACK",
            zone,
            (
                f"price is {distance_atr:.2f} ATR "
                "from EMA200; entry is extended"
            ),
            signal_index=len(x) - 1,
            trigger_level=breakout_level,
            pullback_zone=zone,
        )

    # ========================================================
    # BREAKOUT / BREAKDOWN
    # ========================================================

    current_window = CONFIG[
        "breakout_window"
    ]

    if len(x) > (
        current_window + 1
    ):

        previous_high = float(
            x["high"]
            .iloc[
                -current_window - 1:-1
            ]
            .max()
        )

        previous_low = float(
            x["low"]
            .iloc[
                -current_window - 1:-1
            ]
            .min()
        )

    else:

        previous_high = structure[
            "high"
        ]

        previous_low = structure[
            "low"
        ]

    current_breakout = (
        close > previous_high
        if side == "LONG"
        else close < previous_low
    )

    candle_bullish = (
        close > open_price
    )

    candle_bearish = (
        close < open_price
    )

    if side == "LONG":

        breakout_momentum = (
            macd_hist >= 0
            and
            macd_hist >= previous_hist
        )

        breakout_volume = (
            volume_ratio
            >= CONFIG[
                "volume_ratio_min"
            ]
        )

        confirmed_breakout = (
            current_breakout
            and
            candle_bullish
            and
            breakout_momentum
            and
            breakout_volume
        )

    else:

        breakout_momentum = (
            macd_hist <= 0
            and
            macd_hist <= previous_hist
        )

        breakout_volume = (
            volume_ratio
            >= CONFIG[
                "volume_ratio_min"
            ]
        )

        confirmed_breakout = (
            current_breakout
            and
            candle_bearish
            and
            breakout_momentum
            and
            breakout_volume
        )

    # --------------------------------------------------------
    # Confirmed breakout
    # --------------------------------------------------------

    if confirmed_breakout:

        level = (
            previous_high
            if side == "LONG"
            else previous_low
        )

        extension_from_level = (
            (
                close - level
                if side == "LONG"
                else level - close
            )
            / atr
        )

        # Breakout masih dekat dengan trigger.
        # Boleh dieksekusi jika 15m mengkonfirmasi.
        if (
            extension_from_level
            <= CONFIG[
                "breakout_entry_extension_atr"
            ]
        ):

            return make_setup_result(
                (
                    "BREAKOUT"
                    if side == "LONG"
                    else "BREAKDOWN"
                ),
                "READY",
                [
                    float(level),
                    float(close),
                ],
                (
                    "confirmed 1H breakout with "
                    "volume, momentum and candle close"
                ),
                signal_index=len(x) - 1,
                trigger_level=level,
            )

        # Breakout sudah terlalu jauh.
        # Jangan chase.
        retest_zone = calculate_pullback_zone(
            x,
            side,
            level,
        )

        return make_setup_result(
            (
                "BREAKOUT"
                if side == "LONG"
                else "BREAKDOWN"
            ),
            "WAITING PULLBACK",
            retest_zone,
            (
                f"breakout confirmed but price is "
                f"{extension_from_level:.2f} ATR "
                "above the trigger; wait for retest"
            ),
            signal_index=len(x) - 1,
            trigger_level=level,
            pullback_zone=retest_zone,
        )

    # --------------------------------------------------------
    # Breakout detected but confirmation incomplete.
    # --------------------------------------------------------

    if current_breakout:

        level = (
            previous_high
            if side == "LONG"
            else previous_low
        )

        retest_zone = calculate_pullback_zone(
            x,
            side,
            level,
        )

        return make_setup_result(
            (
                "BREAKOUT"
                if side == "LONG"
                else "BREAKDOWN"
            ),
            "WAITING PULLBACK",
            retest_zone,
            (
                "1H breakout level detected but "
                "volume/momentum confirmation is incomplete"
            ),
            signal_index=len(x) - 1,
            trigger_level=level,
            pullback_zone=retest_zone,
        )

    # ========================================================
    # PULLBACK ENGINE
    # ========================================================

    lookback = max(
        CONFIG["swing_window"],
        CONFIG["momentum_fast_bars"],
        10,
    )

    recent = x.iloc[
        -lookback - 1:-1
    ]

    recent_high = float(
        recent["high"].max()
    )

    recent_low = float(
        recent["low"].min()
    )

    impulse_size = (
        recent_high
        - recent_low
    )

    impulse_atr = (
        impulse_size / atr
    )

    minimum_impulse = CONFIG[
        "minimum_impulse_atr"
    ]

    minimum_retrace = CONFIG[
        "minimum_retrace_atr"
    ]

    ema_zone = calculate_pullback_zone(
        x,
        side,
        None,
    )

    zone_low = ema_zone[0]
    zone_high = ema_zone[1]

    # --------------------------------------------------------
    # LONG PULLBACK
    # --------------------------------------------------------

    if side == "LONG":

        retracement = (
            recent_high - close
        )

        retracement_atr = (
            retracement / atr
        )

        touched_zone = (
            low <= zone_high
            and close >= zone_low
        )

        reached_zone = (
            low <= zone_high
        )

        rejection_strength = (
            close - low
        ) / max(
            high - low,
            atr * 0.10,
        )

        bullish_rejection = (
            close > open_price
            and
            rejection_strength >= 0.45
        ) or (
            close > previous_close
            and
            close > open_price
        )

        momentum_recovery = (
            macd_hist >= previous_hist
            and
            macd_hist > -abs(macd) * 0.50
        )

        valid_pullback = (
            touched_zone
            and
            impulse_atr
            >= minimum_impulse
            and
            retracement_atr
            >= minimum_retrace
            and
            bullish_rejection
            and
            momentum_recovery
            and
            close > ema
        )

        if valid_pullback:

            return make_setup_result(
                "PULLBACK",
                "READY",
                ema_zone,
                (
                    "1H pullback reached the EMA200 "
                    "value area and printed bullish rejection"
                ),
                signal_index=len(x) - 1,
                pullback_zone=ema_zone,
            )

        if (
            reached_zone
            and
            impulse_atr >= minimum_impulse
            and
            retracement_atr >= minimum_retrace
        ):

            return make_setup_result(
                "PULLBACK",
                "WAITING PULLBACK",
                ema_zone,
                (
                    "1H retracement reached the pullback "
                    "area but bullish confirmation is incomplete"
                ),
                signal_index=len(x) - 1,
                pullback_zone=ema_zone,
            )

    # --------------------------------------------------------
    # SHORT PULLBACK
    # --------------------------------------------------------

    else:

        retracement = (
            close - recent_low
        )

        retracement_atr = (
            retracement / atr
        )

        touched_zone = (
            high >= zone_low
            and close <= zone_high
        )

        reached_zone = (
            high >= zone_low
        )

        rejection_strength = (
            high - close
        ) / max(
            high - low,
            atr * 0.10,
        )

        bearish_rejection = (
            close < open_price
            and
            rejection_strength >= 0.45
        ) or (
            close < previous_close
            and
            close < open_price
        )

        momentum_recovery = (
            macd_hist <= previous_hist
            and
            macd_hist < abs(macd) * 0.50
        )

        valid_pullback = (
            touched_zone
            and
            impulse_atr
            >= minimum_impulse
            and
            retracement_atr
            >= minimum_retrace
            and
            bearish_rejection
            and
            momentum_recovery
            and
            close < ema
        )

        if valid_pullback:

            return make_setup_result(
                "PULLBACK",
                "READY",
                ema_zone,
                (
                    "1H pullback reached the EMA200 "
                    "value area and printed bearish rejection"
                ),
                signal_index=len(x) - 1,
                pullback_zone=ema_zone,
            )

        if (
            reached_zone
            and
            impulse_atr >= minimum_impulse
            and
            retracement_atr >= minimum_retrace
        ):

            return make_setup_result(
                "PULLBACK",
                "WAITING PULLBACK",
                ema_zone,
                (
                    "1H retracement reached the pullback "
                    "area but bearish confirmation is incomplete"
                ),
                signal_index=len(x) - 1,
                pullback_zone=ema_zone,
            )

    # ========================================================
    # CONTINUATION ENGINE
    # ========================================================

    # Continuation harus berasal dari compression/pullback
    # pendek, bukan sekadar candle hijau/merah.
    #
    # Kita cek 3 candle sebelum current.
    if len(x) >= 6:

        previous_three = x.iloc[
            -4:-1
        ]

        if side == "LONG":

            pullback_touch = (
                previous_three["low"]
                <= zone_high
            ).any()

            trigger = (
                close
                > float(
                    previous_three["high"].max()
                )
            )

            continuation_candle = (
                close > open_price
            )

            momentum_ok = (
                macd_hist >= 0
                and
                macd_hist >= previous_hist
            )

            continuation_ok = (
                pullback_touch
                and
                trigger
                and
                continuation_candle
                and
                momentum_ok
                and
                close > ema
                and
                st_dir > 0
            )

        else:

            pullback_touch = (
                previous_three["high"]
                >= zone_low
            ).any()

            trigger = (
                close
                < float(
                    previous_three["low"].min()
                )
            )

            continuation_candle = (
                close < open_price
            )

            momentum_ok = (
                macd_hist <= 0
                and
                macd_hist <= previous_hist
            )

            continuation_ok = (
                pullback_touch
                and
                trigger
                and
                continuation_candle
                and
                momentum_ok
                and
                close < ema
                and
                st_dir < 0
            )

        if continuation_ok:

            if (
                volume_ratio
                >= CONFIG[
                    "volume_ratio_min"
                ]
            ):

                return make_setup_result(
                    "CONTINUATION",
                    "READY",
                    [
                        float(close),
                        float(close),
                    ],
                    (
                        "1H continuation triggered after "
                        "a short pullback/compression"
                    ),
                    signal_index=len(x) - 1,
                )

            return make_setup_result(
                "CONTINUATION",
                "WAITING PULLBACK",
                ema_zone,
                (
                    "continuation trigger is present but "
                    "volume confirmation is weak"
                ),
                signal_index=len(x) - 1,
                pullback_zone=ema_zone,
            )

    # ========================================================
    # NO SETUP
    # ========================================================

    return make_setup_result(
        "NO_SETUP",
        "NO_SETUP",
        ema_zone,
        (
            "trend aligned but no clean breakout, "
            "pullback or continuation trigger"
        ),
        signal_index=len(x) - 1,
        pullback_zone=ema_zone,
    )


# ============================================================
# 15M EXECUTION ENGINE
# ============================================================

def execution_confirmation(
    df15,
    side,
    setup,
    setup_status,
    entry_zone,
):

    if (
        df15 is None
        or len(df15) < 30
    ):

        return {
            "aligned": False,
            "triggered": False,
            "reason": "insufficient 15m candles",
            "signal_index": None,
        }

    last = df15.iloc[-1]
    previous = df15.iloc[-2]

    close = safe_float(
        last["close"]
    )

    open_price = safe_float(
        last["open"]
    )

    high = safe_float(
        last["high"]
    )

    low = safe_float(
        last["low"]
    )

    previous_high = safe_float(
        previous["high"]
    )

    previous_low = safe_float(
        previous["low"]
    )

    ema = safe_float(
        last["ema200"]
    )

    st_dir = safe_float(
        last["st_dir"]
    )

    macd = safe_float(
        last["macd"]
    )

    signal = safe_float(
        last["macd_signal"]
    )

    hist = safe_float(
        last["macd_hist"]
    )

    previous_hist = safe_float(
        previous["macd_hist"]
    )

    volume_ratio = safe_float(
        last["volume_ratio"]
    )

    values = [
        close,
        open_price,
        high,
        low,
        previous_high,
        previous_low,
        ema,
        st_dir,
        macd,
        signal,
        hist,
        previous_hist,
        volume_ratio,
    ]

    if not all(
        value is not None
        and np.isfinite(value)
        for value in values
    ):

        return {
            "aligned": False,
            "triggered": False,
            "reason": "invalid 15m execution data",
            "signal_index": None,
        }

    # ========================================================
    # BASIC DIRECTIONAL ALIGNMENT
    # ========================================================

    if side == "LONG":

        aligned = (
            close > ema
            and st_dir > 0
            and macd > signal
        )

    else:

        aligned = (
            close < ema
            and st_dir < 0
            and macd < signal
        )

    if not aligned:

        return {
            "aligned": False,
            "triggered": False,
            "reason": (
                "15m EMA200 / Supertrend / MACD "
                "alignment failed"
            ),
            "signal_index": None,
        }

    # ========================================================
    # WAITING PULLBACK
    # ========================================================
    #
    # Sangat penting:
    #
    # WAITING tidak membutuhkan volume 1.3x.
    # Kita hanya ingin tahu apakah 15m masih
    # menjaga directional bias.
    # ========================================================

    if setup_status == "WAITING PULLBACK":

        if entry_zone is None:

            return {
                "aligned": True,
                "triggered": False,
                "reason": (
                    "15m aligned but entry zone unavailable"
                ),
                "signal_index": None,
            }

        zone_low = float(
            entry_zone[0]
        )

        zone_high = float(
            entry_zone[1]
        )

        # Jangan mempertahankan kandidat jika harga
        # sudah menembus jauh melewati invalidation side.
        if side == "LONG":

            invalidated = (
                close < zone_low
                - 1.25 * float(
                    last["atr"]
                )
            )

        else:

            invalidated = (
                close > zone_high
                + 1.25 * float(
                    last["atr"]
                )
            )

        if invalidated:

            return {
                "aligned": False,
                "triggered": False,
                "reason": (
                    "15m price moved beyond planned "
                    "pullback invalidation area"
                ),
                "signal_index": None,
            }

        return {
            "aligned": True,
            "triggered": False,
            "reason": (
                "15m directional bias remains aligned; "
                "waiting for price to reach entry zone "
                "and produce a trigger"
            ),
            "signal_index": None,
        }

    # ========================================================
    # READY EXECUTION
    # ========================================================

    if side == "LONG":

        candle_ok = (
            close > open_price
        )

        price_trigger = (
            close > previous_high
            or
            (
                close > open_price
                and
                close > previous["close"]
            )
        )

        momentum_ok = (
            hist >= 0
            and
            hist >= previous_hist
        )

    else:

        candle_ok = (
            close < open_price
        )

        price_trigger = (
            close < previous_low
            or
            (
                close < open_price
                and
                close < previous["close"]
            )
        )

        momentum_ok = (
            hist <= 0
            and
            hist <= previous_hist
        )

    if setup in (
        "BREAKOUT",
        "BREAKDOWN",
    ):

        required_volume = CONFIG[
            "execution_breakout_volume_min"
        ]

    else:

        required_volume = CONFIG[
            "execution_volume_min"
        ]

    volume_ok = (
        volume_ratio
        >= required_volume
    )

    triggered = (
        candle_ok
        and
        price_trigger
        and
        momentum_ok
        and
        volume_ok
    )

    if triggered:

        return {
            "aligned": True,
            "triggered": True,
            "reason": (
                "15m execution trigger confirmed "
                "by price action, momentum and volume"
            ),
            "signal_index": len(df15) - 1,
        }

    return {
        "aligned": True,
        "triggered": False,
        "reason": (
            "15m trend remains aligned but execution "
            "trigger is incomplete"
        ),
        "signal_index": None,
    }


# ============================================================
# ENTRY LOGIC
# ============================================================

def calculate_entry(
    setup,
    setup_status,
    setup_entry_zone,
    df15,
    execution,
):

    last15 = df15.iloc[-1]

    current_price = float(
        last15["close"]
    )

    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------
    #
    # Entry selalu berasal dari candle 15m yang
    # benar-benar memberikan trigger.
    # --------------------------------------------------------

    if (
        setup_status == "READY"
        and execution["triggered"]
    ):

        return {

            "entry": current_price,

            "entry_type":
                "15m_trigger",

            "entry_zone": [
                current_price,
                current_price,
            ],

            "signal_index":
                execution[
                    "signal_index"
                ],
        }

    # --------------------------------------------------------
    # WAITING PULLBACK
    # --------------------------------------------------------
    #
    # Tidak boleh menggunakan current price.
    # Gunakan planned zone.
    # --------------------------------------------------------

    if setup_entry_zone is None:

        return None

    zone_low = float(
        setup_entry_zone[0]
    )

    zone_high = float(
        setup_entry_zone[1]
    )

    if (
        zone_low <= 0
        or zone_high <= 0
    ):

        return None

    if zone_low > zone_high:

        zone_low, zone_high = (
            zone_high,
            zone_low,
        )

    planned_entry = (
        zone_low
        + zone_high
    ) / 2.0

    return {

        "entry": planned_entry,

        "entry_type":
            "planned_pullback",

        "entry_zone": [
            zone_low,
            zone_high,
        ],

        "signal_index":
            None,
    }


# ============================================================
# STOP LOSS ENGINE
# ============================================================

def calculate_stop_loss(
    side,
    entry,
    setup_df,
    exec_df,
):

    if (
        setup_df is None
        or exec_df is None
        or len(setup_df) < 10
        or len(exec_df) < 10
    ):

        return None

    setup_last = setup_df.iloc[-1]
    exec_last = exec_df.iloc[-1]

    setup_atr = safe_float(
        setup_last["atr"]
    )

    execution_atr = safe_float(
        exec_last["atr"]
    )

    if (
        setup_atr is None
        or setup_atr <= 0
        or execution_atr is None
        or execution_atr <= 0
    ):

        return None

    structure = recent_structure(
        setup_df,
        CONFIG["swing_window"],
    )

    if structure is None:
        return None

    buffer = (
        CONFIG[
            "sl_structure_buffer_atr"
        ]
        * setup_atr
    )

    execution_buffer = (
        CONFIG[
            "sl_execution_atr"
        ]
        * execution_atr
    )

    if side == "LONG":

        structure_sl = (
            structure["low"]
            - buffer
        )

        atr_sl = (
            entry
            - execution_buffer
        )

        # SL harus berada di bawah struktur
        # dan juga mempunyai ATR breathing room.
        sl = min(
            structure_sl,
            atr_sl,
        )

        risk = (
            entry - sl
        )

        invalidation = (
            "1H structure low lost with ATR buffer"
        )

    else:

        structure_sl = (
            structure["high"]
            + buffer
        )

        atr_sl = (
            entry
            + execution_buffer
        )

        sl = max(
            structure_sl,
            atr_sl,
        )

        risk = (
            sl - entry
        )

        invalidation = (
            "1H structure high reclaimed with ATR buffer"
        )

    if (
        not np.isfinite(sl)
        or sl <= 0
        or risk <= 0
    ):

        return None

    risk_pct = (
        risk / entry
    ) * 100.0

    if (
        not np.isfinite(risk_pct)
        or risk_pct
        > CONFIG["max_risk_pct"]
    ):

        return None

    return {

        "sl": float(sl),

        "risk": float(risk),

        "risk_pct": float(
            risk_pct
        ),

        "invalidation":
            invalidation,
    }


# ============================================================
# TP ENGINE
# ============================================================

def calculate_take_profits(
    side,
    entry,
    risk,
):

    result = []

    for rr in CONFIG[
        "risk_reward"
    ]:

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

        result.append(
            float(target)
        )

    return result


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
                    else pd.Timestamp(
                        value
                    ).isoformat()
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
# CHART MARKERS / ARROWS
# ============================================================

def build_chart_markers(
    data,
    side,
    setup,
    setup_status,
    entry_info,
    sl,
    tp,
):

    markers = {
        "15m": [],
        "1h": [],
        "4h": [],
    }

    # ========================================================
    # SETUP ARROW
    # ========================================================

    df1h = data["1h"]["df"]

    setup_index = setup.get(
        "signal_index"
    )

    if (
        setup_index is not None
        and 0 <= setup_index < len(df1h)
    ):

        candle = df1h.iloc[
            setup_index
        ]

        candle_atr = safe_float(
            candle["atr"]
        )

        if candle_atr is None:
            candle_atr = (
                float(
                    candle["high"]
                )
                - float(
                    candle["low"]
                )
            )

        if side == "LONG":

            arrow_price = (
                float(candle["low"])
                - 0.35 * candle_atr
            )

            markers["1h"].append(
                {
                    "time":
                        candle_time(
                            df1h,
                            setup_index,
                        ),
                    "price":
                        arrow_price,
                    "shape":
                        "arrowUp",
                    "position":
                        "belowBar",
                    "type":
                        "SETUP",
                    "side":
                        "LONG",
                    "text":
                        setup["setup"],
                }
            )

        else:

            arrow_price = (
                float(candle["high"])
                + 0.35 * candle_atr
            )

            markers["1h"].append(
                {
                    "time":
                        candle_time(
                            df1h,
                            setup_index,
                        ),
                    "price":
                        arrow_price,
                    "shape":
                        "arrowDown",
                    "position":
                        "aboveBar",
                    "type":
                        "SETUP",
                    "side":
                        "SHORT",
                    "text":
                        setup["setup"],
                }
            )

    # ========================================================
    # 15M ENTRY ARROW
    # ========================================================

    df15 = data["15m"]["df"]

    if (
        setup_status == "READY"
        and
        entry_info is not None
        and
        entry_info.get(
            "entry_type"
        ) == "15m_trigger"
    ):

        entry_index = (
            entry_info.get(
                "signal_index"
            )
        )

        if (
            entry_index is not None
            and
            0 <= entry_index < len(df15)
        ):

            candle = df15.iloc[
                entry_index
            ]

            candle_atr = safe_float(
                candle["atr"]
            )

            if candle_atr is None:

                candle_atr = (
                    float(
                        candle["high"]
                    )
                    -
                    float(
                        candle["low"]
                    )
                )

            if side == "LONG":

                arrow_price = (
                    float(candle["low"])
                    - 0.50 * candle_atr
                )

                markers["15m"].append(
                    {
                        "time":
                            candle_time(
                                df15,
                                entry_index,
                            ),
                        "price":
                            arrow_price,
                        "shape":
                            "arrowUp",
                        "position":
                            "belowBar",
                        "type":
                            "ENTRY",
                        "side":
                            "LONG",
                        "text":
                            "LONG ENTRY",
                    }
                )

            else:

                arrow_price = (
                    float(candle["high"])
                    + 0.50 * candle_atr
                )

                markers["15m"].append(
                    {
                        "time":
                            candle_time(
                                df15,
                                entry_index,
                            ),
                        "price":
                            arrow_price,
                        "shape":
                            "arrowDown",
                        "position":
                            "aboveBar",
                        "type":
                            "ENTRY",
                        "side":
                            "SHORT",
                        "text":
                            "SHORT ENTRY",
                    }
                )

    # ========================================================
    # WAITING PULLBACK MARKER
    # ========================================================
    #
    # WAITING tidak diberi ENTRY arrow.
    #
    # Kita hanya beri marker planned area pada candle
    # terakhir supaya frontend dapat menunjukkan bahwa
    # area tersebut masih menunggu.
    # ========================================================

    if (
        setup_status == "WAITING PULLBACK"
        and
        entry_info is not None
    ):

        last_index = len(df15) - 1

        last = df15.iloc[
            last_index
        ]

        zone = entry_info.get(
            "entry_zone"
        )

        if zone is not None:

            planned_entry = float(
                entry_info["entry"]
            )

            candle_atr = safe_float(
                last["atr"]
            )

            if candle_atr is None:
                candle_atr = (
                    float(last["high"])
                    - float(last["low"])
                )

            if side == "LONG":

                marker_price = (
                    planned_entry
                )

                markers["15m"].append(
                    {
                        "time":
                            candle_time(
                                df15,
                                last_index,
                            ),
                        "price":
                            marker_price,
                        "shape":
                            "arrowUp",
                        "position":
                            "belowBar",
                        "type":
                            "WAITING",
                        "side":
                            "LONG",
                        "text":
                            "WAIT PULLBACK",
                    }
                )

            else:

                marker_price = (
                    planned_entry
                )

                markers["15m"].append(
                    {
                        "time":
                            candle_time(
                                df15,
                                last_index,
                            ),
                        "price":
                            marker_price,
                        "shape":
                            "arrowDown",
                        "position":
                            "aboveBar",
                        "type":
                            "WAITING",
                        "side":
                            "SHORT",
                        "text":
                            "WAIT PULLBACK",
                    }
                )

    # ========================================================
    # TP / SL LEVEL MARKERS
    # ========================================================

    # TP/SL bukan arrow candle.
    # Mereka dikirim sebagai horizontal chart levels.
    level_markers = {

        "entry": (
            float(entry_info["entry"])
            if entry_info is not None
            else None
        ),

        "sl": (
            float(sl)
            if sl is not None
            else None
        ),

        "tp": [
            float(value)
            for value in tp
        ],
    }

    return {
        "markers": markers,
        "levels": level_markers,
    }


# ============================================================
# MOVEMENT / MOMENTUM SCORE
# ============================================================

def movement_score(df):

    if len(df) < 50:
        return -1.0, None

    if "ema200" not in df.columns:
        x = add_indicators(df)
    else:
        x = df

    last = x.iloc[-1]

    close = safe_float(
        last["close"]
    )

    atr_value = safe_float(
        last["atr"]
    )

    if (
        close is None
        or close <= 0
        or atr_value is None
        or atr_value <= 0
    ):

        return -1.0, None

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

    slow_n = CONFIG[
        "momentum_slow_bars"
    ]

    if len(x) <= slow_n + 2:
        return -1.0, None

    fast_ref = float(
        x["close"].iloc[
            -1 - fast_n
        ]
    )

    slow_ref = float(
        x["close"].iloc[
            -1 - slow_n
        ]
    )

    if (
        fast_ref <= 0
        or slow_ref <= 0
    ):

        return -1.0, None

    fast_return = (
        abs(
            close / fast_ref
            - 1.0
        )
        * 100
    )

    slow_return = (
        abs(
            close / slow_ref
            - 1.0
        )
        * 100
    )

    atr_move = (
        abs(
            close - fast_ref
        )
        / atr_value
    )

    volume_ratio = safe_float(
        last["volume_ratio"]
    )

    if (
        volume_ratio is None
        or not np.isfinite(
            volume_ratio
        )
    ):

        volume_ratio = 1.0

    volume_bonus = min(
        max(
            volume_ratio,
            0.0,
        ),
        4.0,
    )

    window = CONFIG[
        "breakout_window"
    ]

    if len(x) <= window + 1:
        return -1.0, None

    prev_high = float(
        x["high"]
        .iloc[
            -window - 1:-1
        ]
        .max()
    )

    prev_low = float(
        x["low"]
        .iloc[
            -window - 1:-1
        ]
        .min()
    )

    breakout_bonus = 0.0

    if (
        close > prev_high
        or close < prev_low
    ):

        breakout_bonus = 2.0

    direction = (
        1
        if float(last["close"])
        >= float(last["open"])
        else -1
    )

    score = (
        fast_return * 2.0
        + slow_return
        + min(
            atr_move,
            5.0,
        ) * 1.5
        + volume_bonus * 1.25
        + breakout_bonus
    )

    return float(score), {

        "df": x,

        "direction":
            direction,

        "fast_return":
            fast_return,

        "slow_return":
            slow_return,

        "volume_ratio":
            volume_ratio,

        "atr_move":
            atr_move,
    }


# ============================================================
# TIMEFRAME SCORE
# ============================================================

def score_tf(df):

    if "ema200" not in df.columns:
        x = add_indicators(df)
    else:
        x = df

    if len(x) < 210:
        return None

    last = x.iloc[-1]
    previous = x.iloc[-2]

    long_score = 0.0
    short_score = 0.0

    long_reasons = []
    short_reasons = []

    close = safe_float(
        last["close"]
    )

    ema = safe_float(
        last["ema200"]
    )

    atr_value = safe_float(
        last["atr"]
    )

    if close is None:
        return None

    if ema is None:
        return None

    if (
        atr_value is None
        or atr_value <= 0
    ):

        return None

    volume_ratio = safe_float(
        last["volume_ratio"]
    )

    if (
        volume_ratio is None
        or not np.isfinite(
            volume_ratio
        )
    ):

        volume_ratio = 1.0

    # --------------------------------------------------------
    # EMA200
    # --------------------------------------------------------

    if close > ema:

        long_score += 2.0

        long_reasons.append(
            "above EMA200"
        )

    elif close < ema:

        short_score += 2.0

        short_reasons.append(
            "below EMA200"
        )

    # --------------------------------------------------------
    # Supertrend
    # --------------------------------------------------------

    st_dir = int(
        last["st_dir"]
    )

    if st_dir > 0:

        long_score += 2.0

        long_reasons.append(
            "Supertrend bullish"
        )

    else:

        short_score += 2.0

        short_reasons.append(
            "Supertrend bearish"
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = float(
        last["macd"]
    )

    macd_signal = float(
        last["macd_signal"]
    )

    hist_now = float(
        last["macd_hist"]
    )

    hist_previous = float(
        previous["macd_hist"]
    )

    if macd > macd_signal:

        long_score += 1.0

        long_reasons.append(
            "MACD bullish"
        )

        if hist_now > hist_previous:

            long_score += 0.5

            long_reasons.append(
                "MACD histogram rising"
            )

    elif macd < macd_signal:

        short_score += 1.0

        short_reasons.append(
            "MACD bearish"
        )

        if hist_now < hist_previous:

            short_score += 0.5

            short_reasons.append(
                "MACD histogram falling"
            )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    ):

        if close > float(
            last["open"]
        ):

            long_score += 1.5

            long_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

        elif close < float(
            last["open"]
        ):

            short_score += 1.5

            short_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    window = CONFIG[
        "breakout_window"
    ]

    if len(x) > window + 1:

        previous_high = float(
            x["high"]
            .iloc[
                -window - 1:-1
            ]
            .max()
        )

        previous_low = float(
            x["low"]
            .iloc[
                -window - 1:-1
            ]
            .min()
        )

        if close > previous_high:

            long_score += 1.5

            long_reasons.append(
                "20-bar breakout"
            )

        elif close < previous_low:

            short_score += 1.5

            short_reasons.append(
                "20-bar breakdown"
            )

    return {

        "long":
            round(
                long_score,
                3,
            ),

        "short":
            round(
                short_score,
                3,
            ),

        "long_reasons":
            long_reasons,

        "short_reasons":
            short_reasons,

        "close":
            close,

        "ema200":
            ema,

        "atr":
            atr_value,

        "volume_ratio":
            volume_ratio,

        "st_dir":
            st_dir,

        "macd":
            macd,

        "macd_signal":
            macd_signal,

        "macd_hist":
            hist_now,
    }


# ============================================================
# SYMBOL ANALYSIS
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

        df15 = stage1_meta["df"]

        scored_15m = score_tf(
            df15
        )

        if scored_15m:

            data["15m"] = {
                "score":
                    scored_15m,
                "df":
                    df15,
            }

    except Exception as exc:

        logger.debug(
            "MTF %s 15m error: %s",
            symbol,
            exc,
        )

    # ========================================================
    # 1H + 4H
    # ========================================================

    for tf in (
        "1h",
        "4h",
    ):

        try:

            candles = klines(
                symbol,
                tf,
            )

           