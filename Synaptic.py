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
# SYNAPTIC — BINANCE FUTURES MTF SCANNER
#
# FLOW
# ------------------------------------------------------------
# ALL ACTIVE USDT PERPETUAL
#        ↓
# 15m MOMENTUM RANKING
#        ↓
# TOP MOMENTUM CANDIDATES
#        ↓
# 15m + 1H + 4H VALIDATION
#        ↓
# MTF DIRECTION
#        ↓
# 1H SETUP ENGINE
#        ↓
# 15m ENTRY LOGIC
#        ↓
# ENTRY / SL / TP
#        ↓
# JSON OUTPUT
#
# VISUALIZATION / ARROW / MARKER:
#       >>> HANDLED BY vSch.py <<<
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
# CONFIG
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
    # Final candidate
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

    "structure_lookback": 40,
    "swing_window": 8,

    # Pullback zone:
    # EMA200 +/- ATR
    "pullback_atr_min": 0.25,
    "pullback_atr_max": 0.85,

    # Minimum impulse size
    "minimum_impulse_atr": 1.50,

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    "risk_reward": [
        1.5,
        2.25,
        3.0,
    ],

    "sl_atr_buffer": 0.25,
    "max_risk_pct": 8.0,

    # --------------------------------------------------------
    # Extended
    # --------------------------------------------------------

    "extended_atr": 2.50,

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    "api_timeout": 10,
    "api_retries": 2,
    "retry_base_delay": 0.35,

    # --------------------------------------------------------
    # Chart data
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

        session.headers.update(
            HEADERS
        )

        _thread_local.session = session

    return session


# ============================================================
# API
# ============================================================

def _retry_delay(
    attempt,
    response=None,
):

    if response is not None:

        retry_after = response.headers.get(
            "Retry-After"
        )

        if retry_after:

            try:

                return min(
                    max(
                        float(retry_after),
                        0.2,
                    ),
                    5.0,
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

    delay = (
        CONFIG["retry_base_delay"]
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
        timeout = CONFIG["api_timeout"]

    session = get_session()

    last_error = None

    for base_url in BASE_URLS:

        url = base_url + path

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
                # OK
                # ------------------------------------------------

                if status == 200:

                    data = _parse_response(
                        response
                    )

                    if data is None:

                        last_error = (
                            f"{base_url} "
                            "HTTP 200 invalid JSON"
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
                # RATE LIMIT
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
                # OTHER
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


def ticker_price(symbol):

    data = api(
        "/fapi/v1/ticker/price",
        {
            "symbol": symbol,
        },
    )

    try:

        return float(
            data["price"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        return None


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
                last_price,
            )
        )

    if CONFIG["universe_size"] > 0:

        rows = sorted(
            rows,
            key=lambda x: x[2],
            reverse=True,
        )

        rows = rows[
            :CONFIG["universe_size"]
        ]

    logger.info(
        "Universe matched %d active USDT perpetuals (%.2fs).",
        len(rows),
        time.time() - started,
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

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Remove currently forming candle
    # --------------------------------------------------------

    if len(df) > 1:

        now = pd.Timestamp.now(
            tz="UTC"
        )

        if (
            df.iloc[-1]["close_time"]
            > now
        ):

            df = (
                df.iloc[:-1]
                .reset_index(drop=True)
            )

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    x = df.copy()

    # --------------------------------------------------------
    # EMA 200
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
    # MACD 12 / 26 / 9
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

    x["macd"] = fast - slow

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
    # ATR 14
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
            min_periods=CONFIG[
                "volume_ma_period"
            ],
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

        prev_upper = (
            final_upper.iloc[i - 1]
        )

        prev_lower = (
            final_lower.iloc[i - 1]
        )

        if (
            basic_upper.iloc[i]
            < prev_upper
            or
            x["close"].iloc[i - 1]
            > prev_upper
        ):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

        else:

            final_upper.iloc[i] = (
                prev_upper
            )

        if (
            basic_lower.iloc[i]
            > prev_lower
            or
            x["close"].iloc[i - 1]
            < prev_lower
        ):

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

        else:

            final_lower.iloc[i] = (
                prev_lower
            )

        if (
            direction.iloc[i - 1] == -1
            and
            x["close"].iloc[i]
            > prev_upper
        ):

            direction.iloc[i] = 1

        elif (
            direction.iloc[i - 1] == 1
            and
            x["close"].iloc[i]
            < prev_lower
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

        first_atr = float(
            x["atr"].iloc[0]
        )

        if np.isfinite(first_atr):

            supertrend.iloc[0] = (
                hl2.iloc[0]
                - multiplier
                * first_atr
            )

    x["supertrend"] = supertrend
    x["st_dir"] = direction

    return x


# ============================================================
# SERIALIZE
# ============================================================

def serialize_chart_data(df):

    columns = [
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

    records = []

    for _, row in df.iterrows():

        item = {}

        for col in columns:

            if col not in df.columns:
                continue

            value = row[col]

            if col == "time":

                item[col] = (
                    None
                    if pd.isna(value)
                    else pd.Timestamp(
                        value
                    ).isoformat()
                )

            elif pd.isna(value):

                item[col] = None

            elif col == "st_dir":

                item[col] = int(value)

            else:

                item[col] = float(value)

        records.append(item)

    return records


# ============================================================
# STRUCTURE HELPERS
# ============================================================

def recent_structure(
    df,
    side,
):

    lookback = min(
        CONFIG["structure_lookback"],
        len(df) - 1,
    )

    if lookback < 10:
        return None

    x = df.iloc[
        -lookback - 1:-1
    ].copy()

    if side == "LONG":

        swing_low_idx = (
            x["low"].idxmin()
        )

        swing_high_idx = (
            x["high"].idxmax()
        )

        swing_low = float(
            x.loc[
                swing_low_idx,
                "low",
            ]
        )

        swing_high = float(
            x.loc[
                swing_high_idx,
                "high",
            ]
        )

        return {
            "swing_low": swing_low,
            "swing_high": swing_high,
            "low_index": swing_low_idx,
            "high_index": swing_high_idx,
        }

    swing_high_idx = (
        x["high"].idxmax()
    )

    swing_low_idx = (
        x["low"].idxmin()
    )

    swing_high = float(
        x.loc[
            swing_high_idx,
            "high",
        ]
    )

    swing_low = float(
        x.loc[
            swing_low_idx,
            "low",
        ]
    )

    return {
        "swing_low": swing_low,
        "swing_high": swing_high,
        "low_index": swing_low_idx,
        "high_index": swing_high_idx,
    }


def candle_rejection(
    candle,
    side,
):

    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])

    candle_range = h - l

    if candle_range <= 0:
        return False

    body = abs(c - o)

    upper_wick = h - max(
        o,
        c,
    )

    lower_wick = min(
        o,
        c,
    ) - l

    if side == "LONG":

        return (
            c > o
            and
            lower_wick >= body * 0.75
            and
            c >= l
            + candle_range * 0.55
        )

    return (
        c < o
        and
        upper_wick >= body * 0.75
        and
        c <= l
        + candle_range * 0.45
    )


# ============================================================
# SETUP ENGINE
# ============================================================

def determine_setup(
    df,
    side,
):

    if (
        df is None
        or len(df) < 210
    ):

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "insufficient 1H candles",
        }

    x = df.copy()

    last = x.iloc[-1]
    previous = x.iloc[-2]

    close = float(last["close"])
    open_price = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])

    ema = float(last["ema200"])
    atr = float(last["atr"])

    st_dir = int(last["st_dir"])

    macd = float(last["macd"])
    signal = float(last["macd_signal"])
    hist = float(last["macd_hist"])
    prev_hist = float(previous["macd_hist"])

    volume_ratio = float(
        last["volume_ratio"]
    )

    values = [
        close,
        open_price,
        high,
        low,
        ema,
        atr,
        macd,
        signal,
        hist,
        prev_hist,
        volume_ratio,
    ]

    if not all(
        np.isfinite(v)
        for v in values
    ):

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "invalid indicator data",
        }

    if (
        close <= 0
        or ema <= 0
        or atr <= 0
    ):

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "invalid price/ATR",
        }

    # ========================================================
    # 1H TREND ALIGNMENT
    # ========================================================

    if side == "LONG":

        trend_ok = (
            close > ema
            and st_dir > 0
            and macd > signal
        )

    else:

        trend_ok = (
            close < ema
            and st_dir < 0
            and macd < signal
        )

    if not trend_ok:

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": (
                "1H EMA200 / Supertrend / MACD "
                "alignment failed"
            ),
        }

    # ========================================================
    # DISTANCE FROM EMA
    # ========================================================

    distance_atr = (
        abs(close - ema)
        / atr
    )

    # ========================================================
    # EXTENDED
    # ========================================================

    if (
        distance_atr
        > CONFIG["extended_atr"]
    ):

        if side == "LONG":

            zone_low = (
                ema
                - CONFIG[
                    "pullback_atr_min"
                ] * atr
            )

            zone_high = (
                ema
                + CONFIG[
                    "pullback_atr_max"
                ] * atr
            )

        else:

            zone_low = (
                ema
                - CONFIG[
                    "pullback_atr_max"
                ] * atr
            )

            zone_high = (
                ema
                + CONFIG[
                    "pullback_atr_min"
                ] * atr
            )

        return {
            "setup": "EXTENDED",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(zone_low),
                float(zone_high),
            ],
            "reason": (
                f"price is {distance_atr:.2f} ATR "
                "from EMA200; no chase"
            ),
        }

    # ========================================================
    # RECENT STRUCTURE
    # ========================================================

    structure = recent_structure(
        x,
        side,
    )

    if structure is None:

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "structure unavailable",
        }

    swing_low = structure[
        "swing_low"
    ]

    swing_high = structure[
        "swing_high"
    ]

    # ========================================================
    # 20-BAR BREAKOUT / BREAKDOWN
    # ========================================================

    window = CONFIG[
        "breakout_window"
    ]

    previous_high = float(
        x["high"]
        .iloc[-window - 1:-1]
        .max()
    )

    previous_low = float(
        x["low"]
        .iloc[-window - 1:-1]
        .min()
    )

    breakout_long = (
        close > previous_high
    )

    breakout_short = (
        close < previous_low
    )

    volume_ok = (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    )

    # ========================================================
    # BREAKOUT
    # ========================================================

    if side == "LONG" and breakout_long:

        bullish_candle = (
            close > open_price
        )

        momentum_ok = (
            hist >= 0
            and hist >= prev_hist
        )

        if (
            bullish_candle
            and momentum_ok
            and volume_ok
        ):

            # Entry is NOT blindly the breakout close.
            # Wait for retest unless price is still
            # sufficiently close to breakout.
            breakout_distance = (
                close - previous_high
            ) / atr

            if breakout_distance <= 0.35:

                return {
                    "setup": "BREAKOUT",
                    "status": "READY",
                    "entry_zone": [
                        float(previous_high),
                        float(close),
                    ],
                    "reason": (
                        "1H bullish breakout confirmed "
                        "with volume and momentum"
                    ),
                }

        # Breakout exists but price has moved away.
        retest_low = min(
            previous_high,
            ema,
        )

        retest_high = max(
            previous_high,
            ema,
        )

        return {
            "setup": "BREAKOUT",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(retest_low),
                float(retest_high),
            ],
            "reason": (
                "1H breakout detected; wait for "
                "retest instead of chasing"
            ),
        }

    if side == "SHORT" and breakout_short:

        bearish_candle = (
            close < open_price
        )

        momentum_ok = (
            hist <= 0
            and hist <= prev_hist
        )

        if (
            bearish_candle
            and momentum_ok
            and volume_ok
        ):

            breakout_distance = (
                previous_low - close
            ) / atr

            if breakout_distance <= 0.35:

                return {
                    "setup": "BREAKDOWN",
                    "status": "READY",
                    "entry_zone": [
                        float(close),
                        float(previous_low),
                    ],
                    "reason": (
                        "1H bearish breakdown confirmed "
                        "with volume and momentum"
                    ),
                }

        retest_low = min(
            previous_low,
            ema,
        )

        retest_high = max(
            previous_low,
            ema,
        )

        return {
            "setup": "BREAKDOWN",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(retest_low),
                float(retest_high),
            ],
            "reason": (
                "1H breakdown detected; wait for "
                "retest instead of chasing"
            ),
        }

    # ========================================================
    # PULLBACK ZONE
    # ========================================================

    if side == "LONG":

        zone_low = (
            ema
            - CONFIG[
                "pullback_atr_min"
            ] * atr
        )

        zone_high = (
            ema
            + CONFIG[
                "pullback_atr_max"
            ] * atr
        )

        # Recent bullish impulse
        impulse = (
            swing_high
            - swing_low
        )

        impulse_atr = (
            impulse / atr
        )

        touched_zone = (
            low <= zone_high
            and close >= zone_low
        )

        retracement = (
            swing_high - close
        )

        retracement_atr = (
            retracement / atr
        )

        real_pullback = (
            impulse_atr
            >= CONFIG[
                "minimum_impulse_atr"
            ]
            and
            retracement_atr
            >= 0.35
        )

        rejection = candle_rejection(
            last,
            "LONG",
        )

        momentum_recovering = (
            hist >= prev_hist
            or hist >= 0
        )

        if (
            touched_zone
            and real_pullback
            and rejection
            and momentum_recovering
        ):

            return {
                "setup": "PULLBACK",
                "status": "READY",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "1H bullish trend pulled back "
                    "into EMA200 area and printed "
                    "bullish rejection"
                ),
            }

        if (
            touched_zone
            and real_pullback
        ):

            return {
                "setup": "PULLBACK",
                "status": "WAITING PULLBACK",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "price reached the 1H pullback "
                    "area but confirmation is incomplete"
                ),
            }

    else:

        zone_low = (
            ema
            - CONFIG[
                "pullback_atr_max"
            ] * atr
        )

        zone_high = (
            ema
            + CONFIG[
                "pullback_atr_min"
            ] * atr
        )

        impulse = (
            swing_high
            - swing_low
        )

        impulse_atr = (
            impulse / atr
        )

        touched_zone = (
            high >= zone_low
            and close <= zone_high
        )

        retracement = (
            close - swing_low
        )

        retracement_atr = (
            retracement / atr
        )

        real_pullback = (
            impulse_atr
            >= CONFIG[
                "minimum_impulse_atr"
            ]
            and
            retracement_atr
            >= 0.35
        )

        rejection = candle_rejection(
            last,
            "SHORT",
        )

        momentum_recovering = (
            hist <= prev_hist
            or hist <= 0
        )

        if (
            touched_zone
            and real_pullback
            and rejection
            and momentum_recovering
        ):

            return {
                "setup": "PULLBACK",
                "status": "READY",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "1H bearish trend pulled back "
                    "into EMA200 area and printed "
                    "bearish rejection"
                ),
            }

        if (
            touched_zone
            and real_pullback
        ):

            return {
                "setup": "PULLBACK",
                "status": "WAITING PULLBACK",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "price reached the 1H pullback "
                    "area but confirmation is incomplete"
                ),
            }

    # ========================================================
    # CONTINUATION
    # ========================================================

    if side == "LONG":

        continuation = (
            close > open_price
            and close > float(
                previous["close"]
            )
            and hist >= 0
            and hist >= prev_hist
            and close > ema
            and st_dir > 0
        )

    else:

        continuation = (
            close < open_price
            and close < float(
                previous["close"]
            )
            and hist <= 0
            and hist <= prev_hist
            and close < ema
            and st_dir < 0
        )

    if continuation:

        # Continuation is only READY when price is
        # not already extended.
        if (
            distance_atr <= 1.25
            and volume_ok
        ):

            return {
                "setup": "CONTINUATION",
                "status": "READY",
                "entry_zone": [
                    float(close),
                    float(close),
                ],
                "reason": (
                    "1H continuation confirmed without "
                    "excessive extension"
                ),
            }

        return {
            "setup": "CONTINUATION",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(
                    ema
                    - 0.50 * atr
                )
                if side == "LONG"
                else float(
                    ema
                    - 0.85 * atr
                ),
                float(
                    ema
                    + 0.85 * atr
                )
                if side == "LONG"
                else float(
                    ema
                    + 0.50 * atr
                ),
            ],
            "reason": (
                "trend continuation exists but "
                "price should not be chased"
            ),
        }

    return {
        "setup": "NO_SETUP",
        "status": "NO_SETUP",
        "entry_zone": None,
        "reason": (
            "no clean breakout, pullback "
            "or continuation"
        ),
    }


# ============================================================
# 15M ENTRY LOGIC
# ============================================================

def determine_entry(
    df,
    side,
    setup,
    entry_zone,
):

    if (
        df is None
        or len(df) < 30
    ):

        return {
            "status": "NO_ENTRY",
            "entry": None,
            "reason": "insufficient 15m candles",
        }

    if (
        setup == "NO_SETUP"
        or entry_zone is None
    ):

        return {
            "status": "NO_ENTRY",
            "entry": None,
            "reason": "no valid 1H setup",
        }

    x = df.copy()

    last = x.iloc[-1]
    previous = x.iloc[-2]

    close = float(last["close"])
    open_price = float(last["open"])
    high = float(last["high"])
    low = float(last["low"])

    ema = float(last["ema200"])
    atr = float(last["atr"])

    st_dir = int(last["st_dir"])

    macd = float(last["macd"])
    signal = float(last["macd_signal"])
    hist = float(last["macd_hist"])
    prev_hist = float(
        previous["macd_hist"]
    )

    volume_ratio = float(
        last["volume_ratio"]
    )

    if (
        not np.isfinite(close)
        or not np.isfinite(atr)
        or atr <= 0
    ):

        return {
            "status": "NO_ENTRY",
            "entry": None,
            "reason": "invalid 15m data",
        }

    zone_low = float(
        min(
            entry_zone[0],
            entry_zone[1],
        )
    )

    zone_high = float(
        max(
            entry_zone[0],
            entry_zone[1],
        )
    )

    # ========================================================
    # 15M LOCATION
    # ========================================================

    inside_zone = (
        low <= zone_high
        and high >= zone_low
    )

    close_inside_zone = (
        zone_low
        <= close
        <= zone_high
    )

    # ========================================================
    # 15M TREND
    # ========================================================

    if side == "LONG":

        directional = (
            close > ema
            and st_dir > 0
            and macd > signal
        )

    else:

        directional = (
            close < ema
            and st_dir < 0
            and macd < signal
        )

    # ========================================================
    # 15M CANDLE TRIGGER
    # ========================================================

    rejection = candle_rejection(
        last,
        side,
    )

    if side == "LONG":

        bullish_close = (
            close > open_price
        )

        momentum_trigger = (
            hist > prev_hist
            or hist > 0
        )

        breakout_trigger = (
            close > float(
                previous["high"]
            )
        )

        trigger = (
            rejection
            or (
                bullish_close
                and momentum_trigger
                and breakout_trigger
            )
        )

    else:

        bearish_close = (
            close < open_price
        )

        momentum_trigger = (
            hist < prev_hist
            or hist < 0
        )

        breakdown_trigger = (
            close < float(
                previous["low"]
            )
        )

        trigger = (
            rejection
            or (
                bearish_close
                and momentum_trigger
                and breakdown_trigger
            )
        )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_ok = (
        np.isfinite(volume_ratio)
        and
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    )

    # ========================================================
    # READY LOGIC
    # ========================================================

    # PULLBACK:
    # Price MUST interact with the zone.
    # No zone interaction = no entry.
    if setup == "PULLBACK":

        if (
            inside_zone
            and directional
            and trigger
        ):

            return {
                "status": "READY",
                "entry": close,
                "entry_reason": (
                    "15m pullback-zone interaction "
                    "with directional confirmation "
                    "and price-action trigger"
                ),
                "trigger": (
                    "REJECTION"
                    if rejection
                    else "BREAKOUT_CONFIRMATION"
                ),
            }

        return {
            "status": "WAITING PULLBACK",
            "entry": (
                (zone_low + zone_high)
                / 2.0
            ),
            "entry_reason": (
                "15m has not produced a valid "
                "pullback entry trigger"
            ),
            "trigger": None,
        }

    # ========================================================
    # BREAKOUT
    # ========================================================

    if setup in (
        "BREAKOUT",
        "BREAKDOWN",
    ):

        if side == "LONG":

            valid_breakout = (
                close > zone_low
                and directional
                and (
                    trigger
                    or
                    (
                        close_inside_zone
                        and rejection
                    )
                )
            )

        else:

            valid_breakout = (
                close < zone_high
                and directional
                and (
                    trigger
                    or
                    (
                        close_inside_zone
                        and rejection
                    )
                )
            )

        if valid_breakout:

            return {
                "status": "READY",
                "entry": close,
                "entry_reason": (
                    "15m confirms the 1H breakout "
                    "direction"
                ),
                "trigger": (
                    "REJECTION"
                    if rejection
                    else "BREAKOUT_CONFIRMATION"
                ),
            }

        return {
            "status": "WAITING PULLBACK",
            "entry": (
                zone_low + zone_high
            ) / 2.0,
            "entry_reason": (
                "1H breakout exists but 15m "
                "entry trigger is incomplete"
            ),
            "trigger": None,
        }

    # ========================================================
    # CONTINUATION
    # ========================================================

    if setup == "CONTINUATION":

        if (
            directional
            and trigger
            and (
                volume_ok
                or close_inside_zone
            )
        ):

            return {
                "status": "READY",
                "entry": close,
                "entry_reason": (
                    "15m continuation trigger "
                    "agrees with 1H trend"
                ),
                "trigger": (
                    "REJECTION"
                    if rejection
                    else "MOMENTUM_CONFIRMATION"
                ),
            }

        return {
            "status": "WAITING PULLBACK",
            "entry": (
                zone_low + zone_high
            ) / 2.0,
            "entry_reason": (
                "15m continuation is not confirmed "
                "enough to chase"
            ),
            "trigger": None,
        }

    return {
        "status": "NO_ENTRY",
        "entry": None,
        "reason": "unsupported setup",
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

    close = float(last["close"])
    open_price = float(last["open"])
    ema = float(last["ema200"])
    atr = float(last["atr"])

    if (
        not np.isfinite(close)
        or not np.isfinite(ema)
        or not np.isfinite(atr)
        or atr <= 0
    ):

        return None

    long_score = 0.0
    short_score = 0.0

    long_reasons = []
    short_reasons = []

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

    signal = float(
        last["macd_signal"]
    )

    hist = float(
        last["macd_hist"]
    )

    prev_hist = float(
        previous["macd_hist"]
    )

    if macd > signal:

        long_score += 1.0

        long_reasons.append(
            "MACD bullish"
        )

        if hist > prev_hist:

            long_score += 0.5

            long_reasons.append(
                "MACD histogram rising"
            )

    elif macd < signal:

        short_score += 1.0

        short_reasons.append(
            "MACD bearish"
        )

        if hist < prev_hist:

            short_score += 0.5

            short_reasons.append(
                "MACD histogram falling"
            )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_ratio = float(
        last["volume_ratio"]
    )

    if np.isfinite(
        volume_ratio
    ):

        if (
            volume_ratio
            >= CONFIG["volume_ratio_min"]
        ):

            if close > open_price:

                long_score += 1.5

                long_reasons.append(
                    f"volume {volume_ratio:.1f}x"
                )

            elif close < open_price:

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

    previous_high = float(
        x["high"]
        .iloc[-window - 1:-1]
        .max()
    )

    previous_low = float(
        x["low"]
        .iloc[-window - 1:-1]
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
        "long": round(
            long_score,
            3,
        ),
        "short": round(
            short_score,
            3,
        ),
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "close": close,
        "ema200": ema,
        "atr": atr,
        "volume_ratio": volume_ratio,
        "st_dir": st_dir,
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": hist,
    }


# ============================================================
# MOMENTUM SCORE
# ============================================================

def movement_score(df):

    if len(df) < 50:
        return -1.0, None

    x = (
        df
        if "ema200" in df.columns
        else add_indicators(df)
    )

    last = x.iloc[-1]

    close = float(
        last["close"]
    )

    atr = float(
        last["atr"]
    )

    if (
        not np.isfinite(close)
        or close <= 0
        or not np.isfinite(atr)
        or atr <= 0
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

    fast_return = abs(
        close / fast_ref - 1.0
    ) * 100

    slow_return = abs(
        close / slow_ref - 1.0
    ) * 100

    atr_move = abs(
        close - fast_ref
    ) / atr

    volume_ratio = float(
        last["volume_ratio"]
    )

    if not np.isfinite(
        volume_ratio
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

    previous_high = float(
        x["high"]
        .iloc[-window - 1:-1]
        .max()
    )

    previous_low = float(
        x["low"]
        .iloc[-window - 1:-1]
        .min()
    )

    breakout_bonus = 0.0

    if (
        close > previous_high
        or close < previous_low
    ):

        breakout_bonus = 2.0

    direction = (
        1
        if close
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
        "direction": direction,
        "fast_return": fast_return,
        "slow_return": slow_return,
        "volume_ratio": volume_ratio,
        "atr_move": atr_move,
    }


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(
    df,
    side,
    entry,
    setup,
):

    if (
        df is None
        or len(df) < 12
    ):

        return None

    x = df.copy()

    last = x.iloc[-1]

    atr = float(
        last["atr"]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):

        return None

    swing_n = CONFIG[
        "swing_window"
    ]

    recent = x.iloc[
        -swing_n - 1:-1
    ]

    if len(recent) < 3:
        recent = x.iloc[-swing_n:]

    structural_low = float(
        recent["low"].min()
    )

    structural_high = float(
        recent["high"].max()
    )

    buffer = (
        CONFIG["sl_atr_buffer"]
        * atr
    )

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if side == "LONG":

        structure_sl = (
            structural_low
            - buffer
        )

        atr_sl = (
            entry
            - 1.25 * atr
        )

        # Use the protective level that is actually
        # below the structure.
        sl = min(
            structure_sl,
            atr_sl,
        )

        risk = (
            entry - sl
        )

        if risk <= 0:
            return None

        invalidation = (
            f"1H/15m structure invalidation "
            f"below {sl:.8g}"
        )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    else:

        structure_sl = (
            structural_high
            + buffer
        )

        atr_sl = (
            entry
            + 1.25 * atr
        )

        sl = max(
            structure_sl,
            atr_sl,
        )

        risk = (
            sl - entry
        )

        if risk <= 0:
            return None

        invalidation = (
            f"1H/15m structure invalidation "
            f"above {sl:.8g}"
        )

    risk_pct = (
        risk / entry
    ) * 100

    if (
        not np.isfinite(risk_pct)
        or risk_pct <= 0
        or risk_pct
        > CONFIG["max_risk_pct"]
    ):

        return None

    tp = []

    for rr in CONFIG[
        "risk_reward"
    ]:

        if side == "LONG":

            tp.append(
                entry + risk * rr
            )

        else:

            tp.append(
                entry - risk * rr
            )

    return {
        "sl": float(sl),
        "risk": float(risk),
        "risk_pct": float(risk_pct),
        "tp": [
            float(v)
            for v in tp
        ],
        "invalidation": invalidation,
    }


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    change_24h,
    quote_volume_24h,
    live_price,
    stage1_score,
    stage1_meta,
):

    data = {}

    # ========================================================
    # 15M
    # ========================================================

    try:

        df15 = stage1_meta["df"]

        scored15 = score_tf(
            df15
        )

        if scored15:

            data["15m"] = {
                "score": scored15,
                "df": df15,
            }

    except Exception as exc:

        logger.debug(
            "15m error %s: %s",
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

            if len(candles) < 210:
                continue

            enriched = add_indicators(
                candles
            )

            scored = score_tf(
                enriched
            )

            if scored:

                data[tf] = {
                    "score": scored,
                    "df": enriched,
                }

        except Exception as exc:

            logger.debug(
                "%s %s error: %s",
                symbol,
                tf,
                exc,
            )

    # ========================================================
    # MTF COMPLETE
    # ========================================================

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
    # DIRECTION
    # ========================================================

    if long_total > short_total:

        side = "LONG"

    elif short_total > long_total:

        side = "SHORT"

    else:

        return None

    wanted_direction = (
        1
        if side == "LONG"
        else -1
    )

    # ========================================================
    # MTF VOTES
    # ========================================================

    votes = []

    for tf in TFS:

        s = data[tf]["score"]

        if s["long"] > s["short"]:

            votes.append(1)

        elif s["short"] > s["long"]:

            votes.append(-1)

        else:

            votes.append(0)

    agreement = sum(
        vote == wanted_direction
        for vote in votes
    )

    if agreement < 2:
        return None

    # ========================================================
    # 4H HARD BIAS
    # ========================================================

    df4h = data["4h"]["df"]

    last4h = df4h.iloc[-1]

    close4h = float(
        last4h["close"]
    )

    ema4h = float(
        last4h["ema200"]
    )

    st4h = int(
        last4h["st_dir"]
    )

    macd4h = float(
        last4h["macd"]
    )

    signal4h = float(
        last4h["macd_signal"]
    )

    if side == "LONG":

        bias4h = (
            close4h > ema4h
            and st4h > 0
            and macd4h > signal4h
        )

    else:

        bias4h = (
            close4h < ema4h
            and st4h < 0
            and macd4h < signal4h
        )

    if not bias4h:
        return None

    # ========================================================
    # 1H SETUP ENGINE
    # ========================================================

    setup = determine_setup(
        data["1h"]["df"],
        side,
    )

    if (
        setup["setup"]
        == "NO_SETUP"
    ):

        return None

    # ========================================================
    # 15M ENTRY ENGINE
    # ========================================================

    entry_result = determine_entry(
        data["15m"]["df"],
        side,
        setup["setup"],
        setup["entry_zone"],
    )

    if entry_result["status"] == "NO_ENTRY":

        return None

    # ========================================================
    # STATUS
    # ========================================================

    final_status = (
        entry_result["status"]
    )

    # --------------------------------------------------------
    # Important:
    #
    # WAITING PULLBACK from Setup Engine MUST remain
    # WAITING PULLBACK unless 15m actually gives a valid
    # entry trigger.
    # --------------------------------------------------------

    if (
        setup["status"]
        == "WAITING PULLBACK"
        and
        entry_result["status"]
        != "READY"
    ):

        final_status = (
            "WAITING PULLBACK"
        )

    # ========================================================
    # ENTRY
    # ========================================================

    entry = entry_result[
        "entry"
    ]

    if (
        entry is None
        or not np.isfinite(entry)
        or entry <= 0
    ):

        return None

    # ========================================================
    # LIVE PRICE
    # ========================================================

    if (
        live_price is None
        or not np.isfinite(live_price)
        or live_price <= 0
    ):

        live_price = float(
            data["15m"]["df"]
            .iloc[-1]["close"]
        )

    # ========================================================
    # RISK
    # ========================================================

    risk = calculate_risk(
        data["15m"]["df"],
        side,
        entry,
        setup["setup"],
    )

    if risk is None:
        return None

    # ========================================================
    # SCORE
    # ========================================================

    raw_score = (
        long_total
        if side == "LONG"
        else short_total
    )

    momentum_bonus = min(
        stage1_score / 25.0,
        1.5,
    )

    setup_bonus = {
        "BREAKOUT": 1.0,
        "BREAKDOWN": 1.0,
        "PULLBACK": 1.25,
        "CONTINUATION": 0.75,
        "EXTENDED": 0.0,
    }.get(
        setup["setup"],
        0.0,
    )

    ready_bonus = (
        0.75
        if final_status == "READY"
        else 0.0
    )

    final_score = (
        raw_score
        + momentum_bonus
        + setup_bonus
        + ready_bonus
    )

    # ========================================================
    # 15M MOMENTUM
    # ========================================================

    df15 = data["15m"]["df"]

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

    current15 = float(
        df15.iloc[-1]["close"]
    )

    reference15 = float(
        df15.iloc[
            -1 - fast_n
        ]["close"]
    )

    if reference15 <= 0:
        return None

    move15 = (
        current15
        / reference15
        - 1.0
    ) * 100

    # ========================================================
    # REASONS
    # ========================================================

    reasons = (
        data["15m"]["score"][
            "long_reasons"
        ]
        if side == "LONG"
        else data["15m"]["score"][
            "short_reasons"
        ]
    )

    key_points = list(
        reasons[:6]
    )

    key_points.append(
        f"1H {setup['setup']}"
    )

    key_points.append(
        f"15m {final_status}"
    )

    # ========================================================
    # CHART DATA
    #
    # NO ARROW
    # NO MARKER
    # NO PLOT
    #
    # vSch.py owns visualization.
    # ========================================================

    chart_data = {
        tf: serialize_chart_data(
            data[tf]["df"]
        )
        for tf in TFS
    }

    # ========================================================
    # OUTPUT
    # ========================================================

    return {

        "symbol": symbol,

        "side": side,

        "score": round(
            final_score,
            2,
        ),

        "change24h": round(
            change_24h,
            2,
        ),

        "quote_volume24h": round(
            quote_volume_24h,
            2,
        ),

        "live_price": float(
            live_price
        ),

        # ----------------------------------------------------
        # SETUP
        # ----------------------------------------------------

        "setup": setup[
            "setup"
        ],

        "setup_status": final_status,

        "setup_reason": setup[
            "reason"
        ],

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        "entry": float(entry),

        "entry_zone": [
            float(
                min(
                    setup["entry_zone"][0],
                    setup["entry_zone"][1],
                )
            ),
            float(
                max(
                    setup["entry_zone"][0],
                    setup["entry_zone"][1],
                )
            ),
        ],

        "entry_reason": entry_result.get(
            "entry_reason",
            "",
        ),

        "entry_trigger": entry_result.get(
            "trigger"
        ),

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        "sl": risk["sl"],

        "tp": risk["tp"],

        "risk": risk["risk"],

        "risk_pct": round(
            risk["risk_pct"],
            3,
        ),

        "invalidation": risk[
            "invalidation"
        ],

        # ----------------------------------------------------
        # MTF
        # ----------------------------------------------------

        "tf_agreement": agreement,

        "timeframes": {
            tf: data[tf]["score"]
            for tf in TFS
        },

        "momentum_15m": round(
            move15,
            3,
        ),

        # ----------------------------------------------------
        # KEY POINTS
        # ----------------------------------------------------

        "key_points": key_points,

        # ----------------------------------------------------
        # CHART CONFIG
        #
        # These are data flags only.
        # vSch.py handles rendering.
        # ----------------------------------------------------

        "chart": {

            "execution_tf": "15m",

            "setup_tf": "1h",

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

            "show_arrow": False,

            "show_marker": False,
        },

        # ----------------------------------------------------
        # LEVELS FOR vSch.py
        # ----------------------------------------------------

        "chart_levels": {

            "entry": float(entry),

            "entry_zone": [
                float(
                    min(
                        setup["entry_zone"][0],
                        setup["entry_zone"][1],
                    )
                ),
                float(
                    max(
                        setup["entry_zone"][0],
                        setup["entry_zone"][1],
                    )
                ),
            ],

            "sl": float(
                risk["sl"]
            ),

            "tp": [
                float(v)
                for v in risk["tp"]
            ],
        },

        # ----------------------------------------------------
        # RAW CHART DATA
        # ----------------------------------------------------

        "chart_data": chart_data,
    }


# ============================================================
# STAGE 1 WORKER
# ============================================================

def stage1_worker(row):

    (
        symbol,
        change24h,
        quote_volume,
        live_price,
    ) = row

    try:

        candles = klines(
            symbol,
            "15m",
        )

        if len(candles) < 50:
            return None

        enriched = add_indicators(
            candles
        )

        score, meta = movement_score(
            enriched
        )

        if (
            score <= 0
            or meta is None
        ):

            return None

        meta["df"] = enriched

        return (
            score,
            symbol,
            change24h,
            quote_volume,
            live_price,
            meta,
        )

    except Exception as exc:

        logger.debug(
            "Stage 1 %s error: %s",
            symbol,
            exc,
        )

        return None


# ============================================================
# STAGE 2 WORKER
# ============================================================

def stage2_worker(item):

    (
        stage1_score,
        symbol,
        change24h,
        quote_volume,
        live_price,
        stage1_meta,
    ) = item

    try:

        return analyze_symbol(
            symbol,
            change24h,
            quote_volume,
            live_price,
            stage1_score,
            stage1_meta,
        )

    except Exception as exc:

        logger.debug(
            "Stage 2 %s error: %s",
            symbol,
            exc,
        )

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Synaptic Binance Futures "
            "MTF scanner"
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
            "Cannot build universe: %s",
            exc,
        )

        Path(
            args.out
        ).write_text(
            json.dumps(
                {
                    "scanner": "Synaptic",
                    "candidates": [],
                    "error": str(exc),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        raise

    if not universe_rows:

        raise RuntimeError(
            "Universe is empty."
        )

    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_started = time.time()

    logger.info(
        "Stage 1: scanning %d symbols on 15m...",
        len(universe_rows),
    )

    momentum = []

    with ThreadPoolExecutor(
        max_workers=CONFIG[
            "workers_stage1"
        ]
    ) as pool:

        futures = [
            pool.submit(
                stage1_worker,
                row,
            )
            for row in universe_rows
        ]

        for future in as_completed(
            futures
        ):

            try:

                result = future.result()

                if result is not None:

                    momentum.append(
                        result
                    )

            except Exception as exc:

                logger.debug(
                    "Stage 1 future: %s",
                    exc,
                )

    momentum.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    selected = momentum[
        :CONFIG["momentum_pool"]
    ]

    stage1_elapsed = (
        time.time()
        - stage1_started
    )

    logger.info(
        "Stage 1 complete | momentum=%d | selected=%d",
        len(momentum),
        len(selected),
    )

    # ========================================================
    # NO STAGE 1
    # ========================================================

    if not selected:

        payload = {

            "generated_at":
                pd.Timestamp.now(
                    tz="UTC"
                ).isoformat(),

            "scanner":
                "Synaptic",

            "selection_mode":
                "no_stage1_candidates",

            "scan_stats": {

                "universe":
                    len(universe_rows),

                "stage1_momentum":
                    len(momentum),

                "stage1_selected":
                    0,

                "mtf_valid":
                    0,

                "min_score_valid":
                    0,

                "final_candidates":
                    0,

                "elapsed_seconds":
                    round(
                        time.time()
                        - started,
                        2,
                    ),
            },

            "candidates": [],
        }

        Path(
            args.out
        ).write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return

    # ========================================================
    # STAGE 2
    # ========================================================

    stage2_started = time.time()

    logger.info(
        "Stage 2: validating %d candidates on 15m/1H/4H...",
        len(selected),
    )

    mtf_valid = []

    score_valid = []

    with ThreadPoolExecutor(
        max_workers=CONFIG[
            "workers_stage2"
        ]
    ) as pool:

        futures = [
            pool.submit(
                stage2_worker,
                item,
            )
            for item in selected
        ]

        for future in as_completed(
            futures
        ):

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

                    score_valid.append(
                        result
                    )

            except Exception as exc:

                logger.debug(
                    "Stage 2 future: %s",
                    exc,
                )

    stage2_elapsed = (
        time.time()
        - stage2_started
    )

    # ========================================================
    # SORT
    # ========================================================

    mtf_valid.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    score_valid.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # ========================================================
    # FINAL SELECTION
    # ========================================================
    #
    # Prefer >= min_score.
    #
    # If fewer than 2 exist, use strongest valid
    # MTF candidates.
    #
    # Never manufacture candidates.
    # ========================================================

    if len(score_valid) >= CONFIG[
        "min_candidates"
    ]:

        final_results = score_valid[
            :CONFIG["max_results"]
        ]

        selection_mode = (
            "min_score"
        )

    else:

        final_results = mtf_valid[
            :CONFIG["max_results"]
        ]

        selection_mode = (
            "mtf_fallback"
        )

    # ========================================================
    # TOTAL
    # ========================================================

    total_elapsed = (
        time.time()
        - started
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

        "selection_mode":
            selection_mode,

        "config": {

            "timeframes":
                TFS,

            "ema_period":
                CONFIG["ema_period"],

            "volume_ratio_min":
                CONFIG["volume_ratio_min"],

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

            "risk_reward":
                CONFIG["risk_reward"],

            "min_score":
                CONFIG["min_score"],

            "min_candidates":
                CONFIG["min_candidates"],

            "max_results":
                CONFIG["max_results"],

            "momentum_pool":
                CONFIG["momentum_pool"],

            "closed_candle_signals":
                True,

            "visualization_owner":
                "vSch.py",

            "arrows_in_synaptic":
                False,

            "markers_in_synaptic":
                False,
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
                len(score_valid),

            "final_candidates":
                len(final_results),

            "stage1_seconds":
                round(
                    stage1_elapsed,
                    2,
                ),

            "stage2_seconds":
                round(
                    stage2_elapsed,
                    2,
                ),

            "elapsed_seconds":
                round(
                    total_elapsed,
                    2,
                ),

            "workers_stage1":
                CONFIG[
                    "workers_stage1"
                ],

            "workers_stage2":
                CONFIG[
                    "workers_stage2"
                ],
        },

        "candidates":
            final_results,
    }

    # ========================================================
    # SAVE
    # ========================================================

    output_path = Path(
        args.out
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE
    # ========================================================

    print()
    print("=" * 78)
    print("SYNAPTIC SCAN RESULTS")
    print("=" * 78)

    if not final_results:

        print(
            "No valid candidates found."
        )

    else:

        for item in final_results:

            print(
                f"{item['symbol']} "
                f"{item['side']} | "
                f"{item['setup']} | "
                f"{item['setup_status']} | "
                f"Score {item['score']:.2f} | "
                f"TF {item['tf_agreement']}/3"
            )

            print(
                f"  Entry : "
                f"{item['entry']:.8g}"
            )

            print(
                f"  Zone  : "
                f"{item['entry_zone'][0]:.8g}"
                f" - "
                f"{item['entry_zone'][1]:.8g}"
            )

            print(
                f"  SL    : "
                f"{item['sl']:.8g}"
            )

            print(
                f"  TP    : "
                f"{item['tp'][0]:.8g}, "
                f"{item['tp'][1]:.8g}, "
                f"{item['tp'][2]:.8g}"
            )

            print(
                f"  Trigger: "
                f"{item.get('entry_trigger')}"
            )

            print()

    print("=" * 78)
    print(
        f"Universe        : "
        f"{len(universe_rows)}"
    )
    print(
        f"Stage 1         : "
        f"{stage1_elapsed:.2f}s"
    )
    print(
        f"Stage 2         : "
        f"{stage2_elapsed:.2f}s"
    )
    print(
        f"Total           : "
        f"{total_elapsed:.2f}s"
    )
    print(
        f"MTF valid       : "
        f"{len(mtf_valid)}"
    )
    print(
        f"Score valid     : "
        f"{len(score_valid)}"
    )
    print(
        f"Final candidates: "
        f"{len(final_results)}"
    )
    print(
        f"Selection       : "
        f"{selection_mode}"
    )
    print(
        f"Output          : "
        f"{output_path}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()