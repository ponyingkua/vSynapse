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

    # Cukup untuk EMA200 + indikator lain.
    "klines": 240,

    # Concurrency Stage 1.
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
    # Structure / risk
    # --------------------------------------------------------

    "swing_window": 8,

    "risk_reward": [
        1.5,
        2.25,
        3.0,
    ],

    # --------------------------------------------------------
    # Setup Engine
    # --------------------------------------------------------

    "pullback_ema_atr_lower": 0.50,
    "pullback_ema_atr_upper": 1.00,
    "extended_atr": 2.50,

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
    """
    Setiap worker thread mempunyai Session sendiri.
    """

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
    """
    Delay pendek dan terkontrol.
    """

    if response is not None:
        retry_after = response.headers.get(
            "Retry-After"
        )

        if retry_after:
            try:
                value = float(retry_after)
                return min(
                    max(value, 0.2),
                    5.0,
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    base = CONFIG["retry_base_delay"]

    delay = base * (2 ** attempt)

    delay += random.uniform(
        0.05,
        0.20,
    )

    return min(
        delay,
        3.0,
    )


def _parse_response(response):
    """
    Parse response JSON dengan aman.
    """

    try:
        return response.json()
    except ValueError:
        return None


def api(path, params=None, timeout=None):
    """
    Central Binance API request.
    """

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
                # Normal response
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
                # Other HTTP errors
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
# BASIC API ENDPOINTS
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

    period = CONFIG[
        "supertrend_period"
    ]

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
    """
    Setup engine hanya menggunakan candle yang sudah close.
    """

    if len(df) <= 1:
        return df.copy()

    return (
        df.iloc[:-1]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# SETUP ENGINE
# ============================================================

def determine_setup(df, side):
    """
    Setup Engine 1H.

    Klasifikasi:

        BREAKOUT
        PULLBACK
        CONTINUATION
        EXTENDED
        NO_SETUP

    Urutan prioritas:

        1. Validasi data
        2. Trend alignment
        3. Extended detection
        4. Breakout detection
        5. Pullback detection
        6. Continuation detection
        7. No setup

    Prinsip:

        - EMA200 menentukan posisi trend.
        - Supertrend 10/2.5 menentukan arah trend.
        - MACD menentukan momentum.
        - Volume digunakan untuk confirmation.
        - Breakout harus melewati high/low sebelumnya.
        - Pullback harus benar-benar menunjukkan retracement.
        - Continuation tidak boleh sekadar candle hijau/merah biasa.
        - Harga terlalu jauh dari EMA200 tidak dikejar.
    """

    if (
        df is None
        or len(df) < 210
    ):

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "insufficient candles",
        }

    x = df.copy()

    last = x.iloc[-1]

    previous = x.iloc[-2]

    close = float(
        last["close"]
    )

    open_price = float(
        last["open"]
    )

    high = float(
        last["high"]
    )

    low = float(
        last["low"]
    )

    ema = float(
        last["ema200"]
    )

    atr = float(
        last["atr"]
    )

    st_dir = int(
        last["st_dir"]
    )

    macd = float(
        last["macd"]
    )

    macd_signal = float(
        last["macd_signal"]
    )

    macd_hist = float(
        last["macd_hist"]
    )

    previous_macd_hist = float(
        previous["macd_hist"]
    )

    volume_ratio = float(
        last["volume_ratio"]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    values = [
        close,
        open_price,
        high,
        low,
        ema,
        atr,
        macd,
        macd_signal,
        macd_hist,
        previous_macd_hist,
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
            "reason": "invalid price or ATR",
        }

    # --------------------------------------------------------
    # DIRECTIONAL ALIGNMENT
    # --------------------------------------------------------

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

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": (
                "EMA200 / Supertrend / "
                "MACD alignment failed"
            ),
        }

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_confirmed = (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    )

    # --------------------------------------------------------
    # BREAKOUT HISTORY
    # --------------------------------------------------------

    window = CONFIG[
        "breakout_window"
    ]

    if len(x) <= window + 2:

        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": (
                "insufficient breakout history"
            ),
        }

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

    # --------------------------------------------------------
    # DISTANCE FROM EMA200
    # --------------------------------------------------------

    distance_from_ema = abs(
        close - ema
    )

    distance_atr = (
        distance_from_ema / atr
    )

    # --------------------------------------------------------
    # PULLBACK ZONE
    # --------------------------------------------------------

    pullback_lower = CONFIG[
        "pullback_ema_atr_lower"
    ]

    pullback_upper = CONFIG[
        "pullback_ema_atr_upper"
    ]

    if side == "LONG":

        zone_low = (
            ema
            - pullback_lower * atr
        )

        zone_high = (
            ema
            + pullback_upper * atr
        )

    else:

        zone_low = (
            ema
            - pullback_upper * atr
        )

        zone_high = (
            ema
            + pullback_lower * atr
        )

    zone_low = min(
        zone_low,
        zone_high,
    )

    zone_high = max(
        zone_low,
        zone_high,
    )

    in_pullback_zone = (
        zone_low
        <= close
        <= zone_high
    )

    # --------------------------------------------------------
    # EXTENDED
    # --------------------------------------------------------
    #
    # Harga terlalu jauh dari EMA200.
    # Jangan chase harga.
    # --------------------------------------------------------

    extended_limit = CONFIG[
        "extended_atr"
    ]

    if distance_atr > extended_limit:

        return {
            "setup": "EXTENDED",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(zone_low),
                float(zone_high),
            ],
            "reason": (
                f"price {distance_atr:.2f} ATR "
                "from EMA200; wait for pullback"
            ),
        }

    # --------------------------------------------------------
    # BREAKOUT ENGINE
    # --------------------------------------------------------
    #
    # Breakout harus mempunyai:
    #
    # - close melewati 20-bar level
    # - candle searah
    # - volume confirmation
    # - MACD tetap aligned
    #
    # Jika breakout terjadi tetapi volume belum cukup,
    # jangan langsung menganggap READY.
    # --------------------------------------------------------

    if side == "LONG" and breakout_long:

        breakout_candle_ok = (
            close > open_price
        )

        breakout_momentum_ok = (
            macd_hist >= 0
            and macd_hist
            >= previous_macd_hist
        )

        if (
            breakout_candle_ok
            and breakout_momentum_ok
            and volume_confirmed
        ):

            return {
                "setup": "BREAKOUT",
                "status": "READY",
                "entry_zone": [
                    float(close),
                    float(close),
                ],
                "reason": (
                    "1H 20-bar breakout with "
                    "bullish candle, momentum "
                    "and volume confirmation"
                ),
            }

        return {
            "setup": "BREAKOUT",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(zone_low),
                float(zone_high),
            ],
            "reason": (
                "1H breakout detected but "
                "confirmation is incomplete; "
                "wait for pullback"
            ),
        }

    if side == "SHORT" and breakout_short:

        breakout_candle_ok = (
            close < open_price
        )

        breakout_momentum_ok = (
            macd_hist <= 0
            and macd_hist
            <= previous_macd_hist
        )

        if (
            breakout_candle_ok
            and breakout_momentum_ok
            and volume_confirmed
        ):

            return {
                "setup": "BREAKDOWN",
                "status": "READY",
                "entry_zone": [
                    float(close),
                    float(close),
                ],
                "reason": (
                    "1H 20-bar breakdown with "
                    "bearish candle, momentum "
                    "and volume confirmation"
                ),
            }

        return {
            "setup": "BREAKDOWN",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(zone_low),
                float(zone_high),
            ],
            "reason": (
                "1H breakdown detected but "
                "confirmation is incomplete; "
                "wait for pullback"
            ),
        }

    # --------------------------------------------------------
    # PULLBACK ENGINE
    # --------------------------------------------------------
    #
    # Tidak cukup hanya:
    #
    #     close berada dekat EMA200
    #
    # Harus ada retracement nyata dari recent impulse.
    # --------------------------------------------------------

    lookback = max(
        CONFIG["swing_window"],
        CONFIG["momentum_fast_bars"],
        6,
    )

    if len(x) >= lookback + 3:

        recent = x.iloc[
            -lookback - 1:-1
        ]

        recent_high = float(
            recent["high"].max()
        )

        recent_low = float(
            recent["low"].min()
        )

    else:

        recent_high = previous_high
        recent_low = previous_low

    # --------------------------------------------------------
    # LONG PULLBACK
    # --------------------------------------------------------

    if side == "LONG":

        retracement_size = (
            recent_high - close
        )

        retracement_atr = (
            retracement_size / atr
        )

        recent_impulse = (
            recent_high - recent_low
        )

        impulse_atr = (
            recent_impulse / atr
            if atr > 0
            else 0.0
        )

        pullback_retrace = (
            retracement_atr >= 0.50
            and recent_impulse > atr
        )

        zone_interaction = (
            low <= zone_high
            and close >= zone_low
        )

        bullish_rejection = (
            close > open_price
            or close > float(
                previous["close"]
            )
        )

        pullback_momentum = (
            macd_hist >= 0
            or macd_hist
            > previous_macd_hist
        )

        valid_pullback = (
            in_pullback_zone
            and pullback_retrace
            and zone_interaction
            and bullish_rejection
            and pullback_momentum
        )

        if valid_pullback:

            return {
                "setup": "PULLBACK",
                "status": "READY",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "1H bullish retracement reached "
                    "EMA200 pullback zone with "
                    "structure and momentum intact"
                ),
            }

        # Harga sudah memasuki area tetapi belum
        # memberikan confirmation yang cukup.
        if (
            in_pullback_zone
            and zone_interaction
            and pullback_retrace
        ):

            return {
                "setup": "PULLBACK",
                "status": "WAITING PULLBACK",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "price retraced into the "
                    "EMA200 pullback area but "
                    "bullish confirmation is incomplete"
                ),
            }

    # --------------------------------------------------------
    # SHORT PULLBACK
    # --------------------------------------------------------

    else:

        retracement_size = (
            close - recent_low
        )

        retracement_atr = (
            retracement_size / atr
        )

        recent_impulse = (
            recent_high - recent_low
        )

        impulse_atr = (
            recent_impulse / atr
            if atr > 0
            else 0.0
        )

        pullback_retrace = (
            retracement_atr >= 0.50
            and recent_impulse > atr
        )

        zone_interaction = (
            high >= zone_low
            and close <= zone_high
        )

        bearish_rejection = (
            close < open_price
            or close < float(
                previous["close"]
            )
        )

        pullback_momentum = (
            macd_hist <= 0
            or macd_hist
            < previous_macd_hist
        )

        valid_pullback = (
            in_pullback_zone
            and pullback_retrace
            and zone_interaction
            and bearish_rejection
            and pullback_momentum
        )

        if valid_pullback:

            return {
                "setup": "PULLBACK",
                "status": "READY",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "1H bearish retracement reached "
                    "EMA200 pullback zone with "
                    "structure and momentum intact"
                ),
            }

        if (
            in_pullback_zone
            and zone_interaction
            and pullback_retrace
        ):

            return {
                "setup": "PULLBACK",
                "status": "WAITING PULLBACK",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "price retraced into the "
                    "EMA200 pullback area but "
                    "bearish confirmation is incomplete"
                ),
            }

    # --------------------------------------------------------
    # CONTINUATION ENGINE
    # --------------------------------------------------------
    #
    # Continuation hanya digunakan jika:
    #
    # - bukan breakout
    # - bukan pullback
    # - harga tetap berada di trend side
    # - momentum tetap searah
    # - candle current searah
    # - momentum histogram mendukung
    #
    # Volume confirmation tetap diperlukan untuk READY.
    # --------------------------------------------------------

    if side == "LONG":

        candle_direction = (
            close > open_price
        )

        previous_close = float(
            previous["close"]
        )

        continuation_price = (
            close > previous_close
        )

        histogram_support = (
            macd_hist >= 0
        )

        histogram_not_collapsing = (
            macd_hist
            >= previous_macd_hist
        )

        continuation_ok = (
            candle_direction
            and continuation_price
            and histogram_support
            and histogram_not_collapsing
            and close > ema
            and st_dir > 0
        )

    else:

        candle_direction = (
            close < open_price
        )

        previous_close = float(
            previous["close"]
        )

        continuation_price = (
            close < previous_close
        )

        histogram_support = (
            macd_hist <= 0
        )

        histogram_not_collapsing = (
            macd_hist
            <= previous_macd_hist
        )

        continuation_ok = (
            candle_direction
            and continuation_price
            and histogram_support
            and histogram_not_collapsing
            and close < ema
            and st_dir < 0
        )

    if continuation_ok:

        if volume_confirmed:

            return {
                "setup": "CONTINUATION",
                "status": "READY",
                "entry_zone": [
                    float(close),
                    float(close),
                ],
                "reason": (
                    "trend continuation confirmed "
                    "by price action, MACD momentum "
                    "and volume"
                ),
            }

        return {
            "setup": "CONTINUATION",
            "status": "WAITING PULLBACK",
            "entry_zone": [
                float(zone_low),
                float(zone_high),
            ],
            "reason": (
                "trend continuation is present "
                "but volume confirmation is weak; "
                "wait for pullback"
            ),
        }

    # --------------------------------------------------------
    # TREND ALIGNED BUT NO CLEAN SETUP
    # --------------------------------------------------------

    return {
        "setup": "NO_SETUP",
        "status": "NO_SETUP",
        "entry_zone": [
            float(zone_low),
            float(zone_high),
        ],
        "reason": (
            "trend aligned but no clean "
            "breakout, pullback or continuation"
        ),
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

    close = float(
        last["close"]
    )

    atr_value = float(
        last["atr"]
    )

    if (
        not np.isfinite(close)
        or close <= 0
        or not np.isfinite(atr_value)
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
            close / fast_ref - 1.0
        )
        * 100
    )

    slow_return = (
        abs(
            close / slow_ref - 1.0
        )
        * 100
    )

    atr_move = (
        abs(
            close - fast_ref
        )
        / atr_value
    )

    volume_ratio = float(
        last["volume_ratio"]
    )

    if not np.isfinite(
        volume_ratio
    ):

        volume_ratio = 1.0

    volume_bonus = min(
        max(volume_ratio, 0.0),
        4.0,
    )

    window = CONFIG[
        "breakout_window"
    ]

    if len(x) <= window + 1:
        return -1.0, None

    prev_high = float(
        x["high"]
        .iloc[-window - 1:-1]
        .max()
    )

    prev_low = float(
        x["low"]
        .iloc[-window - 1:-1]
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
        + min(atr_move, 5.0) * 1.5
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

    close = float(
        last["close"]
    )

    ema = float(
        last["ema200"]
    )

    atr_value = float(
        last["atr"]
    )

    if not np.isfinite(close):
        return None

    if not np.isfinite(ema):
        return None

    if (
        not np.isfinite(atr_value)
        or atr_value <= 0
    ):

        return None

    volume_ratio = float(
        last["volume_ratio"]
    )

    if not np.isfinite(
        volume_ratio
    ):

        volume_ratio = 1.0

    # --------------------------------------------------------
    # EMA 200
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

        "atr": atr_value,

        "volume_ratio": volume_ratio,

        "st_dir": st_dir,

        "macd": macd,

        "macd_signal": macd_signal,

        "macd_hist": hist_now,
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

    # --------------------------------------------------------
    # 15M
    # --------------------------------------------------------

    try:

        df15 = closed_candles(
            stage1_meta["df"]
        )

        scored_15m = score_tf(
            df15
        )

        if scored_15m:

            data["15m"] = {
                "score": scored_15m,
                "df": df15,
            }

    except Exception as exc:

        logger.debug(
            "MTF %s 15m error: %s",
            symbol,
            exc,
        )

    # --------------------------------------------------------
    # 1H + 4H
    # --------------------------------------------------------

    for tf in [
        "1h",
        "4h",
    ]:

        try:

            candles = klines(
                symbol,
                tf,
            )

            candles = closed_candles(
                candles
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
                "MTF %s %s error: %s",
                symbol,
                tf,
                exc,
            )

    # --------------------------------------------------------
    # Semua timeframe wajib tersedia.
    # --------------------------------------------------------

    if set(data.keys()) != set(TFS):
        return None

    # ========================================================
    # MTF SCORING
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
    # LONG / SHORT DIRECTION
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

    long_4h_bias = (
        close4h > ema4h
        and st4h > 0
        and macd4h > signal4h
    )

    short_4h_bias = (
        close4h < ema4h
        and st4h < 0
        and macd4h < signal4h
    )

    # 4H harus mempunyai directional bias.
    if (
        not long_4h_bias
        and not short_4h_bias
    ):

        return None

    # MTF score menentukan arah kandidat.
    if long_total > short_total:

        side = "LONG"

    elif short_total > long_total:

        side = "SHORT"

    else:

        return None

    # 4H bias wajib sama dengan direction MTF.
    if (
        side == "LONG"
        and not long_4h_bias
    ):

        return None

    if (
        side == "SHORT"
        and not short_4h_bias
    ):

        return None

    raw_score = (
        long_total
        if side == "LONG"
        else short_total
    )

    wanted_direction = (
        1
        if side == "LONG"
        else -1
    )

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

    # Minimal 2 dari 3 timeframe harus searah.
    if agreement < 2:
        return None

    # ========================================================
    # SETUP ENGINE
    # ========================================================
    #
    # SEKARANG BARU dipanggil setelah:
    #
    # 4H + 1H + 15M
    #       ↓
    # scoring
    #       ↓
    # long / short
    #       ↓
    # MTF agreement
    #       ↓
    # SETUP ENGINE
    #
    # Sesuai arsitektur pada diagram.
    # ========================================================

    setup_1h = determine_setup(
        data["1h"]["df"],
        side,
    )

    if (
        setup_1h["setup"]
        == "NO_SETUP"
    ):

        return None

    setup_status = (
        setup_1h["status"]
    )

    # ========================================================
    # 15M EXECUTION
    # ========================================================

    df15 = data["15m"]["df"]

    last15 = df15.iloc[-1]

    current_close = float(
        last15["close"]
    )

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

    if len(df15) <= fast_n + 1:
        return None

    reference_close = float(
        df15.iloc[
            -1 - fast_n
        ]["close"]
    )

    if reference_close <= 0:
        return None

    move_15 = (
        current_close
        / reference_close
        - 1.0
    ) * 100

    volume_ratio_15 = float(
        last15["volume_ratio"]
    )

    macd15 = float(
        last15["macd"]
    )

    signal15 = float(
        last15["macd_signal"]
    )

    st15 = int(
        last15["st_dir"]
    )

    ema15 = float(
        last15["ema200"]
    )

    # ========================================================
    # Mandatory 15m execution filters
    # ========================================================

    if side == "LONG":

        execution_aligned = (
            current_close > ema15
            and st15 > 0
            and macd15 > signal15
            and move_15 > 0
            and volume_ratio_15
            >= CONFIG[
                "volume_ratio_min"
            ]
        )

    else:

        execution_aligned = (
            current_close < ema15
            and st15 < 0
            and macd15 < signal15
            and move_15 < 0
            and volume_ratio_15
            >= CONFIG[
                "volume_ratio_min"
            ]
        )

    if not execution_aligned:
        return None

    # ========================================================
    # FINAL SETUP STATUS
    # ========================================================

    if (
        setup_1h["setup"]
        == "EXTENDED"
    ):

        setup_status = (
            "WAITING PULLBACK"
        )

    elif (
        setup_1h["setup"]
        == "PULLBACK"
    ):

        setup_status = "READY"

    elif (
        setup_1h["setup"]
        in (
            "BREAKOUT",
            "BREAKDOWN",
            "CONTINUATION",
        )
    ):

        # Jangan mengubah WAITING PULLBACK
        # dari Setup Engine menjadi READY
        # hanya karena 15m valid.
        if (
            setup_1h["status"]
            == "WAITING PULLBACK"
        ):

            setup_status = (
                "WAITING PULLBACK"
            )

        else:

            setup_status = "READY"

    else:

        setup_status = (
            setup_1h["status"]
        )

    # ========================================================
    # EXECUTION TIMEFRAME
    # ========================================================

    execution_tf = "15m"

    exec_df = data[
        execution_tf
    ]["df"]

    price = float(
        exec_df.iloc[-1]["close"]
    )

    atr_value = float(
        exec_df.iloc[-1]["atr"]
    )

    if (
        not np.isfinite(price)
        or price <= 0
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):

        return None

    # ========================================================
    # ENTRY
    # ========================================================

    entry_zone = (
        setup_1h["entry_zone"]
    )

    if entry_zone is None:
        return None

    if (
        setup_status
        == "WAITING PULLBACK"
    ):

        entry = (
            float(entry_zone[0])
            + float(entry_zone[1])
        ) / 2.0

    else:

        entry = price

    if (
        not np.isfinite(entry)
        or entry <= 0
    ):

        return None

    # ========================================================
    # RISK
    # ========================================================

    swing_n = CONFIG[
        "swing_window"
    ]

    if len(exec_df) < swing_n:
        return None

    swing_low = float(
        exec_df["low"]
        .iloc[-swing_n:]
        .min()
    )

    swing_high = float(
        exec_df["high"]
        .iloc[-swing_n:]
        .max()
    )

    if side == "LONG":

        sl = min(
            swing_low,
            entry - 1.25 * atr_value,
        )

        risk = (
            entry - sl
        )

        invalidation = (
            f"Close below {sl:.8g} / "
            f"loss of recent "
            f"{execution_tf} swing low"
        )

    else:

        sl = max(
            swing_high,
            entry + 1.25 * atr_value,
        )

        risk = (
            sl - entry
        )

        invalidation = (
            f"Close above {sl:.8g} / "
            f"reclaim of recent "
            f"{execution_tf} swing high"
        )

    if risk <= 0:
        return None

    risk_pct = (
        risk / entry
    ) * 100

    if risk_pct > 8.0:
        return None

    # ========================================================
    # TP
    # ========================================================

    tp = [

        (
            entry + risk * rr
            if side == "LONG"
            else
            entry - risk * rr
        )

        for rr in CONFIG[
            "risk_reward"
        ]
    ]

    # ========================================================
    # FINAL SCORE
    # ========================================================

    momentum_bonus = min(
        stage1_score / 25.0,
        1.5,
    )

    score = (
        raw_score
        + momentum_bonus
    )

    reasons = (
        data["15m"]["score"][
            "long_reasons"
        ]
        if side == "LONG"
        else
        data["15m"]["score"][
            "short_reasons"
        ]
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
    # OUTPUT
    # ========================================================

    return {

        "symbol": symbol,

        "side": side,

        "score": round(
            score,
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

        "execution_tf": execution_tf,

        "setup": setup_1h[
            "setup"
        ],

        "setup_status": setup_status,

        "setup_reason": setup_1h[
            "reason"
        ],

        "entry_zone": (
            [
                float(entry_zone[0]),
                float(entry_zone[1]),
            ]
            if entry_zone is not None
            else None
        ),

        "timeframes": {
            tf: data[tf]["score"]
            for tf in TFS
        },

        "momentum_15m": round(
            move_15,
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

        "chart": {

            "execution_tf": execution_tf,

            "available_timeframes": TFS,

            "analysis_candles": CONFIG[
                "klines"
            ],

            "visible_candles": {
                "15m": 60,
                "1h": 48,
                "4h": 50,
            },

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
# STAGE 1 WORKER
# ============================================================

def stage1_worker(row):

    symbol, change_24h, quote_volume = row

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

        enriched = closed_candles(
            enriched
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
            change_24h,
            quote_volume,
            meta,
        )

    except Exception as exc:

        logger.debug(
            "15m scan error on %s: %s",
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
        change_24h,
        quote_volume,
        stage1_meta,
    ) = item

    try:

        return analyze_symbol(
            symbol,
            change_24h,
            quote_volume,
            stage1_score,
            stage1_meta,
        )

    except Exception as exc:

        logger.debug(
            "MTF scan error on %s: %s",
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
            "Synaptic multi-timeframe "
            "Binance Futures scanner"
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
        "Scanning %d symbols on 15m "
        "(Stage 1, workers=%d)...",
        len(universe_rows),
        CONFIG[
            "workers_stage1"
        ],
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

                result = (
                    future.result()
                )

                if result is not None:

                    momentum.append(
                        result
                    )

            except Exception as exc:

                logger.debug(
                    "Stage 1 future error: %s",
                    exc,
                )

    momentum.sort(
        key=lambda row: row[0],
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
        "Stage 1 completed in %.2fs | "
        "momentum=%d | selected=%d",
        stage1_elapsed,
        len(momentum),
        len(selected),
    )

    if not selected:

        logger.warning(
            "No momentum candidates "
            "found in Stage 1."
        )

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

                "stage1_selected":
                    0,

                "mtf_valid":
                    0,

                "min_score_valid":
                    0,

                "final_candidates":
                    0,

                "stage1_seconds":
                    round(
                        stage1_elapsed,
                        2,
                    ),

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
        "Selected %d symbols for "
        "Stage 2 MTF validation "
        "(workers=%d)...",
        len(selected),
        CONFIG[
            "workers_stage2"
        ],
    )

    results = []

    mtf_valid = []

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

                result = (
                    future.result()
                )

                if result is None:
                    continue

                mtf_valid.append(
                    result
                )

                if (
                    result["score"]
                    >= CONFIG[
                        "min_score"
                    ]
                ):

                    results.append(
                        result
                    )

            except Exception as exc:

                logger.debug(
                    "Stage 2 future error: %s",
                    exc,
                )

    # ========================================================
    # FINAL SORT / TOP 5
    # ========================================================

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    final_candidates = results[
        :CONFIG["max_results"]
    ]

    elapsed_total = (
        time.time()
        - started
    )

    payload = {

        "generated_at":
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat(),

        "scanner":
            "Synaptic",

        "scan_stats": {

            "universe":
                len(universe_rows),

            "stage1_selected":
                len(selected),

            "mtf_valid":
                len(mtf_valid),

            "final_candidates":
                len(final_candidates),

            "elapsed_seconds":
                round(
                    elapsed_total,
                    2,
                ),
        },

        "candidates":
            final_candidates,
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

    logger.info(
        "Scan completed successfully. "
        "Saved %d candidates to %s.",
        len(final_candidates),
        args.out,
    )


if __name__ == "__main__":
    main()