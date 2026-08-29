#!/usr/bin/env python3
"""
Synaptic - Multi-Timeframe Binance Futures Scanner (Optimized Version)
Version: 2.0 (Supertrend Optimized + Refactored + Typed Config)
Author: vSynapse (based on original)
"""

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable,
    Literal, TypedDict
)

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

IGNORED_SYMBOLS: set[str] = {
    "USDCUSDT", "BUSDUSDT", "DAIUSDT", "TUSDUSDT", "FDUSDUSDT",
    "USDPUSDT", "EURUSDT", "USTCUSDT", "PAXGUSDT",
}


# ============================================================
# CONFIGURATION (Typed & Extended)
# ============================================================

class Config(TypedDict):
    """Extended typed configuration for better maintainability."""

    # Universe
    min_quote_volume_24h: int
    universe_size: int

    # Stage 1
    momentum_pool: int
    klines: int
    workers_stage1: int

    # Stage 2
    workers_stage2: int

    # Candidate selection
    min_score: float
    min_candidates: int
    max_results: int

    # Indicators
    ema_period: int
    volume_ma_period: int
    volume_ratio_min: float

    macd_fast: int
    macd_slow: int
    macd_signal: int

    supertrend_period: int
    supertrend_multiplier: float

    atr_period: int
    breakout_window: int

    # Momentum
    momentum_fast_bars: int
    momentum_slow_bars: int

    # Structure / risk
    swing_window: int

    risk_reward: list[float]

    # Setup Engine
    pullback_ema_atr_lower: float
    pullback_ema_atr_upper: float
    extended_atr: float

    # API
    api_timeout: int
    api_retries: int
    retry_base_delay: float

    # Chart defaults
    visible_candles: dict[str, int]


CONFIG: Config = {
    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------
    "min_quote_volume_24h": 500_000,
    "universe_size": 0,  # 0 = all active USDT perpetuals

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
    # Structure / risk
    # --------------------------------------------------------
    "swing_window": 8,

    "risk_reward": [1.5, 2.25, 3.0],

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

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


_thread_local = threading.local()


def get_session() -> requests.Session:
    """Thread-local session to avoid global session contention."""
    session = getattr(_thread_local, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session

    return session


# ============================================================
# API ENGINE
# ============================================================

def _retry_delay(
    attempt: int,
    response: Optional[requests.Response] = None
) -> float:

    if response is not None:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                value = float(retry_after)
                return min(max(value, 0.2), 5.0)
            except (TypeError, ValueError):
                pass

    base = CONFIG["retry_base_delay"]
    delay = base * (2 ** attempt)
    delay += random.uniform(0.05, 0.20)

    return min(delay, 3.0)


def _parse_response(
    response: requests.Response
) -> Optional[Union[Dict, List]]:

    try:
        return response.json()
    except ValueError:
        return None


def api(
    path: str,
    params: Optional[dict] = None,
    timeout: Optional[int] = None,
) -> Any:
    """Central Binance API request with full retry logic."""

    if timeout is None:
        timeout = CONFIG["api_timeout"]

    session = get_session()
    last_error = None

    for base_url in BASE_URLS:

        url = base_url + path

        for attempt in range(CONFIG["api_retries"] + 1):

            try:

                response = session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )

                status = response.status_code

                if status == 200:

                    data = _parse_response(response)

                    if data is None:
                        last_error = (
                            f"{base_url} HTTP 200 but invalid JSON"
                        )
                        break

                    if (
                        isinstance(data, dict)
                        and "code" in data
                        and "msg" in data
                    ):
                        last_error = (
                            f"{data.get('code')}: {data.get('msg')}"
                        )

                        if attempt < CONFIG["api_retries"]:
                            time.sleep(
                                _retry_delay(attempt, response)
                            )
                            continue

                        break

                    return data

                if status == 202:

                    data = _parse_response(response)

                    if isinstance(data, (dict, list)) and not (
                        isinstance(data, dict)
                        and "code" in data
                        and "msg" in data
                    ):
                        return data

                    last_error = f"{base_url} HTTP 202"

                    if attempt < CONFIG["api_retries"]:
                        time.sleep(
                            _retry_delay(attempt, response)
                        )
                        continue

                    break

                if status in (418, 429):

                    last_error = f"{base_url} HTTP {status}"

                    if attempt < CONFIG["api_retries"]:
                        time.sleep(
                            _retry_delay(attempt, response)
                        )
                        continue

                    break

                if status == 451:

                    last_error = f"{base_url} HTTP 451"
                    break

                last_error = f"{base_url} HTTP {status}"

                if attempt < CONFIG["api_retries"]:
                    time.sleep(
                        _retry_delay(attempt, response)
                    )
                    continue

                break

            except requests.Timeout:

                last_error = f"{base_url} timeout"

                if attempt < CONFIG["api_retries"]:
                    time.sleep(_retry_delay(attempt))
                    continue

                break

            except requests.RequestException as exc:

                last_error = f"{base_url}: {exc}"

                if attempt < CONFIG["api_retries"]:
                    time.sleep(_retry_delay(attempt))
                    continue

                break

    raise RuntimeError(
        f"All Binance endpoints failed: {last_error}"
    )


# ============================================================
# BASIC ENDPOINTS
# ============================================================

def exchange_info() -> dict:
    return api(
        "/fapi/v1/exchangeInfo",
        timeout=15
    )


def ticker_24h() -> list:
    return api(
        "/fapi/v1/ticker/24hr",
        timeout=15
    )


# ============================================================
# UNIVERSE
# ============================================================

def universe() -> list[tuple[str, float, float]]:

    started = time.time()

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

        rows.append(
            (
                symbol,
                change_24h,
                quote_volume,
            )
        )

    if CONFIG["universe_size"] > 0:
        rows = rows[:CONFIG["universe_size"]]

    elapsed = time.time() - started

    logger.info(
        "Universe matched %d active USDT-M perpetual symbols globally (%.2fs).",
        len(rows),
        elapsed,
    )

    return rows


# ============================================================
# SUPER TREND
# ============================================================

def calculate_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard SuperTrend calculation.

    Perbaikan hanya pada perhitungan True Range:
    semua array TR sekarang memiliki panjang yang sama.

    Tidak mengubah endpoint atau logic scanner lainnya.
    """

    n = len(high)

    if n < period + 1:
        return (
            np.full(n, np.nan),
            np.full(n, 0, dtype=int),
            np.full(n, np.nan),
        )

    # --------------------------------------------------------
    # Previous close
    # --------------------------------------------------------

    previous_close = np.empty(
        n,
        dtype=float
    )

    previous_close[0] = close[0]
    previous_close[1:] = close[:-1]

    # --------------------------------------------------------
    # True Range
    # --------------------------------------------------------

    tr0 = high - low

    tr1 = np.abs(
        high - previous_close
    )

    tr2 = np.abs(
        low - previous_close
    )

    tr = np.maximum.reduce(
        [tr0, tr1, tr2]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr = (
        pd.Series(tr)
        .ewm(
            alpha=1 / period,
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    # --------------------------------------------------------
    # Basic bands
    # --------------------------------------------------------

    hl2 = (
        high + low
    ) / 2.0

    basic_upper = (
        hl2
        + multiplier * atr
    )

    basic_lower = (
        hl2
        - multiplier * atr
    )

    # --------------------------------------------------------
    # Final bands
    # --------------------------------------------------------

    final_upper = np.full(
        n,
        np.nan,
        dtype=float
    )

    final_lower = np.full(
        n,
        np.nan,
        dtype=float
    )

    direction = np.full(
        n,
        1,
        dtype=int
    )

    supertrend_line = np.full(
        n,
        np.nan,
        dtype=float
    )

    # Initial values
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]

    direction[0] = 1

    supertrend_line[0] = (
        final_lower[0]
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    for i in range(1, n):

        # Final upper
        if (
            basic_upper[i] < final_upper[i - 1]
            or close[i - 1] > final_upper[i - 1]
        ):
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower
        if (
            basic_lower[i] > final_lower[i - 1]
            or close[i - 1] < final_lower[i - 1]
        ):
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction
        if direction[i - 1] == 1:

            if (
                close[i] < final_lower[i - 1]
                and close[i] < final_lower[i]
            ):
                direction[i] = -1
            else:
                direction[i] = 1

        else:

            if (
                close[i] > final_upper[i - 1]
                and close[i] > final_upper[i]
            ):
                direction[i] = 1
            else:
                direction[i] = -1

        # Supertrend line
        if direction[i] > 0:
            supertrend_line[i] = final_lower[i]
        else:
            supertrend_line[i] = final_upper[i]

    return (
        supertrend_line,
        direction,
        atr,
    )


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(
    df: pd.DataFrame
) -> pd.DataFrame:

    x = df.copy()

    # EMA 200
    x["ema200"] = (
        x["close"]
        .ewm(
            span=CONFIG["ema_period"],
            adjust=False
        )
        .mean()
    )

    # MACD
    fast = (
        x["close"]
        .ewm(
            span=CONFIG["macd_fast"],
            adjust=False
        )
        .mean()
    )

    slow = (
        x["close"]
        .ewm(
            span=CONFIG["macd_slow"],
            adjust=False
        )
        .mean()
    )

    x["macd"] = fast - slow

    x["macd_signal"] = (
        x["macd"]
        .ewm(
            span=CONFIG["macd_signal"],
            adjust=False
        )
        .mean()
    )

    x["macd_hist"] = (
        x["macd"]
        - x["macd_signal"]
    )

    # ATR
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
            adjust=False
        )
        .mean()
    )

    # Volume
    x["volume_ma"] = (
        x["volume"]
        .rolling(
            CONFIG["volume_ma_period"],
            min_periods=1
        )
        .mean()
    )

    x["volume_ratio"] = (
        x["volume"]
        / x["volume_ma"].replace(
            0,
            np.nan
        )
    )

    # SuperTrend
    high = x["high"].to_numpy()
    low = x["low"].to_numpy()
    close = x["close"].to_numpy()

    (
        st_line,
        st_dir,
        atr_st
    ) = calculate_supertrend(
        high,
        low,
        close,
        CONFIG["supertrend_period"],
        CONFIG["supertrend_multiplier"],
    )

    x["supertrend"] = st_line
    x["st_dir"] = st_dir

    return x


# ============================================================
# CLOSED CANDLES
# ============================================================

def closed_candles(
    df: pd.DataFrame
) -> pd.DataFrame:

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

def determine_setup(
    df: pd.DataFrame,
    side: Literal["LONG", "SHORT"]
) -> dict[str, Any]:

    if df is None or len(df) < 210:
        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "insufficient candles",
        }

    x = df.copy()
    last = x.iloc[-1]

    close = float(last["close"])
    ema = float(last["ema200"])
    atr = float(last["atr"])
    st_dir = int(last["st_dir"])
    macd = float(last["macd"])
    macd_signal = float(last["macd_signal"])
    macd_hist = float(last["macd_hist"])
    volume_ratio = float(last["volume_ratio"])

    values = [
        close,
        ema,
        atr,
        macd,
        macd_signal,
        macd_hist,
        volume_ratio,
    ]

    if not all(np.isfinite(v) for v in values):
        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "invalid indicator data",
        }

    if close <= 0 or ema <= 0 or atr <= 0:
        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "invalid indicator data",
        }

    # Directional alignment
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
                "EMA200 / Supertrend / MACD "
                "alignment failed"
            ),
        }

    volume_confirmed = (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    )

    # Breakout history
    window = CONFIG["breakout_window"]

    if len(x) <= window + 1:
        return {
            "setup": "NO_SETUP",
            "status": "NO_SETUP",
            "entry_zone": None,
            "reason": "insufficient breakout history",
        }

    previous_high = (
        x["high"]
        .iloc[-window - 1:-1]
        .max()
    )

    previous_low = (
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

    # Distance from EMA200
    distance_from_ema = abs(
        close - ema
    )

    distance_atr = (
        distance_from_ema / atr
    )

    # Pullback zone
    pullback_lower = (
        CONFIG["pullback_ema_atr_lower"]
    )

    pullback_upper = (
        CONFIG["pullback_ema_atr_upper"]
    )

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

    zone_low, zone_high = (
        min(zone_low, zone_high),
        max(zone_low, zone_high),
    )

    in_pullback_zone = (
        zone_low
        <= close
        <= zone_high
    )

    # Extended
    extended_limit = (
        CONFIG["extended_atr"]
    )

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

    # Breakout / Breakdown
    if side == "LONG" and breakout_long:

        if volume_confirmed:

            return {
                "setup": "BREAKOUT",
                "status": "READY",
                "entry_zone": [
                    float(close),
                    float(close),
                ],
                "reason": (
                    "1H 20-bar breakout "
                    "with volume confirmation"
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
                "1H breakout without volume "
                "confirmation; wait for pullback"
            ),
        }

    if side == "SHORT" and breakout_short:

        if volume_confirmed:

            return {
                "setup": "BREAKDOWN",
                "status": "READY",
                "entry_zone": [
                    float(close),
                    float(close),
                ],
                "reason": (
                    "1H 20-bar breakdown "
                    "with volume confirmation"
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
                "1H breakdown without volume "
                "confirmation; wait for pullback"
            ),
        }

    # Pullback
    if in_pullback_zone:

        if side == "LONG":

            valid_structure = (
                close > ema
                and st_dir > 0
                and macd > macd_signal
            )

        else:

            valid_structure = (
                close < ema
                and st_dir < 0
                and macd < macd_signal
            )

        if valid_structure:

            return {
                "setup": "PULLBACK",
                "status": "READY",
                "entry_zone": [
                    float(zone_low),
                    float(zone_high),
                ],
                "reason": (
                    "price returned to EMA200 "
                    "pullback zone with trend alignment"
                ),
            }

    # Continuation
    if side == "LONG":

        momentum_ok = (
            close > last["open"]
            and macd_hist >= 0
        )

    else:

        momentum_ok = (
            close < last["open"]
            and macd_hist <= 0
        )

    if momentum_ok and volume_confirmed:

        return {
            "setup": "CONTINUATION",
            "status": "READY",
            "entry_zone": [
                float(close),
                float(close),
            ],
            "reason": (
                "trend aligned with momentum "
                "and volume confirmation"
            ),
        }

    # Trend aligned but no clean entry
    return {
        "setup": "CONTINUATION",
        "status": "WAITING PULLBACK",
        "entry_zone": [
            float(zone_low),
            float(zone_high),
        ],
        "reason": (
            "trend aligned but momentum confirmation "
            "is insufficient; wait for pullback"
        ),
    }


# ============================================================
# CHART SERIALIZATION
# ============================================================

def serialize_chart_data(
    df: pd.DataFrame
) -> list[dict[str, Any]]:

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

            if pd.isna(value):
                item[col] = None
                continue

            if col == "time":

                item[col] = (
                    None
                    if pd.isna(value)
                    else pd.Timestamp(value).isoformat()
                )

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

def movement_score(
    df: pd.DataFrame
) -> Tuple[
    float,
    Optional[dict[str, Any]]
]:

    if len(df) < 50:
        return -1.0, None

    if "ema200" not in df.columns:
        x = add_indicators(df)
    else:
        x = df

    last = x.iloc[-1]

    close = float(last["close"])
    atr_value = float(last["atr"])

    if (
        not np.isfinite(close)
        or close <= 0
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return -1.0, None

    fast_n = CONFIG["momentum_fast_bars"]
    slow_n = CONFIG["momentum_slow_bars"]

    if len(x) <= slow_n + 2:
        return -1.0, None

    fast_ref = float(
        x["close"].iloc[-1 - fast_n]
    )

    slow_ref = float(
        x["close"].iloc[-1 - slow_n]
    )

    if fast_ref <= 0 or slow_ref <= 0:
        return -1.0, None

    fast_return = (
        abs(close / fast_ref - 1.0)
        * 100
    )

    slow_return = (
        abs(close / slow_ref - 1.0)
        * 100
    )

    atr_move = (
        abs(close - fast_ref)
        / atr_value
    )

    volume_ratio = float(
        last.get("volume_ratio", 1.0)
    )

    volume_bonus = min(
        max(volume_ratio, 0.0),
        4.0
    )

    window = CONFIG["breakout_window"]

    if len(x) <= window + 1:
        return -1.0, None

    prev_high = (
        x["high"]
        .iloc[-window - 1:-1]
        .max()
    )

    prev_low = (
        x["low"]
        .iloc[-window - 1:-1]
        .min()
    )

    breakout_bonus = (
        2.0
        if (
            close > prev_high
            or close < prev_low
        )
        else 0.0
    )

    direction = (
        1
        if close >= last["open"]
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

def score_tf(
    df: pd.DataFrame
) -> Optional[dict[str, Any]]:

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

    long_reasons: list[str] = []
    short_reasons: list[str] = []

    close = float(last["close"])
    ema = float(last["ema200"])
    atr_value = float(last["atr"])

    volume_ratio = float(
        last.get("volume_ratio", 1.0)
    )

    if (
        not np.isfinite(close)
        or not np.isfinite(ema)
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return None

    # EMA 200
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

    # Supertrend
    st_dir = int(last["st_dir"])

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

    # MACD
    macd = float(last["macd"])
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

    # Volume
    if (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    ):

        if close > last["open"]:

            long_score += 1.5
            long_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

        elif close < last["open"]:

            short_score += 1.5
            short_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

    # Breakout
    window = CONFIG["breakout_window"]

    if len(x) > window + 1:

        previous_high = (
            x["high"]
            .iloc[-window - 1:-1]
            .max()
        )

        previous_low = (
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
        "long": round(long_score, 3),
        "short": round(short_score, 3),
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
    symbol: str,
    change_24h: float,
    quote_volume_24h: float,
    stage1_score: float,
    stage1_meta: dict,
) -> Optional[dict[str, Any]]:

    data: dict[str, Any] = {}

    # --------------------------------------------------------
    # 15m MTF
    # --------------------------------------------------------

    try:

        df15 = closed_candles(
            stage1_meta["df"]
        )

        scored_15m = score_tf(df15)

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

    for tf in ["1h", "4h"]:

        try:

            candles = klines(
                symbol,
                tf
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

    if set(data.keys()) != set(TFS):
        return None

    # --------------------------------------------------------
    # 4H Bias
    # --------------------------------------------------------

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

    if not long_4h_bias and not short_4h_bias:
        return None

    side = (
        "LONG"
        if long_4h_bias
        else "SHORT"
    )

    # --------------------------------------------------------
    # 1H Setup
    # --------------------------------------------------------

    setup_1h = determine_setup(
        data["1h"]["df"],
        side
    )

    if setup_1h["setup"] == "NO_SETUP":
        return None

    setup_status = setup_1h["status"]

    # --------------------------------------------------------
    # 15m Execution Filters
    # --------------------------------------------------------

    df15 = data["15m"]["df"]
    last15 = df15.iloc[-1]

    current_close = float(
        last15["close"]
    )

    fast_n = CONFIG["momentum_fast_bars"]

    if len(df15) <= fast_n + 1:
        return None

    reference_close = float(
        df15.iloc[-1 - fast_n]["close"]
    )

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

    if side == "LONG":

        execution_aligned = (
            current_close > ema15
            and st15 > 0
            and macd15 > signal15
            and move_15 > 0
            and volume_ratio_15
            >= CONFIG["volume_ratio_min"]
        )

    else:

        execution_aligned = (
            current_close < ema15
            and st15 < 0
            and macd15 < signal15
            and move_15 < 0
            and volume_ratio_15
            >= CONFIG["volume_ratio_min"]
        )

    if not execution_aligned:
        return None

    if setup_1h["setup"] == "EXTENDED":

        setup_status = "WAITING PULLBACK"

    elif setup_1h["setup"] == "PULLBACK":

        setup_status = "READY"

    else:

        setup_status = "READY"

    # --------------------------------------------------------
    # MTF Score
    # --------------------------------------------------------

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

    if agreement < 2:
        return None

    # --------------------------------------------------------
    # Entry / Risk / TP
    # --------------------------------------------------------

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

    entry_zone = setup_1h[
        "entry_zone"
    ]

    if entry_zone is None:
        return None

    if setup_status == "WAITING PULLBACK":

        entry = (
            entry_zone[0]
            + entry_zone[1]
        ) / 2.0

    else:

        entry = price

    if (
        not np.isfinite(entry)
        or entry <= 0
    ):
        return None

    # --------------------------------------------------------
    # Swing
    # --------------------------------------------------------

    swing_n = CONFIG["swing_window"]

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

    # --------------------------------------------------------
    # Stop Loss
    # --------------------------------------------------------

    if side == "LONG":

        sl = min(
            swing_low,
            entry - 1.25 * atr_value
        )

        risk = entry - sl

        invalidation = (
            f"Close below {sl:.8g} / "
            f"loss of recent {execution_tf} "
            "swing low"
        )

    else:

        sl = max(
            swing_high,
            entry + 1.25 * atr_value
        )

        risk = sl - entry

        invalidation = (
            f"Close above {sl:.8g} / "
            f"reclaim of recent {execution_tf} "
            "swing high"
        )

    if risk <= 0:
        return None

    risk_pct = (
        risk / entry
    ) * 100

    if risk_pct > 8.0:
        return None

    # --------------------------------------------------------
    # Take Profit
    # --------------------------------------------------------

    tp = [
        (
            entry + risk * rr
            if side == "LONG"
            else entry - risk * rr
        )
        for rr in CONFIG["risk_reward"]
    ]

    # --------------------------------------------------------
    # Final Score
    # --------------------------------------------------------

    momentum_bonus = min(
        stage1_score / 25.0,
        1.5
    )

    score = (
        raw_score
        + momentum_bonus
    )

    reasons = (
        data["15m"]["score"]["long_reasons"]
        if side == "LONG"
        else data["15m"]["score"]["short_reasons"]
    )

    # --------------------------------------------------------
    # Chart
    # --------------------------------------------------------

    chart_data = {
        tf: serialize_chart_data(
            data[tf]["df"]
        )
        for tf in TFS
    }

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "symbol": symbol,
        "side": side,
        "score": round(score, 2),
        "change24h": round(change_24h, 2),
        "quote_volume24h": round(
            quote_volume_24h,
            2
        ),
        "execution_tf": execution_tf,
        "setup": setup_1h["setup"],
        "setup_status": setup_status,
        "setup_reason": setup_1h["reason"],
        "entry_zone": [
            float(entry_zone[0]),
            float(entry_zone[1])
        ],
        "timeframes": {
            tf: data[tf]["score"]
            for tf in TFS
        },
        "momentum_15m": round(
            move_15,
            3
        ),
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "risk_pct": round(
            risk_pct,
            3
        ),
        "invalidation": invalidation,
        "key_points": reasons[:6],
        "tf_agreement": agreement,
        "chart": {
            "execution_tf": execution_tf,
            "available_timeframes": TFS,
            "analysis_candles": CONFIG["klines"],
            "visible_candles": CONFIG["visible_candles"],
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

def stage1_worker(
    row: tuple[str, float, float]
) -> Optional[
    tuple[
        float,
        str,
        float,
        float,
        dict
    ]
]:

    symbol, change_24h, quote_volume = row

    try:

        candles = klines(
            symbol,
            "15m"
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

        if score <= 0 or meta is None:
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

def stage2_worker(
    item: tuple[
        float,
        str,
        float,
        float,
        dict
    ]
) -> Optional[dict[str, Any]]:

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
# KLINES
# ============================================================

def klines(
    symbol: str,
    interval: str
) -> pd.DataFrame:

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
        columns=columns
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
            errors="coerce"
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Synaptic multi-timeframe "
            "Binance Futures scanner "
            "(optimized)"
        )
    )

    parser.add_argument(
        "--out",
        default="synaptic_candidates.json"
    )

    args = parser.parse_args()

    started = time.time()

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    try:

        universe_rows = universe()

    except Exception as exc:

        logger.error(
            "Cannot build universe: %s",
            exc,
        )

        Path(args.out).write_text(
            json.dumps(
                {
                    "candidates": [],
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        raise

    if not universe_rows:
        raise RuntimeError(
            "Universe is empty."
        )

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    stage1_started = time.time()

    logger.info(
        "Scanning %d symbols on 15m "
        "(Stage 1, workers=%d)...",
        len(universe_rows),
        CONFIG["workers_stage1"],
    )

    momentum = []

    with ThreadPoolExecutor(
        max_workers=CONFIG["workers_stage1"]
    ) as pool:

        futures = [
            pool.submit(
                stage1_worker,
                row
            )
            for row in universe_rows
        ]

        for future in as_completed(
            futures
        ):

            try:

                result = future.result()

                if result is not None:
                    momentum.append(result)

            except Exception as exc:

                logger.debug(
                    "Stage 1 future error: %s",
                    exc,
                )

    momentum.sort(
        key=lambda row: row[0],
        reverse=True
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

        payload = {
            "generated_at": (
                pd.Timestamp
                .now(tz="UTC")
                .isoformat()
            ),
            "scanner": "Synaptic",
            "selection_mode": (
                "no_stage1_candidates"
            ),
            "scan_stats": {
                "universe": len(
                    universe_rows
                ),
                "stage1_selected": 0,
                "mtf_valid": 0,
                "min_score_valid": 0,
                "final_candidates": 0,
                "stage1_seconds": round(
                    stage1_elapsed,
                    2
                ),
                "elapsed_seconds": round(
                    time.time() - started,
                    2
                ),
            },
            "candidates": [],
        }

        Path(args.out).write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    stage2_started = time.time()

    logger.info(
        "Selected %d symbols for Stage 2 "
        "MTF validation (workers=%d)...",
        len(selected),
        CONFIG["workers_stage2"],
    )

    results = []
    mtf_valid = []

    with ThreadPoolExecutor(
        max_workers=CONFIG["workers_stage2"]
    ) as pool:

        futures = [
            pool.submit(
                stage2_worker,
                item
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

                mtf_valid.append(result)

                if (
                    result["score"]
                    >= CONFIG["min_score"]
                ):
                    results.append(result)

            except Exception as exc:

                logger.debug(
                    "Stage 2 future error: %s",
                    exc,
                )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    final_candidates = results[
        :CONFIG["max_results"]
    ]

    elapsed_total = (
        time.time()
        - started
    )

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    payload = {
        "generated_at": (
            pd.Timestamp
            .now(tz="UTC")
            .isoformat()
        ),
        "scanner": "Synaptic",
        "scan_stats": {
            "universe": len(
                universe_rows
            ),
            "stage1_selected": len(
                selected
            ),
            "mtf_valid": len(
                mtf_valid
            ),
            "final_candidates": len(
                final_candidates
            ),
            "elapsed_seconds": round(
                elapsed_total,
                2
            ),
        },
        "candidates": final_candidates,
    }

    Path(args.out).write_text(
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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()