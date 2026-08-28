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
# CONSTANTS
# ============================================================

TFS = ["15m", "1h", "4h"]

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

    # 0 = seluruh active USDT perpetuals
    "universe_size": 0,


    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    # Kandidat Stage 1 yang diteruskan ke MTF.
    "momentum_pool": 60,

    # Cukup untuk EMA200 + indikator lain.
    "klines": 240,


    # --------------------------------------------------------
    # CONCURRENCY
    # --------------------------------------------------------

    # Naik dari 8.
    "workers_stage1": 24,

    # Stage 2 melakukan request 1H/4H sebagai pekerjaan
    # terpisah, sehingga concurrency efektif lebih baik.
    "workers_stage2": 24,


    # --------------------------------------------------------
    # SELECTION
    # --------------------------------------------------------

    "min_score": 6.0,
    "min_candidates": 2,
    "max_results": 5,


    # --------------------------------------------------------
    # INDICATORS
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
    # MOMENTUM
    # --------------------------------------------------------

    "momentum_fast_bars": 4,
    "momentum_slow_bars": 16,


    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    "swing_window": 8,

    "risk_reward": [
        1.5,
        2.25,
        3.0,
    ],


    # --------------------------------------------------------
    # NETWORK
    # --------------------------------------------------------

    "request_timeout": 8,

    "request_retries": 2,

    # Jangan terlalu agresif jika Binance sedang throttling.
    "retry_backoff": 0.35,


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
# BINANCE ENDPOINTS
# ============================================================

BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]


HEADERS = {
    "User-Agent": "Synaptic/2.0",
    "Accept": "application/json",
}


# ============================================================
# THREAD LOCAL SESSION
# ============================================================

_thread_local = threading.local()


def get_session():
    """
    One requests.Session per worker thread.

    Avoids sharing one Session object between dozens
    of concurrent workers.
    """

    session = getattr(_thread_local, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)

        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
            max_retries=0,
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        _thread_local.session = session

    return session


# ============================================================
# BINANCE API
# ============================================================

def api(path, params=None, timeout=None):
    """
    Fast REST request with short retry/backoff.

    Important:
    - No shared Session.
    - No global ACTIVE_BASE_URL.
    - 429 handled with Retry-After when available.
    - Endpoint fallback only when genuinely necessary.
    """

    if timeout is None:
        timeout = CONFIG["request_timeout"]

    session = get_session()

    last_error = None

    for base_index, base_url in enumerate(BASE_URLS):

        for attempt in range(CONFIG["request_retries"] + 1):

            try:
                response = session.get(
                    base_url + path,
                    params=params,
                    timeout=timeout,
                )

                status = response.status_code

                # ------------------------------------------------
                # SUCCESS
                # ------------------------------------------------

                if status == 200:

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
                    else:
                        return data

                # ------------------------------------------------
                # RATE LIMIT
                # ------------------------------------------------

                elif status in (418, 429):

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    try:
                        wait = float(retry_after)
                    except (TypeError, ValueError):
                        wait = (
                            CONFIG["retry_backoff"]
                            * (attempt + 1)
                        )

                    wait += random.uniform(0.05, 0.20)

                    last_error = f"HTTP {status}"

                    if attempt < CONFIG["request_retries"]:
                        time.sleep(min(wait, 3.0))
                        continue

                    # Move to next endpoint only after retries.
                    break

                # ------------------------------------------------
                # FORBIDDEN / GEO
                # ------------------------------------------------

                elif status == 451:

                    last_error = "HTTP 451"
                    break

                # ------------------------------------------------
                # OTHER HTTP ERROR
                # ------------------------------------------------

                else:

                    last_error = f"HTTP {status}"

                    if attempt < CONFIG["request_retries"]:

                        wait = (
                            CONFIG["retry_backoff"]
                            * (attempt + 1)
                        )

                        time.sleep(wait)
                        continue

                    break

            except (
                requests.Timeout,
                requests.ConnectionError,
            ) as exc:

                last_error = str(exc)

                if attempt < CONFIG["request_retries"]:

                    wait = (
                        CONFIG["retry_backoff"]
                        * (attempt + 1)
                    )

                    time.sleep(wait)
                    continue

                break

            except requests.RequestException as exc:

                last_error = str(exc)
                break

            except ValueError as exc:

                last_error = f"Invalid JSON: {exc}"
                break

        # Small delay before endpoint fallback.
        if base_index < len(BASE_URLS) - 1:
            time.sleep(0.05)

    raise RuntimeError(
        f"All Binance endpoints failed: {last_error}"
    )


# ============================================================
# MARKET DATA
# ============================================================

def exchange_info():
    return api(
        "/fapi/v1/exchangeInfo",
        timeout=10,
    )


def ticker_24h():
    return api(
        "/fapi/v1/ticker/24hr",
        timeout=10,
    )


# ============================================================
# UNIVERSE
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
        # Do NOT filter based on 24h percentage change.
        #
        # Universe remains global.
        # Movement/momentum determines candidates later.

        rows.append(
            (
                symbol,
                change_24h,
                quote_volume,
            )
        )

    if CONFIG["universe_size"] > 0:

        # Sort by liquidity only when hard cap is enabled.
        rows.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        rows = rows[:CONFIG["universe_size"]]

    logger.info(
        f"Universe matched {len(rows)} "
        f"active USDT-M perpetual symbols globally."
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
        timeout=CONFIG["request_timeout"],
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
    ).reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            f"{symbol} {interval}: "
            f"only {len(df)} candles"
        )

    return df


# ============================================================
# ATR
# ============================================================

def add_atr(df):

    previous_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / CONFIG["atr_period"],
        adjust=False,
    ).mean()


# ============================================================
# FULL INDICATORS
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
        x["macd"] -
        x["macd_signal"]
    )


    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    x["atr"] = add_atr(x)


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    x["volume_ma"] = x["volume"].rolling(
        CONFIG["volume_ma_period"]
    ).mean()

    x["volume_ratio"] = (
        x["volume"] /
        x["volume_ma"]
    )


    # --------------------------------------------------------
    # SUPERTREND
    # 10 / 2.5
    # --------------------------------------------------------

    period = CONFIG["supertrend_period"]
    multiplier = CONFIG["supertrend_multiplier"]

    hl2 = (
        x["high"] +
        x["low"]
    ) / 2.0

    basic_upper = (
        hl2 +
        multiplier * x["atr"]
    )

    basic_lower = (
        hl2 -
        multiplier * x["atr"]
    )

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

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

    for i in range(1, len(x)):

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

        if direction.iloc[i - 1] == -1:

            if (
                x["close"].iloc[i]
                > final_upper.iloc[i - 1]
            ):
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        else:

            if (
                x["close"].iloc[i]
                < final_lower.iloc[i - 1]
            ):
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1

        supertrend.iloc[i] = (
            final_lower.iloc[i]
            if direction.iloc[i] > 0
            else final_upper.iloc[i]
        )

    if len(x):

        supertrend.iloc[0] = (
            final_lower.iloc[0]
        )

    x["supertrend"] = supertrend
    x["st_dir"] = direction

    return x


# ============================================================
# LIGHTWEIGHT MOMENTUM SCORING
# ============================================================

def movement_score(df):

    if len(df) < 50:
        return -1.0, None

    x = df

    close = float(
        x["close"].iloc[-1]
    )

    if not np.isfinite(close) or close <= 0:
        return -1.0, None


    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    atr = add_atr(x)

    atr_value = float(
        atr.iloc[-1]
    )

    if (
        not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        return -1.0, None


    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ma = (
        x["volume"]
        .rolling(
            CONFIG["volume_ma_period"]
        )
        .mean()
    )

    volume_ratio = (
        float(
            x["volume"].iloc[-1]
        )
        /
        float(
            volume_ma.iloc[-1]
        )
        if np.isfinite(
            volume_ma.iloc[-1]
        )
        and volume_ma.iloc[-1] > 0
        else 1.0
    )

    volume_bonus = min(
        max(volume_ratio, 0.0),
        4.0,
    )


    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    window = CONFIG["breakout_window"]

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

    breakout_bonus = (
        2.0
        if (
            close > prev_high
            or
            close < prev_low
        )
        else 0.0
    )


    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    direction = (
        1
        if float(x["close"].iloc[-1])
        >= float(x["open"].iloc[-1])
        else -1
    )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = (
        fast_return * 2.0
        +
        slow_return
        +
        min(atr_move, 5.0) * 1.5
        +
        volume_bonus * 1.25
        +
        breakout_bonus
    )

    return float(score), {
        "df": df,
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

    x = df

    if len(x) < 210:
        return None

    last = x.iloc[-1]
    previous = x.iloc[-2]

    long_score = 0.0
    short_score = 0.0

    long_reasons = []
    short_reasons = []


    # --------------------------------------------------------
    # BASIC VALUES
    # --------------------------------------------------------

    close = float(last["close"])
    ema = float(last["ema200"])
    atr_value = float(last["atr"])

    volume_ratio = (
        float(last["volume_ratio"])
        if np.isfinite(last["volume_ratio"])
        else 1.0
    )


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
    # SUPERTREND
    # --------------------------------------------------------

    if int(last["st_dir"]) > 0:

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


    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if volume_ratio >= CONFIG["volume_ratio_min"]:

        if close > float(last["open"]):

            long_score += 1.5

            long_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

        elif close < float(last["open"]):

            short_score += 1.5

            short_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )


    # --------------------------------------------------------
    # BREAKOUT / BREAKDOWN
    # --------------------------------------------------------

    window = CONFIG["breakout_window"]

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

        "st_dir": int(
            last["st_dir"]
        ),

        "macd": macd,

        "macd_signal": macd_signal,

        "macd_hist": hist_now,
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

            item[col] = (
                int(value)
                if col == "st_dir"
                else float(value)
            )

        records.append(item)

    return records


# ============================================================
# STAGE 2 FETCH JOB
# ============================================================

def fetch_stage2_tf(symbol, tf):

    try:

        candles = klines(
            symbol,
            tf,
        )

        enriched = add_indicators(
            candles
        )

        scored = score_tf(
            enriched
        )

        return (
            symbol,
            tf,
            enriched,
            scored,
            None,
        )

    except Exception as exc:

        return (
            symbol,
            tf,
            None,
            None,
            str(exc),
        )


# ============================================================
# ANALYZE SYMBOL
# ============================================================

def analyze_symbol(
    symbol,
    change_24h,
    quote_volume_24h,
    stage1_score,
    stage1_meta,
    stage2_data,
):

    data = {}

    # --------------------------------------------------------
    # 15M
    # --------------------------------------------------------

    try:

        df15 = add_indicators(
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
            f"MTF {symbol} 15m error: {exc}"
        )


    # --------------------------------------------------------
    # 1H / 4H
    # --------------------------------------------------------

    for tf in ("1h", "4h"):

        item = stage2_data.get(tf)

        if not item:
            continue

        enriched = item.get("df")
        scored = item.get("score")

        if enriched is not None and scored:
            data[tf] = {
                "score": scored,
                "df": enriched,
            }


    # --------------------------------------------------------
    # REQUIRE ALL THREE TF
    # --------------------------------------------------------

    if set(data.keys()) != set(TFS):
        return None


    # --------------------------------------------------------
    # MTF WEIGHTS
    # --------------------------------------------------------

    weights = {
        "15m": 0.25,
        "1h": 0.35,
        "4h": 0.40,
    }


    long_total = sum(
        weights[tf]
        *
        data[tf]["score"]["long"]
        for tf in TFS
    )

    short_total = sum(
        weights[tf]
        *
        data[tf]["score"]["short"]
        for tf in TFS
    )


    # --------------------------------------------------------
    # SIDE
    # --------------------------------------------------------

    side = (
        "LONG"
        if long_total > short_total
        else "SHORT"
    )

    raw_score = max(
        long_total,
        short_total,
    )

    wanted_direction = (
        1
        if side == "LONG"
        else -1
    )


    # --------------------------------------------------------
    # TIMEFRAME AGREEMENT
    # --------------------------------------------------------

    votes = []

    for tf in TFS:

        tf_score = data[tf]["score"]

        if (
            tf_score["long"]
            ==
            tf_score["short"]
        ):

            votes.append(0)

        else:

            votes.append(
                1
                if (
                    tf_score["long"]
                    >
                    tf_score["short"]
                )
                else -1
            )


    agreement = sum(
        vote == wanted_direction
        for vote in votes
    )


    # Minimum 2/3 agreement.
    if agreement < 2:
        return None


    # --------------------------------------------------------
    # 15M MOMENTUM DIRECTION
    # --------------------------------------------------------

    current_close = float(
        data["15m"]["df"]
        .iloc[-1]["close"]
    )

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

    reference_close = float(
        data["15m"]["df"]
        .iloc[-1 - fast_n]["close"]
    )

    move_15 = (
        current_close
        /
        reference_close
        -
        1.0
    ) * 100


    if side == "LONG" and move_15 <= 0:
        return None

    if side == "SHORT" and move_15 >= 0:
        return None


    # --------------------------------------------------------
    # EXECUTION TIMEFRAME
    # --------------------------------------------------------

    tf_rank = {
        "1h": 3,
        "4h": 2,
        "15m": 1,
    }

    tf_candidates = []

    for tf in TFS:

        tf_score = (
            data[tf]["score"]["long"]
            if side == "LONG"
            else data[tf]["score"]["short"]
        )

        tf_candidates.append(
            (
                float(tf_score),
                tf_rank[tf],
                tf,
            )
        )


    _, _, execution_tf = max(
        tf_candidates
    )


    # --------------------------------------------------------
    # EXECUTION DATA
    # --------------------------------------------------------

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
        or
        not np.isfinite(atr_value)
        or
        atr_value <= 0
    ):
        return None


    # --------------------------------------------------------
    # SWING / SL
    # --------------------------------------------------------

    swing_n = CONFIG[
        "swing_window"
    ]

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

    entry = price


    if side == "LONG":

        sl = min(
            swing_low,
            entry - 1.25 * atr_value,
        )

        risk = entry - sl

        invalidation = (
            f"Close below "
            f"{sl:.8g} / loss of recent "
            f"{execution_tf} swing low"
        )

    else:

        sl = max(
            swing_high,
            entry + 1.25 * atr_value,
        )

        risk = sl - entry

        invalidation = (
            f"Close above "
            f"{sl:.8g} / reclaim of recent "
            f"{execution_tf} swing high"
        )


    if risk <= 0:
        return None


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_pct = (
        risk / entry
    ) * 100


    if risk_pct > 8.0:
        return None


    # --------------------------------------------------------
    # TARGETS
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
    # MOMENTUM BONUS
    # --------------------------------------------------------

    momentum_bonus = min(
        stage1_score / 25.0,
        1.5,
    )

    score = (
        raw_score
        +
        momentum_bonus
    )


    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    reasons = (
        data["15m"]["score"]["long_reasons"]
        if side == "LONG"
        else data["15m"]["score"]["short_reasons"]
    )


    # --------------------------------------------------------
    # CHART DATA
    # --------------------------------------------------------

    chart_data = {
        tf: serialize_chart_data(
            data[tf]["df"]
        )
        for tf in TFS
    }


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

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

            "visible_candles": (
                CONFIG["visible_candles"]
            ),

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
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Synaptic - fast multi-timeframe "
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


    if not universe_rows:

        logger.warning(
            "Universe is empty."
        )

        Path(args.out).write_text(
            json.dumps(
                {
                    "candidates": [],
                    "error": "Empty universe",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return


    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_started = time.time()

    logger.info(
        f"Scanning {len(universe_rows)} symbols "
        f"on 15m (Stage 1) "
        f"with {CONFIG['workers_stage1']} workers..."
    )


    momentum = []

    completed = 0
    total = len(universe_rows)


    with ThreadPoolExecutor(
        max_workers=CONFIG["workers_stage1"]
    ) as pool:

        jobs = {
            pool.submit(
                klines,
                symbol,
                "15m",
            ): (
                symbol,
                chg,
                q_vol,
            )
            for symbol, chg, q_vol
            in universe_rows
        }


        for future in as_completed(jobs):

            symbol, chg, q_vol = jobs[
                future
            ]

            completed += 1

            try:

                candles = future.result()

                score, meta = (
                    movement_score(
                        candles
                    )
                )

                if (
                    score > 0
                    and meta is not None
                ):

                    momentum.append(
                        (
                            score,
                            symbol,
                            chg,
                            q_vol,
                            meta,
                        )
                    )

            except Exception as exc:

                logger.debug(
                    f"15m scan error on "
                    f"{symbol}: {exc}"
                )


            if (
                completed % 100 == 0
                or completed == total
            ):

                logger.info(
                    f"Stage 1 progress "
                    f"{completed}/{total}"
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
        -
        stage1_started
    )


    logger.info(
        f"Stage 1 completed in "
        f"{stage1_elapsed:.2f}s | "
        f"momentum={len(momentum)} | "
        f"selected={len(selected)}"
    )


    # ========================================================
    # STAGE 2 — PARALLEL MTF FETCH
    # ========================================================

    stage2_started = time.time()

    logger.info(
        f"Stage 2 starting: "
        f"{len(selected)} symbols × "
        f"2 TF = "
        f"{len(selected) * 2} requests"
    )


    stage2_cache = {
        symbol: {}
        for _, symbol, _, _, _
        in selected
    }


    total_stage2_jobs = (
        len(selected) * 2
    )

    completed_stage2 = 0


    with ThreadPoolExecutor(
        max_workers=CONFIG["workers_stage2"]
    ) as pool:

        jobs = {}

        for (
            stage_score,
            symbol,
            chg,
            q_vol,
            meta,
        ) in selected:

            for tf in ("1h", "4h"):

                future = pool.submit(
                    fetch_stage2_tf,
                    symbol,
                    tf,
                )

                jobs[future] = (
                    symbol,
                    tf,
                )


        for future in as_completed(jobs):

            symbol, tf = jobs[
                future
            ]

            completed_stage2 += 1

            try:

                (
                    result_symbol,
                    result_tf,
                    enriched,
                    scored,
                    error,
                ) = future.result()


                if (
                    error is None
                    and enriched is not None
                    and scored is not None
                ):

                    stage2_cache[
                        result_symbol
                    ][result_tf] = {
                        "df": enriched,
                        "score": scored,
                    }

            except Exception as exc:

                logger.debug(
                    f"Stage 2 {symbol} "
                    f"{tf} error: {exc}"
                )


            if (
                completed_stage2 % 20 == 0
                or
                completed_stage2
                == total_stage2_jobs
            ):

                logger.info(
                    f"Stage 2 fetch "
                    f"{completed_stage2}/"
                    f"{total_stage2_jobs}"
                )


    stage2_fetch_elapsed = (
        time.time()
        -
        stage2_started
    )


    logger.info(
        f"Stage 2 data fetch completed "
        f"in {stage2_fetch_elapsed:.2f}s"
    )


    # ========================================================
    # STAGE 2 — MTF VALIDATION
    # ========================================================

    results = []
    mtf_valid = []


    for (
        stage_score,
        symbol,
        chg,
        q_vol,
        meta,
    ) in selected:

        try:

            result = analyze_symbol(
                symbol=symbol,
                change_24h=chg,
                quote_volume_24h=q_vol,
                stage1_score=stage_score,
                stage1_meta=meta,
                stage2_data=stage2_cache.get(
                    symbol,
                    {},
                ),
            )


            if result is None:
                continue


            mtf_valid.append(
                result
            )


            if (
                result["score"]
                >=
                CONFIG["min_score"]
            ):

                results.append(
                    result
                )


        except Exception as exc:

            logger.debug(
                f"MTF validation error "
                f"on {symbol}: {exc}"
            )


    stage2_elapsed = (
        time.time()
        -
        stage2_started
    )


    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    mtf_valid.sort(
        key=lambda item: item["score"],
        reverse=True,
    )


    # ========================================================
    # FINAL SELECTION
    # ========================================================

    final_results = results[
        :CONFIG["max_results"]
    ]

    selection_mode = "min_score"


    # Transparent fallback:
    # If fewer than 2 reach min_score,
    # use strongest MTF-valid candidates.
    if (
        len(final_results)
        <
        CONFIG["min_candidates"]
    ):

        final_results = mtf_valid[
            :CONFIG["max_results"]
        ]

        selection_mode = (
            "mtf_fallback"
        )


    # ========================================================
    # TIMING
    # ========================================================

    elapsed = (
        time.time()
        -
        started
    )


    logger.info(
        f"Stage 2 MTF-valid: "
        f"{len(mtf_valid)} | "
        f"min-score valid: "
        f"{len(results)} | "
        f"selection={selection_mode}"
    )


    logger.info(
        f"Stage 2 completed in "
        f"{stage2_elapsed:.2f}s"
    )


    logger.info(
        f"TOTAL SCAN TIME: "
        f"{elapsed:.2f}s "
        f"({elapsed / 60:.2f} min)"
    )


    logger.info(
        f"Scan completed. "
        f"Found {len(final_results)} "
        f"valid candidates."
    )


    # ========================================================
    # PAYLOAD
    # ========================================================

    payload = {

        "generated_at": (
            pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
        ),

        "scanner": "Synaptic",

        "scanner_version": "2.0-fast",

        "selection_mode": selection_mode,

        "scan_stats": {

            "universe": len(
                universe_rows
            ),

            "stage1_selected": len(
                selected
            ),

            "momentum_pool_size": len(
                momentum
            ),

            "mtf_valid": len(
                mtf_valid
            ),

            "min_score_valid": len(
                results
            ),

            "final_candidates": len(
                final_results
            ),

            "stage1_seconds": round(
                stage1_elapsed,
                2,
            ),

            "stage2_fetch_seconds": round(
                stage2_fetch_elapsed,
                2,
            ),

            "stage2_total_seconds": round(
                stage2_elapsed,
                2,
            ),

            "elapsed_seconds": round(
                elapsed,
                2,
            ),
        },

        "config": {

            "ema_period": CONFIG[
                "ema_period"
            ],

            "volume_ma_period": CONFIG[
                "volume_ma_period"
            ],

            "volume_ratio_min": CONFIG[
                "volume_ratio_min"
            ],

            "macd": [
                CONFIG["macd_fast"],
                CONFIG["macd_slow"],
                CONFIG["macd_signal"],
            ],

            "supertrend": [
                CONFIG["supertrend_period"],
                CONFIG["supertrend_multiplier"],
            ],

            "mtf": TFS,

            "min_tf_agreement": 2,

            "min_score": CONFIG[
                "min_score"
            ],

            "max_results": CONFIG[
                "max_results"
            ],
        },

        "candidates": final_results,
    }


    # ========================================================
    # WRITE OUTPUT
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
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print(
        f"SYNAPTIC SCAN | "
        f"{elapsed / 60:.2f} min"
    )
    print("=" * 78)


    if not final_results:

        print(
            "No valid candidates."
        )

    else:

        for item in final_results:

            print(
                f"{item['symbol']} "
                f"{item['side']} | "
                f"Score {item['score']:.2f} | "
                f"TF "
                f"{item['tf_agreement']}/3 | "
                f"Exec "
                f"{item['execution_tf']} | "
                f"Entry "
                f"{item['entry']:.8g} | "
                f"SL "
                f"{item['sl']:.8g}"
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