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
    # Tidak perlu mengambil ratusan candle tambahan.
    "klines": 240,

    # Concurrency Stage 1.
    # Sengaja tidak dibuat ekstrem agar tidak memukul
    # Binance rate limit.
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
    # Setup engine
    # --------------------------------------------------------

    # Pullback: jarak maksimum close ke EMA200 (dalam ATR)
    # agar tetap dianggap zona retracement yang valid.
    "setup_pullback_atr": 1.0,

    # Extended: jarak minimum close ke EMA200 (dalam ATR)
    # sebelum dianggap terlalu jauh / overextended.
    "setup_extended_atr": 3.0,

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
    # API
    # --------------------------------------------------------

    "api_timeout": 10,

    # Retry cepat.
    # Tujuannya mencegah GitHub Actions tertahan lama.
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
#
# PENTING:
# Endpoint TIDAK DIUBAH dari kode sebelumnya.
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

    Ini menghindari penggunaan requests.Session global
    secara bersamaan oleh banyak thread.
    """

    session = getattr(_thread_local, "session", None)

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

    Retry-After digunakan bila Binance mengirimkannya.
    """

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

    # Jitter kecil supaya banyak worker tidak retry
    # pada milidetik yang sama.
    delay += random.uniform(0.05, 0.20)

    return min(delay, 3.0)


def _parse_response(response):
    """
    Parse response JSON dengan aman.

    HTTP 202 tidak otomatis dianggap sukses.
    Kalau body-nya valid JSON Binance, kita tetap proses.
    """

    try:
        return response.json()
    except ValueError:
        return None


def api(path, params=None, timeout=None):
    """
    Central Binance API request.

    Endpoint tetap menggunakan BASE_URLS yang lama.

    Perbaikan:
    - session per thread
    - retry pendek
    - HTTP 202 tidak langsung fatal
    - 429/418 ditangani khusus
    - timeout dibatasi
    - endpoint berikutnya dicoba hanya setelah retry lokal
    """

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

                # ------------------------------------------------
                # Normal response
                # ------------------------------------------------

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
                            f"{data.get('code')}: "
                            f"{data.get('msg')}"
                        )

                        if attempt < CONFIG["api_retries"]:
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
                #
                # Jangan langsung menganggap endpoint mati.
                #
                # Beberapa gateway/proxy dapat mengembalikan
                # 202 sementara request masih diproses.
                # ------------------------------------------------

                if status == 202:

                    data = _parse_response(response)

                    if isinstance(data, (dict, list)):

                        # Jika ternyata body sudah berisi data
                        # Binance yang bisa dipakai, gunakan.
                        if not (
                            isinstance(data, dict)
                            and "code" in data
                            and "msg" in data
                        ):
                            return data

                    last_error = (
                        f"{base_url} HTTP 202"
                    )

                    if attempt < CONFIG["api_retries"]:
                        time.sleep(
                            _retry_delay(
                                attempt,
                                response,
                            )
                        )
                        continue

                    # Setelah retry lokal habis,
                    # baru pindah endpoint.
                    break

                # ------------------------------------------------
                # Rate limit
                # ------------------------------------------------

                if status in (418, 429):

                    last_error = (
                        f"{base_url} HTTP {status}"
                    )

                    if attempt < CONFIG["api_retries"]:
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

                    # Tidak perlu retry berkali-kali
                    break

                # ------------------------------------------------
                # Other HTTP errors
                # ------------------------------------------------

                last_error = (
                    f"{base_url} HTTP {status}"
                )

                if attempt < CONFIG["api_retries"]:
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

                if attempt < CONFIG["api_retries"]:
                    time.sleep(
                        _retry_delay(attempt)
                    )
                    continue

                break

            except requests.RequestException as exc:

                last_error = (
                    f"{base_url}: {exc}"
                )

                if attempt < CONFIG["api_retries"]:
                    time.sleep(
                        _retry_delay(attempt)
                    )
                    continue

                break

    raise RuntimeError(
        f"All Binance endpoints failed: {last_error}"
    )


# ============================================================
# BASIC API ENDPOINTS
#
# PATH TETAP SAMA.
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

        if (
            quote_volume
            < CONFIG["min_quote_volume_24h"]
        ):
            continue

        if last_price <= 0:
            continue

        # IMPORTANT:
        #
        # Tidak menggunakan change24h sebagai filter
        # trending.
        #
        # Semua liquid USDT perpetual tetap masuk.
        #
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

    elapsed = time.time() - started

    logger.info(
        "Universe matched %d active USDT-M "
        "perpetual symbols globally "
        "(%.2fs).",
        len(rows),
        elapsed,
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
    ).reset_index(drop=True)

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
        x["macd"] - x["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"] - previous_close
            ).abs(),
            (
                x["low"] - previous_close
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
        / x["volume_ma"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Supertrend 10 / 2.5
    # --------------------------------------------------------

    period = CONFIG["supertrend_period"]
    multiplier = CONFIG[
        "supertrend_multiplier"
    ]

    hl2 = (
        x["high"] + x["low"]
    ) / 2.0

    basic_upper = (
        hl2 + multiplier * x["atr"]
    )

    basic_lower = (
        hl2 - multiplier * x["atr"]
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

    # IMPORTANT:
    # df sudah bisa berupa data yang sudah dihitung indikatornya.
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
        last["volume_ratio"]
    )

    if not np.isfinite(volume_ratio):
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

    close = float(last["close"])
    ema = float(last["ema200"])
    atr_value = float(last["atr"])

    if not np.isfinite(close):
        return None

    if not np.isfinite(ema):
        return None

    if not np.isfinite(atr_value) or atr_value <= 0:
        return None

    volume_ratio = float(
        last["volume_ratio"]
    )

    if not np.isfinite(volume_ratio):
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
    # Volume
    # --------------------------------------------------------

    if (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    ):

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
# SETUP ENGINE
#
# Mengklasifikasikan price action pada execution_tf menjadi
# salah satu dari 5 tipe setup, murni dari data yang sudah
# dihitung oleh score_tf() (EMA200, Supertrend, ATR, reasons).
#
# Tidak ada indikator baru yang diperkenalkan di sini.
# ============================================================

def classify_setup(tf_score, side):

    reasons = (
        tf_score["long_reasons"]
        if side == "LONG"
        else tf_score["short_reasons"]
    )

    is_breakout = any(
        "breakout" in reason or "breakdown" in reason
        for reason in reasons
    )

    close = float(tf_score["close"])
    ema = float(tf_score["ema200"])
    atr_value = float(tf_score["atr"])
    st_dir = int(tf_score["st_dir"])

    if not np.isfinite(atr_value) or atr_value <= 0:
        return "NO_SETUP"

    trend_aligned = (
        (side == "LONG" and close > ema and st_dir > 0)
        or
        (side == "SHORT" and close < ema and st_dir < 0)
    )

    # --------------------------------------------------------
    # Jarak ke EMA200 dalam ATR, BERTANDA searah sisi trade.
    #
    # Positif  -> close berada di sisi yang menguntungkan
    #             posisi (di atas EMA untuk LONG, di bawah
    #             EMA untuk SHORT).
    # Negatif  -> close berada di sisi yang berlawanan arah
    #             (mis. LONG tapi close di bawah EMA200).
    #
    # PENTING: EXTENDED wajib pakai jarak bertanda ini, bukan
    # jarak absolut. Kalau pakai abs(), symbol yang close-nya
    # jauh di sisi BERLAWANAN arah trade bisa salah kelabeli
    # EXTENDED padahal seharusnya NO_SETUP.
    # --------------------------------------------------------

    directional_distance_atr = (
        (close - ema) / atr_value
        if side == "LONG"
        else (ema - close) / atr_value
    )

    # --------------------------------------------------------
    # Prioritas: breakout paling eksplisit, lalu extended
    # (terlalu jauh dari EMA200 SEARAH posisi), baru
    # pullback / continuation.
    # --------------------------------------------------------

    if is_breakout:
        return "BREAKOUT"

    if directional_distance_atr >= CONFIG["setup_extended_atr"]:
        return "EXTENDED"

    if (
        trend_aligned
        and directional_distance_atr <= CONFIG["setup_pullback_atr"]
    ):
        return "PULLBACK"

    if trend_aligned:
        return "CONTINUATION"

    return "NO_SETUP"


# ============================================================
# ENTRY LOGIC
#
# Menentukan entry price berdasarkan tipe setup dari
# Setup Engine di atas.
# ============================================================

def build_entry(setup_style, side, price, exec_df, atr_value):

    # PULLBACK: entry mengacu ke level EMA200 execution_tf
    # (zona retracement), dibatasi agar tidak menyimpang
    # jauh dari harga pasar saat ini.
    #
    # BREAKOUT / CONTINUATION / EXTENDED: entry mengikuti
    # harga pasar saat ini, karena price action sudah
    # bergerak sesuai arah sinyal.

    if setup_style != "PULLBACK":
        return price

    ema_level = float(
        exec_df.iloc[-1]["ema200"]
    )

    max_drift = 0.5 * atr_value

    if side == "LONG":
        return max(ema_level, price - max_drift)

    return min(ema_level, price + max_drift)


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

    try:

        df15 = stage1_meta["df"]

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
                "MTF %s %s error: %s",
                symbol,
                tf,
                exc,
            )

    # Semua timeframe wajib tersedia.
    if set(data.keys()) != set(TFS):
        return None

    # --------------------------------------------------------
    # MTF weights
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

    if long_total > short_total:
        side = "LONG"
    else:
        side = "SHORT"

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
    # MTF agreement
    # --------------------------------------------------------

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
    # 15m momentum direction
    # --------------------------------------------------------

    df15 = data["15m"]["df"]

    current_close = float(
        df15.iloc[-1]["close"]
    )

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

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

    if (
        side == "LONG"
        and move_15 <= 0
    ):
        return None

    if (
        side == "SHORT"
        and move_15 >= 0
    ):
        return None

    # --------------------------------------------------------
    # Select execution timeframe
    #
    # Strongest directional timeframe.
    # Tie -> prefer 4H, then 1H, then 15m.
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

    # --------------------------------------------------------
    # SETUP ENGINE
    #
    # Klasifikasi kondisi price action pada execution_tf
    # menjadi salah satu dari:
    # BREAKOUT / PULLBACK / CONTINUATION / EXTENDED / NO_SETUP.
    #
    # Tidak mengarang sinyal: NO_SETUP berarti tidak ada
    # struktur entry yang jelas, kandidat ditolak di sini.
    # --------------------------------------------------------

    exec_score = data[execution_tf]["score"]

    setup_style = classify_setup(
        exec_score,
        side,
    )

    if setup_style == "NO_SETUP":
        return None

    # --------------------------------------------------------
    # ENTRY LOGIC
    #
    # Entry price ditentukan berdasarkan tipe setup dari
    # Setup Engine di atas.
    # --------------------------------------------------------

    entry = build_entry(
        setup_style,
        side,
        price,
        exec_df,
        atr_value,
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

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

        risk = entry - sl

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

        risk = sl - entry

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

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    tp = [
        (
            entry + risk * rr
            if side == "LONG"
            else
            entry - risk * rr
        )
        for rr in CONFIG["risk_reward"]
    ]

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

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
        else data["15m"]["score"][
            "short_reasons"
        ]
    )

    # --------------------------------------------------------
    # Chart data
    # --------------------------------------------------------

    chart_data = {
        tf: serialize_chart_data(
            data[tf]["df"]
        )
        for tf in TFS
    }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {

        "symbol": symbol,

        "side": side,

        "setup_style": setup_style,

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

            # 1H kembali 48 candle.
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

        score, meta = movement_score(
            enriched
        )

        if (
            score <= 0
            or meta is None
        ):
            return None

        # Pastikan metadata menggunakan
        # dataframe yang sudah dihitung.
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
        CONFIG["workers_stage1"],
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
                    momentum.append(result)

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

            "scanner": "Synaptic",

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

    # ========================================================
    # STAGE 2
    # ========================================================

    stage2_started = time.time()

    logger.info(
        "Selected %d symbols for "
        "Stage 2 MTF validation "
        "(workers=%d)...",
        len(selected),
        CONFIG["workers_stage2"],
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
                    "Stage 2 future error: %s",
                    exc,
                )

    stage2_elapsed = (
        time.time()
        - stage2_started
    )

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

    # Transparent fallback.
    #
    # Jika kandidat >=2 tidak mencapai min_score,
    # gunakan kandidat MTF-valid terkuat.
    #
    # Tidak mengarang signal.
    #
    if (
        len(final_results)
        < CONFIG["min_candidates"]
    ):

        final_results = mtf_valid[
            :CONFIG["max_results"]
        ]

        selection_mode = (
            "mtf_fallback"
        )

    # ========================================================
    # FINAL TIMING
    # ========================================================

    total_elapsed = (
        time.time()
        - started
    )

    logger.info(
        "Stage 2 completed in %.2fs",
        stage2_elapsed,
    )

    logger.info(
        "Stage 2 MTF-valid: %d | "
        "min-score valid: %d | "
        "selection=%s",
        len(mtf_valid),
        len(results),
        selection_mode,
    )

    logger.info(
        "Scan completed in %.2fs | "
        "Found %d valid candidates.",
        total_elapsed,
        len(final_results),
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
                CONFIG["workers_stage1"],

            "workers_stage2":
                CONFIG["workers_stage2"],
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
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "SYNAPTIC SCAN RESULTS"
    )
    print("=" * 72)

    if not final_results:

        print(
            "No valid candidates found."
        )

    else:

        for item in final_results:

            print(
                f"{item['symbol']} "
                f"{item['side']} | "
                f"Score "
                f"{item['score']:.2f} | "
                f"TF "
                f"{item['tf_agreement']}/3 | "
                f"EXEC "
                f"{item['execution_tf']} | "
                f"Entry "
                f"{item['entry']:.8g} | "
                f"SL "
                f"{item['sl']:.8g}"
            )

    print("=" * 72)

    print(
        f"Universe       : "
        f"{len(universe_rows)}"
    )

    print(
        f"Stage 1        : "
        f"{stage1_elapsed:.2f}s"
    )

    print(
        f"Stage 2        : "
        f"{stage2_elapsed:.2f}s"
    )

    print(
        f"Total           : "
        f"{total_elapsed:.2f}s"
    )

    print(
        f"Candidates      : "
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

    print("=" * 72)

    logger.info(
        "Output successfully saved to: %s",
        output_path,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()