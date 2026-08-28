#!/usr/bin/env python3

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests


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


CONFIG = {
    "min_quote_volume_24h": 500_000,
    "universe_size": 0,
    "momentum_pool": 60,
    "klines": 240,
    "workers_stage1": 12,
    "workers_stage2": 8,
    "min_score": 6.0,
    "min_candidates": 2,
    "max_results": 5,

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

    "momentum_fast_bars": 4,
    "momentum_slow_bars": 16,

    "swing_window": 8,

    "risk_reward": [1.5, 2.25, 3.0],
}


HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}


BASE_URLS = [
    "https://www.binance.com",
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]


SESSION = requests.Session()
SESSION.headers.update(HEADERS)

ACTIVE_BASE_URL = None


# ============================================================
# BINANCE API
# ============================================================

def api(path, params=None, timeout=15):
    global ACTIVE_BASE_URL

    endpoints = []

    if ACTIVE_BASE_URL:
        endpoints.append(ACTIVE_BASE_URL)

    for base_url in BASE_URLS:
        if base_url not in endpoints:
            endpoints.append(base_url)

    last_error = None

    for base_url in endpoints:
        try:
            response = SESSION.get(
                base_url + path,
                params=params,
                timeout=timeout,
            )

            if response.status_code == 451:
                last_error = "HTTP 451"
                continue

            if response.status_code in (418, 429):
                last_error = f"HTTP {response.status_code}"
                time.sleep(0.5)
                continue

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
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
                continue

            ACTIVE_BASE_URL = base_url
            return data

        except requests.RequestException as exc:
            last_error = str(exc)

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

        rows.append(
            (
                symbol,
                change_24h,
                quote_volume,
            )
        )

    if CONFIG["universe_size"] > 0:
        rows = rows[:CONFIG["universe_size"]]

    print(
        f"[UNIVERSE] "
        f"{len(rows)} active USDT-M perpetual symbols"
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

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
    )

    return (
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


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):
    x = df.copy()

    x["ema200"] = x["close"].ewm(
        span=CONFIG["ema_period"],
        adjust=False,
    ).mean()

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

    x["volume_ma"] = x["volume"].rolling(
        CONFIG["volume_ma_period"]
    ).mean()

    x["volume_ratio"] = (
        x["volume"] /
        x["volume_ma"]
    )

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
            <
            final_upper.iloc[i - 1]
            or
            x["close"].iloc[i - 1]
            >
            final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]

        else:
            final_upper.iloc[i] = (
                final_upper.iloc[i - 1]
            )

        if (
            basic_lower.iloc[i]
            >
            final_lower.iloc[i - 1]
            or
            x["close"].iloc[i - 1]
            <
            final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]

        else:
            final_lower.iloc[i] = (
                final_lower.iloc[i - 1]
            )

        if (
            direction.iloc[i - 1] == -1
            and
            x["close"].iloc[i]
            >
            final_upper.iloc[i - 1]
        ):
            direction.iloc[i] = 1

        elif (
            direction.iloc[i - 1] == 1
            and
            x["close"].iloc[i]
            <
            final_lower.iloc[i - 1]
        ):
            direction.iloc[i] = -1

        else:
            direction.iloc[i] = (
                direction.iloc[i - 1]
            )

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = (
                final_lower.iloc[i]
            )
        else:
            supertrend.iloc[i] = (
                final_upper.iloc[i]
            )

    if len(x):
        supertrend.iloc[0] = final_lower.iloc[0]

    x["supertrend"] = supertrend
    x["st_dir"] = direction

    return x


# ============================================================
# CHART DATA SERIALIZATION
# ============================================================

def serialize_chart_data(df):
    """
    Converts the analyzed DataFrame into JSON-safe chart data.

    vSch MUST NOT calculate or download market data.
    Everything needed for visualization is supplied here.
    """

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
        column
        for column in required_columns
        if column in df.columns
    ]

    records = []

    for _, row in df.iterrows():

        item = {}

        for column in available:

            value = row[column]

            if column == "time":
                if pd.isna(value):
                    item[column] = None
                else:
                    item[column] = (
                        pd.Timestamp(value)
                        .isoformat()
                    )
                continue

            if pd.isna(value):
                item[column] = None
                continue

            if column == "st_dir":
                item[column] = int(value)

            else:
                item[column] = float(value)

        records.append(item)

    return records


# ============================================================
# MOMENTUM SCORE
# ============================================================

def movement_score(df):

    if len(df) < 50:
        return -1.0, None

    x = add_indicators(df)

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

    fast_reference = float(
        x["close"].iloc[-1 - fast_n]
    )

    slow_reference = float(
        x["close"].iloc[-1 - slow_n]
    )

    fast_return = abs(
        close / fast_reference - 1.0
    ) * 100

    slow_return = abs(
        close / slow_reference - 1.0
    ) * 100

    atr_move = (
        abs(close - fast_reference)
        /
        atr_value
    )

    volume_ratio = (
        float(last["volume_ratio"])
        if np.isfinite(last["volume_ratio"])
        else 1.0
    )

    volume_bonus = min(
        max(volume_ratio, 0.0),
        4.0,
    )

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

    breakout_bonus = 0.0

    if close > previous_high:
        breakout_bonus = 2.0

    elif close < previous_low:
        breakout_bonus = 2.0

    direction = (
        1
        if float(last["close"])
        >=
        float(last["open"])
        else -1
    )

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

    x = add_indicators(df)

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

    volume_ratio = (
        float(last["volume_ratio"])
        if np.isfinite(last["volume_ratio"])
        else 1.0
    )

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

    macd = float(last["macd"])
    macd_signal = float(last["macd_signal"])

    hist_now = float(last["macd_hist"])
    hist_previous = float(previous["macd_hist"])

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

    if volume_ratio >= CONFIG["volume_ratio_min"]:

        candle_open = float(last["open"])

        if close > candle_open:

            long_score += 1.5
            long_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

        elif close < candle_open:

            short_score += 1.5
            short_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

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

    if len(x) >= 7:

        current_high = float(
            x["high"].iloc[-1]
        )

        previous_high_short = float(
            x["high"].iloc[-4]
        )

        current_low = float(
            x["low"].iloc[-1]
        )

        previous_low_short = float(
            x["low"].iloc[-4]
        )

        if (
            current_high > previous_high_short
            and
            current_low > previous_low_short
        ):

            long_score += 1.0
            long_reasons.append(
                "higher-high / higher-low"
            )

        elif (
            current_high < previous_high_short
            and
            current_low < previous_low_short
        ):

            short_score += 1.0
            short_reasons.append(
                "lower-high / lower-low"
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

        "st_dir": int(last["st_dir"]),

        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": hist_now,
    }


# ============================================================
# ANALYZE SYMBOL
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
    # 15m
    # --------------------------------------------------------

    try:

        scored_15m = score_tf(
            stage1_meta["df"]
        )

        if scored_15m is not None:

            data["15m"] = {
                "score": scored_15m,
                "df": stage1_meta["df"],
            }

    except Exception as exc:

        print(
            f"[MTF] {symbol} 15m: {exc}"
        )

    # --------------------------------------------------------
    # 1h / 4h
    # --------------------------------------------------------

    for tf in ["1h", "4h"]:

        try:

            candles = klines(
                symbol,
                tf,
            )

            scored = score_tf(
                candles
            )

            if scored is not None:

                data[tf] = {
                    "score": scored,
                    "df": add_indicators(candles),
                }

        except Exception as exc:

            print(
                f"[MTF] {symbol} {tf}: {exc}"
            )

    if set(data.keys()) != set(TFS):
        return None

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

    votes = []

    for tf in TFS:

        tf_score = data[tf]["score"]

        if (
            tf_score["long"]
            ==
            tf_score["short"]
        ):
            votes.append(0)

        elif (
            tf_score["long"]
            >
            tf_score["short"]
        ):
            votes.append(1)

        else:
            votes.append(-1)

    agreement = sum(
        vote == wanted_direction
        for vote in votes
    )

    four_hour_direction = (
        1
        if data["4h"]["score"]["long"]
        >
        data["4h"]["score"]["short"]
        else -1
    )

    if four_hour_direction != wanted_direction:
        return None

    if agreement < 2:
        return None

    # --------------------------------------------------------
    # 15m execution data
    # --------------------------------------------------------

    df15 = data["15m"]["df"]

    current15 = df15.iloc[-1]

    reference15 = df15.iloc[
        -1 - CONFIG["momentum_fast_bars"]
    ]

    current_close = float(
        current15["close"]
    )

    reference_close = float(
        reference15["close"]
    )

    move_15 = (
        current_close /
        reference_close -
        1.0
    ) * 100

    if side == "LONG" and move_15 <= 0:
        return None

    if side == "SHORT" and move_15 >= 0:
        return None

    price = float(
        df15["close"].iloc[-1]
    )

    atr_value = float(
        df15["atr"].iloc[-1]
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
    # Swing
    # --------------------------------------------------------

    swing_n = CONFIG["swing_window"]

    swing_low = float(
        df15["low"]
        .iloc[-swing_n:]
        .min()
    )

    swing_high = float(
        df15["high"]
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
            f"Close below {sl:.8g} / "
            f"loss of recent 15m swing low"
        )

    else:

        sl = max(
            swing_high,
            entry + 1.25 * atr_value,
        )

        risk = sl - entry

        invalidation = (
            f"Close above {sl:.8g} / "
            f"reclaim of recent 15m swing high"
        )

    if risk <= 0:
        return None

    risk_pct = (
        risk /
        entry *
        100
    )

    if risk_pct > 8.0:
        return None

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    if side == "LONG":

        tp = [
            entry + risk * rr
            for rr in CONFIG["risk_reward"]
        ]

    else:

        tp = [
            entry - risk * rr
            for rr in CONFIG["risk_reward"]
        ]

    momentum_bonus = min(
        stage1_score / 25.0,
        1.5,
    )

    score = (
        raw_score +
        momentum_bonus
    )

    if side == "LONG":

        reasons = (
            data["15m"]["score"]["long_reasons"]
        )

    else:

        reasons = (
            data["15m"]["score"]["short_reasons"]
        )

    # ========================================================
    # IMPORTANT:
    # Semua data chart dikirim dari Synaptic.
    # vSch tidak perlu memanggil Binance.
    # ========================================================

    chart_data = {}

    for tf in TFS:

        chart_data[tf] = serialize_chart_data(
            data[tf]["df"]
        )

    # --------------------------------------------------------
    # Return candidate
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

        "execution_tf": "15m",

        # --------------------------------------------
        # Analysis
        # --------------------------------------------

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

        # --------------------------------------------
        # Chart / Visual Layer
        # --------------------------------------------

        "chart": {
            "execution_tf": "15m",

            "available_timeframes": TFS,

            "candles": CONFIG["klines"],

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

        # Semua OHLCV + indikator untuk seluruh timeframe.
        "chart_data": chart_data,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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

        print(
            f"[FATAL] Cannot build universe: {exc}"
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

    print(
        f"[STAGE 1] Scanning "
        f"{len(universe_rows)} symbols on 15m..."
    )

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
                change_24h,
                quote_volume_24h,
            )
            for (
                symbol,
                change_24h,
                quote_volume_24h,
            ) in universe_rows
        }

        for future in as_completed(jobs):

            (
                symbol,
                change_24h,
                quote_volume_24h,
            ) = jobs[future]

            try:

                candles = future.result()

                score, meta = movement_score(
                    candles
                )

                if (
                    score > 0
                    and
                    meta is not None
                ):

                    momentum.append(
                        (
                            score,
                            symbol,
                            change_24h,
                            quote_volume_24h,
                            meta,
                        )
                    )

            except Exception as exc:

                print(
                    f"[15m-SCAN] {symbol}: {exc}"
                )

    momentum.sort(
        key=lambda row: row[0],
        reverse=True,
    )

    selected = momentum[
        :CONFIG["momentum_pool"]
    ]

    print(
        f"[MOMENTUM] "
        f"{len(selected)} selected"
    )

    # ========================================================
    # STAGE 2
    # ========================================================

    results = []

    print(
        "[STAGE 2] "
        "15m / 1h / 4h validation..."
    )

    with ThreadPoolExecutor(
        max_workers=CONFIG["workers_stage2"]
    ) as pool:

        jobs = {
            pool.submit(
                analyze_symbol,
                symbol,
                change_24h,
                quote_volume_24h,
                stage_score,
                stage_meta,
            ): symbol

            for (
                stage_score,
                symbol,
                change_24h,
                quote_volume_24h,
                stage_meta,
            ) in selected
        }

        for future in as_completed(jobs):

            symbol = jobs[future]

            try:

                result = future.result()

                if (
                    result
                    and
                    result["score"]
                    >=
                    CONFIG["min_score"]
                ):

                    results.append(result)

            except Exception as exc:

                print(
                    f"[MTF-SCAN] "
                    f"{symbol}: {exc}"
                )

    # ========================================================
    # SORT
    # ========================================================

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    if len(results) < CONFIG["min_candidates"]:

        final_results = []

        print(
            f"[RESULT] "
            f"{len(results)} valid candidate(s)"
        )

    else:

        final_results = results[
            :CONFIG["max_results"]
        ]

        print(
            f"[RESULT] "
            f"{len(final_results)} candidates"
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

        "architecture": {
            "market_data":
                "Synaptic",

            "analysis":
                "Synaptic",

            "chart_data":
                "Synaptic",

            "visualizer":
                "vSch",

            "vSch_api_access":
                False,

            "vSch_data_source":
                "synaptic_candidates.json",
        },

        "universe":
            "ALL active USDT-M perpetuals",

        "selection_method":
            "15m movement + momentum, "
            "then 15m/1h/4h confirmation",

        "timeframes":
            TFS,

        "indicators": {

            "EMA":
                CONFIG["ema_period"],

            "Volume":
                CONFIG["volume_ma_period"],

            "MACD":
                [
                    CONFIG["macd_fast"],
                    CONFIG["macd_slow"],
                    CONFIG["macd_signal"],
                ],

            "Supertrend":
                [
                    CONFIG["supertrend_period"],
                    CONFIG["supertrend_multiplier"],
                ],

            "ATR":
                CONFIG["atr_period"],
        },

        "config":
            CONFIG,

        "scan_stats": {

            "universe":
                len(universe_rows),

            "momentum_scanned":
                len(momentum),

            "momentum_pool":
                len(selected),

            "valid_before_limit":
                len(results),

            "final_candidates":
                len(final_results),

            "elapsed_seconds":
                round(
                    time.time() - started,
                    2,
                ),
        },

        "candidates":
            final_results,
    }

    # ========================================================
    # WRITE JSON
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
    # TERMINAL OUTPUT
    # ========================================================

    print("=" * 72)

    for item in final_results:

        print(
            f"{item['symbol']} "
            f"{item['side']} | "
            f"Score {item['score']:.2f} | "
            f"TF {item['tf_agreement']}/3 | "
            f"Entry {item['entry']:.8g} | "
            f"TP1 {item['tp'][0]:.8g} | "
            f"TP2 {item['tp'][1]:.8g} | "
            f"TP3 {item['tp'][2]:.8g} | "
            f"SL {item['sl']:.8g}"
        )

        print(
            f"  Chart data: "
            f"15m={len(item['chart_data']['15m'])} "
            f"1h={len(item['chart_data']['1h'])} "
            f"4h={len(item['chart_data']['4h'])}"
        )

    print("=" * 72)

    print(
        f"Output: {args.out}"
    )


if __name__ == "__main__":
    main()