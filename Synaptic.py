from pathlib import Path

code = r'''import time
import requests
import pandas as pd
import numpy as np

# ============================================================
# Synaptic.py
# Binance Futures behavior-first scanner
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
    "https://fapi4.binance.com",
]

SESSION = requests.Session()
SESSION.headers.update(HEADERS)
ACTIVE_BASE_URL = None

CONFIG = {
    # Universe / candidate discovery
    "min_quote_volume": 600_000,
    "candidate_pool": 180,
    "min_price": 0.000001,

    # Behavior scan
    "behavior_lookback_15m": 32,
    "behavior_lookback_1h": 24,
    "volume_baseline": 20,
    "range_baseline": 20,

    # Final MTF
    "timeframes": {
        "15m": 80,
        "1h": 80,
        "4h": 80,
    },

    # Core indicators
    "ema_period": 200,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "supertrend_period": 10,
    "supertrend_multiplier": 2.50,

    # Candidate behavior thresholds
    "min_volume_ratio": 1.20,
    "min_range_expansion": 1.15,
    "min_displacement_atr": 0.80,

    # Output
    "pre_score_keep": 60,
    "final_setups": 10,
}

IGNORED = {
    "USDCUSDT", "BUSDUSDT", "DAIUSDT", "TUSDUSDT",
    "FDUSDUSDT", "USDPUSDT", "EURUSDT", "USTCUSDT",
    "PAXGUSDT",
}


# ============================================================
# BINANCE API
# ============================================================

def binance_get(path, params=None, timeout=10):
    global ACTIVE_BASE_URL

    endpoints = []
    if ACTIVE_BASE_URL:
        endpoints.append(ACTIVE_BASE_URL)

    for base in BASE_URLS:
        if base not in endpoints:
            endpoints.append(base)

    last_error = None

    for base in endpoints:
        try:
            response = SESSION.get(
                base + path,
                params=params,
                timeout=timeout,
            )

            print(f"[API] {response.status_code} {base}{path}")

            if response.status_code == 451:
                continue

            if response.status_code in (418, 429):
                last_error = f"HTTP {response.status_code}"
                time.sleep(1.5)
                continue

            if response.status_code != 200:
                last_error = f"HTTP {response.status_code}"
                continue

            data = response.json()

            if isinstance(data, dict) and "code" in data and "msg" in data:
                last_error = f"{data['code']}: {data['msg']}"
                continue

            ACTIVE_BASE_URL = base
            return data

        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"Binance Futures API gagal. Last error: {last_error}"
    )


def get_futures_symbols():
    data = binance_get("/fapi/v1/exchangeInfo")

    symbols = []

    for item in data.get("symbols", []):
        if (
            item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and item.get("symbol") not in IGNORED
        ):
            symbols.append(item["symbol"])

    return symbols


def get_24h_tickers():
    data = binance_get("/fapi/v1/ticker/24hr")

    result = {}

    for item in data:
        symbol = item.get("symbol")

        try:
            result[symbol] = {
                "price": float(item["lastPrice"]),
                "change_24h": float(item["priceChangePercent"]),
                "quote_volume": float(item["quoteVolume"]),
                "high_24h": float(item["highPrice"]),
                "low_24h": float(item["lowPrice"]),
            }
        except (TypeError, ValueError):
            continue

    return result


def get_klines(symbol, interval, limit=80):
    data = binance_get(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        timeout=10,
    )

    if not isinstance(data, list):
        raise ValueError(f"Kline invalid: {symbol} {interval}")

    df = pd.DataFrame(
        data,
        columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )

    numeric = [
        "open", "high", "low", "close",
        "volume", "quote_volume",
    ]

    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
    )

    return df.dropna(
        subset=["open", "high", "low", "close", "volume"]
    ).reset_index(drop=True)


# ============================================================
# BASIC INDICATORS
# ============================================================

def add_ema(df, period=200):
    df = df.copy()
    df["ema200"] = df["close"].ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()
    return df


def add_macd(df):
    df = df.copy()

    fast = df["close"].ewm(
        span=CONFIG["macd_fast"],
        adjust=False,
    ).mean()

    slow = df["close"].ewm(
        span=CONFIG["macd_slow"],
        adjust=False,
    ).mean()

    df["macd"] = fast - slow
    df["macd_signal"] = df["macd"].ewm(
        span=CONFIG["macd_signal"],
        adjust=False,
    ).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    return df


def add_atr(df, period=14):
    df = df.copy()

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return df


def add_supertrend(df):
    df = add_atr(df, CONFIG["supertrend_period"]).copy()

    multiplier = CONFIG["supertrend_multiplier"]
    hl2 = (df["high"] + df["low"]) / 2

    basic_upper = hl2 + multiplier * df["atr"]
    basic_lower = hl2 - multiplier * df["atr"]

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = pd.Series(1, index=df.index, dtype=int)

    for i in range(1, len(df)):
        if (
            basic_upper.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            basic_lower.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == -1:
            direction.iloc[i] = (
                1 if df["close"].iloc[i] > final_upper.iloc[i]
                else -1
            )
        else:
            direction.iloc[i] = (
                -1 if df["close"].iloc[i] < final_lower.iloc[i]
                else 1
            )

    df["supertrend"] = np.where(
        direction == 1,
        final_lower,
        final_upper,
    )
    df["supertrend_direction"] = direction

    return df


# ============================================================
# MARKET BEHAVIOR
# ============================================================

def behavior_score(df):
    if len(df) < 30:
        return 0.0, {}

    work = add_atr(df)
    close = work["close"]
    high = work["high"]
    low = work["low"]
    volume = work["volume"]
    atr = work["atr"]

    last = -1

    volume_base = volume.iloc[-21:-1].mean()
    range_now = (high.iloc[last] - low.iloc[last])

    range_base = (
        (high - low).iloc[-21:-1].mean()
    )

    volume_ratio = (
        volume.iloc[last] / volume_base
        if volume_base > 0 else 1.0
    )

    range_ratio = (
        range_now / range_base
        if range_base > 0 else 1.0
    )

    atr_now = atr.iloc[last]
    displacement = (
        abs(close.iloc[last] - close.iloc[-2]) / atr_now
        if atr_now and not np.isnan(atr_now)
        else 0.0
    )

    candle_direction = (
        1 if close.iloc[last] > close.iloc[-2]
        else -1
    )

    recent_move = (
        (close.iloc[-1] / close.iloc[-9]) - 1
    ) * 100

    score = 0.0

    if volume_ratio >= 1.2:
        score += min(volume_ratio * 7, 20)

    if range_ratio >= 1.15:
        score += min(range_ratio * 7, 18)

    if displacement >= 0.8:
        score += min(displacement * 6, 18)

    if abs(recent_move) >= 1.0:
        score += min(abs(recent_move) * 2.5, 15)

    # Reward simultaneous expansion.
    if volume_ratio >= 1.3 and range_ratio >= 1.2:
        score += 12

    if volume_ratio >= 1.5 and displacement >= 1.0:
        score += 10

    return min(score, 99.0), {
        "volume_ratio": round(float(volume_ratio), 2),
        "range_ratio": round(float(range_ratio), 2),
        "displacement_atr": round(float(displacement), 2),
        "recent_move_pct": round(float(recent_move), 2),
        "behavior_direction": (
            "LONG" if candle_direction > 0 else "SHORT"
        ),
    }


# ============================================================
# CANDIDATE DISCOVERY
# ============================================================

def discover_candidates():
    symbols = get_futures_symbols()
    tickers = get_24h_tickers()

    candidates = []

    for symbol in symbols:
        ticker = tickers.get(symbol)

        if not ticker:
            continue

        if ticker["price"] < CONFIG["min_price"]:
            continue

        if ticker["quote_volume"] < CONFIG["min_quote_volume"]:
            continue

        day_range = (
            (ticker["high_24h"] - ticker["low_24h"])
            / ticker["low_24h"] * 100
            if ticker["low_24h"] > 0 else 0
        )

        # 24H data is discovery context only.
        liquidity_score = min(
            np.log10(max(ticker["quote_volume"], 1)) * 2.5,
            18,
        )

        range_score = min(day_range * 1.2, 15)

        candidates.append({
            "symbol": symbol,
            "price": ticker["price"],
            "change_24h": ticker["change_24h"],
            "quote_volume": ticker["quote_volume"],
            "range_24h_pct": day_range,
            "discovery_score": liquidity_score + range_score,
        })

    candidates.sort(
        key=lambda x: x["discovery_score"],
        reverse=True,
    )

    # Broad first-pass pool. No top-gainer/top-loser dependency.
    return candidates[:CONFIG["candidate_pool"]]


def behavior_scan(candidates):
    scanned = []

    for i, candidate in enumerate(candidates, 1):
        symbol = candidate["symbol"]

        try:
            df15 = get_klines(
                symbol,
                "15m",
                CONFIG["behavior_lookback_15m"],
            )

            df1h = get_klines(
                symbol,
                "1h",
                CONFIG["behavior_lookback_1h"],
            )

            score15, detail15 = behavior_score(df15)
            score1h, detail1h = behavior_score(df1h)

            combined = (
                score15 * 0.60
                + score1h * 0.40
                + candidate["discovery_score"]
            )

            candidate = {
                **candidate,
                "behavior_score_15m": round(score15, 2),
                "behavior_score_1h": round(score1h, 2),
                "behavior_score": round(combined, 2),
                "behavior_15m": detail15,
                "behavior_1h": detail1h,
            }

            scanned.append(candidate)

            print(
                f"[{i}] {symbol} "
                f"behavior={combined:.1f} "
                f"15m={score15:.1f} "
                f"1h={score1h:.1f}"
            )

        except Exception as exc:
            print(f"[SKIP] {symbol}: {exc}")

    scanned.sort(
        key=lambda x: x["behavior_score"],
        reverse=True,
    )

    return scanned[:CONFIG["pre_score_keep"]]


# ============================================================
# MTF DIRECTION ENGINE
# ============================================================

def analyze_timeframe(df):
    if len(df) < CONFIG["ema_period"]:
        return {
            "direction": "NEUTRAL",
            "score": 0,
        }

    df = add_ema(df, CONFIG["ema_period"])
    df = add_macd(df)
    df = add_supertrend(df)

    row = df.iloc[-1]

    long_score = 0
    short_score = 0

    if row["close"] > row["ema200"]:
        long_score += 2
    elif row["close"] < row["ema200"]:
        short_score += 2

    if row["macd"] > row["macd_signal"]:
        long_score += 1
    elif row["macd"] < row["macd_signal"]:
        short_score += 1

    if row["macd_hist"] > 0:
        long_score += 1
    elif row["macd_hist"] < 0:
        short_score += 1

    if row["supertrend_direction"] > 0:
        long_score += 2
    else:
        short_score += 2

    if row["volume"] > df["volume"].iloc[-21:-1].mean() * 1.2:
        if row["close"] > row["open"]:
            long_score += 1
        elif row["close"] < row["open"]:
            short_score += 1

    if long_score > short_score:
        direction = "LONG"
    elif short_score > long_score:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "long_score": long_score,
        "short_score": short_score,
        "score": max(long_score, short_score),
        "ema200": float(row["ema200"]),
        "volume": float(row["volume"]),
        "macd": float(row["macd"]),
        "macd_signal": float(row["macd_signal"]),
        "supertrend_direction": int(row["supertrend_direction"]),
    }


def determine_direction(symbol):
    results = {}

    for tf, limit in CONFIG["timeframes"].items():
        df = get_klines(symbol, tf, limit)
        results[tf] = analyze_timeframe(df)

    directions = [
        results[tf]["direction"]
        for tf in ("15m", "1h", "4h")
    ]

    long_count = directions.count("LONG")
    short_count = directions.count("SHORT")

    if long_count >= 2:
        direction = "LONG"
    elif short_count >= 2:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    return direction, results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SYNAPTIC — BINANCE BEHAVIOR SCANNER")
    print("=" * 60)

    candidates = discover_candidates()

    print(f"\nUniverse candidates: {len(candidates)}")

    behavior = behavior_scan(candidates)

    print(f"Behavior survivors: {len(behavior)}")

    final = []

    for candidate in behavior:
        try:
            direction, mtf = determine_direction(
                candidate["symbol"]
            )

            if direction == "NEUTRAL":
                continue

            item = {
                **candidate,
                "direction": direction,
                "mtf": mtf,
            }

            final.append(item)

        except Exception as exc:
            print(
                f"[MTF SKIP] "
                f"{candidate['symbol']}: {exc}"
            )

    final.sort(
        key=lambda x: (
            x["behavior_score"],
            x["mtf"]["1h"]["score"],
            x["mtf"]["4h"]["score"],
        ),
        reverse=True,
    )

    print("\nFINAL CANDIDATES")
    print("-" * 60)

    for item in final[:CONFIG["final_setups"]]:
        print(
            f"{item['symbol']:14} "
            f"{item['direction']:5} "
            f"behavior={item['behavior_score']:5.1f} "
            f"15m={item['mtf']['15m']['direction']:7} "
            f"1h={item['mtf']['1h']['direction']:7} "
            f"4h={item['mtf']['4h']['direction']:7}"
        )

    return final


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/Synaptic.py")
path.write_text(code, encoding="utf-8")

print(f"Created: {path}")
