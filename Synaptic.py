#!/usr/bin/env python3
"""
Synaptic.py
Main Binance Futures candidate scanner.

Pipeline:
1) Build active/liquid USDT perpetual universe.
2) Score candidates using 15m / 1h / 4h.
3) Analyze EMA200, Volume, MACD, RSI, Supertrend(10, 2.50).
4) Select LONG/SHORT direction.
5) Calculate Entry / TP1-TP3 / SL and invalidation.
6) Export JSON for vSch.py.

This is a research scanner, not financial advice.
"""

import argparse, json, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://fapi.binance.com"
TFs = ["15m", "1h", "4h"]

CONFIG = {
    "min_quote_volume_24h": 5_000_000,
    "min_abs_change_24h": 4.0,
    "universe_size": 80,
    "klines": 240,
    "min_score": 6.0,
    "max_results": 15,
    "supertrend_period": 10,
    "supertrend_multiplier": 2.50,
    "atr_period": 14,
    "risk_reward": [1.5, 2.25, 3.0],
    "max_entry_atr_distance": 2.75,
}

S = requests.Session()
S.headers.update({"User-Agent": "Synaptic/1.0"})


def api(path, params=None):
    r = S.get(BASE + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def universe():
    tick = api("/fapi/v1/ticker/24hr")
    rows = []
    for x in tick:
        sym = x["symbol"]
        if not sym.endswith("USDT") or "_" in sym:
            continue
        qv = float(x["quoteVolume"])
        ch = float(x["priceChangePercent"])
        if qv < CONFIG["min_quote_volume_24h"] or abs(ch) < CONFIG["min_abs_change_24h"]:
            continue
        activity = abs(ch) * math.log10(max(qv, 1))
        rows.append((sym, ch, qv, activity))
    rows.sort(key=lambda z: z[3], reverse=True)
    return rows[:CONFIG["universe_size"]]


def klines(symbol, interval):
    raw = api("/fapi/v1/klines", {
        "symbol": symbol, "interval": interval, "limit": CONFIG["klines"]
    })
    c = ["time","open","high","low","close","volume","close_time",
         "quote_volume","trades","tb","tq","ignore"]
    df = pd.DataFrame(raw, columns=c)
    for col in ["open","high","low","close","volume","quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        df.high-df.low,
        (df.high-pc).abs(),
        (df.low-pc).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def indicators(df):
    x = df.copy()
    x["ema200"] = x.close.ewm(span=200, adjust=False).mean()

    # MACD 12/26/9
    e12 = x.close.ewm(span=12, adjust=False).mean()
    e26 = x.close.ewm(span=26, adjust=False).mean()
    x["macd"] = e12 - e26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x.macd - x.macd_signal

    # RSI 14
    delta = x.close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / down.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    x["rsi"] = 100 - 100/(1+rs)

    x["atr"] = atr(x, CONFIG["atr_period"])
    x["vol_ma20"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume / x.vol_ma20

    # Supertrend 10 / 2.50
    n = CONFIG["supertrend_period"]
    m = CONFIG["supertrend_multiplier"]
    hl2 = (x.high+x.low)/2
    basic_ub = hl2 + m*x.atr
    basic_lb = hl2 - m*x.atr
    fub = basic_ub.copy()
    flb = basic_lb.copy()
    direction = pd.Series(index=x.index, dtype=float)
    st = pd.Series(index=x.index, dtype=float)

    direction.iloc[0] = 1
    fub.iloc[0] = basic_ub.iloc[0]
    flb.iloc[0] = basic_lb.iloc[0]
    st.iloc[0] = flb.iloc[0]

    for i in range(1, len(x)):
        fub.iloc[i] = basic_ub.iloc[i] if (
            basic_ub.iloc[i] < fub.iloc[i-1] or x.close.iloc[i-1] > fub.iloc[i-1]
        ) else fub.iloc[i-1]
        flb.iloc[i] = basic_lb.iloc[i] if (
            basic_lb.iloc[i] > flb.iloc[i-1] or x.close.iloc[i-1] < flb.iloc[i-1]
        ) else flb.iloc[i-1]

        if direction.iloc[i-1] == -1 and x.close.iloc[i] > fub.iloc[i]:
            direction.iloc[i] = 1
        elif direction.iloc[i-1] == 1 and x.close.iloc[i] < flb.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

        st.iloc[i] = flb.iloc[i] if direction.iloc[i] == 1 else fub.iloc[i]

    x["supertrend"] = st
    x["st_dir"] = direction
    return x


def score_tf(x):
    last = x.iloc[-1]
    prev = x.iloc[-2]
    score_l = 0.0
    score_s = 0.0
    rl, rs = [], []

    # EMA200
    if last.close > last.ema200:
        score_l += 2
        rl.append("above EMA200")
    elif last.close < last.ema200:
        score_s += 2
        rs.append("below EMA200")

    # Supertrend
    if last.st_dir > 0:
        score_l += 2
        rl.append("Supertrend bullish")
    else:
        score_s += 2
        rs.append("Supertrend bearish")

    # MACD
    if last.macd > last.macd_signal and last.macd_hist > prev.macd_hist:
        score_l += 1.5
        rl.append("MACD bullish")
    elif last.macd < last.macd_signal and last.macd_hist < prev.macd_hist:
        score_s += 1.5
        rs.append("MACD bearish")

    # RSI: avoid using extreme as automatic direction.
    if 52 <= last.rsi <= 68:
        score_l += 1
        rl.append(f"RSI {last.rsi:.0f} supportive")
    elif 32 <= last.rsi <= 48:
        score_s += 1
        rs.append(f"RSI {last.rsi:.0f} supportive")
    elif last.rsi > 72:
        score_l += 0.25
        rl.append(f"RSI {last.rsi:.0f} overheated")
    elif last.rsi < 28:
        score_s += 0.25
        rs.append(f"RSI {last.rsi:.0f} oversold")

    # Volume
    vr = float(last.vol_ratio) if np.isfinite(last.vol_ratio) else 1
    if vr >= 1.5:
        if last.close > last.open:
            score_l += 1.5
            rl.append(f"volume {vr:.1f}x")
        elif last.close < last.open:
            score_s += 1.5
            rs.append(f"volume {vr:.1f}x")

    # Price action
    hi20 = x.high.iloc[-21:-1].max()
    lo20 = x.low.iloc[-21:-1].min()
    if last.close > hi20:
        score_l += 2
        rl.append("20-bar breakout")
    elif last.close < lo20:
        score_s += 2
        rs.append("20-bar breakdown")

    # Recent 5-bar structure
    recent = x.iloc[-6:]
    if recent.high.iloc[-1] > recent.high.iloc[-3] and recent.low.iloc[-1] > recent.low.iloc[-3]:
        score_l += 1
        rl.append("higher-high/higher-low")
    if recent.high.iloc[-1] < recent.high.iloc[-3] and recent.low.iloc[-1] < recent.low.iloc[-3]:
        score_s += 1
        rs.append("lower-high/lower-low")

    return {
        "long": score_l, "short": score_s,
        "long_reasons": rl, "short_reasons": rs,
        "close": float(last.close),
        "atr": float(last.atr),
        "ema200": float(last.ema200),
        "rsi": float(last.rsi),
        "vol_ratio": vr,
        "st_dir": int(last.st_dir),
    }


def analyze_symbol(symbol, ch24, qv24):
    tf_data = {}
    for tf in TFs:
        try:
            d = indicators(klines(symbol, tf))
            tf_data[tf] = score_tf(d)
        except Exception:
            pass

    if len(tf_data) < 2:
        return None

    # 4H has the most context, 1H execution structure, 15m timing.
    weights = {"15m": 0.25, "1h": 0.35, "4h": 0.40}
    L = sum(weights[t]*tf_data[t]["long"] for t in tf_data)
    Sh = sum(weights[t]*tf_data[t]["short"] for t in tf_data)

    side = "LONG" if L > Sh else "SHORT"
    score = max(L, Sh)

    # Require directional agreement from at least 2/3 TFs.
    votes = [
        1 if tf_data[t]["long"] > tf_data[t]["short"] else -1
        for t in tf_data
    ]
    agreement = sum(v == (1 if side == "LONG" else -1) for v in votes)

    if agreement < 2:
        score -= 1.5

    # Pick execution TF: strongest directional score, preferring 1h.
    candidates = []
    for tf, d in tf_data.items():
        s = d["long"] if side == "LONG" else d["short"]
        candidates.append((s, {"1h": 0.15, "15m": 0.05, "4h": 0.10}[tf], tf))
    _, _, exec_tf = max(candidates)

    d = indicators(klines(symbol, exec_tf))
    last = d.iloc[-1]
    price = float(last.close)
    a = float(last.atr)

    # Entry around current market price, with a small ATR pullback bias.
    entry = price

    if side == "LONG":
        swing_low = float(d.low.iloc[-8:].min())
        sl = min(swing_low, price - 1.15*a)
        risk = entry - sl
        if risk <= 0:
            return None
        tps = [entry + risk*r for r in CONFIG["risk_reward"]]
        invalid = f"Close below {sl:.8g} / loss of recent swing low"
    else:
        swing_high = float(d.high.iloc[-8:].max())
        sl = max(swing_high, price + 1.15*a)
        risk = sl - entry
        if risk <= 0:
            return None
        tps = [entry - risk*r for r in CONFIG["risk_reward"]]
        invalid = f"Close above {sl:.8g} / reclaim of recent swing high"

    # Avoid absurdly wide risk.
    risk_pct = abs(entry-sl)/entry*100
    if risk_pct > 8:
        return None

    return {
        "symbol": symbol,
        "side": side,
        "score": round(score, 2),
        "change24h": round(ch24, 2),
        "quote_volume24h": round(qv24, 2),
        "execution_tf": exec_tf,
        "timeframes": tf_data,
        "entry": entry,
        "tp": tps,
        "sl": sl,
        "risk_pct": risk_pct,
        "invalidation": invalid,
        "key_points": (
            tf_data[exec_tf]["long_reasons"]
            if side == "LONG" else tf_data[exec_tf]["short_reasons"]
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synaptic_candidates.json")
    args = ap.parse_args()

    uni = universe()
    results = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(analyze_symbol, *u[:3]) for u in uni]
        for j in as_completed(jobs):
            try:
                r = j.result()
                if r and r["score"] >= CONFIG["min_score"]:
                    results.append(r)
            except Exception:
                pass

    results.sort(key=lambda r: r["score"], reverse=True)
    results = results[:CONFIG["max_results"]]

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "config": CONFIG,
        "candidates": results,
    }

    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Candidates: {len(results)}")
    for r in results:
        print(
            f'{r["symbol"]:14} {r["side"]:5} '
            f'Score {r["score"]:4.1f} | TF {r["execution_tf"]:>3} | '
            f'Entry {r["entry"]:.8g} | TP {r["tp"][0]:.8g}/{r["tp"][1]:.8g}/{r["tp"][2]:.8g} | '
            f'SL {r["sl"]:.8g}'
        )


if __name__ == "__main__":
    main()
