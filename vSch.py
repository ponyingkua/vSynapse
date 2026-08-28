import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import requests


BASE_URLS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

S = requests.Session()

S.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
})

ACTIVE_BASE = None


def api(path, params=None):
    global ACTIVE_BASE

    endpoints = []

    if ACTIVE_BASE:
        endpoints.append(ACTIVE_BASE)

    for base in BASE_URLS:
        if base not in endpoints:
            endpoints.append(base)

    last_error = None

    for base in endpoints:
        try:
            r = S.get(
                base + path,
                params=params,
                timeout=20,
            )

            if r.status_code == 451:
                last_error = f"{base}: HTTP 451"
                continue

            if r.status_code in (418, 429):
                last_error = f"{base}: HTTP {r.status_code}"
                time.sleep(1)
                continue

            if r.status_code != 200:
                last_error = f"{base}: HTTP {r.status_code}"
                continue

            data = r.json()

            if isinstance(data, dict) and "code" in data:
                last_error = (
                    f"{base}: "
                    f"{data.get('code')} "
                    f"{data.get('msg', '')}"
                )
                continue

            ACTIVE_BASE = base

            print(f"Binance endpoint: {base}")

            return data

        except requests.RequestException as e:
            last_error = f"{base}: {e}"

    raise RuntimeError(
        f"Binance API failed on all endpoints: {last_error}"
    )


def klines(symbol, interval, limit=120):
    raw = api(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
    )

    if not raw:
        raise RuntimeError(
            f"No kline data for {symbol} {interval}"
        )

    cols = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "tb",
        "tq",
        "ignore",
    ]

    d = pd.DataFrame(
        raw,
        columns=cols,
    )

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        d[c] = pd.to_numeric(
            d[c],
            errors="coerce",
        )

    d["time"] = pd.to_datetime(
        d["time"],
        unit="ms",
        utc=True,
    )

    d = d.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    ).reset_index(drop=True)

    return d


def ema(s, n):
    return s.ewm(
        span=n,
        adjust=False,
    ).mean()


def render(c):
    symbol = c["symbol"]
    tf = c["execution_tf"]

    d = klines(
        symbol,
        tf,
        120,
    )

    d["ema200"] = ema(
        d.close,
        200,
    )

    fig = plt.figure(
        figsize=(13, 7.5)
    )

    ax = fig.add_axes(
        [0.07, 0.16, 0.90, 0.74]
    )

    axv = fig.add_axes(
        [0.07, 0.07, 0.90, 0.08],
        sharex=ax,
    )

    fig.patch.set_facecolor(
        "#0b0f14"
    )

    ax.set_facecolor(
        "#0b0f14"
    )

    axv.set_facecolor(
        "#0b0f14"
    )

    for i, row in d.iterrows():

        up = row.close >= row.open

        body_color = (
            "#20c997"
            if up
            else "#ff5c5c"
        )

        wick_color = "#c8d1dc"

        ax.plot(
            [i, i],
            [row.low, row.high],
            color=wick_color,
            linewidth=0.8,
        )

        lo = min(
            row.open,
            row.close,
        )

        h = abs(
            row.close -
            row.open
        )

        if h == 0:
            h = max(
                (row.high - row.low) * 0.01,
                1e-12,
            )

        ax.add_patch(
            Rectangle(
                (i - 0.32, lo),
                0.64,
                h,
                facecolor=body_color,
                edgecolor=body_color,
                linewidth=0.5,
            )
        )

        axv.bar(
            i,
            row.volume,
            width=0.65,
            color=body_color,
            alpha=0.55,
        )

    ax.plot(
        range(len(d)),
        d.ema200,
        linewidth=1.2,
        color="#f1c40f",
        label="EMA200",
    )

    entry = c["entry"]
    sl = c["sl"]
    tps = c["tp"]
    side = c["side"]

    ax.axhline(
        entry,
        linewidth=1.4,
        color="#4dabf7",
        label="Entry",
    )

    ax.axhline(
        sl,
        linewidth=1.2,
        color="#ff6b6b",
        linestyle="--",
        label="SL",
    )

    for i, tp in enumerate(
        tps,
        1,
    ):
        ax.axhline(
            tp,
            linewidth=1.0,
            color="#51cf66",
            linestyle="--",
            label=(
                f"TP{i}"
                if i == 1
                else None
            ),
        )

    fig.text(
        0.07,
        0.94,
        f"{symbol}  |  {side}  |  {tf}",
        fontsize=18,
        fontweight="bold",
        color="white",
    )

    fig.text(
        0.07,
        0.905,
        f"Entry {entry:.8g}    "
        f"TP1 {tps[0]:.8g}    "
        f"TP2 {tps[1]:.8g}    "
        f"TP3 {tps[2]:.8g}    "
        f"SL {sl:.8g}",
        fontsize=10.5,
        color="#c8d1dc",
    )

    points = c.get(
        "key_points",
        [],
    )

    point_text = (
        " • ".join(points[:5])
        if points
        else "Price-action confirmation"
    )

    fig.text(
        0.07,
        0.025,
        f"Key points: {point_text}",
        fontsize=9.5,
        color="#c8d1dc",
    )

    ax.text(
        0.985,
        0.985,
        f'Invalidation: {c["invalidation"]}',
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#ff8787",
    )

    ax.grid(
        True,
        alpha=0.12,
        linewidth=0.7,
    )

    axv.grid(
        True,
        axis="y",
        alpha=0.10,
    )

    ax.set_xlim(
        max(0, len(d) - 90),
        len(d),
    )

    ax.tick_params(
        colors="#9aa7b5",
        labelsize=8,
    )

    axv.tick_params(
        colors="#9aa7b5",
        labelsize=8,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    for spine in axv.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper left",
        ncol=5,
        frameon=False,
        fontsize=8,
        labelcolor="white",
    )

    out = Path(
        f"{symbol}_{tf}_{side}_chart.png"
    )

    fig.savefig(
        out,
        dpi=170,
        facecolor=fig.get_facecolor(),
    )

    plt.close(fig)

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="synaptic_candidates.json",
    )

    ap.add_argument(
        "--symbol",
        default="",
    )

    args = ap.parse_args()

    data = json.loads(
        Path(args.input).read_text(
            encoding="utf-8"
        )
    )

    candidates = data.get(
        "candidates",
        [],
    )

    if not candidates:
        raise SystemExit(
            "No candidates in input JSON."
        )

    if args.symbol:

        candidates = [
            x
            for x in candidates
            if x["symbol"].upper()
            == args.symbol.upper()
        ]

        if not candidates:
            raise SystemExit(
                "Symbol not found."
            )

    c = candidates[0]

    print("=" * 72)

    print(
        f'{c["symbol"]}  '
        f'{c["side"]}  | '
        f'Score {c["score"]} | '
        f'TF {c["execution_tf"]}'
    )

    print(
        f'Entry: {c["entry"]:.10g}'
    )

    print(
        f'TP1:   {c["tp"][0]:.10g}'
    )

    print(
        f'TP2:   {c["tp"][1]:.10g}'
    )

    print(
        f'TP3:   {c["tp"][2]:.10g}'
    )

    print(
        f'SL:    {c["sl"]:.10g}'
    )

    print(
        f'Invalidation: '
        f'{c["invalidation"]}'
    )

    print(
        "Key points:",
        " | ".join(
            c.get(
                "key_points",
                [],
            )
        ),
    )

    print("=" * 72)

    out = render(c)

    print(
        f"Chart: {out}"
    )


if __name__ == "__main__":
    main()