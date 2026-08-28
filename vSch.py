import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_candidates(path):
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    candidates = data.get("candidates", [])

    if not candidates:
        raise SystemExit(
            "No candidates in synaptic_candidates.json"
        )

    return data, candidates


def fmt(value):
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return str(value)


def render(c):
    symbol = c["symbol"]
    side = c["side"]

    entry = float(c["entry"])
    sl = float(c["sl"])
    tps = [float(x) for x in c["tp"]]

    score = float(c.get("score", 0))
    momentum = float(c.get("momentum_15m", 0))
    agreement = int(c.get("tf_agreement", 0))

    timeframes = c.get("timeframes", {})

    fig = plt.figure(
        figsize=(14, 8),
        facecolor="#0b0f14"
    )

    ax = fig.add_axes(
        [0.06, 0.12, 0.64, 0.76]
    )

    ax_info = fig.add_axes(
        [0.73, 0.12, 0.23, 0.76]
    )

    ax.set_facecolor("#0b0f14")
    ax_info.set_facecolor("#0b0f14")

    levels = [
        ("SL", sl, "#ff6b6b"),
        ("Entry", entry, "#4dabf7"),
        ("TP1", tps[0], "#51cf66"),
        ("TP2", tps[1], "#51cf66"),
        ("TP3", tps[2], "#51cf66"),
    ]

    all_prices = [entry, sl] + tps

    low_price = min(all_prices)
    high_price = max(all_prices)

    span = high_price - low_price

    if span <= 0:
        span = max(abs(entry) * 0.05, 1e-8)

    padding = span * 0.20

    ax.set_ylim(
        low_price - padding,
        high_price + padding
    )

    ax.set_xlim(0, 100)

    ax.grid(
        True,
        alpha=0.12,
        linewidth=0.7
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        colors="#9aa7b5",
        labelsize=9
    )

    ax.set_ylabel(
        "Price",
        color="#9aa7b5",
        fontsize=9
    )

    for name, price, color in levels:

        linestyle = (
            "-"
            if name == "Entry"
            else "--"
        )

        linewidth = (
            1.6
            if name == "Entry"
            else 1.1
        )

        ax.axhline(
            price,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle
        )

        ax.text(
            2,
            price,
            f"  {name} {fmt(price)}",
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold"
        )

    if side == "LONG":
        zone_low = entry
        zone_high = tps[2]
    else:
        zone_low = tps[2]
        zone_high = entry

    ax.fill_between(
        [0, 100],
        zone_low,
        zone_high,
        alpha=0.035
    )

    ax.text(
        50,
        entry,
        side,
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
        color="#ffffff",
        alpha=0.08
    )

    ax.set_xticks([])

    fig.text(
        0.06,
        0.94,
        f"{symbol}  |  {side}",
        fontsize=20,
        fontweight="bold",
        color="white"
    )

    fig.text(
        0.06,
        0.905,
        (
            f"Score {score:.2f}   |   "
            f"MTF agreement {agreement}/3   |   "
            f"15m momentum {momentum:+.3f}%"
        ),
        fontsize=10.5,
        color="#c8d1dc"
    )

    ax_info.axis("off")

    y = 0.96

    ax_info.text(
        0.02,
        y,
        "MULTI-TIMEFRAME",
        fontsize=11,
        fontweight="bold",
        color="white",
        transform=ax_info.transAxes
    )

    y -= 0.07

    for tf in ["15m", "1h", "4h"]:

        item = timeframes.get(tf)

        if not item:
            continue

        long_score = float(
            item.get("long", 0)
        )

        short_score = float(
            item.get("short", 0)
        )

        if long_score > short_score:
            direction = "LONG"
            tf_score = long_score
        elif short_score > long_score:
            direction = "SHORT"
            tf_score = short_score
        else:
            direction = "NEUTRAL"
            tf_score = long_score

        ax_info.text(
            0.02,
            y,
            f"{tf}   {direction}",
            fontsize=10.5,
            fontweight="bold",
            color="white",
            transform=ax_info.transAxes
        )

        y -= 0.045

        ax_info.text(
            0.02,
            y,
            f"Long {long_score:.2f}   Short {short_score:.2f}",
            fontsize=8.5,
            color="#9aa7b5",
            transform=ax_info.transAxes
        )

        y -= 0.075

    ax_info.text(
        0.02,
        y,
        "TRADE LEVELS",
        fontsize=11,
        fontweight="bold",
        color="white",
        transform=ax_info.transAxes
    )

    y -= 0.06

    trade_levels = [
        ("Entry", entry),
        ("TP1", tps[0]),
        ("TP2", tps[1]),
        ("TP3", tps[2]),
        ("SL", sl),
    ]

    for name, price in trade_levels:

        ax_info.text(
            0.02,
            y,
            f"{name:<6} {fmt(price)}",
            fontsize=9.5,
            color="#c8d1dc",
            transform=ax_info.transAxes
        )

        y -= 0.045

    y -= 0.025

    ax_info.text(
        0.02,
        y,
        "INVALIDATION",
        fontsize=11,
        fontweight="bold",
        color="white",
        transform=ax_info.transAxes
    )

    y -= 0.055

    invalidation = c.get(
        "invalidation",
        "Not provided"
    )

    ax_info.text(
        0.02,
        y,
        invalidation,
        fontsize=8.5,
        color="#ff8787",
        wrap=True,
        transform=ax_info.transAxes
    )

    y -= 0.12

    ax_info.text(
        0.02,
        y,
        "KEY POINTS",
        fontsize=11,
        fontweight="bold",
        color="white",
        transform=ax_info.transAxes
    )

    y -= 0.055

    points = c.get(
        "key_points",
        []
    )

    for point in points[:6]:

        ax_info.text(
            0.02,
            y,
            f"• {point}",
            fontsize=8.3,
            color="#c8d1dc",
            wrap=True,
            transform=ax_info.transAxes
        )

        y -= 0.045

    fig.text(
        0.06,
        0.035,
        "Synaptic data visualization",
        fontsize=8.5,
        color="#6f7d8c"
    )

    out = Path(
        f"{symbol}_{side}_chart.png"
    )

    fig.savefig(
        out,
        dpi=170,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    plt.close(fig)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="synaptic_candidates.json"
    )

    parser.add_argument(
        "--symbol",
        default=""
    )

    args = parser.parse_args()

    data, candidates = load_candidates(
        args.input
    )

    if args.symbol:

        candidates = [
            item
            for item in candidates
            if item["symbol"].upper()
            == args.symbol.upper()
        ]

        if not candidates:
            raise SystemExit(
                f"{args.symbol} not found."
            )

    candidates.sort(
        key=lambda item: float(
            item.get("score", 0)
        ),
        reverse=True
    )

    candidate = candidates[0]

    print("=" * 60)

    print(
        f"Visualizing: "
        f"{candidate['symbol']} "
        f"{candidate['side']}"
    )

    print(
        f"Score: "
        f"{candidate.get('score', 0)}"
    )

    print(
        f"TF agreement: "
        f"{candidate.get('tf_agreement', 0)}/3"
    )

    print(
        f"Entry: "
        f"{fmt(candidate['entry'])}"
    )

    print(
        f"TP1: "
        f"{fmt(candidate['tp'][0])}"
    )

    print(
        f"TP2: "
        f"{fmt(candidate['tp'][1])}"
    )

    print(
        f"TP3: "
        f"{fmt(candidate['tp'][2])}"
    )

    print(
        f"SL: "
        f"{fmt(candidate['sl'])}"
    )

    print("=" * 60)

    out = render(candidate)

    print(f"Chart: {out}")


if __name__ == "__main__":
    main()