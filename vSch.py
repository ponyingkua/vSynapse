import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


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

    # Mengambil chart_data asli dari payload yang disediakan Synaptic
    exec_tf = c.get("execution_tf", "15m")
    chart_records = c.get("chart_data", {}).get(exec_tf, [])

    if not chart_records:
        raise SystemExit(f"No chart data found for timeframe {exec_tf} in candidate {symbol}")

    df = pd.DataFrame(chart_records)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    fig = plt.figure(
        figsize=(14, 9),
        facecolor="#0b0f14"
    )

    # Layout: Chart Utama di atas, Volume di bawah, Panel Info di kanan
    ax_main = fig.add_axes([0.05, 0.20, 0.68, 0.70])
    ax_vol = fig.add_axes([0.05, 0.08, 0.68, 0.10])
    ax_info = fig.add_axes([0.75, 0.08, 0.22, 0.82])

    ax_main.set_facecolor("#0b0f14")
    ax_vol.set_facecolor("#0b0f14")
    ax_info.set_facecolor("#0b0f14")

    # Batas Harga Chart berdasarkan data asli Synaptic
    all_prices = [entry, sl] + tps + list(df['high']) + list(df['low'])
    if 'ema200' in df.columns:
        all_prices += [p for p in df['ema200'].dropna() if p > 0]
    if 'supertrend' in df.columns:
        all_prices += [p for p in df['supertrend'].dropna() if p > 0]

    low_price = min(all_prices)
    high_price = max(all_prices)
    span = high_price - low_price
    padding = span * 0.15

    ax_main.set_ylim(low_price - padding, high_price + padding)
    ax_main.set_xlim(-1, len(df))

    # --- PLOTTING CANDLESTICK ASLI ---
    width = 0.6
    for i in range(len(df)):
        o = df['open'].iloc[i]
        cl = df['close'].iloc[i]
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]

        color = '#0ecb81' if cl >= o else '#f6465d'

        ax_main.plot([i, i], [l, h], color=color, linewidth=1, zorder=1)
        rect = Rectangle(
            (i - width/2, min(o, cl)), width, abs(cl - o),
            facecolor=color, edgecolor=color, zorder=2
        )
        ax_main.add_patch(rect)

    # --- PLOTTING INDIKATOR DARI SYNAPTIC ---
    if 'ema200' in df.columns:
        ax_main.plot(range(len(df)), df['ema200'], color='#f08c00', linewidth=1.2, label='EMA 200', zorder=3)

    if 'supertrend' in df.columns:
        ax_main.plot(range(len(df)), df['supertrend'], color='#748ffc', linewidth=1.0, linestyle='--', label='Supertrend', zorder=3)

    # --- GARIS LEVEL TRADING (Entry, TP, SL) ---
    levels = [
        ("SL", sl, "#ff6b6b", "-."),
        ("Entry", entry, "#4dabf7", "-"),
        ("TP1", tps[0], "#51cf66", "-."),
        ("TP2", tps[1], "#51cf66", "-."),
        ("TP3", tps[2], "#51cf66", "-."),
    ]

    for name, price, color, linestyle in levels:
        ax_main.axhline(
            price,
            color=color,
            linewidth=1.3,
            linestyle=linestyle,
            zorder=4
        )
        ax_main.text(
            len(df) - 1,
            price,
            f"  {name} {fmt(price)}",
            color=color,
            fontsize=8.5,
            va="center",
            fontweight="bold"
        )

    # --- VOLUME PLOT ---
    if 'volume' in df.columns:
        vol_colors = ['#0ecb81' if df['close'].iloc[i] >= df['open'].iloc[i] else '#f6465d' for i in range(len(df))]
        ax_vol.bar(range(len(df)), df['volume'], color=vol_colors, alpha=0.8, width=0.8)
    
    ax_vol.set_xticks([])
    ax_vol.set_yticks([])

    # --- STYLING AXES ---
    for ax in [ax_main, ax_vol]:
        ax.grid(True, alpha=0.12, linewidth=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors="#9aa7b5", labelsize=8)

    ax_main.set_ylabel("Price", color="#9aa7b5", fontsize=9)
    ax_main.set_xticks([])

    # --- HEADER & INFO PANEL KANAN ---
    fig.text(
        0.05, 0.94,
        f"{symbol}  |  {side}",
        fontsize=18,
        fontweight="bold",
        color="white"
    )

    fig.text(
        0.05, 0.905,
        f"Score {score:.2f}   |   MTF agreement {agreement}/3   |   15m momentum {momentum:+.3f}%",
        fontsize=10,
        color="#c8d1dc"
    )

    ax_info.axis("off")
    y = 0.96

    ax_info.text(0.02, y, "MULTI-TIMEFRAME", fontsize=10.5, fontweight="bold", color="white", transform=ax_info.transAxes)
    y -= 0.06

    for tf in ["15m", "1h", "4h"]:
        item = timeframes.get(tf)
        if not item:
            continue
        long_score = float(item.get("long", 0))
        short_score = float(item.get("short", 0))
        direction = "LONG" if long_score > short_score else ("SHORT" if short_score > long_score else "NEUTRAL")
        
        ax_info.text(0.02, y, f"{tf}   {direction}", fontsize=10, fontweight="bold", color="white", transform=ax_info.transAxes)
        y -= 0.04
        ax_info.text(0.02, y, f"Long {long_score:.2f}   Short {short_score:.2f}", fontsize=8, color="#9aa7b5", transform=ax_info.transAxes)
        y -= 0.065

    ax_info.text(0.02, y, "TRADE LEVELS", fontsize=10.5, fontweight="bold", color="white", transform=ax_info.transAxes)
    y -= 0.05

    for name, price in [("Entry", entry), ("TP1", tps[0]), ("TP2", tps[1]), ("TP3", tps[2]), ("SL", sl)]:
        ax_info.text(0.02, y, f"{name:<6} {fmt(price)}", fontsize=9, color="#c8d1dc", transform=ax_info.transAxes)
        y -= 0.04

    y -= 0.02
    ax_info.text(0.02, y, "INVALIDATION", fontsize=10.5, fontweight="bold", color="white", transform=ax_info.transAxes)
    y -= 0.05
    ax_info.text(0.02, y, c.get("invalidation", "Not provided"), fontsize=8, color="#ff8787", wrap=True, transform=ax_info.transAxes)

    y -= 0.11
    ax_info.text(0.02, y, "KEY POINTS", fontsize=10.5, fontweight="bold", color="white", transform=ax_info.transAxes)
    y -= 0.05
    for point in c.get("key_points", [])[:5]:
        ax_info.text(0.02, y, f"• {point}", fontsize=8, color="#c8d1dc", wrap=True, transform=ax_info.transAxes)
        y -= 0.04

    fig.text(
        0.05, 0.03,
        "Synaptic data visualization | Powered by vSch",
        fontsize=8,
        color="#6f7d8c"
    )

    out = Path(f"{symbol}_{side}_chart.png")
    fig.savefig(out, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="synaptic_candidates.json")
    parser.add_argument("--symbol", default="")
    args = parser.parse_args()

    data, candidates = load_candidates(args.input)

    if args.symbol:
        candidates = [item for item in candidates if item["symbol"].upper() == args.symbol.upper()]
        if not candidates:
            raise SystemExit(f"{args.symbol} not found.")

    candidates.sort(key=lambda item: float(item.get("score", 0)), reverse=True)
    candidate = candidates[0]

    print("=" * 60)
    print(f"Visualizing: {candidate['symbol']} {candidate['side']}")
    print(f"Score: {candidate.get('score', 0)}")
    print(f"TF agreement: {candidate.get('tf_agreement', 0)}/3")
    print(f"Entry: {fmt(candidate['entry'])}")
    print(f"SL: {fmt(candidate['sl'])}")
    print("=" * 60)

    out = render(candidate)
    print(f"Chart: {out}")


if __name__ == "__main__":
    main()
