"""
vSch.py
Generate visual chart dengan Header Info, Box Transparan, Label HH/LL/HL/LH,
dan membaca data klines dari 'chart_data' di dalam JSON Synaptic.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def format_price(val, decimals):
    return f"{val:.{decimals}f}"


def draw_visual_chart(df, symbol, setup, output_path, config=None):
    if config is None:
        config = {}

    ema_fast = config.get('ema_fast', 20)
    ema_slow = config.get('ema_slow', 50)
    chart_candles = config.get('chart_candles', 48)

    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=ema_fast).mean()
    df['EMA50'] = df['close'].ewm(span=ema_slow).mean()
    df['VOL_MA'] = df['volume'].rolling(20, min_periods=1).mean()

    n_show = min(chart_candles, len(df))
    df = df.iloc[-n_show:].reset_index(drop=True)

    # Layout dengan ruang untuk Header di bagian atas
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 5.5),
        gridspec_kw={'height_ratios': [4.0, 0.8], 'hspace': 0.05},
        sharex=True
    )
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

    # --- HEADER INFO (Simbol, Side, Change 24h, Vol 24h, Harga) ---
    side = setup.get('side', 'LONG').upper()
    change_24h = setup.get('change24h', 0.0)
    q_vol = setup.get('quote_volume24h', 0.0)
    current_price = df['close'].iloc[-1]
    dec = setup.get('decimals', 4)

    chg_color = '#2e7d32' if change_24h >= 0 else '#c62828'
    side_color = '#1b5e20' if side == 'LONG' else '#b71c1c'

    header_text = (
        f"  {symbol}  |  "
        f"BIAS: {side}  |  "
        f"Price: {format_price(current_price, dec)}  |  "
        f"24h Change: {change_24h:+.2f}%  |  "
        f"24h Vol: ${q_vol:,.0f}"
    )
    fig.text(
        0.06, 0.96, header_text,
        fontsize=9.0, fontweight='bold', color='#ffffff',
        bbox=dict(facecolor=side_color, edgecolor='none', boxstyle='round,pad=0.4', alpha=0.95),
        transform=fig.transFigure, zorder=10
    )

    x_indices = np.arange(len(df))
    last_x = int(x_indices[-1])

    # Plot Candlesticks & Volume
    for i in range(len(df)):
        open_p = df['open'].iloc[i]
        close_p = df['close'].iloc[i]
        high_p = df['high'].iloc[i]
        low_p = df['low'].iloc[i]
        color = '#26a69a' if close_p >= open_p else '#ef5350'
        ax1.plot([i, i], [low_p, high_p], color=color, linewidth=1.3, solid_capstyle='round')
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.012)
        ax1.add_patch(Rectangle((i - 0.36, body_bottom), 0.72, body_height, color=color, alpha=0.92, linewidth=0))
        vol_color = '#26a69a4d' if close_p >= open_p else '#ef53504d'
        ax2.bar(i, df['volume'].iloc[i], color=vol_color, width=0.72, linewidth=0)

    ax2.plot(x_indices, df['VOL_MA'], color='#e65100', linewidth=1.2, alpha=0.7)
    ax1.plot(x_indices, df['EMA20'], color='#1565c0', linewidth=1.4, label='EMA 20', zorder=3)
    ax1.plot(x_indices, df['EMA50'], color='#ef6c00', linewidth=1.4, label='EMA 50', zorder=3)

    gap_from_candle = 4.0
    label_width_est = 9.5
    gap_from_edge = 1.8
    extra_margin = gap_from_candle + label_width_est + gap_from_edge
    label_x = last_x + gap_from_candle
    ax1.set_xlim(-0.6, last_x + extra_margin)
    ax2.set_xlim(-0.6, last_x + extra_margin)

    entry = setup['entry']
    sl = setup['sl']
    tps = setup.get('tp', [])
    tp1 = tps[0] if len(tps) > 0 else setup.get('tp1', entry)
    tp2 = tps[1] if len(tps) > 1 else setup.get('tp2', tp1)

    all_levels = [float(df['low'].min()), float(df['high'].max()), entry, sl, tp1, tp2]
    y_min, y_max = min(all_levels), max(all_levels)
    y_span = max(y_max - y_min, abs(y_min) * 0.01 if y_min != 0 else 0.01)
    y_padding = y_span * 0.16
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)

    # --- DINAMIS MARKET STRUCTURE (HH, LH, HL, LL) ---
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_w = 3
    
    raw_highs = []
    for i in range(swing_w, n - swing_w):
        if highs[i] == highs[i - swing_w:i + swing_w + 1].max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            raw_highs.append((i, float(highs[i])))

    raw_lows = []
    for i in range(swing_w, n - swing_w):
        if lows[i] == lows[i - swing_w:i + swing_w + 1].min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            raw_lows.append((i, float(lows[i])))

    labeled_highs = []
    last_sh_price = None
    for idx, price in raw_highs:
        label = 'HH' if (last_sh_price is None or price > last_sh_price) else 'LH'
        labeled_highs.append((idx, price, label))
        last_sh_price = price

    labeled_lows = []
    last_sl_price = None
    for idx, price in raw_lows:
        label = 'HL' if (last_sl_price is None or price > last_sl_price) else 'LL'
        labeled_lows.append((idx, price, label))
        last_sl_price = price

    # --- TRANSPARANT SUPPORT / DEMAND BOX ---
    is_long = (side == "LONG")
    
    if is_long and labeled_lows:
        box_idx, box_price, _ = labeled_lows[-1]
        box_low = box_price
        box_high = box_low + (y_span * 0.025)
        box_color = '#26a69a'
        box_label = "DEMAND"
    elif not is_long and labeled_highs:
        box_idx, box_price, _ = labeled_highs[-1]
        box_high = box_price
        box_low = box_high - (y_span * 0.025)
        box_color = '#ef5350'
        box_label = "SUPPLY"
    else:
        box_idx, box_high, box_low, box_color, box_label = None, 0, 0, '#9e9e9e', ""

    if box_label and box_idx is not None:
        rect = Rectangle((box_idx - 1, box_low), last_x - (box_idx - 1) + gap_from_candle, box_high - box_low,
                         facecolor=box_color, alpha=0.18, edgecolor=box_color, linewidth=0.8, linestyle='--', zorder=2)
        ax1.add_patch(rect)
        ax1.text(box_idx, box_high + (y_span * 0.005), f"{box_label}", color=box_color, fontsize=6.5, 
                 fontweight='bold', va='bottom', ha='left', zorder=4)

    for idx, price, label in labeled_highs[-2:]:
        color = '#2e7d32' if label == 'HH' else '#c62828'
        ax1.text(idx, price + (y_span * 0.028), label, color=color, fontsize=6.5, fontweight='bold', ha='center', va='bottom', zorder=7)

    for idx, price, label in labeled_lows[-2:]:
        color = '#26a69a' if label == 'HL' else '#c62828'
        ax1.text(idx, price - (y_span * 0.028), label, color=color, fontsize=6.5, fontweight='bold', ha='center', va='top', zorder=7)

    levels = [
        (entry, '#1565c0', f"ENTRY  {format_price(entry, dec)}"),
        (tp1, '#00897b', f"TP1  {format_price(tp1, dec)}"),
        (tp2, '#00695c', f"TP2  {format_price(tp2, dec)}"),
        (sl, '#c62828', f"SL  {format_price(sl, dec)}"),
    ]

    for val, color, _ in levels:
        ax1.axhline(y=val, color=color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    # Label harga di sebelah kanan
    ylim = ax1.get_ylim()
    view_span = ylim[1] - ylim[0]
    min_gap = view_span * 0.035
    sorted_lv = sorted([(val, color, label) for val, color, label in levels], key=lambda t: t[0])
    text_ys = [t[0] for t in sorted_lv]
    
    for i in range(1, len(text_ys)):
        if text_ys[i] - text_ys[i - 1] < min_gap:
            text_ys[i] = text_ys[i - 1] + min_gap
    for i in range(len(text_ys)):
        text_ys[i] = min(max(text_ys[i], ylim[0] + view_span * 0.02), ylim[1] - view_span * 0.02)
    for i in range(len(text_ys) - 2, -1, -1):
        if text_ys[i + 1] - text_ys[i] < min_gap:
            text_ys[i] = text_ys[i + 1] - min_gap

    for (val, color, label), ty in zip(sorted_lv, text_ys):
        ax1.text(label_x, ty, f" {label} ", color='#ffffff',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.32', alpha=0.95),
                 va='center', ha='left', fontweight='bold', fontsize=7.0, zorder=8, clip_on=False)

    grid_c = '#9e9e9e'
    axis_c = '#555555'
    spine_c = '#9e9e9e'
    ax1.grid(True, linestyle='-', alpha=0.35, color=grid_c)
    ax2.grid(True, linestyle='-', alpha=0.35, color=grid_c)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    leg = ax1.legend(loc='upper left', fontsize=7.0, framealpha=0.95, facecolor='#ffffff',
                     edgecolor='#bdbdbd', labelcolor='#333333', borderpad=0.4)
    leg.get_frame().set_linewidth(0.7)

    ax1.tick_params(colors=axis_c, labelcolor=axis_c, labelsize=7.0)
    ax2.tick_params(colors=axis_c, labelcolor=axis_c, labelsize=7.0)
    ax1.tick_params(labelbottom=False)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(spine_c)
        ax.spines['bottom'].set_color(spine_c)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    ticks_idx = np.linspace(0, len(df) - 1, min(6, len(df)), dtype=int)
    ax2.set_xticks(ticks_idx)
    ax2.set_xticklabels([df['timestamp'].iloc[t].strftime('%d %b  %H:%M') for t in ticks_idx],
                        fontsize=7.0, color=axis_c)

    plt.subplots_adjust(left=0.06, right=0.95, top=0.90, bottom=0.12)
    plt.savefig(output_path, dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="vSch Visualizer")
    parser.add_argument("--input", default="synaptic_candidates.json", help="Path to candidates json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    data = json.loads(input_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])

    if not candidates:
        print("No candidates found in JSON.")
        return

    charts_dir = Path("charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        symbol = c['symbol']
        side = c.get('side', 'LONG')
        print(f"Rendering chart for {symbol} ({side}) from JSON data...")

        chart_data = c.get('chart_data', {})
        candles_raw = chart_data.get('15m', [])

        if not candles_raw:
            print(f"No candle history found in JSON for {symbol}")
            continue

        df = pd.DataFrame(candles_raw)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        else:
            df['timestamp'] = pd.to_datetime('now')

        price_val = df['close'].iloc[-1]
        if price_val < 1:
            decimals = 5
        elif price_val < 10:
            decimals = 4
        elif price_val < 100:
            decimals = 3
        else:
            decimals = 2
        c['decimals'] = decimals

        output_file = charts_dir / f"{symbol}_{side}_chart.png"
        draw_visual_chart(df, symbol, c, str(output_file))


if __name__ == "__main__":
    main()
