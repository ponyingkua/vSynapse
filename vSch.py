"""
vSch.py
Generate visual chart untuk setup trading dengan dukungan Supply/Demand Box & Market Structure.
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
import requests


def format_price(val, decimals):
    return f"{val:.{decimals}f}"


def draw_visual_chart(df, symbol, setup, output_path, config=None):
    """
    Draw chart setup dengan tema terang, kotak S/D transparan, dan struktur pasar dinamis.
    """
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

    # Menggunakan layout ukuran 12 x 6.8 dengan rasio grid [4.2, 0.75] sesuai kode kedua Anda
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.8),
        gridspec_kw={'height_ratios': [4.2, 0.75], 'hspace': 0.06},
        sharex=True
    )
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

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

    # Ambil data level dari setup
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

    # --- TRANSPARANT SUPPORT / DEMAND BOX (ZONA ORDER BLOCK) ---
    bias = setup.get('side', setup.get('bias', 'LONG')).upper()
    is_long = (bias == "LONG")
    
    if is_long and labeled_lows:
        box_idx, box_price, _ = labeled_lows[-1]
        box_low = box_price
        box_high = box_low + (y_span * 0.025)
        box_color = '#26a69a'
        box_label = "DEMAND ZONE"
    elif not is_long and labeled_highs:
        box_idx, box_price, _ = labeled_highs[-1]
        box_high = box_price
        box_low = box_high - (y_span * 0.025)
        box_color = '#ef5350'
        box_label = "SUPPLY ZONE"
    else:
        box_idx, box_high, box_low, box_color, box_label = None, 0, 0, '#9e9e9e', ""

    if box_label and box_idx is not None:
        rect = Rectangle((box_idx - 1, box_low), last_x - (box_idx - 1) + gap_from_candle, box_high - box_low,
                         facecolor=box_color, alpha=0.18, edgecolor=box_color, linewidth=0.8, linestyle='--', zorder=2)
        ax1.add_patch(rect)
        ax1.text(box_idx, box_high + (y_span * 0.005), f"{box_label}", color=box_color, fontsize=7, 
                 fontweight='bold', va='bottom', ha='left', zorder=4)

    # Render label Market Structure di chart
    for idx, price, label in labeled_highs[-2:]:
        color = '#2e7d32' if label == 'HH' else '#c62828'
        ax1.text(idx, price + (y_span * 0.028), label, color=color, fontsize=7, fontweight='bold',
                 ha='center', va='bottom', zorder=7)

    for idx, price, label in labeled_lows[-2:]:
        color = '#2e7d32' if label == 'HL' else '#c62828'
        ax1.text(idx, price - (y_span * 0.028), label, color=color, fontsize=7, fontweight='bold',
                 ha='center', va='top', zorder=7)

    # Panah indikasi setup
    arrow_color = '#00897b' if is_long else '#c62828'
    rad = -0.28 if is_long else 0.28
    ax1.annotate('', xy=(last_x + gap_from_candle * 0.7, tp1),
                xytext=(last_x, entry),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.35, linestyle='--',
                                connectionstyle=f'arc3,rad={rad}', mutation_scale=12, alpha=0.8))

    dec = setup.get('decimals', 4)
    levels = [
        (entry, '#1565c0', f"ENTRY  {format_price(entry, dec)}"),
        (tp1, '#00897b', f"TP1  {format_price(tp1, dec)}"),
        (tp2, '#00695c', f"TP2  {format_price(tp2, dec)}"),
        (sl, '#c62828', f"SL  {format_price(sl, dec)}"),
    ]

    for val, color, _ in levels:
        ax1.axhline(y=val, color=color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

    # Label harga di sebelah kanan agar tidak saling bertumpuk
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
                 va='center', ha='left', fontweight='bold', fontsize=7.5, zorder=8, clip_on=False)

    coin = symbol.replace('USDT', '')
    style_label = setup.get('setup_style', 'continuation').upper()
    structure_label = setup.get('structure', 'neutral').upper()
    conf = setup.get('confidence', setup.get('score', 0))

    fig.text(0.08, 0.965, f"${coin}/USDT 1H - {bias} SETUP",
             fontsize=13, fontweight='bold', color='#212121', ha='left', va='top')
    fig.text(0.08, 0.932, f"{structure_label}  ·  {style_label}  ·  Conf {conf}%",
             fontsize=9, color='#455a64', ha='left', va='top', fontweight='medium')

    grid_c = '#9e9e9e'
    axis_c = '#555555'
    spine_c = '#9e9e9e'
    ax1.grid(True, linestyle='-', alpha=0.35, color=grid_c)
    ax2.grid(True, linestyle='-', alpha=0.35, color=grid_c)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    leg = ax1.legend(loc='upper left', fontsize=7.5, framealpha=0.95, facecolor='#ffffff',
                     edgecolor='#bdbdbd', labelcolor='#333333', borderpad=0.4)
    leg.get_frame().set_linewidth(0.7)

    ax1.set_ylabel('Price (USDT)', fontsize=8.5, color=axis_c, labelpad=5)
    ax2.set_ylabel('Vol', fontsize=8, color=axis_c, labelpad=5)
    ax1.tick_params(colors=axis_c, labelcolor=axis_c, labelsize=7.5)
    ax2.tick_params(colors=axis_c, labelcolor=axis_c, labelsize=7.5)
    ax1.tick_params(labelbottom=False)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(spine_c)
        ax.spines['bottom'].set_color(spine_c)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    ticks_idx = np.linspace(0, len(df) - 1, 6, dtype=int)
    ax2.set_xticks(ticks_idx)
    ax2.set_xticklabels([df['timestamp'].iloc[t].strftime('%d %b  %H:%M') for t in ticks_idx],
                        fontsize=7.5, color=axis_c)

    fig.text(0.08, 0.012, f"BINANCE FUTURES  ·  ${coin}/USDT  ·  1H",
             fontsize=7, color='#555555', ha='left', va='bottom')
    fig.text(0.92, 0.012, "Not financial advice",
             fontsize=7, color='#555555', ha='right', va='bottom')

    plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.07)
    plt.savefig(output_path, dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.18)
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

    # Pastikan direktori charts/ dibuat otomatis agar artifact tidak kosong
    charts_dir = Path("charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        symbol = c['symbol']
        side = c.get('side', c.get('bias', 'LONG'))
        print(f"Rendering chart for {symbol} ({side})...")

        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100"
        try:
            resp = requests.get(url, timeout=10)
            raw = resp.json()
            if not isinstance(raw, list):
                print(f"Failed to fetch klines for {symbol}")
                continue

            df = pd.DataFrame(raw, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

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
        except Exception as e:
            print(f"Error rendering {symbol}: {e}")


if __name__ == "__main__":
    main()
