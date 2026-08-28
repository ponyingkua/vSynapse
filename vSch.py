"""
vSch.py
Generate visual chart lengkap dengan panel samping, S/D box transparan, dan Market Structure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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

    # Menggunakan rasio layout yang pas (lebar 13, tinggi 7.2) untuk memberi ruang panel kanan
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7.2),
        gridspec_kw={'height_ratios': [4.2, 0.75], 'hspace': 0.06},
        sharex=True
    )
    
    # Warna latar belakang ala tema bersih/terang atau disesuaikan dengan selera
    bg_color = '#121212'
    fig.patch.set_facecolor(bg_color)
    ax1.set_facecolor(bg_color)
    ax2.set_facecolor(bg_color)

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

    ax2.plot(x_indices, df['VOL_MA'], color='#ff9800', linewidth=1.2, alpha=0.7)
    ax1.plot(x_indices, df['EMA20'], color='#29b6f6', linewidth=1.4, label='EMA 20', zorder=3)
    ax1.plot(x_indices, df['EMA50'], color='#ffa726', linewidth=1.4, label='EMA 50', zorder=3)

    # Batasi sumbu X agar ada ruang untuk label di kanan
    gap_from_candle = 3.0
    label_width_est = 7.0
    extra_margin = gap_from_candle + label_width_est
    label_x = last_x + gap_from_candle
    ax1.set_xlim(-0.6, last_x + extra_margin)
    ax2.set_xlim(-0.6, last_x + extra_margin)

    # --- PENGATURAN SUMBU Y AGAR TIDAK TERLALU PANJANG (FIX OVER-STRETCHED) ---
    entry = setup['entry']
    sl = setup['sl']
    price_min = min(df['low'].min(), entry, sl)
    price_max = max(df['high'].max(), entry, sl)
    y_span = price_max - price_min
    if y_span == 0:
        y_span = price_max * 0.05
    y_padding = y_span * 0.15
    ax1.set_ylim(price_min - y_padding, price_max + y_padding)
    # ------------------------------------------------------------------------

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

    # --- TAMBAHAN: TRANSPARANT SUPPLY / DEMAND BOX ---
    is_long = setup['bias'] == "LONG"
    if is_long and labeled_lows:
        box_idx, box_price, _ = labeled_lows[-1]
        box_low = box_price
        box_high = box_low + (y_span * 0.03)
        box_color = '#26a69a'
        box_label = "DEMAND ZONE"
    elif not is_long and labeled_highs:
        box_idx, box_price, _ = labeled_highs[-1]
        box_high = box_price
        box_low = box_high - (y_span * 0.03)
        box_color = '#ef5350'
        box_label = "SUPPLY ZONE"
    else:
        box_idx, box_high, box_low, box_color, box_label = None, 0, 0, '#9e9e9e', ""

    if box_label:
        rect = Rectangle((box_idx - 1, box_low), last_x - (box_idx - 1) + gap_from_candle, box_high - box_low,
                         facecolor=box_color, alpha=0.15, edgecolor=box_color, linewidth=0.8, linestyle='--', zorder=2)
        ax1.add_patch(rect)
        ax1.text(box_idx, box_high + (y_span * 0.005), f"{box_label}", color=box_color, fontsize=6.5, 
                 fontweight='bold', va='bottom', ha='left', zorder=4)

    # Render label Market Structure
    for idx, price, label in labeled_highs[-2:]:
        color = '#2e7d32' if label == 'HH' else '#ef5350'
        ax1.text(idx, price + (y_span * 0.02), label, color=color, fontsize=7, fontweight='bold', ha='center', va='bottom', zorder=7)

    for idx, price, label in labeled_lows[-2:]:
        color = '#26a69a' if label == 'HL' else '#ef5350'
        ax1.text(idx, price - (y_span * 0.02), label, color=color, fontsize=7, fontweight='bold', ha='center', va='top', zorder=7)

    # Garis Level (Entry, SL, TP)
    dec = setup['decimals']
    levels = [
        (setup['entry'], '#29b6f6', f"Entry {format_price(setup['entry'], dec)}"),
        (setup['sl'], '#ef5350', f"SL {format_price(setup['sl'], dec)}"),
    ]
    if 'tp1' in setup and setup['tp1']:
        levels.append((setup['tp1'], '#26a69a', f"TP1 {format_price(setup['tp1'], dec)}"))
    if 'tp2' in setup and setup['tp2']:
        levels.append((setup['tp2'], '#26a69a', f"TP2 {format_price(setup['tp2'], dec)}"))
    if 'tp3' in setup and setup['tp3']:
        levels.append((setup['tp3'], '#26a69a', f"TP3 {format_price(setup['tp3'], dec)}"))

    for val, color, label in levels:
        ax1.axhline(y=val, color=color, linestyle='-.', linewidth=1.0, alpha=0.8, zorder=2)
        ax1.text(label_x, val, f" {label} ", color=color, fontsize=7, fontweight='bold', va='center', ha='left', zorder=5)

    # --- PANEL INFORMASI DI SISI KANAN LUAR CHART ---
    coin = symbol.replace('USDT', '')
    bias = setup['bias']
    score = setup.get('score', 0)
    
    # Judul Utama di Kiri Atas
    fig.text(0.08, 0.95, f"{symbol}  |  {bias}", fontsize=15, fontweight='bold', color='#ffffff', ha='left', va='top')
    fig.text(0.08, 0.915, f"Score {score}  |  Setup Kualitas Tinggi", fontsize=8.5, color='#b0bec5', ha='left', va='top')

    # Panel Kanan (Trade Levels & Invalidation)
    r_x = 0.74
    fig.text(r_x, 0.95, "TRADE LEVELS", fontsize=9, fontweight='bold', color='#ffffff', ha='left', va='top')
    fig.text(r_x, 0.92, f"Entry  {format_price(setup['entry'], dec)}", fontsize=8, color='#29b6f6', ha='left', va='top')
    fig.text(r_x, 0.89, f"SL     {format_price(setup['sl'], dec)}", fontsize=8, color='#ef5350', ha='left', va='top')
    if 'tp1' in setup:
        fig.text(r_x, 0.86, f"TP1    {format_price(setup['tp1'], dec)}", fontsize=8, color='#26a69a', ha='left', va='top')
    if 'tp2' in setup:
        fig.text(r_x, 0.83, f"TP2    {format_price(setup['tp2'], dec)}", fontsize=8, color='#26a69a', ha='left', va='top')

    fig.text(r_x, 0.78, "INVALIDATION", fontsize=9, fontweight='bold', color='#ffffff', ha='left', va='top')
    fig.text(r_x, 0.75, setup.get('invalidation', 'Close beyond SL level'), fontsize=7.5, color='#ef5350', ha='left', va='top', wrap=True)

    # Styling Grid & Axis
    grid_c = '#262626'
    axis_c = '#888888'
    ax1.grid(True, linestyle='-', alpha=0.3, color=grid_c)
    ax2.grid(True, linestyle='-', alpha=0.3, color=grid_c)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    ax1.set_ylabel('Price', fontsize=8, color=axis_c)
    ax2.set_ylabel('Vol', fontsize=8, color=axis_c)
    ax1.tick_params(colors=axis_c, labelsize=7)
    ax2.tick_params(colors=axis_c, labelsize=7)
    ax1.tick_params(labelbottom=False)

    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444444')
        ax.spines['bottom'].set_color('#444444')

    ticks_idx = np.linspace(0, len(df) - 1, 5, dtype=int)
    ax2.set_xticks(ticks_idx)
    ax2.set_xticklabels([df['timestamp'].iloc[t].strftime('%d %b %H:%M') for t in ticks_idx], fontsize=7, color=axis_c)

    fig.text(0.08, 0.015, "Synaptic data visualization | Powered by vSch", fontsize=7, color='#777777', ha='left', va='bottom')

    plt.subplots_adjust(left=0.08, right=0.71, top=0.88, bottom=0.08)
    plt.savefig(output_path, dpi=140, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
