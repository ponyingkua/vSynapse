"""
vSch.py (Final with JSON Support & Custom Chart Settings)
- 25 Candles
- EMA 200 & Supertrend (10, 2.5)
- No Supply/Demand box
- Header: Black color, flush with chart top, aligned to last candle
- Automatic JSON file handling from result/synaptic_candidates.json
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


def calculate_supertrend(df, period=10, multiplier=2.5):
    hl2 = (df['high'] + df['low']) / 2
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            continue
        
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_band.iloc[i]
        elif upper_band.iloc[i] < upper_band.iloc[i-1] and df['close'].iloc[i-1] <= upper_band.iloc[i-1]:
            upper_band.iloc[i] = upper_band.iloc[i]
        else:
            upper_band.iloc[i] = min(upper_band.iloc[i], upper_band.iloc[i-1])

        if df['close'].iloc[i] < lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_band.iloc[i]
        elif lower_band.iloc[i] > lower_band.iloc[i-1] and df['close'].iloc[i-1] >= lower_band.iloc[i-1]:
            lower_band.iloc[i] = lower_band.iloc[i]
        else:
            lower_band.iloc[i] = max(lower_band.iloc[i], lower_band.iloc[i-1])

        if direction.iloc[i-1] == 1:
            if df['close'].iloc[i] < lower_band.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > upper_band.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, direction


def draw_visual_chart(df, symbol, setup, output_path, config=None):
    if config is None:
        config = {}

    chart_candles = config.get('chart_candles', 25)

    df = df.copy()
    df['EMA200'] = df['close'].ewm(span=200, min_periods=1).mean()
    st_vals, st_dir = calculate_supertrend(df, period=10, multiplier=2.5)
    df['ST'] = st_vals
    df['ST_DIR'] = st_dir
    df['VOL_MA'] = df['volume'].rolling(20, min_periods=1).mean()

    n_show = min(chart_candles, len(df))
    df = df.iloc[-n_show:].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 5.5),
        gridspec_kw={'height_ratios': [4.0, 0.8], 'hspace': 0.05},
        sharex=True
    )
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#ffffff')
    ax2.set_facecolor('#ffffff')

    x_indices = np.arange(len(df))
    last_x = int(x_indices[-1])

    for i in range(len(df)):
        open_p = df['open'].iloc[i]
        close_p = df['close'].iloc[i]
        high_p = df['high'].iloc[i]
        low_p = df['low'].iloc[i]
        color = '#26a69a' if close_p >= open_p else '#ef5350'
        ax1.plot([i, i], [low_p, high_p], color=color, linewidth=1.5, solid_capstyle='round')
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.015)
        ax1.add_patch(Rectangle((i - 0.42, body_bottom), 0.84, body_height, color=color, alpha=0.95, linewidth=0))
        vol_color = '#26a69a55' if close_p >= open_p else '#ef535055'
        ax2.bar(i, df['volume'].iloc[i], color=vol_color, width=0.84, linewidth=0)

    ax2.plot(x_indices, df['VOL_MA'], color='#e65100', linewidth=1.2, alpha=0.7)
    
    ax1.plot(x_indices, df['EMA200'], color='#673ab7', linewidth=1.5, label='EMA 200', zorder=3)
    
    st_up = np.where(df['ST_DIR'] == 1, df['ST'], np.nan)
    st_down = np.where(df['ST_DIR'] == -1, df['ST'], np.nan)
    ax1.plot(x_indices, st_up, color='#2e7d32', linewidth=1.5, label='Supertrend', zorder=3)
    ax1.plot(x_indices, st_down, color='#c62828', linewidth=1.5, zorder=3)

    gap_from_candle = 3.0
    label_width_est = 8.5
    gap_from_edge = 1.5
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
    y_padding = y_span * 0.12
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)

    # Market Structure Labels (HH, LH, HL, LL)
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_w = 2
    
    raw_highs = []
    for i in range(swing_w, n - swing_w):
        if highs[i] == highs[i - swing_w:i + swing_w + 1].max() and highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            raw_highs.append((i, float(highs[i])))

    raw_lows = []
    for i in range(swing_w, n - swing_w):
        if lows[i] == lows[i - swing_w:i + swing_w + 1].min() and lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
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

    for idx, price, label in labeled_highs[-2:]:
        color = '#2e7d32' if label == 'HH' else '#c62828'
        ax1.text(idx, price + (y_span * 0.030), label, color=color, fontsize=7.0, fontweight='bold', ha='center', va='bottom', zorder=7)

    for idx, price, label in labeled_lows[-2:]:
        color = '#26a69a' if label == 'HL' else '#c62828'
        ax1.text(idx, price - (y_span * 0.030), label, color=color, fontsize=7.0, fontweight='bold', ha='center', va='top', zorder=7)

    levels = [
        (entry, '#1565c0', f"ENTRY  {format_price(entry, setup.get('decimals', 4))}"),
        (tp1, '#00897b', f"TP1  {format_price(tp1, setup.get('decimals', 4))}"),
        (tp2, '#00695c', f"TP2  {format_price(tp2, setup.get('decimals', 4))}"),
        (sl, '#c62828', f"SL  {format_price(sl, setup.get('decimals', 4))}"),
    ]

    for val, color, _ in levels:
        ax1.axhline(y=val, color=color, linestyle='--', linewidth=1.0, alpha=0.7, zorder=2)

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
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.35', alpha=0.95),
                 va='center', ha='left', fontweight='bold', fontsize=7.5, zorder=8, clip_on=False)

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

    ticks_idx = np.linspace(0, len(df) - 1, min(6, len(df)), dtype=int)
    ax2.set_xticks(ticks_idx)
    ax2.set_xticklabels([df['timestamp'].iloc[t].strftime('%d %b  %H:%M') for t in ticks_idx],
                        fontsize=7.5, color=axis_c)

    # --- HEADER INFO ---
    plt.subplots_adjust(left=0.06, right=0.88, top=0.95, bottom=0.12)
    
    symbol = setup.get('symbol', 'BTCUSDT')
    side = setup.get('side', 'LONG').upper()
    exec_tf = setup.get('execution_tf', '15m').upper()
    change_24h = setup.get('change24h', 0.0)
    q_vol = setup.get('quote_volume24h', 0.0)
    current_price = df['close'].iloc[-1]
    dec = setup.get('decimals', 4)

    header_text = (
        f" {symbol} ({exec_tf})  |  "
        f"BIAS: {side}  |  "
        f"Price: {format_price(current_price, dec)}  |  "
        f"24h Change: {change_24h:+.2f}%  |  "
        f"24h Vol: ${q_vol:,.0f} "
    )

    ax1.text(
        0.0, 1.015, header_text,
        fontsize=8.5, fontweight='bold', color='#ffffff',
        bbox=dict(facecolor='#111111', edgecolor='none', boxstyle='square,pad=0.5', alpha=0.95),
        transform=ax1.transAxes, zorder=15, ha='left', va='bottom'
    )

    plt.savefig(output_path, dpi=140, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="vSch Visualizer")
    parser.add_argument("--input", default="result/synaptic_candidates.json", help="Path to candidates json")
    args = parser.parse_args()

    input_path = Path(args.input)
    
    # Otomatis pastikan folder result/ ada jika belum ada
    input_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Input file {input_path} not found. Membuat contoh template JSON...")
        sample_data = {
            "candidates": [
                {
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "execution_tf": "15m",
                    "change24h": 2.5,
                    "quote_volume24h": 1200000000,
                    "entry": 65000.0,
                    "sl": 64000.0,
                    "tp": [67000.0, 69000.0],
                    "chart_data": {
                        "15m": [
                            {"time": "2026-08-29 00:00:00", "open": 64500, "high": 65200, "low": 64400, "close": 65000, "volume": 100}
                        ]
                    }
                }
            ]
        }
        input_path.write_text(json.dumps(sample_data, indent=4), encoding="utf-8")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", [])

    if not candidates:
        print("No candidates found in JSON.")
        return

    charts_dir = Path("result/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        symbol = c['symbol']
        side = c.get('side', 'LONG')
        exec_tf = c.get('execution_tf', '15m')
        print(f"Rendering chart for {symbol} ({side}) on timeframe {exec_tf}...")

        chart_data = c.get('chart_data', {})
        candles_raw = chart_data.get(exec_tf, [])

        if not candles_raw:
            for fallback_tf in ['15m', '1h', '4h']:
                if fallback_tf in chart_data:
                    candles_raw = chart_data[fallback_tf]
                    exec_tf = fallback_tf
                    c['execution_tf'] = exec_tf
                    break

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

        output_file = charts_dir / f"{symbol}_{side}_{exec_tf}_chart.png"
        draw_visual_chart(df, symbol, c, str(output_file))


if __name__ == "__main__":
    main()
