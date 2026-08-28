"""
vSch.py (Final with JSON Support & Custom Chart Settings)
- 25 Candles
- EMA 200 & Supertrend (10, 2.5)
- No Supply/Demand box
- Header: Black color, flush with chart top, aligned to last candle
- Reads the JSON produced directly by Synaptic.py (synaptic_candidates.json)
- Writes charts to charts/ by default
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
    tp3 = tps[2] if len(tps) > 2 else setup.get('tp3', tp2)

    all_levels = [float(df['low'].min()), float(df['high'].max()), entry, sl, tp1, tp2, tp3]
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
        (tp3, '#004d40', f"TP3  {format_price(tp3, setup.get('decimals', 4))}"),
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


def _load_candidates(input_path):
    """Load the exact JSON contract emitted by Synaptic.py."""
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {input_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"{input_path} must contain a JSON object.")

    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        raise RuntimeError("JSON field 'candidates' must be a list.")

    return data, candidates


def _build_dataframe(candidate, execution_tf):
    """Read candles from Synaptic's chart_data without fetching Binance again."""
    chart_data = candidate.get("chart_data")
    if not isinstance(chart_data, dict):
        raise ValueError("missing 'chart_data'")

    candles = chart_data.get(execution_tf)

    # Be tolerant if execution_tf uses a different case.
    if not candles:
        for key, value in chart_data.items():
            if str(key).lower() == str(execution_tf).lower():
                candles = value
                execution_tf = key
                break

    if not candles:
        raise ValueError(f"no candles for timeframe '{execution_tf}'")

    df = pd.DataFrame(candles)

    # Synaptic serializes the timestamp as "time". Accept timestamp too,
    # but never require a second Binance/API request.
    time_col = "time" if "time" in df.columns else "timestamp"
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing candle fields: {', '.join(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = (
        pd.to_datetime(df[time_col], utc=True, errors="coerce")
        if time_col in df.columns
        else pd.NaT
    )

    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("not enough valid candles")

    # Synaptic already calculated indicators and stored them in JSON.
    # vSch will use these values when available; otherwise it calculates
    # them locally from the same candles.
    for src, dst in [
        ("ema200", "EMA200"),
        ("supertrend", "ST"),
        ("st_dir", "ST_DIR"),
        ("volume_ma", "VOL_MA"),
    ]:
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    if "EMA200" not in df:
        df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    if "ST" not in df or "ST_DIR" not in df:
        st_vals, st_dir = calculate_supertrend(
            df, period=10, multiplier=2.5
        )
        df["ST"] = st_vals
        df["ST_DIR"] = st_dir

    if "VOL_MA" not in df:
        df["VOL_MA"] = df["volume"].rolling(20, min_periods=1).mean()

    return df, execution_tf


def _decimals_from_price(price):
    price = abs(float(price))
    if price < 0.0001:
        return 8
    if price < 0.001:
        return 7
    if price < 0.01:
        return 6
    if price < 0.1:
        return 5
    if price < 1:
        return 5
    if price < 10:
        return 4
    if price < 100:
        return 3
    return 2


def main():
    parser = argparse.ArgumentParser(description="vSch - Synaptic JSON chart renderer")
    parser.add_argument(
        "--input",
        default="synaptic_candidates.json",
        help="JSON produced by Synaptic.py",
    )
    parser.add_argument(
        "--output-dir",
        default="charts",
        help="Directory for generated PNG charts",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Render only this symbol",
    )
    parser.add_argument(
        "--chart-candles",
        type=int,
        default=25,
        help="Number of candles visible in the chart",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Synaptic JSON not found: {input_path}. "
            "Run Synaptic.py first."
        )

    _, candidates = _load_candidates(input_path)

    if not candidates:
        print("No candidates found in Synaptic JSON. Nothing to render.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for candidate in candidates:
        symbol = str(candidate.get("symbol", "")).upper()
        if args.symbol and symbol != args.symbol.upper():
            continue

        side = str(candidate.get("side", "LONG")).upper()
        execution_tf = str(candidate.get("execution_tf", "15m"))

        try:
            # The price levels MUST come from Synaptic's JSON.
            for field in ("entry", "sl", "tp"):
                if field not in candidate:
                    raise ValueError(f"missing top-level field '{field}'")

            entry = float(candidate["entry"])
            sl = float(candidate["sl"])
            tps = candidate.get("tp", [])
            if not isinstance(tps, list):
                raise ValueError("'tp' must be a list")
            if len(tps) < 1:
                raise ValueError("'tp' contains no targets")

            candidate["decimals"] = int(
                candidate.get("decimals", _decimals_from_price(entry))
            )

            df, actual_tf = _build_dataframe(candidate, execution_tf)
            candidate["execution_tf"] = str(actual_tf)

            output_file = output_dir / f"{symbol}_{side}_{actual_tf}_chart.png"

            print(
                f"Rendering {symbol} {side} {actual_tf} "
                f"from Synaptic JSON -> {output_file}"
            )
            draw_visual_chart(
                df,
                symbol,
                candidate,
                str(output_file),
                config={"chart_candles": max(5, args.chart_candles)},
            )
            rendered += 1

        except Exception as exc:
            print(f"Skipping {symbol or 'UNKNOWN'}: {exc}")

    print(f"Chart generation completed: {rendered} chart(s).")


if __name__ == "__main__":
    main()
