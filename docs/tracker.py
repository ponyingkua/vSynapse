#!/usr/bin/env python3
"""tracker.py - Win-rate tracker for Synaptic Futures Journey candidates.

Reads new candidates from scan_results/<timestamp>/synaptic_candidates.json,
tracks them through a persistent state file (trade_log.json), and resolves
each one to WIN / LOSS / EXPIRED / NEVER_TRIGGERED by walking forward through
real Binance klines (candles) on the candidate's own execution_tf.

State machine per candidate:
  PENDING  -> waiting for price to touch the entry level (WAITING_RETEST /
              WAITING_PULLBACK). Expires to NEVER_TRIGGERED if entry is never
              touched within --pending-expiry-candles candles.
  OPEN     -> entry has occurred (or was ENTRY_READY at scan time). Walks
              forward candle by candle: whichever of SL / any TP is touched
              first decides the outcome. If neither is touched within
              --max-hold-candles candles, it EXPIRES (no result either way).
  CLOSED   -> terminal state, one of WIN / LOSS / EXPIRED / NEVER_TRIGGERED.

Run this once per scan cycle (same cron as the scanner), ideally right after
a new scan_results/<timestamp>/ folder is written and before belenggu.py
rebuilds the dashboard.

Usage:
    python3 tracker.py --results-dir scan_results --log trade_log.json \\
        --stats-out winrate_stats.json
"""

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

# Milliseconds per candle, used to convert "candles" into elapsed time.
TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


# TIME HELPERS

def iso_to_ms(iso_str):
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# BINANCE KLINES

def fetch_klines(symbol, interval, start_time_ms, limit=1000, end_time_ms=None):
    """Fetch OHLC candles from Binance USDT-M futures. Returns [] on any
    failure so the caller can safely retry next cycle instead of crashing
    the whole run over one bad symbol/network hiccup."""

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": str(start_time_ms),
        "limit": str(limit),
    }
    if end_time_ms is not None:
        params["endTime"] = str(end_time_ms)

    url = f"{BINANCE_KLINES_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"  ! kline fetch failed for {symbol} {interval}: {exc}")
        return []

    if isinstance(data, dict) and data.get("code"):
        print(f"  ! Binance error for {symbol} {interval}: {data}")
        return []

    return data


# OUTCOME LOGIC

def check_entry_triggered(entry_price, klines):
    """Entry counts as triggered the first candle whose [low, high] range
    contains the entry price - direction-agnostic, works for retest and
    pullback setups alike."""

    entry_price = float(entry_price)

    for k in klines:
        low = float(k[3])
        high = float(k[2])
        if low <= entry_price <= high:
            return k[6]  # close_time of the triggering candle

    return None


def simulate_outcome(record, klines):
    """Walk candles forward from entry. First touch of SL or any TP decides
    the outcome; if both are touched in the same candle, SL wins (the
    conservative assumption - you can't know which came first intra-candle).
    Returns None if unresolved within the given klines."""

    side = record["side"]
    sl = float(record["sl"])
    tps = [float(t) for t in (record.get("tp") or []) if t is not None]

    for idx, k in enumerate(klines):
        high = float(k[2])
        low = float(k[3])
        close_time = k[6]

        sl_hit = (low <= sl) if side == "LONG" else (high >= sl)

        tp_hit_index = None
        for i, tp in enumerate(tps):
            hit = (high >= tp) if side == "LONG" else (low <= tp)
            if hit:
                tp_hit_index = i
                break

        if sl_hit:
            return {
                "outcome": "LOSS",
                "exit_price": sl,
                "tp_index": None,
                "bar_time": close_time,
                "bars_held": idx + 1,
            }

        if tp_hit_index is not None:
            return {
                "outcome": "WIN",
                "exit_price": tps[tp_hit_index],
                "tp_index": tp_hit_index,
                "bar_time": close_time,
                "bars_held": idx + 1,
            }

    return None


def compute_r_multiple(record):
    """R multiple realized: -1.0 on a loss (exited exactly at SL), positive
    on a win sized by which TP was hit relative to initial risk."""

    try:
        entry = float(record["entry"])
        sl = float(record["sl"])
        exit_price = float(record["exit_price"])
    except (TypeError, ValueError, KeyError):
        return None

    risk = abs(entry - sl)
    if risk == 0:
        return None

    if record["side"] == "LONG":
        return round((exit_price - entry) / risk, 2)
    return round((entry - exit_price) / risk, 2)


# SCAN RESULT LOADING (self-contained - mirrors belenggu.py's loader)

def _iter_run_dirs_ascending(results_dir):
    if not results_dir.exists():
        return []
    dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.name)  # oldest first: ingest chronologically
    return dirs


def _load_run(run_dir):
    json_path = run_dir / "synaptic_candidates.json"
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skip {run_dir.name}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    return {"run_id": run_dir.name, "dir": run_dir, "data": data}


# TRADE LOG PERSISTENCE

def load_trade_log(path):
    if path.exists():
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
            log.setdefault("pending", [])
            log.setdefault("open", [])
            log.setdefault("closed", [])
            log.setdefault("ingested_runs", [])
            return log
        except json.JSONDecodeError as exc:
            print(f"trade_log corrupt ({exc}), starting fresh")
    return {"pending": [], "open": [], "closed": [], "ingested_runs": []}


def save_trade_log(path, trade_log):
    path.write_text(json.dumps(trade_log, indent=2), encoding="utf-8")


# INGEST NEW CANDIDATES

def ingest_new_runs(trade_log, results_dir):
    ingested = set(trade_log.get("ingested_runs", []))
    run_dirs = _iter_run_dirs_ascending(results_dir)
    new_run_names = []

    for run_dir in run_dirs:
        if run_dir.name in ingested:
            continue

        run = _load_run(run_dir)
        if run is None:
            ingested.add(run_dir.name)
            continue

        candidates = run["data"].get("candidates", [])
        generated_at = run["data"].get("generated_at") or ms_to_iso(now_ms())

        for c in candidates:
            symbol = c.get("symbol")
            side = c.get("side")
            tf = c.get("execution_tf")
            entry_state = c.get("entry_state")

            if not (symbol and side and tf and entry_state):
                continue
            if entry_state == "NO_SETUP":
                continue

            record = {
                "id": f"{symbol}_{side}_{tf}_{run['run_id']}",
                "symbol": symbol,
                "side": side,
                "execution_tf": tf,
                "setup_style": c.get("setup_style"),
                "entry_state": entry_state,
                "entry": c.get("entry"),
                "sl": c.get("sl"),
                "tp": c.get("tp", []),
                "decimals": c.get("decimals"),
                "run_id": run["run_id"],
                "first_seen": generated_at,
            }

            if entry_state == "ENTRY_READY":
                record["triggered_at"] = generated_at
                trade_log["open"].append(record)
            else:
                record["added_at"] = generated_at
                trade_log["pending"].append(record)

        ingested.add(run_dir.name)
        new_run_names.append(run_dir.name)

    trade_log["ingested_runs"] = sorted(ingested)
    return new_run_names


# UPDATE PENDING -> OPEN / NEVER_TRIGGERED

def update_pending(trade_log, pending_expiry_candles, klines_limit):
    still_pending = []
    current_ms = now_ms()

    for record in trade_log["pending"]:
        added_ms = iso_to_ms(record["added_at"])
        tf_ms = TF_MS.get(record["execution_tf"], TF_MS["1h"])

        klines = fetch_klines(record["symbol"], record["execution_tf"], added_ms, limit=klines_limit)

        if not klines:
            still_pending.append(record)
            continue

        trigger_time = check_entry_triggered(record["entry"], klines)
        elapsed_candles = (current_ms - added_ms) / tf_ms

        if trigger_time is not None:
            record["triggered_at"] = ms_to_iso(trigger_time)
            trade_log["open"].append(record)
        elif elapsed_candles >= pending_expiry_candles:
            record["outcome"] = "NEVER_TRIGGERED"
            record["closed_at"] = ms_to_iso(current_ms)
            trade_log["closed"].append(record)
        else:
            still_pending.append(record)

    trade_log["pending"] = still_pending


# UPDATE OPEN -> CLOSED (WIN / LOSS / EXPIRED)

def update_open(trade_log, max_hold_candles, klines_limit):
    still_open = []
    current_ms = now_ms()

    for record in trade_log["open"]:
        triggered_ms = iso_to_ms(record["triggered_at"])
        tf_ms = TF_MS.get(record["execution_tf"], TF_MS["1h"])

        klines = fetch_klines(record["symbol"], record["execution_tf"], triggered_ms, limit=klines_limit)

        if not klines:
            still_open.append(record)
            continue

        result = simulate_outcome(record, klines)

        if result is not None:
            record.update(result)
            record["closed_at"] = ms_to_iso(result["bar_time"])
            record["r_multiple"] = compute_r_multiple(record)
            trade_log["closed"].append(record)
            continue

        elapsed_candles = (current_ms - triggered_ms) / tf_ms
        if elapsed_candles >= max_hold_candles:
            last_close = float(klines[-1][4])
            record["outcome"] = "EXPIRED"
            record["exit_price"] = last_close
            record["closed_at"] = ms_to_iso(klines[-1][6])
            record["r_multiple"] = compute_r_multiple(record)
            trade_log["closed"].append(record)
        else:
            still_open.append(record)

    trade_log["open"] = still_open


# STATS

def _group_win_rate(records, key_fn):
    groups = {}
    for r in records:
        key = key_fn(r) or "UNKNOWN"
        g = groups.setdefault(key, {"wins": 0, "losses": 0})
        g["wins" if r["outcome"] == "WIN" else "losses"] += 1

    for g in groups.values():
        total = g["wins"] + g["losses"]
        g["win_rate_pct"] = round(g["wins"] / total * 100, 1) if total else None

    return groups


def compute_stats(trade_log):
    closed = trade_log.get("closed", [])
    resolved = [t for t in closed if t.get("outcome") in ("WIN", "LOSS")]
    wins = [t for t in resolved if t["outcome"] == "WIN"]
    losses = [t for t in resolved if t["outcome"] == "LOSS"]

    r_values = [t["r_multiple"] for t in resolved if t.get("r_multiple") is not None]
    avg_r = round(sum(r_values) / len(r_values), 2) if r_values else None

    return {
        "generated_at": ms_to_iso(now_ms()),
        "resolved_trades": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(resolved) * 100, 1) if resolved else None,
        "avg_r_multiple": avg_r,
        "by_setup_style": _group_win_rate(resolved, lambda t: t.get("setup_style")),
        "by_side": _group_win_rate(resolved, lambda t: t.get("side")),
        "expired_unresolved": len([t for t in closed if t.get("outcome") == "EXPIRED"]),
        "never_triggered": len([t for t in closed if t.get("outcome") == "NEVER_TRIGGERED"]),
        "currently_open": len(trade_log.get("open", [])),
        "currently_pending": len(trade_log.get("pending", [])),
    }


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="tracker - win-rate tracker for Synaptic Futures Journey candidates"
    )
    parser.add_argument("--results-dir", default="scan_results", help="Same folder belenggu.py reads.")
    parser.add_argument("--log", default="trade_log.json", help="Persistent trade state file.")
    parser.add_argument("--stats-out", default="winrate_stats.json", help="Aggregate stats output, read by belenggu.py.")
    parser.add_argument("--max-hold-candles", type=int, default=60, help="Bars an OPEN trade can run before it EXPIREs unresolved.")
    parser.add_argument("--pending-expiry-candles", type=int, default=15, help="Bars a PENDING entry can wait before NEVER_TRIGGERED.")
    parser.add_argument("--klines-limit", type=int, default=1000, help="Max candles fetched per API call (Binance cap is 1000).")

    args = parser.parse_args()

    log_path = Path(args.log)
    trade_log = load_trade_log(log_path)

    new_runs = ingest_new_runs(trade_log, Path(args.results_dir))
    if new_runs:
        print(f"Ingested {len(new_runs)} new run(s): {', '.join(new_runs)}")
    else:
        print("No new runs to ingest.")

    print(f"Checking {len(trade_log['pending'])} pending entr(y/ies)...")
    update_pending(trade_log, args.pending_expiry_candles, args.klines_limit)

    print(f"Checking {len(trade_log['open'])} open trade(s)...")
    update_open(trade_log, args.max_hold_candles, args.klines_limit)

    save_trade_log(log_path, trade_log)

    stats = compute_stats(trade_log)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"Resolved: {stats['resolved_trades']} "
        f"(W{stats['wins']}/L{stats['losses']}, "
        f"win rate {stats['win_rate_pct']}%), "
        f"open={stats['currently_open']}, pending={stats['currently_pending']}"
    )


if __name__ == "__main__":
    main()
