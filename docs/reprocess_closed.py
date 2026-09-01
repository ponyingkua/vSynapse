#!/usr/bin/env python3
"""reprocess_closed.py - ONE-OFF: re-evaluate already-CLOSED trades in
trade_log.json using the patched simulate_outcome() (fix bug #3: entry-vs-SL
same-candle disambiguation), without touching "pending" or "open".

Why this exists: winrate_stats.json's closed trades were computed with the
OLD simulate_outcome() before the entry-vs-SL disambiguation patch. New
trades ingested from now on will automatically use the patched logic via the
normal tracker.py cron run - but the 10 trades already sitting in "closed"
with bars_held == 1 were judged with the old (buggy) logic and won't be
touched again by the normal pipeline, since CLOSED is terminal.

This script:
  1. Loads trade_log.json (does NOT touch "pending" or "open").
  2. Finds closed WIN/LOSS records that (a) resolved on the very first bar
     after trigger (bars_held == 1) and (b) went through the entry-touch
     flow (no partial_check_from - i.e. not an ENTRY_READY candidate,
     which never had a discrete "entry touch" moment to disambiguate).
  3. Re-fetches the SAME historical klines (Binance history is immutable,
     so this is safe) starting at triggered_at, on execution_tf.
  4. Re-runs the patched simulate_outcome() against those candles.
  5. Overwrites the record's outcome/exit_price/r_multiple/flags with the
     new result, and prints a before/after diff for every record whose
     outcome actually changed.
  6. Recomputes winrate_stats.json from the updated trade_log.

This script is meant to be run ONCE, then discarded - it is not part of the
regular tracker.py pipeline and nothing depends on it existing afterwards.

Usage:
    python3 reprocess_closed.py --log trade_log.json --stats-out winrate_stats.json
"""

import argparse
import json
from pathlib import Path

from tracker import (
    TF_MS,
    compute_r_multiple,
    compute_stats,
    fetch_klines,
    iso_to_ms,
    ms_to_iso,
    simulate_outcome,
)


def _is_reprocess_target(record):
    if record.get("outcome") not in ("WIN", "LOSS"):
        return False
    if record.get("bars_held") != 1:
        return False
    if record.get("partial_check_from"):
        # ENTRY_READY flow - never had a discrete entry-touch moment,
        # fix bug #3 doesn't apply here.
        return False
    if "triggered_at" not in record:
        return False
    return True


def reprocess(trade_log, klines_limit):
    targets = [r for r in trade_log.get("closed", []) if _is_reprocess_target(r)]
    print(f"Found {len(targets)} closed first-bar record(s) eligible for reprocessing.")

    changed = 0
    unchanged = 0
    skipped = 0

    for record in targets:
        triggered_ms = iso_to_ms(record["triggered_at"])
        tf = record["execution_tf"]

        klines = fetch_klines(record["symbol"], tf, triggered_ms, limit=klines_limit)
        record_klines = [k for k in klines if k[0] >= triggered_ms]

        if not record_klines:
            print(f"  ! {record['id']}: no candles refetched, skipping")
            skipped += 1
            continue

        result = simulate_outcome(record, record_klines)

        if result is None:
            print(f"  ! {record['id']}: patched logic didn't resolve within refetched window, skipping")
            skipped += 1
            continue

        old_outcome = record.get("outcome")
        old_exit = record.get("exit_price")
        old_bars_held = record.get("bars_held")

        record.update(result)
        record["closed_at"] = ms_to_iso(result["bar_time"])
        record["r_multiple"] = compute_r_multiple(record)
        record["reprocessed"] = True

        if record["outcome"] != old_outcome or record.get("bars_held") != old_bars_held:
            changed += 1
            print(
                f"  CHANGED {record['id']}: {old_outcome} (exit {old_exit}, bar1) "
                f"-> {record['outcome']} (exit {record['exit_price']}, "
                f"bars_held={record['bars_held']}, resolved_via={result.get('resolved_via')})"
            )
        else:
            unchanged += 1

    print(f"\nDone: {changed} changed, {unchanged} unchanged (confirmed same outcome), {skipped} skipped.")
    return changed, unchanged, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="trade_log.json")
    parser.add_argument("--stats-out", default="winrate_stats.json")
    parser.add_argument("--klines-limit", type=int, default=1000)
    args = parser.parse_args()

    log_path = Path(args.log)
    trade_log = json.loads(log_path.read_text(encoding="utf-8"))

    reprocess(trade_log, args.klines_limit)

    log_path.write_text(json.dumps(trade_log, indent=2), encoding="utf-8")

    stats = compute_stats(trade_log)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(
        f"\nUpdated win rate: {stats['win_rate_pct']}% "
        f"(W{stats['wins']}/L{stats['losses']}), "
        f"resolved_at_first_bar={stats['resolved_at_first_bar']} "
        f"(W{stats['resolved_at_first_bar_wins']}/L{stats['resolved_at_first_bar_losses']})"
    )


if __name__ == "__main__":
    main()
