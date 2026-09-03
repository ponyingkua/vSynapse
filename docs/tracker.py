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
  CLOSED   -> terminal state, one of WIN / LOSS / BREAKEVEN / EXPIRED /
              NEVER_TRIGGERED.

Run this once per scan cycle (same cron as the scanner), ideally right after
a new scan_results/<timestamp>/ folder is written and before belenggu.py
rebuilds the dashboard.

PATCH NOTES - dua bug bias-ke-LOSS yang ditemukan dari winrate_stats.json
(23.4% win rate yang mencurigakan rendah), keduanya diperbaiki di revisi ini:

  1. "Entry candle hilang dari simulasi" - check_entry_triggered() dulu
     mengembalikan close_time candle entry, lalu update_open() memfilter
     `k[0] >= triggered_ms`. Karena open_time candle manapun selalu <
     close_time-nya sendiri, candle entry OTOMATIS terpotong dari
     simulasi - move impulsif (entry disentuh lalu langsung lanjut ke TP
     di candle yang sama) tidak pernah tercatat WIN. FIX: sekarang
     mengembalikan open_time, jadi candle entry ikut disimulasikan.

  2. "SL vs TP dalam satu candle selalu dimenangkan SL" - kalau dalam satu
     candle SL dan TP sama-sama tersentuh (umum di TF kecil / candle
     bervolatilitas tinggi), tracker dulu selalu asumsi SL duluan. FIX:
     sekarang dicoba dibedah dulu pakai candle di timeframe yang lebih
     halus (DISAMBIGUATION_TF) untuk melihat urutan sebenarnya. Kalau
     tetap tidak bisa dibedah (sudah di TF paling halus / fetch gagal),
     baru fallback ke asumsi SL-duluan yang lama - tapi record ditandai
     intrabar_ambiguous=True supaya kelihatan di stats, bukan diam-diam
     dianggap pasti.

  3. "Entry vs SL di candle YANG SAMA juga selalu dimenangkan SL" - efek
     samping dari fix #1: begitu candle entry ikut disimulasikan (idx=0
     di simulate_outcome), kalau di candle itu juga SL kesentuh, kode
     lama langsung declare LOSS tanpa tahu urutan sebenarnya. FIX: mirip
     fix #2, candle entry yang juga kena SL sekarang dibedah dulu pakai
     sub-candle (DISAMBIGUATION_TF) - dicari sub-candle pertama yang
     benar-benar menyentuh harga entry, lalu urutan SL/TP dicek HANYA
     dari titik itu ke depan. Kalau tidak bisa dibedah, fallback ke
     perilaku lama (asumsi SL duluan) tapi ditandai
     entry_sl_ambiguous_fallback=True.

  4. (REVISI INI) "Breakeven-after-TP1 dari Synaptic_fixed.py tidak
     pernah disimulasikan" - begitu TP mana pun tersentuh, simulate_outcome
     dulu langsung `return` menutup trade saat itu juga. Akibatnya field
     `breakeven_after_tp1` / `sl_after_tp1` yang dikirim scanner
     (CONFIG["move_sl_to_breakeven_after_tp1"]) murni metadata pasif -
     tidak pernah benar-benar disimulasikan, jadi salah satu perbaikan
     risk-management scanner efektif TIDAK berpengaruh ke hasil backtest
     sama sekali. FIX: kalau breakeven_after_tp1=True dan candle HANYA
     menyentuh TP1 (bukan TP yang lebih jauh sekaligus, dan masih ada TP
     lain di atasnya), simulasi TIDAK berhenti - level stop efektif
     dipindah dari sl awal ke sl_after_tp1 (breakeven), lalu simulasi
     lanjut ke candle berikutnya. Outcome baru "BREAKEVEN" dipakai kalau
     posisi akhirnya keluar di level breakeven itu (bukan LOSS - risikonya
     sudah dikunci ke ~0 sejak TP1 tersentuh). Kalau breakeven_after_tp1
     tidak ada di record lama (trade sebelum fix ini), perilaku persis
     sama seperti sebelumnya (tidak ada regresi).

  5. (REVISI INI) "Field skor/regime tidak ikut dicatat" - ingest_new_runs()
     dulu cuma menyalin field yang dipakai untuk menentukan entry/exit
     (entry/sl/tp/setup_style/entry_state/htf_bias). score, btc_regime,
     stage1_score, funding_alert, tf_agreement, momentum_15m dari
     Synaptic_fixed.py tidak pernah ikut ke trade_log.json, padahal semua
     itu sudah tersedia di synaptic_candidates.json yang sama yang dibaca
     ingest_new_runs(). FIX: field-field itu sekarang ikut disalin,
     supaya compute_stats() (dan analisis manual di luar tracker) bisa
     mengecek apakah skor/regime tinggi benar-benar berkorelasi dengan
     win rate - sebelumnya mustahil dibuktikan karena datanya tidak
     pernah disimpan.

Usage:
    python3 tracker.py --results-dir scan_results --log trade_log.json \\
        --stats-out winrate_stats.json --config-version v4
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# BINANCE ENDPOINTS
#
# Ported from Synaptic.py: fallback antar-endpoint, retry per
# endpoint, penanganan status 451 / 429 / 418.
# ============================================================

BASE_URLS = [
    # www.binance.com ditaruh PERTAMA: dari log diagnostik run terakhir,
    # 100% fetch berhasil lewat endpoint ini dan 0% berhasil lewat 4
    # endpoint fapi* di bawah - IP runner GitHub Actions tampaknya
    # diblokir/di-throttle oleh fapi*.binance.com. Endpoint fapi* tetap
    # dipertahankan sebagai fallback (kalau www.binance.com suatu saat
    # kena rate limit/berubah), tapi sekarang di urutan belakang supaya
    # tidak membuang waktu retry ke endpoint yang hampir pasti gagal.
    "https://www.binance.com",
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

API_TIMEOUT = 10
API_RETRIES = 2
RETRY_BASE_DELAY = 0.35

# Berapa banyak fetch klines yang boleh berjalan bersamaan. Ini network-bound
# (nunggu response Binance), bukan CPU-bound, jadi thread biasa (bukan
# multiprocessing) sudah cukup - GIL tidak jadi bottleneck di sini.
MAX_FETCH_WORKERS = 8

# Dua record dengan symbol+timeframe yang sama dan start-time berdekatan
# (dalam window ini) digabung jadi SATU fetch klines, lalu hasilnya dipotong
# ulang per-record. Ini yang memangkas jumlah request untuk simbol yang
# sering muncul berulang (mis. DASHUSDT/LITUSDT muncul di banyak run
# berturut-turut). Nilainya sengaja kecil relatif terhadap
# --max-hold-candles/--pending-expiry-candles default supaya tidak
# memotong candle yang sebetulnya perlu dicek (lihat catatan di
# _bucket_groups).
DEDUPE_WINDOW_MS = 6 * 60 * 60 * 1000  # 6 jam

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

# Dipetakan ke timeframe yang lebih halus, dipakai _resolve_ambiguous_candle()
# dan _resolve_entry_candle_order() untuk membedah candle di mana beberapa
# level (entry/SL/TP) tersentuh sekaligus. None berarti sudah di TF paling
# halus yang didukung - tidak ada lagi TF di bawahnya untuk membedah lebih
# jauh, jadi harus fallback ke asumsi konservatif.
DISAMBIGUATION_TF = {
    "1m": None,
    "3m": "1m",
    "5m": "1m",
    "15m": "1m",
    "30m": "5m",
    "1h": "5m",
    "2h": "15m",
    "4h": "15m",
    "6h": "30m",
    "8h": "30m",
    "12h": "1h",
    "1d": "1h",
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


# ============================================================
# API ENGINE (ported from Synaptic.py)
# ============================================================

def _retry_delay(attempt, retry_after=None):
    if retry_after is not None:
        try:
            value = float(retry_after)
            return min(max(value, 0.2), 5.0)
        except (TypeError, ValueError):
            pass

    base = RETRY_BASE_DELAY
    delay = base * (2 ** attempt)
    delay += random.uniform(0.05, 0.20)
    return min(delay, 3.0)


def _parse_response(raw_body):
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None


def fetch_klines(symbol, interval, start_time_ms, limit=1000, end_time_ms=None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": str(start_time_ms),
        "limit": str(limit),
    }
    if end_time_ms is not None:
        params["endTime"] = str(end_time_ms)

    query = urllib.parse.urlencode(params)
    path = f"/fapi/v1/klines?{query}"

    last_error = None
    fetch_started = time.monotonic()

    for base_url in BASE_URLS:

        url = base_url + path

        for attempt in range(API_RETRIES + 1):

            try:

                req = urllib.request.Request(url, headers=HEADERS)

                with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:

                    status = resp.status
                    raw_body = resp.read()

                    if status == 200:

                        data = _parse_response(raw_body)

                        if data is None:
                            last_error = f"{base_url} HTTP 200 but invalid JSON"
                            break

                        if isinstance(data, dict) and "code" in data and "msg" in data:
                            last_error = f"{data.get('code')}: {data.get('msg')}"
                            if attempt < API_RETRIES:
                                time.sleep(_retry_delay(attempt))
                                continue
                            break

                        _log_fetch_timing(symbol, interval, base_url, fetch_started, ok=True)
                        return data

                    if status == 202:

                        data = _parse_response(raw_body)

                        if isinstance(data, (dict, list)):
                            if not (isinstance(data, dict) and "code" in data and "msg" in data):
                                _log_fetch_timing(symbol, interval, base_url, fetch_started, ok=True)
                                return data

                        last_error = f"{base_url} HTTP 202"

                        if attempt < API_RETRIES:
                            time.sleep(_retry_delay(attempt))
                            continue

                        break

                    if status in (418, 429):

                        last_error = f"{base_url} HTTP {status}"
                        retry_after = resp.headers.get("Retry-After")

                        if attempt < API_RETRIES:
                            time.sleep(_retry_delay(attempt, retry_after))
                            continue

                        break

                    if status == 451:
                        last_error = f"{base_url} HTTP 451"
                        break

                    last_error = f"{base_url} HTTP {status}"

                    if attempt < API_RETRIES:
                        time.sleep(_retry_delay(attempt))
                        continue

                    break

            except urllib.error.HTTPError as exc:

                status = exc.code
                retry_after = None

                try:
                    retry_after = exc.headers.get("Retry-After")
                except Exception:
                    pass

                if status in (418, 429):
                    last_error = f"{base_url} HTTP {status}"
                    if attempt < API_RETRIES:
                        time.sleep(_retry_delay(attempt, retry_after))
                        continue
                    break

                if status == 451:
                    last_error = f"{base_url} HTTP 451"
                    break

                if status == 202:
                    try:
                        raw_body = exc.read()
                        data = _parse_response(raw_body)
                        if isinstance(data, (dict, list)):
                            if not (isinstance(data, dict) and "code" in data and "msg" in data):
                                _log_fetch_timing(symbol, interval, base_url, fetch_started, ok=True)
                                return data
                    except Exception:
                        pass

                    last_error = f"{base_url} HTTP 202"
                    if attempt < API_RETRIES:
                        time.sleep(_retry_delay(attempt))
                        continue
                    break

                last_error = f"{base_url} HTTP {status}"
                if attempt < API_RETRIES:
                    time.sleep(_retry_delay(attempt))
                    continue
                break

            except TimeoutError:
                last_error = f"{base_url} timeout"
                if attempt < API_RETRIES:
                    time.sleep(_retry_delay(attempt))
                    continue
                break

            except urllib.error.URLError as exc:
                last_error = f"{base_url}: {exc}"
                if attempt < API_RETRIES:
                    time.sleep(_retry_delay(attempt))
                    continue
                break

            except Exception as exc:
                last_error = f"{base_url}: {exc}"
                if attempt < API_RETRIES:
                    time.sleep(_retry_delay(attempt))
                    continue
                break

    _log_fetch_timing(symbol, interval, None, fetch_started, ok=False, error=last_error)
    print(f"  ! kline fetch failed for {symbol} {interval}: All Binance endpoints failed: {last_error}")
    return []


def _log_fetch_timing(symbol, interval, base_url, started_at, ok, error=None):
    """Diagnostik ringan (poin 'c') - kelihatan langsung di log job Actions,
    tidak perlu perubahan apapun di workflow.yml karena stdout job selalu
    ter-capture otomatis. Kalau di log ternyata banyak baris 'slow=' atau
    endpoint selain fapi.binance.com yang sering menang, itu tandanya
    endpoint utama kena throttle/geo-block dari runner GitHub Actions -
    reorder BASE_URLS supaya endpoint yang menang duluan ditaruh di depan."""

    elapsed = time.monotonic() - started_at
    if ok:
        slow_tag = " slow=1" if elapsed > 3.0 else ""
        print(f"  fetch {symbol} {interval}: {elapsed:.2f}s via {base_url}{slow_tag}")
    else:
        print(f"  fetch {symbol} {interval}: {elapsed:.2f}s FAILED ({error})")


# OUTCOME LOGIC

def check_entry_triggered(entry_price, klines):
    """Return open_time (k[0]) dari candle pertama yang menyentuh
    entry_price - BUKAN close_time (k[6]) seperti sebelumnya.

    FIX bug #1: update_open() memfilter klines simulasi dengan
    `k[0] >= triggered_ms`. Kalau triggered_ms = close_time candle entry,
    candle entry itu sendiri otomatis kepotong (open_time-nya selalu <
    close_time-nya sendiri) dan tidak pernah ikut disimulasikan.
    Mengembalikan open_time membuat candle entry ikut disertakan
    (k[0] >= triggered_ms == True untuk dirinya sendiri).
    """
    entry_price = float(entry_price)
    for k in klines:
        low = float(k[3])
        high = float(k[2])
        if low <= entry_price <= high:
            return k[0]
    return None


def _resolve_ambiguous_candle(record, candle, sl_override=None):
    """FIX bug #2: kalau dalam satu candle SL dan TP sama-sama tersentuh,
    coba bedah candle itu pakai candle di timeframe lebih halus
    (DISAMBIGUATION_TF) untuk melihat urutan sebenarnya, alih-alih
    langsung asumsi SL duluan.

    sl_override (fix #4/breakeven): kalau posisi sudah "diarmed" ke
    breakeven setelah TP1 tersentuh di candle sebelumnya, level stop yang
    relevan BUKAN lagi record["sl"] asli melainkan level breakeven itu.
    Caller mengoper level efektif yang sedang aktif lewat parameter ini;
    kalau None, fungsi ini memakai record["sl"] seperti sebelumnya (tidak
    ada regresi untuk trade yang belum/tidak memakai breakeven).

    Return dict {"outcome", "exit_price", "tp_index", "resolved_via"} kalau
    berhasil dibedah. Return None kalau tidak bisa dibedah (TF sudah paling
    halus, atau fetch sub-candle gagal/kosong) - caller-nya (simulate_outcome)
    yang bertanggung jawab fallback ke asumsi SL-duluan sambil menandai
    intrabar_ambiguous=True.

    Rekursif turun satu TF lagi kalau di sub-candle pun keduanya masih
    tersentuh bersamaan, sampai TF paling halus (DISAMBIGUATION_TF[tf] is
    None) atau berhasil dibedah.
    """
    side = record["side"]
    sl = float(sl_override) if sl_override is not None else float(record["sl"])
    tps = [float(t) for t in (record.get("tp") or []) if t is not None]

    sub_tf = DISAMBIGUATION_TF.get(record["execution_tf"])
    if sub_tf is None:
        return None

    open_time, close_time = candle[0], candle[6]
    sub_klines = fetch_klines(
        record["symbol"], sub_tf, open_time, limit=1000, end_time_ms=close_time
    )
    if not sub_klines:
        return None

    for sub_k in sub_klines:
        sub_high = float(sub_k[2])
        sub_low = float(sub_k[3])

        sub_sl_hit = (sub_low <= sl) if side == "LONG" else (sub_high >= sl)

        sub_tp_hit_index = None
        for i, tp in enumerate(tps):
            hit = (sub_high >= tp) if side == "LONG" else (sub_low <= tp)
            if hit:
                sub_tp_hit_index = i  # farthest TP tersentuh di sub-candle ini

        if sub_sl_hit and sub_tp_hit_index is not None:
            # Masih ambigu di TF ini juga - coba turun satu tingkat lagi.
            deeper_record = {**record, "execution_tf": sub_tf}
            deeper = _resolve_ambiguous_candle(deeper_record, sub_k, sl_override=sl_override)
            return deeper  # None kalau tetap tidak bisa dibedah

        if sub_sl_hit:
            return {
                "outcome": "LOSS",
                "exit_price": sl,
                "tp_index": None,
                "resolved_via": "subcandle",
            }

        if sub_tp_hit_index is not None:
            return {
                "outcome": "WIN",
                "exit_price": tps[sub_tp_hit_index],
                "tp_index": sub_tp_hit_index,
                "resolved_via": "subcandle",
            }

    return None


def _resolve_entry_candle_order(record, candle):
    """FIX bug #3: candle entry (idx=0 di simulate_outcome) yang JUGA
    menyentuh SL di rentang high/low-nya tidak otomatis berarti "entry lalu
    langsung SL" - bisa saja urutan sebenarnya di dalam candle itu berbeda.
    Fungsi ini membedah candle tersebut memakai sub-candle (DISAMBIGUATION_TF)
    untuk mencari titik pertama harga benar-benar menyentuh level entry,
    lalu mengevaluasi SL/TP HANYA dari titik itu ke depan (bukan dari awal
    candle, supaya tidak menghukum pergerakan yang terjadi SEBELUM entry
    benar-benar tersentuh).

    Return salah satu dari:
      - dict {"outcome", "exit_price", "tp_index", "resolved_via"} kalau
        entry ditemukan DAN outcome (SL/TP) juga resolve di dalam rentang
        candle yang sama.
      - {"no_exit": True} kalau entry ditemukan tapi tidak ada SL/TP yang
        tersentuh SETELAH titik entry itu (artinya sinyal SL-hit di level
        full-candle terjadi sebelum entry benar-benar tersentuh, jadi tidak
        relevan - simulate_outcome harus lanjut ke candle berikutnya tanpa
        mencatat outcome apa pun untuk candle ini).
      - None kalau tidak bisa dibedah sama sekali (TF sudah paling halus,
        fetch sub-candle gagal/kosong, atau titik entry tidak ketemu di
        sub-candle manapun) - caller fallback ke asumsi lama (SL duluan)
        sambil menandai entry_sl_ambiguous_fallback=True.

    Catatan (fix #4/breakeven): di idx==0 posisi belum pernah "diarmed" ke
    breakeven (baru saja triggered), jadi level stop yang relevan di sini
    selalu record["sl"] asli - tidak perlu sl_override.
    """
    side = record["side"]
    sl = float(record["sl"])
    tps = [float(t) for t in (record.get("tp") or []) if t is not None]
    entry_price = float(record["entry"])

    sub_tf = DISAMBIGUATION_TF.get(record["execution_tf"])
    if sub_tf is None:
        return None

    open_time, close_time = candle[0], candle[6]
    sub_klines = fetch_klines(
        record["symbol"], sub_tf, open_time, limit=1000, end_time_ms=close_time
    )
    if not sub_klines:
        return None

    entered = False

    for i, sub_k in enumerate(sub_klines):
        sub_high = float(sub_k[2])
        sub_low = float(sub_k[3])

        if not entered:
            if sub_low <= entry_price <= sub_high:
                entered = True
            else:
                # Belum menyentuh entry di sub-candle ini - lanjut, jangan
                # evaluasi SL/TP dulu (itu sebelum posisi benar-benar ada).
                continue

        sub_sl_hit = (sub_low <= sl) if side == "LONG" else (sub_high >= sl)

        sub_tp_hit_index = None
        for j, tp in enumerate(tps):
            hit = (sub_high >= tp) if side == "LONG" else (sub_low <= tp)
            if hit:
                sub_tp_hit_index = j

        if sub_sl_hit and sub_tp_hit_index is not None:
            # Ambigu lagi di sub-candle ini juga - coba bedah lebih dalam
            # pakai mesin disambiguasi SL-vs-TP yang sudah ada.
            deeper_record = {**record, "execution_tf": sub_tf}
            deeper = _resolve_ambiguous_candle(deeper_record, sub_k)
            if deeper is not None:
                return {**deeper, "resolved_via": "subcandle_entry"}
            return None

        if sub_sl_hit:
            return {
                "outcome": "LOSS",
                "exit_price": sl,
                "tp_index": None,
                "resolved_via": "subcandle_entry",
            }

        if sub_tp_hit_index is not None:
            return {
                "outcome": "WIN",
                "exit_price": tps[sub_tp_hit_index],
                "tp_index": sub_tp_hit_index,
                "resolved_via": "subcandle_entry",
            }

    if entered:
        # Entry ketemu, tapi tidak ada SL/TP tersentuh SETELAH titik itu
        # sampai akhir candle - sinyal SL-hit di level full-candle terjadi
        # sebelum entry beneran tersentuh, jadi tidak relevan untuk candle
        # ini.
        return {"no_exit": True}

    # Titik entry tidak ketemu sama sekali di sub-candle (seharusnya jarang,
    # karena parent candle sudah lolos check_entry_triggered) - tidak bisa
    # dibedah dengan yakin.
    return None


def simulate_outcome(record, klines):
    """Jalan maju candle demi candle sampai SL / TP / breakeven resolve,
    atau klines habis (caller menangani EXPIRED kalau max-hold terlampaui).

    FIX #4 (breakeven-after-TP1): sebelumnya, begitu TP mana pun tersentuh,
    fungsi ini langsung `return` menutup trade. Kalau
    record["breakeven_after_tp1"] True, sekarang HANYA-TP1-tersentuh
    (bukan TP yang lebih jauh sekaligus di candle yang sama) tidak menutup
    trade - level stop efektif dipindah ke record["sl_after_tp1"]
    (breakeven) dan simulasi berlanjut ke candle selanjutnya. Kalau posisi
    akhirnya keluar di level breakeven itu, outcome-nya "BREAKEVEN" (bukan
    "LOSS" - risiko sudah dikunci ke ~0 R sejak TP1 tersentuh) dengan
    tp_index=0 supaya tetap terlihat bahwa TP1 sempat kena.

    Kalau record TIDAK punya breakeven_after_tp1 (trade lama, sebelum fix
    ini ada di scanner) atau nilainya False/kosong, perilaku persis sama
    seperti sebelumnya - tidak ada regresi untuk data historis.
    """
    side = record["side"]
    original_sl = float(record["sl"])
    tps = [float(t) for t in (record.get("tp") or []) if t is not None]

    breakeven_enabled = bool(record.get("breakeven_after_tp1")) and len(tps) > 1
    breakeven_level = None
    if breakeven_enabled:
        breakeven_level = float(record.get("sl_after_tp1", record.get("entry", original_sl)))

    # Disambiguasi entry-vs-SL (fix bug #3) hanya berlaku untuk candle
    # pertama dalam simulasi ini (idx==0), dan hanya untuk record yang
    # memang menunggu harga menyentuh level entry secara diskrit (jalur
    # PENDING -> OPEN via check_entry_triggered). Record ENTRY_READY sudah
    # "di dalam" zona entry sejak awal (ditangani terpisah lewat
    # _process_partial_entry_windows), jadi tidak ada momen "sentuh entry"
    # yang perlu dibedah di sini.
    is_entry_touch_flow = not record.get("partial_check_from")

    tp1_reached = False
    active_sl = original_sl

    def _stop_outcome(exit_price):
        """Label yang benar untuk 'posisi berhenti di stop level' -
        BREAKEVEN kalau TP1 sudah kena & breakeven sudah diarmed
        (risiko sudah dikunci ke ~0 R), LOSS kalau belum."""
        if tp1_reached and breakeven_enabled:
            return "BREAKEVEN"
        return "LOSS"

    for idx, k in enumerate(klines):
        high = float(k[2])
        low = float(k[3])
        close_time = k[6]

        sl_hit = (low <= active_sl) if side == "LONG" else (high >= active_sl)

        tp_hit_index = None
        for i, tp in enumerate(tps):
            hit = (high >= tp) if side == "LONG" else (low <= tp)
            if hit:
                tp_hit_index = i  # keep going - don't stop at the first
                                  # (nearest) target; a single impulsive
                                  # candle can blow through TP1+TP2+TP3 at
                                  # once, and the outcome should reflect
                                  # the FARTHEST one actually reached, not
                                  # just the nearest.

        if idx == 0 and is_entry_touch_flow and sl_hit:
            # FIX bug #3 - candle entry ini juga menyentuh SL. Jangan
            # langsung asumsi "entry lalu SL" - bedah dulu urutan
            # sebenarnya pakai sub-candle. (tp1_reached selalu False di
            # sini - ini candle pertama - jadi active_sl == original_sl,
            # tidak ada interaksi dengan breakeven di titik ini.)
            entry_resolved = _resolve_entry_candle_order(record, k)

            if entry_resolved is not None:
                if entry_resolved.get("no_exit"):
                    # SL-hit di level candle ini terjadi SEBELUM entry
                    # benar-benar tersentuh - tidak relevan, lanjut ke
                    # candle berikutnya tanpa mencatat outcome di sini.
                    continue

                # FIX #4: kalau disambiguasi menemukan TP1 (index 0) yang
                # tersentuh duluan, dan masih ada TP lebih jauh, dan
                # breakeven aktif - arm breakeven & lanjut simulasi,
                # jangan langsung tutup trade di TP1.
                if (
                    entry_resolved.get("outcome") == "WIN"
                    and entry_resolved.get("tp_index") == 0
                    and breakeven_enabled
                ):
                    tp1_reached = True
                    active_sl = breakeven_level
                    continue

                return {
                    **entry_resolved,
                    "bar_time": close_time,
                    "bars_held": idx + 1,
                    "intrabar_ambiguous": False,
                    "entry_sl_ambiguous_fallback": False,
                }

            # Tidak bisa dibedah - fallback ke asumsi lama (SL duluan),
            # tapi ditandai supaya kelihatan di stats bahwa ini masih
            # under-determined, sama seperti pola intrabar_ambiguous.
            if tp_hit_index is not None:
                return {
                    "outcome": "LOSS",
                    "exit_price": original_sl,
                    "tp_index": None,
                    "bar_time": close_time,
                    "bars_held": idx + 1,
                    "intrabar_ambiguous": True,
                    "entry_sl_ambiguous_fallback": True,
                }
            return {
                "outcome": "LOSS",
                "exit_price": original_sl,
                "tp_index": None,
                "bar_time": close_time,
                "bars_held": idx + 1,
                "intrabar_ambiguous": False,
                "entry_sl_ambiguous_fallback": True,
            }

        if sl_hit and tp_hit_index is not None:
            # FIX bug #2 - ambigu: SL/stop-aktif dan TP tersentuh di candle
            # yang sama. Jangan langsung asumsi stop menang, coba bedah
            # pakai sub-candle dulu. sl_override diteruskan supaya kalau
            # posisi sudah diarmed ke breakeven, disambiguasi memakai
            # level yang benar-benar aktif sekarang (bukan sl asli).
            resolved = _resolve_ambiguous_candle(
                record, k, sl_override=(active_sl if tp1_reached else None)
            )
            if resolved is not None:
                # FIX #4: kalau disambiguasi menemukan TP1 duluan dan masih
                # ada TP lebih jauh, arm breakeven & lanjut - jangan tutup.
                if (
                    resolved.get("outcome") == "WIN"
                    and resolved.get("tp_index") == 0
                    and breakeven_enabled
                    and not tp1_reached
                ):
                    tp1_reached = True
                    active_sl = breakeven_level
                    continue

                if resolved.get("outcome") == "LOSS":
                    resolved = {**resolved, "outcome": _stop_outcome(resolved["exit_price"])}

                return {
                    **resolved,
                    "bar_time": close_time,
                    "bars_held": idx + 1,
                    "intrabar_ambiguous": False,
                    "entry_sl_ambiguous_fallback": False,
                }
            # Tidak bisa dibedah - fallback ke asumsi konservatif lama
            # (stop duluan), tapi DITANDAI supaya kelihatan di stats.
            return {
                "outcome": _stop_outcome(active_sl),
                "exit_price": active_sl,
                "tp_index": 0 if (tp1_reached and breakeven_enabled) else None,
                "bar_time": close_time,
                "bars_held": idx + 1,
                "intrabar_ambiguous": True,
                "entry_sl_ambiguous_fallback": False,
            }

        if sl_hit:
            return {
                "outcome": _stop_outcome(active_sl),
                "exit_price": active_sl,
                "tp_index": 0 if (tp1_reached and breakeven_enabled) else None,
                "bar_time": close_time,
                "bars_held": idx + 1,
                "intrabar_ambiguous": False,
                "entry_sl_ambiguous_fallback": False,
            }

        if tp_hit_index is not None:
            # FIX #4: candle ini HANYA menyentuh TP1 (farthest == 0), masih
            # ada TP lebih jauh, dan breakeven aktif -> jangan tutup trade,
            # arm breakeven dan lanjut mensimulasikan candle berikutnya.
            if tp_hit_index == 0 and breakeven_enabled and not tp1_reached:
                tp1_reached = True
                active_sl = breakeven_level
                continue

            return {
                "outcome": "WIN",
                "exit_price": tps[tp_hit_index],
                "tp_index": tp_hit_index,
                "bar_time": close_time,
                "bars_held": idx + 1,
                "intrabar_ambiguous": False,
                "entry_sl_ambiguous_fallback": False,
            }

    return None


def compute_r_multiple(record):
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


# SCAN RESULT LOADING

def _iter_run_dirs_ascending(results_dir):
    if not results_dir.exists():
        return []
    dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.name)
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


# CONFIG VERSION SAFETY CHECK (FIX #3)

def _known_config_versions(trade_log):
    """Kumpulkan semua config_version yang pernah tercatat di pending/open/
    closed, supaya operator bisa langsung lihat kalau --config-version yang
    dipakai run ini kebetulan sama dengan versi lama (lupa dinaikkan) atau
    beda dari yang paling baru dipakai run sebelumnya."""
    versions = set()
    for bucket in ("pending", "open", "closed"):
        for record in trade_log.get(bucket, []):
            v = record.get("config_version")
            if v:
                versions.add(v)
    return versions


def _warn_if_config_version_stale(trade_log, config_version):
    """Cetak peringatan yang jelas (bukan diam-diam) kalau config_version
    run ini kemungkinan lupa dinaikkan. Tidak menghentikan run - cuma
    supaya kesalahan ini tidak lolos tanpa disadari, karena tag ini tidak
    dibaca otomatis dari Synaptic_fixed.py (murni argumen CLI)."""
    existing = _known_config_versions(trade_log)

    if not existing:
        print(f"config_version untuk run ini: '{config_version}' (belum ada versi lain tercatat).")
        return

    print(f"config_version untuk run ini: '{config_version}'. Versi yang sudah pernah tercatat: {sorted(existing)}")

    if config_version == "unversioned":
        print(
            "  PERINGATAN: --config-version tidak di-set (memakai default "
            "'unversioned'). Kalau baru saja mengubah CONFIG di "
            "Synaptic_fixed.py, trade baru ini tidak akan bisa dibedakan "
            "dari versi sebelumnya di winrate_stats.json."
        )
    elif config_version in existing:
        print(
            f"  PERINGATAN: '{config_version}' sudah pernah dipakai sebelumnya. "
            "Kalau Synaptic_fixed.py baru saja diubah secara berarti, "
            "pastikan ini memang disengaja - bukan lupa menaikkan versi."
        )


# INGEST NEW CANDIDATES

def ingest_new_runs(trade_log, results_dir, config_version="unversioned"):
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
                # Bias EMA200 harian (BULLISH/BEARISH/NEUTRAL/None) pada
                # saat kandidat muncul - dipakai compute_stats() untuk
                # mengelompokkan trade "selaras" vs "melawan" tren harian.
                "htf_bias": c.get("htf_bias"),
                # Tag config_version dari luar (bukan dibaca otomatis dari
                # Synaptic.py) - naikkan versinya di workflow tiap kali
                # CONFIG di Synaptic.py diubah secara berarti, supaya
                # evaluasi win-rate bisa dipecah "sebelum vs sesudah"
                # tanpa perlu filter manual by tanggal. Lihat
                # _warn_if_config_version_stale() di main() untuk
                # pengingat kalau ini lupa dinaikkan.
                "config_version": config_version,
                # --- FIX #5: field skor/regime dari Synaptic_fixed.py,
                # sebelumnya tidak pernah ikut tercatat sama sekali,
                # sehingga tidak mungkin membuktikan komponen skor mana
                # yang benar-benar berkorelasi dengan win rate. Semuanya
                # sudah tersedia di synaptic_candidates.json yang sama,
                # jadi ini bukan API call tambahan - cuma field yang
                # sebelumnya dibuang begitu saja. ---
                "score": c.get("score"),
                "stage1_score": c.get("stage1_score"),
                "btc_regime": c.get("btc_regime"),
                "funding_alert": c.get("funding_alert"),
                "tf_agreement": c.get("tf_agreement"),
                "momentum_15m": c.get("momentum_15m"),
                # --- FIX #4: field breakeven-after-TP1 dari
                # Synaptic_fixed.py. sl_after_tp1 default ke sl asli kalau
                # tidak ada (mis. trade dari versi scanner sebelum fitur
                # ini ada) - simulate_outcome() sudah menjaga supaya
                # breakeven_after_tp1 falsy berarti perilaku lama persis
                # sama, tidak ada regresi. ---
                "breakeven_after_tp1": c.get("breakeven_after_tp1", False),
                "sl_after_tp1": c.get("sl_after_tp1", c.get("sl")),
            }

            if entry_state == "ENTRY_READY":
                # FIX (varian bug #1, revisi ke-2): percobaan pertama
                # (floor ke AWAL candle) ternyata overshoot ke arah
                # sebaliknya - itu bisa membuat simulasi melihat MUNDUR ke
                # harga SEBELUM setup dinyatakan ENTRY_READY, dan salah
                # menghukum SL yang sebenarnya tersentuh sebelum sinyal
                # muncul (padahal secara real-time kamu belum masuk posisi
                # saat itu). Fix yang benar: jangan floor mundur, tapi catat
                # partial_check_from = generated_at (titik sinyal muncul)
                # dan triggered_at = batas AWAL candle penuh BERIKUTNYA.
                # update_open() akan mengecek jendela [partial_check_from,
                # triggered_at) itu pakai timeframe lebih halus dulu
                # (_process_partial_entry_windows) - jadi gerakan persis
                # setelah sinyal tetap tertangkap, tanpa melihat mundur ke
                # sebelum sinyal.
                tf_ms = TF_MS.get(tf, TF_MS["1h"])
                generated_ms = iso_to_ms(generated_at)
                next_boundary_ms = ((generated_ms // tf_ms) + 1) * tf_ms
                record["triggered_at"] = ms_to_iso(next_boundary_ms)
                record["partial_check_from"] = generated_at
                trade_log["open"].append(record)
            else:
                record["added_at"] = generated_at
                trade_log["pending"].append(record)

        ingested.add(run_dir.name)
        new_run_names.append(run_dir.name)

    trade_log["ingested_runs"] = sorted(ingested)
    return new_run_names


# ============================================================
# FETCH DEDUPE / GROUPING (poin b)
#
# Symbol yang sama sering nongol di banyak record pending/open sekaligus
# (mis. DASHUSDT muncul di 5+ run beruntun sebagai trade terpisah-pisah).
# Daripada fetch klines satu-satu per record, kelompokkan record dengan
# symbol+timeframe sama yang start-time-nya berdekatan (dalam
# DEDUPE_WINDOW_MS), fetch SEKALI dari titik start paling awal di grup itu,
# lalu setiap record memakai potongan klines yang relevan buat dirinya
# (klines dengan open_time >= start_time record itu sendiri).
#
# Window dedupe sengaja dibuat kecil relatif terhadap panjang hidup
# trade (lihat --max-hold-candles / --pending-expiry-candles) supaya
# fetch tunggal (limit=1000 candle) masih cukup jauh menjangkau ke masa
# sekarang untuk SEMUA anggota grup - tidak ada candle yang "terpotong".
# ============================================================

def _bucket_groups(records, start_field):
    """Mengelompokkan records ke dalam grup (symbol, tf, bucket) dimana
    bucket = start_time // DEDUPE_WINDOW_MS. Return dict
    {(symbol, tf, bucket): [records...]}."""

    groups = {}
    for r in records:
        start_ms = iso_to_ms(r[start_field])
        bucket = start_ms // DEDUPE_WINDOW_MS
        key = (r["symbol"], r["execution_tf"], bucket)
        groups.setdefault(key, []).append(r)
    return groups


def _fetch_grouped_klines(groups, klines_limit, max_workers=MAX_FETCH_WORKERS, start_field="added_at"):
    """Fetch satu kali per grup (paralel via ThreadPoolExecutor - network
    bound, aman diparalel walau ada GIL). Return dict
    {group_key: (min_start_ms, klines)}."""

    def _fetch(key, records):
        symbol, tf, _bucket = key
        min_start = min(iso_to_ms(r[start_field]) for r in records)
        klines = fetch_klines(symbol, tf, min_start, limit=klines_limit)
        return key, min_start, klines

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch, key, records): key
            for key, records in groups.items()
        }
        for fut in as_completed(futures):
            key, min_start, klines = fut.result()
            results[key] = (min_start, klines)

    return results


# UPDATE PENDING -> OPEN / NEVER_TRIGGERED

def update_pending(trade_log, pending_expiry_candles, klines_limit):
    still_pending = []
    current_ms = now_ms()

    groups = _bucket_groups(trade_log["pending"], start_field="added_at")
    print(f"  {len(trade_log['pending'])} pending record(s) -> {len(groups)} fetch group(s)")
    fetched = _fetch_grouped_klines(groups, klines_limit, start_field="added_at")

    for key, records in groups.items():
        _min_start, klines = fetched[key]

        if not klines:
            still_pending.extend(records)
            continue

        for record in records:
            added_ms = iso_to_ms(record["added_at"])
            tf_ms = TF_MS.get(record["execution_tf"], TF_MS["1h"])

            # Potong ke candle yang relevan buat record ini saja (grup bisa
            # berisi beberapa record dengan added_at sedikit berbeda).
            record_klines = [k for k in klines if k[0] >= added_ms]

            if not record_klines:
                still_pending.append(record)
                continue

            trigger_time = check_entry_triggered(record["entry"], record_klines)
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


# UPDATE OPEN -> CLOSED (WIN / LOSS / BREAKEVEN / EXPIRED)

def _process_partial_entry_windows(trade_log, klines_limit):
    """Untuk record ENTRY_READY yang masih punya jendela parsial belum
    dicek (partial_check_from -> triggered_at = sisa waktu dari saat
    sinyal muncul sampai candle penuh berikutnya mulai), cek dulu pakai
    timeframe yang lebih halus SEBELUM masuk ke alur grouped-fetch normal
    di execution_tf.

    Ini sengaja dipisah dari alur utama supaya dua hal tetap terjaga:
      - tidak melihat mundur ke SEBELUM saat sinyal muncul (window mulai
        persis di partial_check_from, bukan awal candle penuh)
      - tidak melewatkan gerakan yang terjadi PERSIS setelah sinyal, di
        sisa candle penuh yang sama (ini varian dari bug #1)

    Record yang resolve di jendela ini langsung dipindah ke closed
    (ditandai closed_in_partial_window=True untuk transparansi). Record
    yang tidak resolve ditandai partial_check_done=True dan lanjut lewat
    alur grouped normal seperti biasa, mulai dari triggered_at.
    """
    still_open = []
    for record in trade_log["open"]:
        partial_from = record.get("partial_check_from")
        if not partial_from or record.get("partial_check_done"):
            still_open.append(record)
            continue

        sub_tf = DISAMBIGUATION_TF.get(record["execution_tf"]) or record["execution_tf"]
        from_ms = iso_to_ms(partial_from)
        until_ms = iso_to_ms(record["triggered_at"])

        sub_klines = fetch_klines(record["symbol"], sub_tf, from_ms, limit=klines_limit, end_time_ms=until_ms)
        result = simulate_outcome(record, sub_klines) if sub_klines else None

        if result is not None:
            record.update(result)
            record["closed_in_partial_window"] = True
            record["closed_at"] = ms_to_iso(result["bar_time"])
            record["r_multiple"] = compute_r_multiple(record)
            trade_log["closed"].append(record)
        else:
            record["partial_check_done"] = True
            still_open.append(record)

    trade_log["open"] = still_open


def update_open(trade_log, max_hold_candles, klines_limit):
    _process_partial_entry_windows(trade_log, klines_limit)

    still_open = []
    current_ms = now_ms()

    groups = _bucket_groups(trade_log["open"], start_field="triggered_at")
    print(f"  {len(trade_log['open'])} open record(s) -> {len(groups)} fetch group(s)")
    fetched = _fetch_grouped_klines(groups, klines_limit, start_field="triggered_at")

    for key, records in groups.items():
        _min_start, klines = fetched[key]

        if not klines:
            still_open.extend(records)
            continue

        for record in records:
            triggered_ms = iso_to_ms(record["triggered_at"])
            tf_ms = TF_MS.get(record["execution_tf"], TF_MS["1h"])

            # Sekarang triggered_ms adalah open_time candle entry (lihat fix
            # di check_entry_triggered), jadi filter ">=" ini otomatis ikut
            # menyertakan candle entry itu sendiri - bug #1 selesai di sini.
            record_klines = [k for k in klines if k[0] >= triggered_ms]

            if not record_klines:
                still_open.append(record)
                continue

            result = simulate_outcome(record, record_klines)

            if result is not None:
                record.update(result)
                record["closed_at"] = ms_to_iso(result["bar_time"])
                record["r_multiple"] = compute_r_multiple(record)
                trade_log["closed"].append(record)
                continue

            elapsed_candles = (current_ms - triggered_ms) / tf_ms
            if elapsed_candles >= max_hold_candles:
                last_close = float(record_klines[-1][4])
                record["outcome"] = "EXPIRED"
                record["exit_price"] = last_close
                record["closed_at"] = ms_to_iso(record_klines[-1][6])
                record["r_multiple"] = compute_r_multiple(record)
                trade_log["closed"].append(record)
            else:
                still_open.append(record)

    trade_log["open"] = still_open


# STATS

def _htf_alignment(record):
    """Classifies a resolved trade as ALIGNED (side agrees with the daily
    EMA200 bias) or COUNTER (side fights it). Missing/NEUTRAL bias data
    (funding/API hiccups, or a genuinely flat daily trend) goes to
    UNKNOWN_OR_NEUTRAL rather than being silently dropped, so the group
    sizes in by_htf_alignment always add up to the full resolved count."""

    bias = record.get("htf_bias")
    side = record.get("side")

    if not bias or bias == "NEUTRAL" or not side:
        return "UNKNOWN_OR_NEUTRAL"

    if (side == "LONG" and bias == "BULLISH") or (side == "SHORT" and bias == "BEARISH"):
        return "ALIGNED"

    if (side == "LONG" and bias == "BEARISH") or (side == "SHORT" and bias == "BULLISH"):
        return "COUNTER"

    return "UNKNOWN_OR_NEUTRAL"


def _group_win_rate(records, key_fn):
    """FIX #4: sekarang menangani 3 outcome (WIN/LOSS/BREAKEVEN), bukan cuma
    2. win_rate_pct dihitung sebagai wins / (wins+losses+breakeven) - trade
    breakeven tetap masuk penyebut (itu tetap "hasil", bukan dibuang begitu
    saja) tapi tidak dihitung sebagai win. Jumlah breakeven per grup tetap
    dilaporkan terpisah supaya tidak tersembunyi."""
    groups = {}
    for r in records:
        key = key_fn(r) or "UNKNOWN"
        g = groups.setdefault(key, {"wins": 0, "losses": 0, "breakeven": 0})
        outcome = r.get("outcome")
        if outcome == "WIN":
            g["wins"] += 1
        elif outcome == "BREAKEVEN":
            g["breakeven"] += 1
        else:
            g["losses"] += 1

    for g in groups.values():
        total = g["wins"] + g["losses"] + g["breakeven"]
        g["win_rate_pct"] = round(g["wins"] / total * 100, 1) if total else None

    return groups


def compute_stats(trade_log):
    closed = trade_log.get("closed", [])
    # FIX #4: BREAKEVEN sekarang termasuk "resolved" (itu hasil nyata dari
    # simulasi, r_multiple-nya ~0) - sebelumnya cuma WIN/LOSS yang dianggap
    # resolved, jadi outcome baru ini tidak akan pernah muncul di stats
    # kalau tidak ditambahkan di sini.
    resolved = [t for t in closed if t.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")]
    wins = [t for t in resolved if t["outcome"] == "WIN"]
    losses = [t for t in resolved if t["outcome"] == "LOSS"]
    breakeven = [t for t in resolved if t["outcome"] == "BREAKEVEN"]

    r_values = [t["r_multiple"] for t in resolved if t.get("r_multiple") is not None]
    avg_r = round(sum(r_values) / len(r_values), 2) if r_values else None

    # Transparansi utk fix bug #2: berapa trade yang outcome-nya berhasil
    # dibedah pakai sub-candle (resolved_via="subcandle") vs yang terpaksa
    # masih pakai fallback konservatif SL-duluan karena tidak bisa dibedah
    # lebih jauh (intrabar_ambiguous=True). Kalau angka kedua ini besar,
    # win rate yang dilaporkan masih under-estimate.
    disambiguated = len([t for t in resolved if t.get("resolved_via") == "subcandle"])
    ambiguous_fallback = len([t for t in resolved if t.get("intrabar_ambiguous")])

    # Transparansi utk fix bug #3: sama seperti di atas, tapi khusus untuk
    # ambiguitas urutan entry-vs-SL di candle entry itu sendiri.
    entry_disambiguated = len([t for t in resolved if t.get("resolved_via") == "subcandle_entry"])
    entry_ambiguous_fallback = len([t for t in resolved if t.get("entry_sl_ambiguous_fallback")])

    resolved_in_partial_window = len([t for t in resolved if t.get("closed_in_partial_window")])
    first_bar_trades = [t for t in resolved if t.get("bars_held") == 1]
    resolved_at_bar_1 = len(first_bar_trades)
    resolved_at_bar_1_wins = len([t for t in first_bar_trades if t["outcome"] == "WIN"])
    resolved_at_bar_1_losses = len([t for t in first_bar_trades if t["outcome"] == "LOSS"])

    # FIX #5: breakdown baru berbasis field skor/regime yang sekarang ikut
    # tercatat (lihat ingest_new_runs). Kalau field-nya kosong di record
    # lama (sebelum fix ini), key_fn mengembalikan None -> masuk ke bucket
    # "UNKNOWN" oleh _group_win_rate, tidak error dan tidak tercampur salah.
    def _score_bucket(t):
        score = t.get("score")
        if score is None:
            return None
        try:
            score = float(score)
        except (TypeError, ValueError):
            return None
        if score < 7:
            return "6.0-6.9"
        if score < 8:
            return "7.0-7.9"
        if score < 9:
            return "8.0-8.9"
        return "9.0+"

    return {
        "generated_at": ms_to_iso(now_ms()),
        "resolved_trades": len(resolved),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": round(len(wins) / len(resolved) * 100, 1) if resolved else None,
        "avg_r_multiple": avg_r,
        "by_setup_style": _group_win_rate(resolved, lambda t: t.get("setup_style")),
        "by_side": _group_win_rate(resolved, lambda t: t.get("side")),
        "by_config_version": _group_win_rate(resolved, lambda t: t.get("config_version")),
        "by_htf_alignment": _group_win_rate(resolved, _htf_alignment),
        # --- FIX #5: breakdown baru, tidak mungkin dihitung sebelumnya
        # karena field-nya belum pernah tersimpan di trade_log.json. ---
        "by_score_bucket": _group_win_rate(resolved, _score_bucket),
        "by_btc_regime": _group_win_rate(resolved, lambda t: t.get("btc_regime")),
        "by_funding_alert": _group_win_rate(resolved, lambda t: "ALERT" if t.get("funding_alert") else "NORMAL"),
        "expired_unresolved": len([t for t in closed if t.get("outcome") == "EXPIRED"]),
        "never_triggered": len([t for t in closed if t.get("outcome") == "NEVER_TRIGGERED"]),
        "currently_open": len(trade_log.get("open", [])),
        "currently_pending": len(trade_log.get("pending", [])),
        "intrabar_disambiguated_via_subcandle": disambiguated,
        "intrabar_ambiguous_fallback_to_sl": ambiguous_fallback,
        # Baru (fix bug #3): sama seperti di atas, tapi untuk ambiguitas
        # urutan entry-vs-SL di dalam candle entry itu sendiri, bukan
        # SL-vs-TP.
        "entry_sl_disambiguated_via_subcandle": entry_disambiguated,
        "entry_sl_ambiguous_fallback_to_sl": entry_ambiguous_fallback,
        "resolved_in_partial_entry_window": resolved_in_partial_window,
        # Trade yang resolve di CANDLE PERTAMA setelah trigger (bars_held
        # == 1). Kalau ini porsinya besar dan mayoritas LOSS, itu sinyal
        # kuat "immediate stop-out"/false-breakout - bukan tanda tracker
        # rusak, tapi tanda banyak setup langsung gagal begitu entry
        # tersentuh (lihat penjelasan di chat). Sejak fix bug #3, angka
        # LOSS di sini seharusnya lebih dipercaya karena urutan entry-vs-SL
        # sudah dibedah, bukan diasumsikan.
        "resolved_at_first_bar": resolved_at_bar_1,
        "resolved_at_first_bar_wins": resolved_at_bar_1_wins,
        "resolved_at_first_bar_losses": resolved_at_bar_1_losses,
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
    parser.add_argument(
        "--config-version",
        default="unversioned",
        help=(
            "Free-form label stamped on every new trade ingested this run. "
            "Bump it (e.g. v1 -> v2) whenever CONFIG in Synaptic.py changes "
            "meaningfully, so winrate_stats.json's by_config_version can "
            "compare before/after without manual date filtering. main() "
            "prints a warning if this looks stale (default value, or a "
            "version already used before) - see _warn_if_config_version_stale."
        ),
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    trade_log = load_trade_log(log_path)

    # FIX #3: peringatan eksplisit kalau config_version kemungkinan lupa
    # dinaikkan - tag ini murni argumen CLI, tidak pernah dibaca otomatis
    # dari Synaptic_fixed.py, jadi ini satu-satunya safety net di sini.
    _warn_if_config_version_stale(trade_log, args.config_version)

    new_runs = ingest_new_runs(trade_log, Path(args.results_dir), config_version=args.config_version)
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
        f"(W{stats['wins']}/L{stats['losses']}/BE{stats['breakeven']}, "
        f"win rate {stats['win_rate_pct']}%), "
        f"open={stats['currently_open']}, pending={stats['currently_pending']}"
    )


if __name__ == "__main__":
    main()
