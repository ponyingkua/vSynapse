#!/usr/bin/env python3

import argparse
import json
import logging
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("Synaptic")


# ============================================================
# REJECTION TRAIL (DEBUG)
#
# Sebelumnya setiap kandidat yang gugur di Stage 1 / Stage 2
# cuma "return None" tanpa jejak -- begitu funnel mengecil dari
# ratusan symbol jadi 0 kandidat, tidak ada cara untuk tahu di
# titik mana dan KENAPA mereka gugur (kecuali baca ulang kode).
#
# RejectionTracker mencatat setiap penolakan dengan:
#   stage   : "stage1" / "stage2"
#   symbol
#   reason  : kode alasan singkat & stabil (bukan kalimat bebas),
#             supaya bisa di-agregasi jadi funnel counter.
#   detail  : angka pendukung (mis. score, threshold terkait)
#             untuk beberapa sample pertama tiap reason.
#
# Thread-safe (dipakai dari dalam ThreadPoolExecutor worker),
# overhead-nya kecil (cuma counter + sample list dibatasi).
# ============================================================

class RejectionTracker:

    def __init__(self, max_samples_per_reason=5):
        self._lock = threading.Lock()
        self._counts = defaultdict(Counter)
        self._samples = defaultdict(list)
        self._max_samples = max_samples_per_reason

    def note(self, stage, symbol, reason, **detail):

        with self._lock:

            self._counts[stage][reason] += 1

            bucket = self._samples[(stage, reason)]

            if len(bucket) < self._max_samples:
                entry = {"symbol": symbol}
                entry.update(detail)
                bucket.append(entry)

        if detail:
            logger.debug(
                "[%s] %s gugur (%s): %s",
                stage, symbol, reason, detail,
            )
        else:
            logger.debug(
                "[%s] %s gugur (%s)",
                stage, symbol, reason,
            )

    def summary(self):
        with self._lock:
            return {
                stage: dict(
                    counter.most_common()
                )
                for stage, counter in self._counts.items()
            }

    def samples(self):
        with self._lock:
            return {
                f"{stage}:{reason}": list(entries)
                for (stage, reason), entries in self._samples.items()
            }

    def reset(self):
        with self._lock:
            self._counts.clear()
            self._samples.clear()


REJECTIONS = RejectionTracker()


def _log_rejection_funnel():
    """Cetak ringkasan alasan gugur per stage ke logger.

    Dipanggil di akhir scan (baik yang berhasil dapat kandidat
    maupun yang berakhir kosong) supaya funnel selalu terlihat
    di log, bukan cuma bisa diakses lewat --debug.
    """

    summary = REJECTIONS.summary()

    if not summary:
        return

    for stage in ("universe", "stage1", "stage2"):

        reasons = summary.get(stage)

        if not reasons:
            continue

        total = sum(reasons.values())

        logger.info(
            "Rejection funnel [%s] -- total gugur: %d",
            stage, total,
        )

        for reason, count in reasons.items():
            logger.info(
                "  - %-32s %d",
                reason, count,
            )


# ============================================================
# TIMEFRAMES
# ============================================================

TFS = ["15m", "1h", "4h"]


# ============================================================
# SYMBOL FILTER
# ============================================================

IGNORED_SYMBOLS = {
    "USDCUSDT",
    "BUSDUSDT",
    "DAIUSDT",
    "TUSDUSDT",
    "FDUSDUSDT",
    "USDPUSDT",
    "EURUSDT",
    "USTCUSDT",
    "PAXGUSDT",
}


# ============================================================
# PRICE DECIMALS
#
# Dipakai untuk membulatkan entry/sl/tp SEBELUM ditulis ke
# JSON output, supaya semua konsumen data (vSch.py, summary.txt,
# notifikasi bot, dll) melihat angka yang sama-sama rapi --
# bukan float mentah hasil kalkulasi (mis. 1.053276482234).
#
# Logika identik dengan decimals_from_price() di vSch.py,
# supaya chart dan JSON tidak pernah "beda pembulatan".
# ============================================================

def decimals_from_price(price):

    p = abs(float(price))

    if p < 0.0001:
        return 8

    if p < 0.001:
        return 7

    if p < 0.01:
        return 6

    if p < 0.1:
        return 5

    if p < 1:
        return 5

    if p < 10:
        return 4

    if p < 100:
        return 3

    return 2


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {

    # --------------------------------------------------------
    # Universe
    # --------------------------------------------------------

    "min_quote_volume_24h": 500_000,

    # 0 = seluruh active USDT perpetual
    "universe_size": 0,

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    "momentum_pool": 60,

    # Cukup untuk EMA200 + indikator lain.
    # Tidak perlu mengambil ratusan candle tambahan.
    "klines": 240,

    # Concurrency Stage 1.
    # Sengaja tidak dibuat ekstrem agar tidak memukul
    # Binance rate limit.
    "workers_stage1": 16,

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    "workers_stage2": 12,

    # --------------------------------------------------------
    # Candidate selection
    # --------------------------------------------------------

    # POIN 3: longgarkan min_candidates jadi 0.
    # Jangan backfill dari mtf_valid demi angka "Top 5".
    # Kalau tidak ada setup yang mencapai min_score, biarkan
    # final_candidates = 0 (atau sebanyak yang benar-benar
    # lolos min_score, tanpa memaksa isi dari fallback).
    "min_score": 6.0,
    "min_candidates": 0,
    "max_results": 5,

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    "ema_period": 200,

    "volume_ma_period": 20,
    "volume_ratio_min": 1.30,

    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,

    "supertrend_period": 10,
    "supertrend_multiplier": 2.50,

    "atr_period": 14,

    "breakout_window": 20,

    # --------------------------------------------------------
    # Repaint guard
    # --------------------------------------------------------

    # Kalau True, semua deteksi sinyal (EMA/Supertrend/MACD/
    # breakout/volume) memakai candle TERAKHIR YANG SUDAH
    # CLOSED, bukan candle yang masih berjalan. Mencegah sinyal
    # muncul lalu hilang lagi sebelum candle-nya selesai
    # (repaint/whiplash). Harga live tetap dipakai terpisah
    # untuk keputusan entry_state / ideal_entry, jadi tidak
    # kehilangan presisi harga saat ini.
    "confirm_on_closed_bar": True,

    # --------------------------------------------------------
    # Stage 1 -- extension penalty
    # --------------------------------------------------------

    # movement_score() secara desain menghargai pergerakan yang
    # SUDAH terjadi (fast_return/atr_move/breakout_bonus), jadi
    # secara struktural condong ke koin yang sudah lari jauh.
    # Untuk symbol yang jaraknya ke EMA200 (dalam ATR) sudah
    # sebesar ini, skor Stage 1-nya didiskon supaya tidak
    # mendominasi momentum_pool dan menutup ruang setup yang
    # masih segar.
    "extended_momentum_penalty_atr": 4.0,

    "extended_momentum_penalty_factor": 0.6,

    # --------------------------------------------------------
    # Stage 1 -- liquidity / volatility sanity gate
    # --------------------------------------------------------

    # Filter coin yang secara teknikal "mati"/choppy: ATR
    # (candle 15m) terlalu kecil dibanding harga. Volume 24h
    # saja tidak cukup -- symbol likuid tapi range-nya sempit
    # tetap bisa lolos ke momentum_pool dan menghasilkan setup
    # yang secara struktural tidak layak ditradingkan. Nilai
    # dalam persen (ATR / close * 100).
    "min_atr_pct": 0.15,

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    "momentum_fast_bars": 4,
    "momentum_slow_bars": 16,

    # Stage 1 harus mempertimbangkan ARAH, bukan hanya besar gerak.
    # Trend lokal yang searah EMA200 + Supertrend mendapat bonus.
    # Kandidat yang bergerak melawan regime mendapat penalti.
    "stage1_trend_bonus": 1.50,
    "stage1_countertrend_penalty_factor": 0.55,

    # Filter minimum kekuatan momentum 15m searah sisi trade
    # (dalam persen). Dulu cuma cek tanda (>0 / <0) tanpa
    # ambang -> pergerakan 0.01% pun lolos sebagai konfirmasi.
    "min_momentum_15m_pct": 0.15,

    # --------------------------------------------------------
    # Setup engine
    # --------------------------------------------------------

    # Pullback: jarak maksimum close ke EMA200 (dalam ATR)
    # agar tetap dianggap zona retracement yang valid.
    "setup_pullback_atr": 1.0,

    # Extended: jarak minimum close ke EMA200 (dalam ATR)
    # sebelum dianggap terlalu jauh / overextended.
    "setup_extended_atr": 3.0,

    # Continuation "near" band: batas jarak ke EMA200 (dalam ATR)
    # yang masih dianggap wajar untuk entry di harga pasar
    # (market/chase ringan). Lebih dari ini -> tunggu pullback,
    # jangan chase.
    # CATATAN (POIN 2): CONTINUATION sekarang SELALU diarahkan
    # ke pullback zone (sama seperti EXTENDED). Parameter ini
    # masih disimpan untuk kompatibilitas, tapi tidak lagi
    # dipakai untuk mengizinkan ENTRY_READY di market price.
    "continuation_near_atr": 1.5,

    # Entry Engine -----------------------------------------------
    #
    # ENTRY_READY vs WAITING_*: seberapa dekat harga sekarang
    # boleh menyimpang dari entry ideal (dalam ATR) sebelum
    # dianggap "belum sampai", bukan "sudah siap".
    "entry_ready_atr": 0.35,

    # BREAKOUT / BREAKDOWN: buffer di atas/bawah level breakout
    # untuk entry retest (dalam ATR). Ini yang mencegah entry
    # chase jauh setelah candle impulsif besar -- entry ideal
    # dijepit balik ke dekat level yang ditembus, bukan ikut
    # harga yang sudah lari jauh.
    "breakout_chase_buffer_atr": 0.25,

    # BREAKOUT / BREAKDOWN -- retest requirement.
    #
    # Sebelumnya ideal_entry = min(market_price, level+buffer),
    # yang berarti kalau harga baru SEDIKIT menembus level,
    # ideal_entry = market_price -> ENTRY_READY seketika, tanpa
    # retest sungguhan. Ini pola klasik "beli di fakeout".
    #
    # Sekarang breakout WAJIB menunjukkan retest minimal
    # sejauh ini (dalam ATR) balik ke arah level sebelum
    # dianggap ENTRY_READY -- entry tetap dijepit ke area
    # retest (level +/- buffer), tapi entry_state tidak akan
    # ENTRY_READY kalau harga belum benar-benar mendekat lagi
    # ke level yang ditembus.
    "breakout_min_retest_atr": 0.15,

    # BREAKOUT/BREAKDOWN baru valid setelah minimal N closed bars
    # bertahan di luar level yang ditembus.
    "breakout_follow_through_bars": 2,

    # --------------------------------------------------------
    # Entry Engine -- sanity guard
    # --------------------------------------------------------
    #
    # FIX: sebelumnya tidak ada batas atas untuk seberapa jauh
    # ideal_entry boleh berada dari harga pasar saat ini. Untuk
    # BREAKOUT/BREAKDOWN yang levelnya sudah lama ditinggalkan
    # harga (candle impulsif jauh + tren berlanjut), atau
    # EXTENDED/CONTINUATION yang zona pullback-nya jauh di
    # belakang, kandidat tetap lolos sebagai WAITING_RETEST/
    # WAITING_PULLBACK selamanya walau entry-nya sudah tidak
    # realistis tersentuh dalam kondisi market yang relevan.
    # Kandidat dengan jarak entry > ambang ini (dalam ATR)
    # dibuang sama sekali, bukan cuma ditandai WAITING_*.
    "max_entry_distance_atr": 2.0,

    # --------------------------------------------------------
    # Structure / risk
    # --------------------------------------------------------

    "swing_window": 8,

    # TP2 diturunkan karena 2.25R/3R terlalu jarang tercapai.
    # Setelah TP1 tercapai, SL dipindahkan ke breakeven.
    "risk_reward": [
        1.5,
        1.9,
        2.25,
    ],
    "move_sl_to_breakeven_after_tp1": True,

    # Batas atas risk_pct (risk/entry * 100) yang masih dianggap
    # layak ditradingkan. Kandidat dengan SL yang berarti risiko
    # lebih besar dari ini dibuang (lihat "risk_too_high").
    "max_risk_pct": 8.0,

    # --------------------------------------------------------
    # Higher-timeframe (1D) bias
    # --------------------------------------------------------

    # Konteks makro tambahan di luar TFS (15m/1h/4h). Bukan
    # hard filter -- kalau sisi trade (LONG/SHORT) melawan tren
    # EMA200 di timeframe ini, skor akhirnya didiskon. Kalau
    # data 1D gagal diambil, bias dianggap netral (tidak ada
    # penalty), supaya scan tetap jalan.
    "htf_bias_tf": "1d",

    # Ambil sedikit lebih banyak history khusus 1D agar EMA200
    # tidak gagal hanya karena data terlalu pendek.
    "daily_bias_klines": 260,
    "daily_bias_retries": 2,

    # Kandidat tanpa data 1D yang valid tidak boleh dianggap
    # netral secara diam-diam; gate harus benar-benar bekerja.
    "require_daily_bias": True,

    # POIN 1: soft-penalty lama 0.85 diganti jadi jauh lebih
    # berat (0.5). Selain itu, saat konflik dengan daily bias
    # sekarang WAJIB TF agreement 3/3 (gate tegas), bukan cuma
    # diskon skor.
    "htf_bias_penalty_factor": 0.50,

    # --------------------------------------------------------
    # Market regime (BTC) -- POIN 4
    # --------------------------------------------------------
    #
    # EMA200 pada BTCUSDT (4h + 1D) dipakai sebagai regime
    # check tambahan. Kalau BTC bearish, semua kandidat LONG
    # mendapat penalty ekstra (dan opsional hard gate).
    # Tidak mempengaruhi SHORT.
    "btc_regime_symbol": "BTCUSDT",
    "btc_regime_tfs": ["4h", "1d"],
    # Penalti diperkuat; hard gate tetap OFF sampai sample lebih panjang.
    "btc_regime_penalty_factor": 0.35,
    # True = hard reject LONG saat BTC bearish (lebih tegas).
    # False = cuma soft penalty.
    "btc_regime_hard_gate": False,

    # --------------------------------------------------------
    # Cooldown per-symbol -- POIN 5
    # --------------------------------------------------------
    #
    # Setelah sebuah symbol muncul sebagai kandidat final
    # (entah win/loss), skip N scan berikutnya supaya tidak
    # spam entry di simbol yang sama saat market choppy.
    # State disimpan di file JSON agar persisten antar-run
    # (mis. GitHub Actions cron).
    "cooldown_scans": 3,
    "cooldown_file": "synaptic_cooldown.json",

    # Persistent feedback loop. Satu record JSON per kandidat final
    # agar komponen score/regime/funding/MTF bisa dianalisis ulang.
    "trade_log_file": "synaptic_trade_log.jsonl",

    # --------------------------------------------------------
    # Diversity / correlation filter
    # --------------------------------------------------------

    # Tanpa filter ini, 5 kandidat final bisa saja semua alt
    # yang gerak bareng (mis. sekumpulan alt yang sama-sama
    # ngikut BTC) -- kelihatan "5 sinyal" padahal secara market
    # cuma 1 gerakan yang sama diulang-ulang. Korelasi dihitung
    # dari return candle timeframe di bawah ini, HANYA dibanding
    # kandidat lain yang searah (LONG vs LONG, SHORT vs SHORT)
    # -- dua koin berkorelasi tinggi tapi beda arah bukan
    # duplikasi, itu tetap dua bet yang berbeda.
    "diversity_correlation_tf": "1h",

    # Jumlah candle (dari data yang sudah difetch, tidak ada
    # API call tambahan) yang dipakai untuk menghitung korelasi.
    "diversity_lookback": 50,

    # Kandidat dibuang dari daftar final kalau korelasi
    # return-nya terhadap kandidat lain yang SUDAH terpilih
    # (searah) >= ambang ini.
    "diversity_max_correlation": 0.85,

    # --------------------------------------------------------
    # Funding rate (crowded-trade guard)
    # --------------------------------------------------------

    # Funding rate (per 8 jam, dari /fapi/v1/premiumIndex) yang
    # SEARAH sisi trade dan melewati ambang ini (dalam persen)
    # dianggap sinyal crowded trade -- risiko funding-squeeze
    # naik. Bukan hard veto (supaya tidak "mengarang" penolakan
    # di luar data teknikal), tapi skor akhirnya didiskon.
    "funding_rate_alert_pct": 0.05,

    "funding_rate_penalty_factor": 0.7,

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    "api_timeout": 10,

    # Retry cepat.
    # Tujuannya mencegah GitHub Actions tertahan lama.
    "api_retries": 2,

    "retry_base_delay": 0.35,

    # --------------------------------------------------------
    # Chart defaults
    # --------------------------------------------------------

    "visible_candles": {
        "15m": 60,
        "1h": 48,
        "4h": 50,
    },
}


# ============================================================
# BINANCE ENDPOINTS
#
# PENTING:
# Endpoint TIDAK DIUBAH dari kode sebelumnya.
# ============================================================

BASE_URLS = [
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


# ============================================================
# THREAD LOCAL SESSION
# ============================================================

_thread_local = threading.local()


def get_session():
    """
    Setiap worker thread mempunyai Session sendiri.

    Ini menghindari penggunaan requests.Session global
    secara bersamaan oleh banyak thread.
    """

    session = getattr(_thread_local, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session

    return session


# ============================================================
# API ENGINE
# ============================================================

def _retry_delay(attempt, response=None):
    """
    Delay pendek dan terkontrol.

    Retry-After digunakan bila Binance mengirimkannya.
    """

    if response is not None:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                value = float(retry_after)
                return min(max(value, 0.2), 5.0)
            except (TypeError, ValueError):
                pass

    base = CONFIG["retry_base_delay"]
    delay = base * (2 ** attempt)

    # Jitter kecil supaya banyak worker tidak retry
    # pada milidetik yang sama.
    delay += random.uniform(0.05, 0.20)

    return min(delay, 3.0)


def _parse_response(response):
    """
    Parse response JSON dengan aman.

    HTTP 202 tidak otomatis dianggap sukses.
    Kalau body-nya valid JSON Binance, kita tetap proses.
    """

    try:
        return response.json()
    except ValueError:
        return None


def api(path, params=None, timeout=None):
    """
    Central Binance API request.

    Endpoint tetap menggunakan BASE_URLS yang lama.

    Perbaikan:
    - session per thread
    - retry pendek
    - HTTP 202 tidak langsung fatal
    - 429/418 ditangani khusus
    - timeout dibatasi
    - endpoint berikutnya dicoba hanya setelah retry lokal
    """

    if timeout is None:
        timeout = CONFIG["api_timeout"]

    session = get_session()

    last_error = None

    for base_url in BASE_URLS:

        url = base_url + path

        for attempt in range(CONFIG["api_retries"] + 1):

            try:

                response = session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )

                status = response.status_code

                # ------------------------------------------------
                # Normal response
                # ------------------------------------------------

                if status == 200:

                    data = _parse_response(response)

                    if data is None:
                        last_error = (
                            f"{base_url} HTTP 200 but invalid JSON"
                        )
                        break

                    if (
                        isinstance(data, dict)
                        and "code" in data
                        and "msg" in data
                    ):
                        last_error = (
                            f"{data.get('code')}: "
                            f"{data.get('msg')}"
                        )

                        if attempt < CONFIG["api_retries"]:
                            time.sleep(
                                _retry_delay(
                                    attempt,
                                    response,
                                )
                            )
                            continue

                        break

                    return data

                # ------------------------------------------------
                # 202
                # ------------------------------------------------
                #
                # Jangan langsung menganggap endpoint mati.
                #
                # Beberapa gateway/proxy dapat mengembalikan
                # 202 sementara request masih diproses.
                # ------------------------------------------------

                if status == 202:

                    data = _parse_response(response)

                    if isinstance(data, (dict, list)):

                        # Jika ternyata body sudah berisi data
                        # Binance yang bisa dipakai, gunakan.
                        if not (
                            isinstance(data, dict)
                            and "code" in data
                            and "msg" in data
                        ):
                            return data

                    last_error = (
                        f"{base_url} HTTP 202"
                    )

                    if attempt < CONFIG["api_retries"]:
                        time.sleep(
                            _retry_delay(
                                attempt,
                                response,
                            )
                        )
                        continue

                    # Setelah retry lokal habis,
                    # baru pindah endpoint.
                    break

                # ------------------------------------------------
                # Rate limit
                # ------------------------------------------------

                if status in (418, 429):

                    last_error = (
                        f"{base_url} HTTP {status}"
                    )

                    if attempt < CONFIG["api_retries"]:
                        time.sleep(
                            _retry_delay(
                                attempt,
                                response,
                            )
                        )
                        continue

                    break

                # ------------------------------------------------
                # 451
                # ------------------------------------------------

                if status == 451:

                    last_error = (
                        f"{base_url} HTTP 451"
                    )

                    # Tidak perlu retry berkali-kali
                    break

                # ------------------------------------------------
                # Other HTTP errors
                # ------------------------------------------------

                last_error = (
                    f"{base_url} HTTP {status}"
                )

                if attempt < CONFIG["api_retries"]:
                    time.sleep(
                        _retry_delay(
                            attempt,
                            response,
                        )
                    )
                    continue

                break

            except requests.Timeout:

                last_error = (
                    f"{base_url} timeout"
                )

                if attempt < CONFIG["api_retries"]:
                    time.sleep(
                        _retry_delay(attempt)
                    )
                    continue

                break

            except requests.RequestException as exc:

                last_error = (
                    f"{base_url}: {exc}"
                )

                if attempt < CONFIG["api_retries"]:
                    time.sleep(
                        _retry_delay(attempt)
                    )
                    continue

                break

    raise RuntimeError(
        f"All Binance endpoints failed: {last_error}"
    )


# ============================================================
# BASIC API ENDPOINTS
#
# PATH TETAP SAMA.
# ============================================================

def exchange_info():
    return api(
        "/fapi/v1/exchangeInfo",
        timeout=15,
    )


def ticker_24h():
    return api(
        "/fapi/v1/ticker/24hr",
        timeout=15,
    )


def premium_index():
    """
    Funding rate + mark price seluruh symbol dalam satu call.

    Dipakai untuk crowded-trade guard (funding_rate_alert_pct).
    Endpoint baru -- tidak mengubah endpoint yang sudah ada.
    """
    return api(
        "/fapi/v1/premiumIndex",
        timeout=15,
    )


# ============================================================
# UNIVERSE
# ============================================================

def universe():

    started = time.time()

    info = exchange_info()
    tickers = ticker_24h()

    ticker_map = {
        str(item.get("symbol", "")): item
        for item in tickers
        if isinstance(item, dict)
    }

    # Funding rate map. Kalau endpoint ini gagal, scan tetap
    # jalan -- funding rate cuma dipakai sebagai penalty
    # tambahan, bukan syarat wajib.
    funding_map = {}

    try:

        premium = premium_index()

        funding_map = {
            str(item.get("symbol", "")): float(
                item.get("lastFundingRate", 0) or 0
            )
            for item in premium
            if isinstance(item, dict)
        }

    except Exception as exc:

        logger.warning(
            "Funding rate unavailable, "
            "continuing without it: %s",
            exc,
        )

    rows = []

    for item in info.get("symbols", []):

        if not isinstance(item, dict):
            continue

        symbol = str(item.get("symbol", ""))

        if not symbol:
            continue

        if item.get("contractType") != "PERPETUAL":
            REJECTIONS.note(
                "universe", symbol, "not_perpetual",
                contract_type=item.get("contractType"),
            )
            continue

        if item.get("quoteAsset") != "USDT":
            REJECTIONS.note(
                "universe", symbol, "quote_not_usdt",
                quote_asset=item.get("quoteAsset"),
            )
            continue

        if item.get("status") != "TRADING":
            REJECTIONS.note(
                "universe", symbol, "not_trading",
                status=item.get("status"),
            )
            continue

        if symbol in IGNORED_SYMBOLS:
            REJECTIONS.note(
                "universe", symbol, "in_ignored_list",
            )
            continue

        ticker = ticker_map.get(symbol)

        if not ticker:
            REJECTIONS.note(
                "universe", symbol, "no_ticker_data",
            )
            continue

        try:

            quote_volume = float(
                ticker.get("quoteVolume", 0)
            )

            change_24h = float(
                ticker.get("priceChangePercent", 0)
            )

            last_price = float(
                ticker.get("lastPrice", 0)
            )

        except (TypeError, ValueError) as exc:

            REJECTIONS.note(
                "universe", symbol, "invalid_ticker_values",
                error=str(exc),
            )

            continue

        if (
            quote_volume
            < CONFIG["min_quote_volume_24h"]
        ):
            REJECTIONS.note(
                "universe", symbol, "quote_volume_below_min",
                quote_volume=quote_volume,
                min_quote_volume_24h=CONFIG["min_quote_volume_24h"],
            )
            continue

        if last_price <= 0:
            REJECTIONS.note(
                "universe", symbol, "invalid_last_price",
                last_price=last_price,
            )
            continue

        funding_rate = funding_map.get(
            symbol,
            0.0,
        )

        # IMPORTANT:
        #
        # Tidak menggunakan change24h sebagai filter
        # trending.
        #
        # Semua liquid USDT perpetual tetap masuk.
        #
        rows.append(
            (
                symbol,
                change_24h,
                quote_volume,
                funding_rate,
            )
        )

    if CONFIG["universe_size"] > 0:
        rows = rows[
            :CONFIG["universe_size"]
        ]

    elapsed = time.time() - started

    logger.info(
        "Universe matched %d active USDT-M "
        "perpetual symbols globally "
        "(%.2fs).",
        len(rows),
        elapsed,
    )

    return rows


# ============================================================
# KLINES
# ============================================================

def klines(symbol, interval, limit=None):

    if limit is None:
        limit = CONFIG["klines"]

    raw = api(
        "/fapi/v1/klines",
        {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        timeout=CONFIG["api_timeout"],
    )

    columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(
        raw,
        columns=columns,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# HIGHER-TIMEFRAME (1D) BIAS
#
# Konteks makro tambahan, terpisah dari TFS (15m/1h/4h) yang
# dipakai untuk MTF agreement -- tidak mengubah skema voting
# yang sudah ada. Cuma EMA200 harian, dihitung dari close
# (tidak perlu ATR/MACD/Supertrend harian).
# ============================================================

def daily_bias(symbol):

    retries = CONFIG.get("daily_bias_retries", 0)
    candles = None

    for attempt in range(retries + 1):
        try:
            candles = klines(
                symbol,
                CONFIG["htf_bias_tf"],
                limit=CONFIG.get(
                    "daily_bias_klines",
                    CONFIG["klines"],
                ),
            )
            break
        except Exception as exc:
            if attempt >= retries:
                logger.debug(
                    "Daily bias %s failed after %d retries: %s",
                    symbol, retries, exc,
                )
                return None

            time.sleep(_retry_delay(attempt))

    if candles is None:
        return None

    if len(candles) < CONFIG["ema_period"] + 5:
        logger.debug(
            "Daily bias %s insufficient history: %d",
            symbol, len(candles),
        )
        return None

    try:
        closes = candles["close"]

        ema = closes.ewm(
            span=CONFIG["ema_period"],
            adjust=False,
        ).mean()

        signal_offset = (
            1 if CONFIG["confirm_on_closed_bar"] else 0
        )

        pos = len(closes) - 1 - signal_offset

        if pos < 1:
            return None

        close_value = float(closes.iloc[pos])
        ema_value = float(ema.iloc[pos])

        if (
            not np.isfinite(close_value)
            or not np.isfinite(ema_value)
        ):
            return None

        if close_value > ema_value:
            return "BULLISH"

        if close_value < ema_value:
            return "BEARISH"

        return "NEUTRAL"

    except Exception as exc:
        logger.debug(
            "Daily bias %s calculation error: %s",
            symbol, exc,
        )
        return None


# ============================================================
# BTC MARKET REGIME -- POIN 4
#
# Cek EMA200 BTCUSDT pada 4h dan 1D.
# Return "BEARISH" kalau mayoritas TF bearish, "BULLISH"
# kalau mayoritas bullish, None kalau data gagal.
# ============================================================

def btc_market_regime():

    symbol = CONFIG["btc_regime_symbol"]
    tfs = CONFIG["btc_regime_tfs"]

    votes = []

    for tf in tfs:

        try:

            candles = klines(symbol, tf)

            if len(candles) < CONFIG["ema_period"] + 5:
                continue

            closes = candles["close"]

            ema = closes.ewm(
                span=CONFIG["ema_period"],
                adjust=False,
            ).mean()

            signal_offset = (
                1 if CONFIG["confirm_on_closed_bar"] else 0
            )

            pos = len(closes) - 1 - signal_offset

            if pos < 1:
                continue

            close_value = float(closes.iloc[pos])
            ema_value = float(ema.iloc[pos])

            if (
                not np.isfinite(close_value)
                or not np.isfinite(ema_value)
            ):
                continue

            if close_value > ema_value:
                votes.append("BULLISH")
            elif close_value < ema_value:
                votes.append("BEARISH")
            else:
                votes.append("NEUTRAL")

        except Exception as exc:

            logger.debug(
                "BTC regime %s/%s error: %s",
                symbol, tf, exc,
            )

    if not votes:
        return None

    bearish = sum(1 for v in votes if v == "BEARISH")
    bullish = sum(1 for v in votes if v == "BULLISH")

    if bearish > bullish:
        return "BEARISH"

    if bullish > bearish:
        return "BULLISH"

    return "NEUTRAL"


# ============================================================
# COOLDOWN PER-SYMBOL -- POIN 5
#
# Setelah symbol muncul di final candidates, skip N scan
# berikutnya. State disimpan ke file JSON.
# ============================================================

def load_cooldown():
    """Load cooldown state dari file. Format:
    {
      "SYMBOL": {"remaining": N, "last_seen": "..."},
      ...
    }
    """

    path = Path(CONFIG["cooldown_file"])

    if not path.exists():
        return {}

    try:

        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, dict):
            return data

    except Exception as exc:

        logger.warning(
            "Failed to load cooldown file: %s",
            exc,
        )

    return {}


def save_cooldown(state):
    """Simpan cooldown state ke file."""

    path = Path(CONFIG["cooldown_file"])

    try:

        path.write_text(
            json.dumps(
                state,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except Exception as exc:

        logger.warning(
            "Failed to save cooldown file: %s",
            exc,
        )


def apply_cooldown_decay(state):
    """Kurangi remaining tiap symbol, hapus yang sudah 0."""

    new_state = {}

    for symbol, info in state.items():

        remaining = int(info.get("remaining", 0))

        if remaining > 1:
            new_state[symbol] = {
                "remaining": remaining - 1,
                "last_seen": info.get("last_seen"),
            }
        # remaining == 1 -> setelah decay jadi 0, dihapus
        # remaining <= 0 -> sudah expired, dihapus

    return new_state


def is_on_cooldown(symbol, state):
    """True kalau symbol masih dalam masa cooldown."""

    info = state.get(symbol)

    if not info:
        return False

    return int(info.get("remaining", 0)) > 0


def register_cooldown(state, symbols):
    """Daftarkan symbol yang baru muncul sebagai kandidat."""

    n = CONFIG["cooldown_scans"]

    if n <= 0:
        return state

    now = pd.Timestamp.now(tz="UTC").isoformat()

    for symbol in symbols:

        state[symbol] = {
            "remaining": n,
            "last_seen": now,
        }

    return state


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    x = df.copy()

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    x["ema200"] = (
        x["close"]
        .ewm(
            span=CONFIG["ema_period"],
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    fast = (
        x["close"]
        .ewm(
            span=CONFIG["macd_fast"],
            adjust=False,
        )
        .mean()
    )

    slow = (
        x["close"]
        .ewm(
            span=CONFIG["macd_slow"],
            adjust=False,
        )
        .mean()
    )

    x["macd"] = fast - slow

    x["macd_signal"] = (
        x["macd"]
        .ewm(
            span=CONFIG["macd_signal"],
            adjust=False,
        )
        .mean()
    )

    x["macd_hist"] = (
        x["macd"] - x["macd_signal"]
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = x["close"].shift(1)

    true_range = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"] - previous_close
            ).abs(),
            (
                x["low"] - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr"] = (
        true_range
        .ewm(
            alpha=1 / CONFIG["atr_period"],
            adjust=False,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    x["volume_ma"] = (
        x["volume"]
        .rolling(
            CONFIG["volume_ma_period"],
            min_periods=1,
        )
        .mean()
    )

    x["volume_ratio"] = (
        x["volume"]
        / x["volume_ma"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Supertrend 10 / 2.5
    # --------------------------------------------------------

    period = CONFIG["supertrend_period"]
    multiplier = CONFIG[
        "supertrend_multiplier"
    ]

    hl2 = (
        x["high"] + x["low"]
    ) / 2.0

    basic_upper = (
        hl2 + multiplier * x["atr"]
    )

    basic_lower = (
        hl2 - multiplier * x["atr"]
    )

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    direction = pd.Series(
        1,
        index=x.index,
        dtype=int,
    )

    supertrend = pd.Series(
        np.nan,
        index=x.index,
        dtype=float,
    )

    for i in range(1, len(x)):

        if (
            basic_upper.iloc[i]
            < final_upper.iloc[i - 1]
            or
            x["close"].iloc[i - 1]
            > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )
        else:
            final_upper.iloc[i] = (
                final_upper.iloc[i - 1]
            )

        if (
            basic_lower.iloc[i]
            > final_lower.iloc[i - 1]
            or
            x["close"].iloc[i - 1]
            < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )
        else:
            final_lower.iloc[i] = (
                final_lower.iloc[i - 1]
            )

        if (
            direction.iloc[i - 1] == -1
            and
            x["close"].iloc[i]
            > final_upper.iloc[i - 1]
        ):
            direction.iloc[i] = 1

        elif (
            direction.iloc[i - 1] == 1
            and
            x["close"].iloc[i]
            < final_lower.iloc[i - 1]
        ):
            direction.iloc[i] = -1

        else:
            direction.iloc[i] = (
                direction.iloc[i - 1]
            )

        if direction.iloc[i] > 0:
            supertrend.iloc[i] = (
                final_lower.iloc[i]
            )
        else:
            supertrend.iloc[i] = (
                final_upper.iloc[i]
            )

    if len(x):

        supertrend.iloc[0] = (
            final_lower.iloc[0]
        )

    x["supertrend"] = supertrend
    x["st_dir"] = direction

    return x


# ============================================================
# CHART SERIALIZATION
# ============================================================

def serialize_chart_data(df):

    required_columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ema200",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "volume_ma",
        "volume_ratio",
        "supertrend",
        "st_dir",
    ]

    available = [
        col
        for col in required_columns
        if col in df.columns
    ]

    records = []

    for _, row in df.iterrows():

        item = {}

        for col in available:

            value = row[col]

            if col == "time":

                item[col] = (
                    None
                    if pd.isna(value)
                    else pd.Timestamp(
                        value
                    ).isoformat()
                )

                continue

            if pd.isna(value):

                item[col] = None
                continue

            if col == "st_dir":

                item[col] = int(value)

            else:

                item[col] = float(value)

        records.append(item)

    return records


# ============================================================
# MOVEMENT / MOMENTUM SCORE
# ============================================================

def movement_score(df, symbol=None):

    if len(df) < 51:
        REJECTIONS.note(
            "stage1", symbol, "insufficient_candles",
            have=len(df), need=51,
        )
        return -1.0, None

    # IMPORTANT:
    # df sudah bisa berupa data yang sudah dihitung indikatornya.
    if "ema200" not in df.columns:
        x = add_indicators(df)
    else:
        x = df

    n = len(x)

    # --------------------------------------------------------
    # Repaint guard.
    #
    # signal_offset = 1 -> bar acuan adalah candle terakhir yang
    # SUDAH CLOSED (x.iloc[-2]), bukan candle yang masih
    # berjalan. Mencegah momentum score naik-turun sendiri
    # sebelum candle terakhir selesai.
    # --------------------------------------------------------

    signal_offset = (
        1 if CONFIG["confirm_on_closed_bar"] else 0
    )

    signal_pos = n - 1 - signal_offset

    if signal_pos < 1:
        REJECTIONS.note(
            "stage1", symbol, "insufficient_signal_position",
            signal_pos=signal_pos,
        )
        return -1.0, None

    last = x.iloc[signal_pos]

    close = float(last["close"])
    atr_value = float(last["atr"])

    if (
        not np.isfinite(close)
        or close <= 0
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        REJECTIONS.note(
            "stage1", symbol, "invalid_price_or_atr",
            close=close, atr=atr_value,
        )
        return -1.0, None

    # --------------------------------------------------------
    # Liquidity / volatility sanity gate.
    #
    # Volume 24h (di universe()) tidak menjamin range candle
    # cukup lebar untuk ditradingkan. Symbol dengan ATR/close
    # terlalu kecil (choppy/dead) dibuang di sini, sebelum
    # sempat mendominasi momentum_pool.
    # --------------------------------------------------------

    atr_pct = (atr_value / close) * 100

    if atr_pct < CONFIG["min_atr_pct"]:
        REJECTIONS.note(
            "stage1", symbol, "atr_below_min",
            atr_pct=round(atr_pct, 4),
            min_atr_pct=CONFIG["min_atr_pct"],
        )
        return -1.0, None

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

    slow_n = CONFIG[
        "momentum_slow_bars"
    ]

    if signal_pos <= slow_n + 1:
        REJECTIONS.note(
            "stage1", symbol, "insufficient_history_for_momentum",
            signal_pos=signal_pos, slow_n=slow_n,
        )
        return -1.0, None

    fast_ref = float(
        x["close"].iloc[
            signal_pos - fast_n
        ]
    )

    slow_ref = float(
        x["close"].iloc[
            signal_pos - slow_n
        ]
    )

    if fast_ref <= 0 or slow_ref <= 0:
        REJECTIONS.note(
            "stage1", symbol, "invalid_reference_price",
            fast_ref=fast_ref, slow_ref=slow_ref,
        )
        return -1.0, None

    # --------------------------------------------------------
    # Directional momentum.
    #
    # Sebelumnya fast/slow return memakai abs(), sehingga gerak
    # +5% dan -5% punya nilai sama. Akibatnya Stage 1 bisa penuh
    # kandidat LONG walau regime pasar sedang melemah.
    # Sekarang kekuatan dihitung per arah dan Stage 1 memilih
    # sisi yang benar-benar lebih kuat.
    # --------------------------------------------------------

    fast_return_signed = (
        close / fast_ref - 1.0
    ) * 100

    slow_return_signed = (
        close / slow_ref - 1.0
    ) * 100

    fast_return = abs(fast_return_signed)
    slow_return = abs(slow_return_signed)

    atr_move = (
        abs(close - fast_ref)
        / atr_value
    )

    volume_ratio = float(
        last["volume_ratio"]
    )

    if not np.isfinite(volume_ratio):
        volume_ratio = 1.0

    volume_bonus = min(
        max(volume_ratio, 0.0),
        4.0,
    )

    window = CONFIG[
        "breakout_window"
    ]

    if signal_pos <= window:
        REJECTIONS.note(
            "stage1", symbol, "insufficient_history_for_breakout",
            signal_pos=signal_pos, window=window,
        )
        return -1.0, None

    prev_high = float(
        x["high"]
        .iloc[signal_pos - window:signal_pos]
        .max()
    )

    prev_low = float(
        x["low"]
        .iloc[signal_pos - window:signal_pos]
        .min()
    )

    breakout_bonus = 0.0

    if (
        close > prev_high
        or close < prev_low
    ):
        breakout_bonus = 2.0

    direction = (
        1
        if float(last["close"])
        >= float(last["open"])
        else -1
    )

    # --------------------------------------------------------
    # Extension penalty.
    #
    # movement_score menghargai pergerakan yang SUDAH terjadi,
    # jadi tanpa penalti, Stage 1 struktural condong ke koin
    # yang sudah lari jauh dari EMA200 -- mendesak keluar setup
    # yang masih segar. Kalau jarak ke EMA200 (dalam ATR) sudah
    # ekstrem, skor didiskon dengan faktor tetap.
    # --------------------------------------------------------

    ema_value = float(last["ema200"])

    penalty = 1.0

    if np.isfinite(ema_value) and ema_value > 0:

        extension_atr = (
            abs(close - ema_value) / atr_value
        )

        if (
            extension_atr
            >= CONFIG["extended_momentum_penalty_atr"]
        ):

            penalty = CONFIG[
                "extended_momentum_penalty_factor"
            ]

    # Directional base scores.
    long_score = (
        max(fast_return_signed, 0.0) * 2.0
        + max(slow_return_signed, 0.0)
        + min(atr_move, 5.0) * (
            1.5 if fast_return_signed > 0 else 0.0
        )
        + volume_bonus * 1.25
    )

    short_score = (
        max(-fast_return_signed, 0.0) * 2.0
        + max(-slow_return_signed, 0.0)
        + min(atr_move, 5.0) * (
            1.5 if fast_return_signed < 0 else 0.0
        )
        + volume_bonus * 1.25
    )

    # Breakout bonus belongs only to its actual direction.
    if close > prev_high:
        long_score += breakout_bonus
    elif close < prev_low:
        short_score += breakout_bonus

    # Local regime/arah is now part of Stage 1, before momentum_pool.
    trend_long = (
        close > ema_value
        and int(last["st_dir"]) > 0
    )
    trend_short = (
        close < ema_value
        and int(last["st_dir"]) < 0
    )

    trend_bonus = CONFIG["stage1_trend_bonus"]
    if trend_long:
        long_score += trend_bonus
    elif not trend_short:
        long_score *= CONFIG["stage1_countertrend_penalty_factor"]

    if trend_short:
        short_score += trend_bonus
    elif not trend_long:
        short_score *= CONFIG["stage1_countertrend_penalty_factor"]

    # Apply the existing extension penalty after directional scoring.
    long_score *= penalty
    short_score *= penalty

    preferred_side = (
        "LONG" if long_score > short_score
        else "SHORT"
        if short_score > long_score
        else None
    )

    score = max(long_score, short_score)

    return float(score), {
        "df": x,
        "direction": direction,
        "preferred_side": preferred_side,
        "long_score": float(long_score),
        "short_score": float(short_score),
        "fast_return": fast_return,
        "slow_return": slow_return,
        "fast_return_signed": fast_return_signed,
        "slow_return_signed": slow_return_signed,
        "volume_ratio": volume_ratio,
        "atr_move": atr_move,
    }


# ============================================================
# TIMEFRAME SCORE
# ============================================================

def score_tf(df, symbol=None, tf=None):

    if "ema200" not in df.columns:
        x = add_indicators(df)
    else:
        x = df

    if len(x) < 210:
        REJECTIONS.note(
            "stage2", symbol, "tf_insufficient_candles",
            tf=tf, have=len(x), need=210,
        )
        return None

    n = len(x)

    # --------------------------------------------------------
    # Repaint guard.
    #
    # signal_offset = 1 -> semua sinyal (EMA/Supertrend/MACD/
    # volume/breakout) dihitung dari candle terakhir yang
    # SUDAH CLOSED (x.iloc[-2]), bukan candle yang masih
    # berjalan. Harga live (x.iloc[-1]) tetap diekspos sebagai
    # "live_close" untuk keputusan entry_state/ideal_entry di
    # Setup Engine, jadi presisi harga saat ini tidak hilang --
    # yang distabilkan cuma klasifikasi regime-nya.
    # --------------------------------------------------------

    signal_offset = (
        1 if CONFIG["confirm_on_closed_bar"] else 0
    )

    signal_pos = n - 1 - signal_offset

    if signal_pos <= CONFIG["breakout_window"]:
        REJECTIONS.note(
            "stage2", symbol, "tf_insufficient_signal_position",
            tf=tf, signal_pos=signal_pos,
            breakout_window=CONFIG["breakout_window"],
        )
        return None

    last = x.iloc[signal_pos]
    previous = x.iloc[signal_pos - 1]

    live_close = float(x.iloc[-1]["close"])

    long_score = 0.0
    short_score = 0.0

    long_reasons = []
    short_reasons = []

    close = float(last["close"])
    ema = float(last["ema200"])
    atr_value = float(last["atr"])

    if not np.isfinite(close):
        REJECTIONS.note(
            "stage2", symbol, "tf_invalid_close",
            tf=tf, close=close,
        )
        return None

    if not np.isfinite(ema):
        REJECTIONS.note(
            "stage2", symbol, "tf_invalid_ema",
            tf=tf, ema=ema,
        )
        return None

    if not np.isfinite(atr_value) or atr_value <= 0:
        REJECTIONS.note(
            "stage2", symbol, "tf_invalid_atr",
            tf=tf, atr=atr_value,
        )
        return None

    if not np.isfinite(live_close) or live_close <= 0:
        live_close = close

    volume_ratio = float(
        last["volume_ratio"]
    )

    if not np.isfinite(volume_ratio):
        volume_ratio = 1.0

    # --------------------------------------------------------
    # EMA 200
    # --------------------------------------------------------

    if close > ema:

        long_score += 2.0
        long_reasons.append(
            "above EMA200"
        )

    elif close < ema:

        short_score += 2.0
        short_reasons.append(
            "below EMA200"
        )

    # --------------------------------------------------------
    # Supertrend
    # --------------------------------------------------------

    st_dir = int(last["st_dir"])

    if st_dir > 0:

        long_score += 2.0
        long_reasons.append(
            "Supertrend bullish"
        )

    else:

        short_score += 2.0
        short_reasons.append(
            "Supertrend bearish"
        )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = float(last["macd"])
    macd_signal = float(
        last["macd_signal"]
    )

    hist_now = float(
        last["macd_hist"]
    )

    hist_previous = float(
        previous["macd_hist"]
    )

    if macd > macd_signal:

        long_score += 1.0
        long_reasons.append(
            "MACD bullish"
        )

        if hist_now > hist_previous:

            long_score += 0.5
            long_reasons.append(
                "MACD histogram rising"
            )

    elif macd < macd_signal:

        short_score += 1.0
        short_reasons.append(
            "MACD bearish"
        )

        if hist_now < hist_previous:

            short_score += 0.5
            short_reasons.append(
                "MACD histogram falling"
            )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if (
        volume_ratio
        >= CONFIG["volume_ratio_min"]
    ):

        if close > float(last["open"]):

            long_score += 1.5

            long_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

        elif close < float(last["open"]):

            short_score += 1.5

            short_reasons.append(
                f"volume {volume_ratio:.1f}x"
            )

    # --------------------------------------------------------
    # Breakout / breakdown with follow-through confirmation.
    #
    # Satu closed bar di luar level tidak lagi cukup. Untuk N=2,
    # bar breakout dan satu bar sesudahnya harus sama-sama close
    # di luar level yang dihitung sebelum bar-bar konfirmasi.
    # --------------------------------------------------------

    window = CONFIG[
        "breakout_window"
    ]

    required_follow_through = max(
        1,
        int(CONFIG["breakout_follow_through_bars"]),
    )

    breakout_level_up = None
    breakout_level_down = None
    breakout_confirmed_long = False
    breakout_confirmed_short = False
    breakout_follow_through_bars = 0

    # Exclude the confirmation bars from the reference range.
    level_end = signal_pos - required_follow_through + 1

    if level_end >= window:

        level_start = level_end - window

        previous_high = float(
            x["high"]
            .iloc[level_start:level_end]
            .max()
        )

        previous_low = float(
            x["low"]
            .iloc[level_start:level_end]
            .min()
        )

        breakout_level_up = previous_high
        breakout_level_down = previous_low

        confirm_closes = x["close"].iloc[
            signal_pos - required_follow_through + 1:
            signal_pos + 1
        ]

        if (
            len(confirm_closes) >= required_follow_through
            and bool((confirm_closes > previous_high).all())
        ):
            breakout_confirmed_long = True
            breakout_follow_through_bars = required_follow_through
            long_score += 1.5
            long_reasons.append(
                f"20-bar breakout ({required_follow_through}-bar follow-through)"
            )

        if (
            len(confirm_closes) >= required_follow_through
            and bool((confirm_closes < previous_low).all())
        ):
            breakout_confirmed_short = True
            breakout_follow_through_bars = required_follow_through
            short_score += 1.5
            short_reasons.append(
                f"20-bar breakdown ({required_follow_through}-bar follow-through)"
            )

    return {
        "long": round(
            long_score,
            3,
        ),

        "short": round(
            short_score,
            3,
        ),

        "long_reasons": long_reasons,

        "short_reasons": short_reasons,

        "close": close,

        "live_close": live_close,

        "ema200": ema,

        "atr": atr_value,

        "volume_ratio": volume_ratio,

        "st_dir": st_dir,

        "macd": macd,

        "macd_signal": macd_signal,

        "macd_hist": hist_now,

        "breakout_level_up": breakout_level_up,

        "breakout_level_down": breakout_level_down,

        "breakout_confirmed_long": breakout_confirmed_long,

        "breakout_confirmed_short": breakout_confirmed_short,

        "breakout_follow_through_bars": breakout_follow_through_bars,
    }


# ============================================================
# SETUP ENGINE
#
# Mengklasifikasikan price action pada execution_tf menjadi
# salah satu dari 5 tipe setup, murni dari data yang sudah
# dihitung oleh score_tf() (EMA200, Supertrend, ATR, reasons).
#
# Tidak ada indikator baru yang diperkenalkan di sini.
# ============================================================

def _no_setup_result():

    return {
        "setup_style": "NO_SETUP",
        "entry_state": "NO_SETUP",
        "ideal_entry": None,
        "reference_level": None,
        "retest_zone_low": None,
        "retest_zone_high": None,
        "setup_invalidation_level": None,
    }


def classify_setup(tf_score, side, live_price=None):

    # --------------------------------------------------------
    # Return value sekarang berupa dict, bukan string tunggal:
    #
    #   setup_style     : BREAKOUT / BREAKDOWN / PULLBACK /
    #                      CONTINUATION (disabled) / EXTENDED / NO_SETUP
    #                      -> regime / konteks price action.
    #                      Dihitung dari "close" di tf_score,
    #                      yaitu candle yang SUDAH CLOSED kalau
    #                      confirm_on_closed_bar aktif -> stabil,
    #                      tidak flip-flop sebelum candle selesai.
    #
    #   entry_state     : ENTRY_READY / WAITING_PULLBACK /
    #                      WAITING_RETEST / NO_SETUP
    #                      -> apakah harga SAAT INI (live_price,
    #                         bukan closed-bar) sudah di zona
    #                         entry ideal, atau belum.
    #
    #   ideal_entry     : level harga acuan untuk build_entry().
    #                      Untuk BREAKOUT/BREAKDOWN dan
    #                      EXTENDED/CONTINUATION-jauh, ini BUKAN
    #                      harga saat ini -> supaya entry tidak
    #                      chase candle yang sudah lari jauh.
    #
    #   reference_level : level breakout atau EMA200 yang jadi
    #                      acuan (untuk ditampilkan di chart).
    #
    #   retest_zone_low /
    #   retest_zone_high : batas bawah/atas zona retest yang
    #                      SEBENARNYA dipakai Entry Engine untuk
    #                      menentukan ENTRY_READY (ideal_entry
    #                      +/- entry_ready_atr, dalam ATR harga).
    #                      Hanya relevan untuk BREAKOUT/BREAKDOWN
    #                      -- diisi None untuk setup style lain.
    #                      vSch merender field ini APA ADANYA,
    #                      bukan menebak ulang lebar zona dari
    #                      reference_level saja -- supaya chart
    #                      dan requirement retest yang sebenarnya
    #                      (breakout_min_retest_atr / entry_ready_atr)
    #                      tidak pernah berbeda.
    # --------------------------------------------------------

    reasons = (
        tf_score["long_reasons"]
        if side == "LONG"
        else tf_score["short_reasons"]
    )

    is_breakout = (
        tf_score.get(
            "breakout_confirmed_long"
            if side == "LONG"
            else "breakout_confirmed_short"
        )
        is True
    )

    close = float(tf_score["close"])
    ema = float(tf_score["ema200"])
    atr_value = float(tf_score["atr"])
    st_dir = int(tf_score["st_dir"])

    if not np.isfinite(atr_value) or atr_value <= 0:
        return _no_setup_result()

    # market_price = harga LIVE (tick sekarang), dipakai khusus
    # untuk menentukan seberapa dekat harga saat ini ke entry
    # ideal (entry_state). Regime (setup_style) tetap pakai
    # "close" (closed-bar) di atas supaya tidak repaint.
    market_price = (
        float(live_price)
        if live_price is not None and np.isfinite(live_price)
        else close
    )

    trend_aligned = (
        (side == "LONG" and close > ema and st_dir > 0)
        or
        (side == "SHORT" and close < ema and st_dir < 0)
    )

    # --------------------------------------------------------
    # Jarak ke EMA200 dalam ATR, BERTANDA searah sisi trade.
    #
    # Positif  -> close berada di sisi yang menguntungkan
    #             posisi (di atas EMA untuk LONG, di bawah
    #             EMA untuk SHORT).
    # Negatif  -> close berada di sisi yang berlawanan arah
    #             (mis. LONG tapi close di bawah EMA200).
    #
    # PENTING: EXTENDED wajib pakai jarak bertanda ini, bukan
    # jarak absolut. Kalau pakai abs(), symbol yang close-nya
    # jauh di sisi BERLAWANAN arah trade bisa salah kelabeli
    # EXTENDED padahal seharusnya NO_SETUP.
    # --------------------------------------------------------

    directional_distance_atr = (
        (close - ema) / atr_value
        if side == "LONG"
        else (ema - close) / atr_value
    )

    entry_ready_band = (
        CONFIG["entry_ready_atr"] * atr_value
    )

    # --------------------------------------------------------
    # EXTENDED
    #
    # FIX (urutan pengecekan): cabang ini SEKARANG dicek
    # SEBELUM is_breakout. Sebelumnya is_breakout selalu menang
    # duluan -- kalau score_tf() mendeteksi 20-bar breakout/
    # breakdown, symbol langsung diberi label BREAKOUT/BREAKDOWN
    # walau close-nya sudah jauh sekali (>= setup_extended_atr)
    # dari EMA200. Padahal breakout yang muncul setelah harga
    # sudah overextended bukan lagi "fresh setup" -- itu
    # kelanjutan tren yang sudah lari jauh, dan entry retest ke
    # level breakout lama (yang sudah lama ditinggalkan harga)
    # jadi tidak realistis tersentuh (lihat kasus TRIA/RIVER:
    # entry dijepit ke level lama, padahal harga sudah menuju
    # TP1). Sekarang begitu directional_distance_atr sudah
    # extended, setup diklasifikasikan EXTENDED (entry diarahkan
    # ke zona pullback EMA200 yang jauh lebih relevan) terlepas
    # dari ada tidaknya sinyal breakout di reasons.
    # --------------------------------------------------------

    # FIX: EXTENDED wajib trend_aligned juga (EMA200 + arah
    # Supertrend), bukan cuma directional_distance_atr. Sebelumnya
    # cabang ini hanya mengecek jarak bertanda ke EMA200 --
    # directional_distance_atr tidak melibatkan Supertrend sama
    # sekali, jadi symbol yang close-nya kebetulan jauh di sisi
    # yang "menguntungkan" versus EMA200 tapi Supertrend belum/
    # tidak searah tetap lolos sebagai EXTENDED, padahal trend-nya
    # sendiri belum align.
    if (
        trend_aligned
        and directional_distance_atr >= CONFIG["setup_extended_atr"]
    ):

        pullback_level = (
            ema + CONFIG["setup_pullback_atr"] * atr_value
            if side == "LONG"
            else ema - CONFIG["setup_pullback_atr"] * atr_value
        )

        distance = abs(market_price - pullback_level)

        entry_state = (
            "ENTRY_READY"
            if distance <= entry_ready_band
            else "WAITING_PULLBACK"
        )

        return {
            "setup_style": "EXTENDED",
            "entry_state": entry_state,
            "ideal_entry": pullback_level,
            "reference_level": ema,
            "retest_zone_low": None,
            "retest_zone_high": None,
        }

    # --------------------------------------------------------
    # BREAKOUT / BREAKDOWN
    #
    # Entry ideal dijepit ke area RETEST level yang ditembus
    # (level + buffer kecil), bukan ke close saat ini. Kalau
    # candle breakout sudah lari jauh dari level (impulsif),
    # ideal_entry akan jauh dari harga sekarang -> entry_state
    # jadi WAITING_RETEST, bukan dipaksa ENTRY_READY di harga
    # yang sudah "telat"/chase.
    #
    # FIX (retest sungguhan): sebelumnya kalau harga baru
    # SEDIKIT menembus level, ideal_entry = market_price ->
    # ENTRY_READY seketika tanpa retest apapun (rawan fakeout).
    # Sekarang wajib ada mundur minimal breakout_min_retest_atr
    # dari titik ekstrem candle breakout balik ke arah level,
    # baru dianggap ENTRY_READY.
    #
    # retest_zone_low/high dihitung persis dari band yang dipakai
    # untuk menentukan ENTRY_READY (ideal_entry +/- entry_ready_
    # band) -- ini zona yang SAMA yang menentukan status
    # entry_state di bawah, bukan estimasi terpisah. vSch tinggal
    # merender apa adanya, tidak perlu menebak ±0.15% lagi.
    # --------------------------------------------------------

    if is_breakout:

        setup_style = (
            "BREAKOUT" if side == "LONG" else "BREAKDOWN"
        )

        level = (
            tf_score.get("breakout_level_up")
            if side == "LONG"
            else tf_score.get("breakout_level_down")
        )

        if level is None or not np.isfinite(level):
            return _no_setup_result()

        follow_through = int(
            tf_score.get("breakout_follow_through_bars", 0)
        )
        required_follow_through = int(
            CONFIG["breakout_follow_through_bars"]
        )

        if follow_through < required_follow_through:
            return _no_setup_result()

        # Breakout is invalid once live price crosses back through
        # the broken level. A retest that stays on the original side
        # remains valid; a full reclaim of the old range does not.
        breakout_invalidated = (
            side == "LONG" and market_price < level
        ) or (
            side == "SHORT" and market_price > level
        )

        if breakout_invalidated:
            return _no_setup_result()

        buffer_ = (
            CONFIG["breakout_chase_buffer_atr"] * atr_value
        )

        min_retest = (
            CONFIG["breakout_min_retest_atr"] * atr_value
        )

        if side == "LONG":
            ideal_entry = min(market_price, level + buffer_)
        else:
            ideal_entry = max(market_price, level - buffer_)

        distance = abs(market_price - ideal_entry)

        # Retest sungguhan: harga sekarang harus sudah mundur
        # setidaknya min_retest dari close breakout (bukan cuma
        # "kebetulan" market_price == ideal_entry karena baru
        # sedikit menembus level).
        if side == "LONG":
            retest_progress = close - market_price
        else:
            retest_progress = market_price - close

        has_retested = (
            retest_progress >= min_retest
            and (
                (side == "LONG" and market_price >= level)
                or
                (side == "SHORT" and market_price <= level)
            )
        )

        entry_state = (
            "ENTRY_READY"
            if (distance <= entry_ready_band and has_retested)
            else "WAITING_RETEST"
        )

        return {
            "setup_style": setup_style,
            "entry_state": entry_state,
            "ideal_entry": ideal_entry,
            "reference_level": level,
            "setup_invalidation_level": level,
            "retest_zone_low": ideal_entry - entry_ready_band,
            "retest_zone_high": ideal_entry + entry_ready_band,
        }

    # --------------------------------------------------------
    # PULLBACK
    #
    # Harga sudah berada di zona retracement yang valid.
    #
    # FIX: sebelumnya entry_state di-hardcode "ENTRY_READY"
    # begitu closed-bar berada di zona pullback, TANPA mengecek
    # apakah harga LIVE (market_price) masih di zona yang sama.
    # Beda dengan cabang BREAKOUT/EXTENDED yang eksplisit
    # membandingkan distance ke entry_ready_band. Sekarang
    # PULLBACK ikut memakai pengecekan yang sama supaya
    # konsisten: kalau harga live sudah kabur jauh dari EMA200
    # (closed-bar tadinya di zona pullback, tapi live price
    # sudah melesat), status jadi WAITING_PULLBACK, bukan
    # dipaksa ENTRY_READY di harga yang sudah basi.
    # --------------------------------------------------------

    if (
        trend_aligned
        and directional_distance_atr <= CONFIG["setup_pullback_atr"]
    ):

        live_distance_atr = (
            (market_price - ema) / atr_value
            if side == "LONG"
            else (ema - market_price) / atr_value
        )

        live_in_zone = (
            live_distance_atr
            <= CONFIG["setup_pullback_atr"]
            + CONFIG["entry_ready_atr"]
        )

        entry_state = (
            "ENTRY_READY" if live_in_zone else "WAITING_PULLBACK"
        )

        return {
            "setup_style": "PULLBACK",
            "entry_state": entry_state,
            "ideal_entry": market_price,
            "reference_level": ema,
            "retest_zone_low": None,
            "retest_zone_high": None,
        }

    # --------------------------------------------------------
    # CONTINUATION -- DISABLED SEMENTARA
    #
    # Data feedback menunjukkan continuation memiliki win rate
    # terendah dan avg R negatif pada kedua arah. Jangan membuat
    # continuation menjadi trade hanya karena trend masih align.
    # Kandidat di luar zona PULLBACK dan belum cukup extended
    # sekarang ditolak sampai data baru membuktikan sebaliknya.
    # --------------------------------------------------------

    return _no_setup_result()


# ============================================================
# ENTRY LOGIC
#
# Menentukan entry price berdasarkan tipe setup dari
# Setup Engine di atas.
# ============================================================

def build_entry(setup_info, side, price, exec_df, atr_value):

    # setup_info datang dari classify_setup() dan sudah berisi
    # "ideal_entry" yang dihitung sesuai tipe setup:
    #
    # - BREAKOUT/BREAKDOWN : dijepit ke area retest level yang
    #                        ditembus (tidak chase candle
    #                        impulsif).
    # - EXTENDED / CONTINUATION : diarahkan ke zona
    #                        pullback EMA200, bukan harga
    #                        puncak/dasar extension.
    # - PULLBACK : boleh entry di harga
    #                        pasar saat ini.
    #
    # PULLBACK di-refine lagi di sini memakai EMA200 execution_tf
    # yang paling baru + max_drift, identik dengan perilaku versi
    # sebelumnya, supaya tidak ada regresi pada setup PULLBACK.

    ideal_entry = setup_info.get("ideal_entry")

    if ideal_entry is None or not np.isfinite(ideal_entry):
        return price

    if setup_info["setup_style"] != "PULLBACK":
        return ideal_entry

    ema_level = float(
        exec_df.iloc[-1]["ema200"]
    )

    max_drift = 0.5 * atr_value

    if side == "LONG":
        return max(ema_level, price - max_drift)

    return min(ema_level, price + max_drift)


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(
    symbol,
    change_24h,
    quote_volume_24h,
    stage1_score,
    stage1_meta,
    funding_rate=0.0,
    btc_regime=None,
):

    data = {}

    try:

        df15 = stage1_meta["df"]

        scored_15m = score_tf(df15, symbol=symbol, tf="15m")

        if scored_15m:

            data["15m"] = {
                "score": scored_15m,
                "df": df15,
            }

    except Exception as exc:

        REJECTIONS.note(
            "stage2", symbol,
            f"tf_exception:{type(exc).__name__}",
            tf="15m", error=str(exc),
        )

    # --------------------------------------------------------
    # 1H + 4H
    # --------------------------------------------------------

    for tf in ["1h", "4h"]:

        try:

            candles = klines(
                symbol,
                tf,
            )

            if len(candles) < 210:
                REJECTIONS.note(
                    "stage2", symbol, "tf_insufficient_klines_raw",
                    tf=tf, have=len(candles), need=210,
                )
                continue

            enriched = add_indicators(
                candles
            )

            scored = score_tf(
                enriched,
                symbol=symbol,
                tf=tf,
            )

            if scored:

                data[tf] = {
                    "score": scored,
                    "df": enriched,
                }

        except Exception as exc:

            REJECTIONS.note(
                "stage2", symbol,
                f"tf_exception:{type(exc).__name__}",
                tf=tf, error=str(exc),
            )

    # Semua timeframe wajib tersedia.
    if set(data.keys()) != set(TFS):
        REJECTIONS.note(
            "stage2", symbol, "missing_timeframe_data",
            have=sorted(data.keys()),
            missing=sorted(set(TFS) - set(data.keys())),
        )
        return None

    # --------------------------------------------------------
    # MTF weights
    # --------------------------------------------------------

    weights = {
        "15m": 0.25,
        "1h": 0.35,
        "4h": 0.40,
    }

    long_total = sum(
        weights[tf]
        * data[tf]["score"]["long"]
        for tf in TFS
    )

    short_total = sum(
        weights[tf]
        * data[tf]["score"]["short"]
        for tf in TFS
    )

    if long_total > short_total:
        side = "LONG"
    else:
        side = "SHORT"

    raw_score = max(
        long_total,
        short_total,
    )

    wanted_direction = (
        1
        if side == "LONG"
        else -1
    )

    # --------------------------------------------------------
    # MTF agreement
    # --------------------------------------------------------

    votes = []

    for tf in TFS:

        tf_score = data[tf]["score"]

        if (
            tf_score["long"]
            == tf_score["short"]
        ):

            votes.append(0)

        elif (
            tf_score["long"]
            > tf_score["short"]
        ):

            votes.append(1)

        else:

            votes.append(-1)

    agreement = sum(
        vote == wanted_direction
        for vote in votes
    )

    # --------------------------------------------------------
    # Higher-timeframe (1D) bias -- ambil dulu supaya bisa
    # dipakai sebagai gate agreement (POIN 1).
    # --------------------------------------------------------

    htf_bias = daily_bias(symbol)

    if (
        CONFIG.get("require_daily_bias", False)
        and htf_bias is None
    ):
        REJECTIONS.note(
            "stage2", symbol, "daily_bias_unavailable",
        )
        return None

    htf_conflict = (
        (htf_bias == "BEARISH" and side == "LONG")
        or
        (htf_bias == "BULLISH" and side == "SHORT")
    )

    # POIN 1: kalau bertentangan dengan daily bias, WAJIB
    # TF agreement 3/3 (bukan 2/3). Soft-penalty lama diganti
    # menjadi gate tegas + penalty jauh lebih berat (0.5).
    min_agreement_required = 3 if htf_conflict else 2

    if agreement < min_agreement_required:
        REJECTIONS.note(
            "stage2", symbol, "insufficient_tf_agreement",
            side=side, agreement=agreement,
            required=min_agreement_required,
            htf_bias=htf_bias,
            htf_conflict=htf_conflict,
        )
        return None

    # --------------------------------------------------------
    # BTC market regime -- POIN 4
    #
    # Kalau BTC bearish, LONG mendapat penalty ekstra
    # (atau hard reject kalau btc_regime_hard_gate = True).
    # SHORT tidak terpengaruh.
    # --------------------------------------------------------

    btc_penalty = 1.0

    if (
        btc_regime == "BEARISH"
        and side == "LONG"
    ):

        if CONFIG["btc_regime_hard_gate"]:
            REJECTIONS.note(
                "stage2", symbol, "btc_regime_bearish_long",
                side=side, btc_regime=btc_regime,
            )
            return None

        btc_penalty = CONFIG["btc_regime_penalty_factor"]

    # --------------------------------------------------------
    # 15m momentum direction
    # --------------------------------------------------------

    df15 = data["15m"]["df"]

    # Repaint guard: harus ikut confirm_on_closed_bar seperti
    # semua deteksi sinyal lain (lihat daily_bias / score_tf).
    # Sebelumnya bagian ini selalu memakai df15.iloc[-1] (candle
    # 15m yang masih berjalan) walau confirm_on_closed_bar aktif
    # -- jadi move_15 bisa naik-turun sendiri dan sinyal
    # weak_momentum_15m bisa berubah beberapa kali sebelum candle
    # 15m selesai (repaint bocor lewat filter ini).
    signal_offset_15m = (
        1 if CONFIG["confirm_on_closed_bar"] else 0
    )

    signal_pos_15m = len(df15) - 1 - signal_offset_15m

    fast_n = CONFIG[
        "momentum_fast_bars"
    ]

    if signal_pos_15m - fast_n < 0:
        REJECTIONS.note(
            "stage2", symbol, "insufficient_15m_bars",
            signal_pos_15m=signal_pos_15m, fast_n=fast_n,
        )
        return None

    current_close = float(
        df15.iloc[signal_pos_15m]["close"]
    )

    reference_close = float(
        df15.iloc[
            signal_pos_15m - fast_n
        ]["close"]
    )

    if reference_close <= 0:
        REJECTIONS.note(
            "stage2", symbol, "invalid_reference_close",
            reference_close=reference_close,
        )
        return None

    move_15 = (
        current_close
        / reference_close
        - 1.0
    ) * 100

    # Dulu cuma cek tanda (>0 / <0) tanpa ambang minimum, jadi
    # pergerakan sekecil apapun (bisa cuma noise) lolos sebagai
    # "konfirmasi arah". Sekarang wajib minimal
    # min_momentum_15m_pct searah sisi trade.
    min_momentum = CONFIG[
        "min_momentum_15m_pct"
    ]

    if (
        side == "LONG"
        and move_15 < min_momentum
    ):
        REJECTIONS.note(
            "stage2", symbol, "weak_momentum_15m",
            side=side, move_15=round(move_15, 4),
            min_momentum=min_momentum,
        )
        return None

    if (
        side == "SHORT"
        and move_15 > -min_momentum
    ):
        REJECTIONS.note(
            "stage2", symbol, "weak_momentum_15m",
            side=side, move_15=round(move_15, 4),
            min_momentum=min_momentum,
        )
        return None

    # --------------------------------------------------------
    # Select execution timeframe
    #
    # Strongest directional timeframe.
    # Tie -> prefer 4H, then 1H, then 15m.
    #
    # FIX: rank sebelumnya (1h=3, 4h=2, 15m=1) BERLAWANAN
    # dengan komentar di atas -- pada skor seri, 1h akan
    # menang duluan padahal seharusnya 4h. Rank sekarang
    # disamakan dengan urutan yang dimaksud: 4h > 1h > 15m.
    # --------------------------------------------------------

    tf_rank = {
        "4h": 3,
        "1h": 2,
        "15m": 1,
    }

    tf_candidates = []

    for tf in TFS:

        tf_score = (
            data[tf]["score"]["long"]
            if side == "LONG"
            else data[tf]["score"]["short"]
        )

        tf_candidates.append(
            (
                float(tf_score),
                tf_rank[tf],
                tf,
            )
        )

    _, _, execution_tf = max(
        tf_candidates
    )

    exec_df = data[
        execution_tf
    ]["df"]

    price = float(
        exec_df.iloc[-1]["close"]
    )

    # --------------------------------------------------------
    # ATR untuk entry/SL/TP.
    #
    # FIX: sebelumnya diambil dari baris TERAKHIR (candle yang
    # masih berjalan / live), padahal classify_setup() di bawah
    # memakai ATR dari CLOSED bar (tf_score["atr"], via
    # score_tf() yang menghormati repaint guard). Kalau candle
    # yang sedang berjalan punya wick besar, dua ATR ini bisa
    # beda -> SL/TP jadi tidak konsisten dengan setup yang
    # baru saja diklasifikasikan. Sekarang exec_score["atr"]
    # (closed-bar, sudah dihitung score_tf) dipakai untuk risk,
    # sama seperti yang dipakai Setup Engine.
    # --------------------------------------------------------

    exec_score = data[execution_tf]["score"]

    atr_value = float(exec_score["atr"])

    if (
        not np.isfinite(price)
        or price <= 0
        or not np.isfinite(atr_value)
        or atr_value <= 0
    ):
        REJECTIONS.note(
            "stage2", symbol, "invalid_price_or_atr_exec",
            execution_tf=execution_tf,
            price=price, atr=atr_value,
        )
        return None

    # --------------------------------------------------------
    # SETUP ENGINE
    #
    # Klasifikasi kondisi price action pada execution_tf
    # menjadi salah satu dari:
    # BREAKOUT / PULLBACK / CONTINUATION / EXTENDED / NO_SETUP.
    #
    # Tidak mengarang sinyal: NO_SETUP berarti tidak ada
    # struktur entry yang jelas, kandidat ditolak di sini.
    # --------------------------------------------------------

    setup_info = classify_setup(
        exec_score,
        side,
        live_price=price,
    )

    if setup_info["setup_style"] == "NO_SETUP":
        REJECTIONS.note(
            "stage2", symbol, "no_setup",
            side=side, execution_tf=execution_tf,
        )
        return None

    setup_style = setup_info["setup_style"]
    entry_state = setup_info["entry_state"]
    reference_level = setup_info["reference_level"]
    retest_zone_low = setup_info.get("retest_zone_low")
    retest_zone_high = setup_info.get("retest_zone_high")
    setup_invalidation_level = setup_info.get(
        "setup_invalidation_level"
    )

    # --------------------------------------------------------
    # ENTRY LOGIC
    #
    # Entry price ditentukan berdasarkan tipe setup dari
    # Setup Engine di atas.
    # --------------------------------------------------------

    entry = build_entry(
        setup_info,
        side,
        price,
        exec_df,
        atr_value,
    )

    # --------------------------------------------------------
    # Entry distance sanity guard.
    #
    # FIX: sebelumnya tidak ada batas atas untuk seberapa jauh
    # ideal_entry boleh berada dari harga pasar SEKARANG. Kalau
    # level breakout/breakdown atau zona pullback sudah lama
    # ditinggalkan harga (candle impulsif jauh + tren berlanjut,
    # mis. kasus TRIA/RIVER), kandidat tetap lolos terus sebagai
    # WAITING_RETEST/WAITING_PULLBACK walau entry-nya sudah tidak
    # realistis tersentuh tanpa reversal besar. Sekarang kandidat
    # semacam ini dibuang sama sekali di sini, bukan cuma
    # ditandai "waiting".
    # --------------------------------------------------------

    entry_distance_atr = abs(price - entry) / atr_value

    max_entry_distance_atr = CONFIG["max_entry_distance_atr"]

    if entry_distance_atr > max_entry_distance_atr:
        REJECTIONS.note(
            "stage2", symbol, "entry_too_far",
            side=side, execution_tf=execution_tf,
            setup_style=setup_style,
            distance_atr=round(entry_distance_atr, 2),
            max_distance_atr=max_entry_distance_atr,
        )
        return None

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    swing_n = CONFIG[
        "swing_window"
    ]

    if len(exec_df) < swing_n:
        REJECTIONS.note(
            "stage2", symbol, "insufficient_swing_data",
            execution_tf=execution_tf,
            have=len(exec_df), need=swing_n,
        )
        return None

    swing_low = float(
        exec_df["low"]
        .iloc[-swing_n:]
        .min()
    )

    swing_high = float(
        exec_df["high"]
        .iloc[-swing_n:]
        .max()
    )

    if side == "LONG":

        sl = min(
            swing_low,
            entry - 1.25 * atr_value,
        )

        risk = entry - sl

        invalidation = (
            f"Close below {sl:.8g} / "
            f"loss of recent "
            f"{execution_tf} swing low"
        )

    else:

        sl = max(
            swing_high,
            entry + 1.25 * atr_value,
        )

        risk = sl - entry

        invalidation = (
            f"Close above {sl:.8g} / "
            f"reclaim of recent "
            f"{execution_tf} swing high"
        )

    if (
        setup_invalidation_level is not None
        and np.isfinite(setup_invalidation_level)
    ):
        if side == "LONG":
            invalidation += (
                f" / breakout invalid if price falls below "
                f"{setup_invalidation_level:.8g}"
            )
        else:
            invalidation += (
                f" / breakdown invalid if price rises above "
                f"{setup_invalidation_level:.8g}"
            )

    if risk <= 0:
        REJECTIONS.note(
            "stage2", symbol, "non_positive_risk",
            side=side, entry=entry, sl=sl,
        )
        return None

    risk_pct = (
        risk / entry
    ) * 100

    if risk_pct > CONFIG["max_risk_pct"]:
        REJECTIONS.note(
            "stage2", symbol, "risk_too_high",
            side=side, risk_pct=round(risk_pct, 3),
            max_risk_pct=CONFIG["max_risk_pct"],
        )
        return None

    # --------------------------------------------------------
    # TP
    # --------------------------------------------------------

    tp_rr = list(CONFIG["risk_reward"])

    tp = [
        (
            entry + risk * rr
            if side == "LONG"
            else
            entry - risk * rr
        )
        for rr in tp_rr
    ]

    # TP1 remains the initial risk objective. Once TP1 is reached,
    # management moves SL to entry/breakeven to protect the trade.
    breakeven_after_tp1 = bool(
        CONFIG.get("move_sl_to_breakeven_after_tp1", False)
    )

    # --------------------------------------------------------
    # Price-already-beyond-TP1 guard.
    #
    # FIX: sebelumnya tidak ada pengecekan apakah harga PASAR
    # SAAT INI sudah melewati TP1 sebelum entry sempat tersentuh.
    # Untuk setup BREAKOUT/BREAKDOWN/EXTENDED yang entry-nya
    # dijepit jauh ke belakang (level lama / zona pullback),
    # harga bisa saja sudah menuju atau bahkan melewati TP1
    # sebelum retracement ke entry terjadi (kasus TRIA: harga
    # sudah dekat TP1 padahal entry masih jauh di atas). Kalau
    # itu terjadi, risk/reward yang dihitung sudah tidak relevan
    # dengan kondisi market saat ini -- kandidat dibuang.
    # --------------------------------------------------------

    if side == "LONG" and price >= tp[0]:
        REJECTIONS.note(
            "stage2", symbol, "price_beyond_tp1",
            side=side, price=price, tp1=tp[0],
        )
        return None

    if side == "SHORT" and price <= tp[0]:
        REJECTIONS.note(
            "stage2", symbol, "price_beyond_tp1",
            side=side, price=price, tp1=tp[0],
        )
        return None

    # --------------------------------------------------------
    # Rounding harga -- supaya entry/sl/tp konsisten dengan
    # chart (vSch.py) dan tidak menampilkan noise float
    # (mis. 1.053276482234) di JSON/summary output.
    #
    # Jumlah desimal ditentukan dari magnitude harga entry,
    # sama persis dengan decimals_from_price() di vSch.py, dan
    # disimpan sebagai "decimals" supaya semua konsumen data
    # (chart, notifikasi, summary) pakai angka yang sama.
    #
    # retest_zone_low/high dibulatkan dengan presisi yang SAMA
    # supaya kalau ditampilkan berdampingan dengan entry/sl/tp
    # di chart atau notifikasi, tidak ada mismatch desimal.
    # --------------------------------------------------------

    price_decimals = decimals_from_price(entry)

    entry = round(entry, price_decimals)
    sl = round(sl, price_decimals)
    tp = [round(t, price_decimals) for t in tp]

    if retest_zone_low is not None and np.isfinite(retest_zone_low):
        retest_zone_low = round(retest_zone_low, price_decimals)
    else:
        retest_zone_low = None

    if retest_zone_high is not None and np.isfinite(retest_zone_high):
        retest_zone_high = round(retest_zone_high, price_decimals)
    else:
        retest_zone_high = None

    # --------------------------------------------------------
    # Funding rate -- crowded trade guard.
    #
    # Funding SEARAH sisi trade di atas ambang berarti mayoritas
    # posisi terbuka sudah searah kita (crowded) -> risiko
    # funding-squeeze / mean-reversion mendadak naik. Bukan hard
    # veto (data teknikal tetap valid), tapi skor akhirnya
    # didiskon dan ditandai transparan di output.
    # --------------------------------------------------------

    funding_pct = float(funding_rate) * 100

    funding_alert = False

    funding_penalty = 1.0

    if (
        side == "LONG"
        and funding_pct >= CONFIG["funding_rate_alert_pct"]
    ) or (
        side == "SHORT"
        and funding_pct <= -CONFIG["funding_rate_alert_pct"]
    ):

        funding_alert = True

        funding_penalty = CONFIG[
            "funding_rate_penalty_factor"
        ]

    # --------------------------------------------------------
    # Higher-timeframe (1D) bias -- macro trend guard.
    #
    # POIN 1: penalty diperberat jauh (0.5, bukan 0.85).
    # Gate agreement 3/3 sudah diterapkan di atas.
    # --------------------------------------------------------

    htf_penalty = 1.0

    if htf_conflict:

        htf_penalty = CONFIG[
            "htf_bias_penalty_factor"
        ]

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    momentum_bonus = min(
        stage1_score / 25.0,
        1.5,
    )

    score = (
        raw_score
        + momentum_bonus
    ) * funding_penalty * htf_penalty * btc_penalty

    reasons = (
        data["15m"]["score"][
            "long_reasons"
        ]
        if side == "LONG"
        else data["15m"]["score"][
            "short_reasons"
        ]
    )

    # --------------------------------------------------------
    # Chart data
    # --------------------------------------------------------

    chart_data = {
        tf: serialize_chart_data(
            data[tf]["df"]
        )
        for tf in TFS
    }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {

        "symbol": symbol,

        "side": side,

        "setup_style": setup_style,

        "entry_state": entry_state,

        "reference_level": reference_level,

        "retest_zone_low": retest_zone_low,

        "retest_zone_high": retest_zone_high,

        "decimals": price_decimals,

        "score": round(
            score,
            2,
        ),

        "change24h": round(
            change_24h,
            2,
        ),

        "quote_volume24h": round(
            quote_volume_24h,
            2,
        ),

        "funding_rate_pct": round(
            funding_pct,
            4,
        ),

        "funding_alert": funding_alert,

        "htf_bias": htf_bias,

        "btc_regime": btc_regime,

        "execution_tf": execution_tf,

        "timeframes": {
            tf: data[tf]["score"]
            for tf in TFS
        },

        "momentum_15m": round(
            move_15,
            3,
        ),

        "entry": entry,

        "tp": tp,
        "tp_rr": tp_rr,

        "sl": sl,
        "sl_initial": sl,
        "sl_after_tp1": entry if breakeven_after_tp1 else sl,
        "breakeven_after_tp1": breakeven_after_tp1,

        "risk_pct": round(
            risk_pct,
            3,
        ),

        "invalidation": invalidation,

        "setup_invalidation_level": setup_invalidation_level,

        "key_points": reasons[:6],

        "tf_agreement": agreement,
        "stage1_score": round(stage1_score, 3),

        "chart": {
            "execution_tf": execution_tf,

            "available_timeframes": TFS,

            "analysis_candles": CONFIG[
                "klines"
            ],

            # 1H kembali 48 candle.
            "visible_candles": {
                "15m": 60,
                "1h": 48,
                "4h": 50,
            },

            "show_ema200": True,

            "show_supertrend": True,

            "show_volume": True,

            "show_entry": True,

            "show_sl": True,

            "show_tp": True,
        },

        "chart_levels": {
            "entry": entry,
            "sl": sl,
            "tp": tp,
        },

        "chart_data": chart_data,
    }


# ============================================================
# STAGE 1 WORKER
# ============================================================

def stage1_worker(row, btc_regime=None):

    symbol, change_24h, quote_volume, funding_rate = row

    try:

        candles = klines(
            symbol,
            "15m",
        )

        if len(candles) < 51:
            REJECTIONS.note(
                "stage1", symbol, "insufficient_klines_raw",
                have=len(candles), need=51,
            )
            return None

        enriched = add_indicators(
            candles
        )

        score, meta = movement_score(
            enriched,
            symbol=symbol,
        )

        # Regime is enforced BEFORE momentum_pool so the pool itself
        # is less likely to be dominated by counter-regime LONGs.
        if meta is not None and btc_regime in ("BULLISH", "BEARISH"):
            preferred_side = meta.get("preferred_side")
            if (
                (btc_regime == "BEARISH" and preferred_side == "LONG")
                or
                (btc_regime == "BULLISH" and preferred_side == "SHORT")
            ):
                score *= CONFIG["btc_regime_penalty_factor"]
                meta["btc_regime_penalty"] = CONFIG[
                    "btc_regime_penalty_factor"
                ]
            else:
                meta["btc_regime_penalty"] = 1.0

            meta["btc_regime"] = btc_regime

        if (
            score <= 0
            or meta is None
        ):
            # Reason spesifik sudah dicatat di dalam
            # movement_score(); ini fallback kalau ada jalur
            # yang lolos tanpa note (mis. score memang <=0).
            REJECTIONS.note(
                "stage1", symbol, "non_positive_score",
                score=score,
            )
            return None

        # Pastikan metadata menggunakan
        # dataframe yang sudah dihitung.
        meta["df"] = enriched

        return (
            score,
            symbol,
            change_24h,
            quote_volume,
            funding_rate,
            meta,
        )

    except Exception as exc:

        REJECTIONS.note(
            "stage1", symbol,
            f"exception:{type(exc).__name__}",
            error=str(exc),
        )

        return None


# ============================================================
# STAGE 2 WORKER
# ============================================================

def stage2_worker(item, btc_regime=None):

    (
        stage1_score,
        symbol,
        change_24h,
        quote_volume,
        funding_rate,
        stage1_meta,
    ) = item

    try:

        return analyze_symbol(
            symbol,
            change_24h,
            quote_volume,
            stage1_score,
            stage1_meta,
            funding_rate=funding_rate,
            btc_regime=btc_regime,
        )

    except Exception as exc:

        REJECTIONS.note(
            "stage2", symbol,
            f"exception:{type(exc).__name__}",
            error=str(exc),
        )

        return None


# ============================================================
# RANKING HELPER
#
# ENTRY_READY diprioritaskan di atas WAITING_RETEST/
# WAITING_PULLBACK -- kandidat yang paling siap dieksekusi
# SEKARANG naik ke atas daftar, bukan cuma diurutkan dari
# score mentah (yang bisa saja tinggi justru karena setup-nya
# lagi extended/chase).
# ============================================================

_ENTRY_STATE_RANK = {
    "ENTRY_READY": 0,
    "WAITING_RETEST": 1,
    "WAITING_PULLBACK": 2,
}

# Feedback menunjukkan PULLBACK dan BREAKDOWN memiliki kombinasi
# hasil yang paling sehat. Keduanya diprioritaskan saat memilih
# kandidat final, lalu entry_state dan score menjadi tie-breaker.
_SETUP_STYLE_RANK = {
    "PULLBACK": 0,
    "BREAKDOWN": 0,
    "BREAKOUT": 1,
    "EXTENDED": 2,
    "CONTINUATION": 3,
}


def _entry_state_rank(entry_state):

    return _ENTRY_STATE_RANK.get(
        entry_state,
        99,
    )


def _setup_style_rank(setup_style):

    return _SETUP_STYLE_RANK.get(
        setup_style,
        99,
    )


# ============================================================
# DIVERSITY / CORRELATION FILTER
#
# Dihitung dari chart_data yang SUDAH ada di tiap kandidat
# (hasil analyze_symbol) -- tidak ada API call tambahan.
# ============================================================

def _closes_from_chart(candidate, tf, window):

    records = (
        candidate.get("chart_data", {})
        .get(tf, [])
    )

    closes = [
        record["close"]
        for record in records[-(window + 1):]
        if record.get("close") is not None
    ]

    return closes


def _pct_returns(closes):

    if len(closes) < 5:
        return None

    arr = np.asarray(
        closes,
        dtype=float,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        returns = (
            np.diff(arr) / arr[:-1]
        )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) < 5:
        return None

    return returns


def _return_correlation(candidate_a, candidate_b, tf, window):

    returns_a = _pct_returns(
        _closes_from_chart(candidate_a, tf, window)
    )

    returns_b = _pct_returns(
        _closes_from_chart(candidate_b, tf, window)
    )

    if returns_a is None or returns_b is None:
        return None

    n = min(
        len(returns_a),
        len(returns_b),
    )

    if n < 5:
        return None

    a = returns_a[-n:]
    b = returns_b[-n:]

    if np.std(a) == 0 or np.std(b) == 0:
        return None

    corr = float(
        np.corrcoef(a, b)[0, 1]
    )

    if not np.isfinite(corr):
        return None

    return corr


def select_diverse_candidates(candidates, max_results):

    # Greedy: jalan dari kandidat terkuat ke terlemah (list
    # input sudah terurut). Kandidat dilewati (bukan dibuang
    # permanen dari pertimbangan) kalau korelasinya terlalu
    # tinggi dengan kandidat SEARAH yang sudah lolos -- jadi
    # slot yang kosong tetap bisa diisi kandidat berikutnya
    # yang lebih "berbeda" secara pergerakan harga.

    tf = CONFIG["diversity_correlation_tf"]
    window = CONFIG["diversity_lookback"]
    threshold = CONFIG["diversity_max_correlation"]

    selected = []

    for candidate in candidates:

        if len(selected) >= max_results:
            break

        too_correlated = False

        for chosen in selected:

            if chosen["side"] != candidate["side"]:
                continue

            corr = _return_correlation(
                candidate,
                chosen,
                tf,
                window,
            )

            if corr is not None and corr >= threshold:

                too_correlated = True

                logger.debug(
                    "%s dropped: correlated %.2f "
                    "with %s (already selected)",
                    candidate["symbol"],
                    corr,
                    chosen["symbol"],
                )

                break

        if too_correlated:
            continue

        selected.append(candidate)

    return selected


def append_trade_log(candidates, generated_at=None):
    """Append final candidates to a compact JSONL feedback log."""
    if not candidates:
        return

    path = Path(CONFIG["trade_log_file"])
    timestamp = (
        generated_at
        or pd.Timestamp.now(tz="UTC").isoformat()
    )

    fields = (
        "symbol",
        "side",
        "setup_style",
        "entry_state",
        "score",
        "stage1_score",
        "btc_regime",
        "htf_bias",
        "funding_rate_pct",
        "funding_alert",
        "tf_agreement",
        "momentum_15m",
        "execution_tf",
        "entry",
        "sl_initial",
        "sl_after_tp1",
        "tp",
        "tp_rr",
        "risk_pct",
        "invalidation",
        "setup_invalidation_level",
    )

    try:
        with path.open("a", encoding="utf-8") as handle:
            for candidate in candidates:
                record = {"logged_at": timestamp}
                record.update({
                    key: candidate.get(key)
                    for key in fields
                })
                record["outcome"] = None
                record["realized_r"] = None
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
    except Exception as exc:
        logger.warning(
            "Failed to append trade log: %s",
            exc,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Synaptic multi-timeframe "
            "Binance Futures scanner"
        )
    )

    parser.add_argument(
        "--out",
        default="synaptic_candidates.json",
    )

    args = parser.parse_args()

    started = time.time()

    # Reset rejection trail supaya funnel di run ini bersih
    # (penting kalau main() pernah dipanggil lebih dari sekali
    # dalam proses yang sama, mis. dari test/REPL).
    REJECTIONS.reset()

    # ========================================================
    # COOLDOWN -- POIN 5
    #
    # Load state, decay remaining, filter symbol yang masih
    # cooldown dari universe.
    # ========================================================

    cooldown_state = load_cooldown()
    cooldown_state = apply_cooldown_decay(cooldown_state)

    on_cooldown = {
        sym
        for sym, info in cooldown_state.items()
        if int(info.get("remaining", 0)) > 0
    }

    if on_cooldown:
        logger.info(
            "Cooldown active for %d symbols: %s",
            len(on_cooldown),
            ", ".join(sorted(on_cooldown)[:15])
            + ("..." if len(on_cooldown) > 15 else ""),
        )

    # ========================================================
    # BTC MARKET REGIME -- POIN 4
    # ========================================================

    btc_regime = None

    try:

        btc_regime = btc_market_regime()

        logger.info(
            "BTC market regime: %s",
            btc_regime or "UNKNOWN",
        )

    except Exception as exc:

        logger.warning(
            "BTC regime check failed, continuing without it: %s",
            exc,
        )

    # ========================================================
    # UNIVERSE
    # ========================================================

    try:

        universe_rows = universe()

    except Exception as exc:

        logger.error(
            "Cannot build universe: %s",
            exc,
        )

        Path(args.out).write_text(
            json.dumps(
                {
                    "candidates": [],
                    "error": str(exc),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        raise

    if not universe_rows:

        raise RuntimeError(
            "Universe is empty."
        )

    # Filter symbol yang masih cooldown (POIN 5)
    if on_cooldown:

        before = len(universe_rows)

        universe_rows = [
            row
            for row in universe_rows
            if row[0] not in on_cooldown
        ]

        skipped = before - len(universe_rows)

        if skipped:
            logger.info(
                "Skipped %d symbols still on cooldown.",
                skipped,
            )

            for sym in sorted(on_cooldown):
                REJECTIONS.note(
                    "universe", sym, "on_cooldown",
                    remaining=cooldown_state.get(sym, {}).get("remaining"),
                )

    # ========================================================
    # STAGE 1
    # ========================================================

    stage1_started = time.time()

    logger.info(
        "Scanning %d symbols on 15m "
        "(Stage 1, workers=%d)...",
        len(universe_rows),
        CONFIG["workers_stage1"],
    )

    momentum = []

    with ThreadPoolExecutor(
        max_workers=CONFIG[
            "workers_stage1"
        ]
    ) as pool:

        futures = [
            pool.submit(
                stage1_worker,
                row,
                btc_regime,
            )
            for row in universe_rows
        ]

        for future in as_completed(
            futures
        ):

            try:

                result = future.result()

                if result is not None:
                    momentum.append(result)

            except Exception as exc:

                logger.debug(
                    "Stage 1 future error: %s",
                    exc,
                )

    momentum.sort(
        key=lambda row: row[0],
        reverse=True,
    )

    selected = momentum[
        :CONFIG["momentum_pool"]
    ]

    stage1_elapsed = (
        time.time()
        - stage1_started
    )

    logger.info(
        "Stage 1 completed in %.2fs | "
        "momentum=%d | selected=%d",
        stage1_elapsed,
        len(momentum),
        len(selected),
    )

    if not selected:

        logger.warning(
            "No momentum candidates "
            "found in Stage 1."
        )

        _log_rejection_funnel()

        # Save cooldown state even on empty run
        save_cooldown(cooldown_state)

        payload = {
            "generated_at":
                pd.Timestamp.now(
                    tz="UTC"
                ).isoformat(),

            "scanner": "Synaptic",

            "selection_mode":
                "no_stage1_candidates",

            "btc_regime": btc_regime,

            "scan_stats": {
                "universe":
                    len(universe_rows),

                "stage1_selected":
                    0,

                "mtf_valid":
                    0,

                "min_score_valid":
                    0,

                "final_candidates":
                    0,

                "stage1_seconds":
                    round(
                        stage1_elapsed,
                        2,
                    ),

                "elapsed_seconds":
                    round(
                        time.time()
                        - started,
                        2,
                    ),
                "trade_log_file":
                    CONFIG["trade_log_file"],
            },

            "rejection_stats": REJECTIONS.summary(),

            "candidates": [],
        }

        Path(args.out).write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return

    # ========================================================
    # STAGE 2
    # ========================================================

    stage2_started = time.time()

    logger.info(
        "Selected %d symbols for "
        "Stage 2 MTF validation "
        "(workers=%d)...",
        len(selected),
        CONFIG["workers_stage2"],
    )

    results = []
    mtf_valid = []

    with ThreadPoolExecutor(
        max_workers=CONFIG[
            "workers_stage2"
        ]
    ) as pool:

        futures = [
            pool.submit(
                stage2_worker,
                item,
                btc_regime,
            )
            for item in selected
        ]

        for future in as_completed(
            futures
        ):

            try:

                result = future.result()

                if result is None:
                    continue

                mtf_valid.append(
                    result
                )

                if (
                    result["score"]
                    >= CONFIG["min_score"]
                ):

                    results.append(
                        result
                    )

            except Exception as exc:

                logger.debug(
                    "Stage 2 future error: %s",
                    exc,
                )

    stage2_elapsed = (
        time.time()
        - stage2_started
    )

    results.sort(
        key=lambda item: (
            _setup_style_rank(
                item["setup_style"]
            ),
            _entry_state_rank(
                item["entry_state"]
            ),
            -item["score"],
        ),
    )

    mtf_valid.sort(
        key=lambda item: (
            _setup_style_rank(
                item["setup_style"]
            ),
            _entry_state_rank(
                item["entry_state"]
            ),
            -item["score"],
        ),
    )

    # ========================================================
    # FINAL SELECTION -- POIN 3
    #
    # min_candidates = 0. JANGAN backfill dari mtf_valid
    # demi angka "Top 5". Kalau tidak ada setup yang mencapai
    # min_score, biarkan final_candidates = 0.
    # ========================================================

    final_results = select_diverse_candidates(
        results,
        CONFIG["max_results"],
    )

    selection_mode = "min_score"

    # POIN 3: hapus fallback ke mtf_valid.
    # Sebelumnya kalau len(final_results) < min_candidates,
    # diisi dari mtf_valid. Sekarang min_candidates = 0 dan
    # fallback dimatikan -- hanya kandidat yang benar-benar
    # lolos min_score yang masuk.

    for item in final_results:
        item["from_fallback"] = False

    # ========================================================
    # REGISTER COOLDOWN -- POIN 5
    #
    # Symbol yang muncul di final candidates didaftarkan
    # supaya di-skip N scan berikutnya.
    # ========================================================

    if final_results:

        symbols_to_cool = [
            item["symbol"] for item in final_results
        ]

        cooldown_state = register_cooldown(
            cooldown_state,
            symbols_to_cool,
        )

        logger.info(
            "Registered cooldown (%d scans) for: %s",
            CONFIG["cooldown_scans"],
            ", ".join(symbols_to_cool),
        )

    save_cooldown(cooldown_state)

    # Persistent feedback loop: log every final trade with the
    # variables needed to measure which score components predict outcome.
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    append_trade_log(
        final_results,
        generated_at=generated_at,
    )

    # ========================================================
    # FINAL TIMING
    # ========================================================

    total_elapsed = (
        time.time()
        - started
    )

    logger.info(
        "Stage 2 completed in %.2fs",
        stage2_elapsed,
    )

    logger.info(
        "Stage 2 MTF-valid: %d | "
        "min-score valid: %d | "
        "selection=%s",
        len(mtf_valid),
        len(results),
        selection_mode,
    )

    logger.info(
        "Scan completed in %.2fs | "
        "Found %d valid candidates.",
        total_elapsed,
        len(final_results),
    )

    _log_rejection_funnel()

    # ========================================================
    # PAYLOAD
    # ========================================================

    payload = {

        "generated_at":
            generated_at,

        "scanner":
            "Synaptic",

        "selection_mode":
            selection_mode,

        "btc_regime":
            btc_regime,

        "scan_stats": {

            "universe":
                len(universe_rows),

            "stage1_momentum":
                len(momentum),

            "stage1_selected":
                len(selected),

            "mtf_valid":
                len(mtf_valid),

            "min_score_valid":
                len(results),

            "final_candidates":
                len(final_results),

            "stage1_seconds":
                round(
                    stage1_elapsed,
                    2,
                ),

            "stage2_seconds":
                round(
                    stage2_elapsed,
                    2,
                ),

            "elapsed_seconds":
                round(
                    total_elapsed,
                    2,
                ),

            "workers_stage1":
                CONFIG["workers_stage1"],

            "workers_stage2":
                CONFIG["workers_stage2"],

            "cooldown_active":
                len(on_cooldown),
            "trade_log_file":
                CONFIG["trade_log_file"],
        },

        # Funnel lengkap: berapa banyak symbol gugur di tiap
        # stage, per alasan. Contoh konkret tiap alasan (symbol +
        # angka pendukung, dibatasi beberapa sample) ada di
        # REJECTIONS.samples() -- sengaja TIDAK dimasukkan ke
        # payload utama supaya file output tetap ringkas; kalau
        # perlu debug mendalam, panggil REJECTIONS.samples()
        # secara terpisah atau jalankan dengan level DEBUG.
        "rejection_stats": REJECTIONS.summary(),

        "candidates":
            final_results,
    }

    # ========================================================
    # SAVE
    # ========================================================

    output_path = Path(
        args.out
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print(
        "SYNAPTIC SCAN RESULTS"
    )
    print("=" * 72)

    if not final_results:

        print(
            "No valid candidates found."
        )

    else:

        for item in final_results:

            print(
                f"{item['symbol']} "
                f"{item['side']} | "
                f"Score "
                f"{item['score']:.2f} | "
                f"TF "
                f"{item['tf_agreement']}/3 | "
                f"EXEC "
                f"{item['execution_tf']} | "
                f"Entry "
                f"{item['entry']:.8g} | "
                f"SL "
                f"{item['sl']:.8g}"
            )

    print("=" * 72)

    print(
        f"Universe       : "
        f"{len(universe_rows)}"
    )

    print(
        f"BTC regime     : "
        f"{btc_regime or 'UNKNOWN'}"
    )

    print(
        f"Stage 1        : "
        f"{stage1_elapsed:.2f}s"
    )

    print(
        f"Stage 2        : "
        f"{stage2_elapsed:.2f}s"
    )

    print(
        f"Total           : "
        f"{total_elapsed:.2f}s"
    )

    print(
        f"Candidates      : "
        f"{len(final_results)}"
    )

    print(
        f"Selection       : "
        f"{selection_mode}"
    )

    print(
        f"Cooldown active : "
        f"{len(on_cooldown)}"
    )

    print(
        f"Output          : "
        f"{output_path}"
    )

    print("=" * 72)

    logger.info(
        "Output successfully saved to: %s",
        output_path,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
