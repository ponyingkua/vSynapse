# vSynapse

vSynapse adalah scanner kandidat Binance Futures berbasis price action + konfirmasi indikator. Sistem ini dirancang untuk menyaring koin yang sedang aktif, menentukan bias LONG/SHORT, menghitung level Entry/TP/SL/Invalidation, lalu membuat visual chart melalui `vSch.py`.

> Untuk riset/edukasi. Bukan financial advice. Hasil scanner bukan jaminan profit.

## Struktur

```text
.
├── Synaptic.py
├── vSch.py
├── vSynapse.yml
├── synaptic_candidates.json
└── charts/
```

### `Synaptic.py`

Mesin utama.

Tugasnya:

1. Mengambil data Binance Futures USDT perpetual.
2. Menyaring kandidat berdasarkan aktivitas dan likuiditas 24H.
3. Menganalisis timeframe:
   - 15m
   - 1h
   - 4h
4. Menggunakan kombinasi indikator:
   - EMA200
   - Volume
   - MACD
   - Supertrend 10 / 2.50
5. Memberikan scoring LONG vs SHORT.
6. Memerlukan keselarasan arah minimal 2 timeframe.
7. Memilih timeframe eksekusi terbaik.
8. Menghasilkan:
   - Direction
   - Score
   - Entry
   - TP1
   - TP2
   - TP3
   - SL
   - Invalidation
   - Key points
9. Menyimpan hasil ke `synaptic_candidates.json`.

### `vSch.py`

Lapisan visualisasi.

Tugasnya:

1. Membaca hasil `Synaptic.py`.
2. Memilih kandidat.
3. Mengambil data candle timeframe yang dipilih.
4. Membuat chart candlestick custom bergaya terminal Binance.
5. Menampilkan:
   - Candle
   - Volume
   - EMA200
   - Entry
   - TP1/TP2/TP3
   - SL
   - Invalidation
   - Key points
6. Menyimpan PNG ke folder kerja.

### `vSynapse.yml`

GitHub Actions workflow.

Workflow dijalankan:

- manual melalui `workflow_dispatch`
- otomatis setiap 15 menit melalui cron

Alurnya:

```text
GitHub Actions
      ↓
Install Python + dependencies
      ↓
Synaptic.py
      ↓
synaptic_candidates.json
      ↓
vSch.py
      ↓
charts/*.png
      ↓
Upload artifact
      ↓
Commit hasil scan
```

## Logika kandidat

vSynapse tidak mencoba memindai seluruh pasar secara sama rata.

Tahap pertama mencari koin yang:

- memiliki volume futures 24H memadai,
- mengalami perubahan harga yang cukup signifikan,
- memiliki aktivitas relatif tinggi.

Tujuannya adalah menemukan koin yang sedang "hidup" terlebih dahulu.

Kandidat kemudian dianalisis pada 15m, 1h, dan 4h.

## Kombinasi indikator

### EMA200

EMA200 digunakan sebagai filter regime.

Secara umum:

```text
Price > EMA200 → bias bullish
Price < EMA200 → bias bearish
```

EMA200 bukan sinyal entry tunggal.

### Volume

Volume digunakan untuk melihat apakah pergerakan harga memiliki dukungan aktivitas.

Default:

```text
Volume baseline = MA20
Confirmation = Volume >= 1.5x baseline
```

Volume bullish dan bearish dibedakan berdasarkan arah candle.

### MACD

Default:

```text
Fast   = 12
Slow   = 26
Signal = 9
```

MACD digunakan sebagai konfirmasi momentum, bukan sebagai trigger tunggal.

Bullish:

```text
MACD > Signal
dan histogram menguat
```

Bearish:

```text
MACD < Signal
dan histogram melemah
```

### Supertrend

Parameter:

```text
Period     = 10
Multiplier = 2.50
```

Supertrend digunakan sebagai konfirmasi arah/trend.

### RSI

RSI tersedia di engine tetapi default:

```yaml
enabled: false
```

Alasannya adalah menghindari terlalu banyak konfirmasi yang sebenarnya membawa informasi momentum yang mirip.

Jika pengujian historis menunjukkan RSI meningkatkan kualitas kandidat, RSI dapat diaktifkan kembali.

## Scoring

Scoring menggunakan bobot indikator dan price action.

Bobot dasar:

```text
EMA200       2.0
Supertrend   2.0
MACD         1.5
Volume       1.5
Breakout     2.0
Structure    1.0
```

Scanner menghitung skor LONG dan SHORT secara terpisah.

Contoh konseptual:

```text
LONG
EMA200 bullish       +2
Supertrend bullish   +2
MACD bullish         +1.5
Volume bullish       +1.5
Breakout             +2
Structure            +1
-------------------------
Total                10
```

SHORT menggunakan logika kebalikan.

Scanner tidak mengharuskan seluruh indikator memberikan sinyal yang sama. Tujuannya adalah mendapatkan **konfluensi**, bukan membuat filter yang terlalu ketat.

## Multi-timeframe

Bobot timeframe:

```text
15m = 25%
1h  = 35%
4h  = 40%
```

Interpretasi:

- **4H** memberikan konteks struktur terbesar.
- **1H** menjadi timeframe struktur utama.
- **15m** membantu membaca momentum/timing yang lebih dekat.

Minimal dua dari tiga timeframe harus mendukung arah yang sama.

Contoh:

```text
4H  → LONG
1H  → LONG
15m → SHORT
```

Masih dapat menghasilkan kandidat LONG karena mayoritas timeframe mendukung LONG.

Sebaliknya:

```text
4H  → LONG
1H  → SHORT
15m → SHORT
```

cenderung menghasilkan SHORT.

## Entry / TP / SL

Scanner saat ini menggunakan harga terakhir sebagai basis Entry.

Stop loss menggunakan kombinasi recent swing dan ATR.

Risk-reward default:

```text
TP1 = 1.50R
TP2 = 2.25R
TP3 = 3.00R
```

Untuk LONG:

```text
Risk = Entry - SL

TP1 = Entry + 1.50 × Risk
TP2 = Entry + 2.25 × Risk
TP3 = Entry + 3.00 × Risk
```

Untuk SHORT:

```text
Risk = SL - Entry

TP1 = Entry - 1.50 × Risk
TP2 = Entry - 2.25 × Risk
TP3 = Entry - 3.00 × Risk
```

## Invalidation

Invalidation bukan sekadar "SL kena".

Scanner juga memberikan kondisi struktural yang membatalkan bias.

Contoh LONG:

```text
Close below recent swing low
```

Contoh SHORT:

```text
Close above recent swing high
```

Artinya pengguna dapat membedakan antara level risiko numerik dan kondisi chart yang membuat tesis arah tidak lagi valid.

## Konfigurasi utama

Parameter utama berada di `Synaptic.py` dan workflow `vSynapse.yml`.

Contoh:

```yaml
scanner:
  min_quote_volume_24h: 5000000
  min_abs_change_24h: 4.0
  universe_size: 80
  max_results: 15
  min_score: 6.0
```

Timeframe:

```yaml
timeframes:
  primary:
    - 15m
    - 1h
    - 4h
```

Indikator:

```yaml
indicators:
  ema200:
    enabled: true

  volume:
    enabled: true

  macd:
    enabled: true

  supertrend:
    enabled: true
    period: 10
    multiplier: 2.50

  rsi:
    enabled: false
```

## Menjalankan lokal

Install dependency:

```bash
pip install requests pandas numpy matplotlib pyyaml
```

Jalankan scanner:

```bash
python Synaptic.py --out synaptic_candidates.json
```

Kemudian buat chart:

```bash
python vSch.py --input synaptic_candidates.json
```

Untuk kandidat tertentu:

```bash
python vSch.py --input synaptic_candidates.json --symbol BTCUSDT
```

## Menjalankan di GitHub

Repository minimal berisi:

```text
Synaptic.py
vSch.py
vSynapse.yml
```

Pastikan workflow berada pada:

```text
.github/workflows/vSynapse.yml
```

GitHub Actions kemudian dapat dijalankan manual atau mengikuti jadwal cron.

Workflow membutuhkan:

```yaml
permissions:
  contents: write
```

karena workflow melakukan commit hasil scan ke repository.

## Output JSON

Contoh struktur:

```json
{
  "symbol": "TOKENUSDT",
  "side": "LONG",
  "score": 8.4,
  "execution_tf": "1h",
  "entry": 1.2345,
  "tp": [
    1.3000,
    1.3500,
    1.4000
  ],
  "sl": 1.1900,
  "risk_pct": 3.61,
  "invalidation": "Close below recent swing low",
  "key_points": [
    "above EMA200",
    "Supertrend bullish",
    "MACD bullish",
    "volume 1.8x"
  ]
}
```

## Filosofi scanner

vSynapse dibuat dengan prinsip:

```text
ACTIVE COIN
    ↓
STRUCTURE
    ↓
MULTI-TIMEFRAME
    ↓
EMA200
    ↓
VOLUME
    ↓
MACD + SUPERTREND
    ↓
LONG / SHORT
    ↓
ENTRY / TP / SL
    ↓
VISUAL CHART
```

Scanner tidak dirancang untuk menghasilkan sebanyak mungkin sinyal.

Targetnya adalah menemukan **kandidat yang sedang aktif dan memiliki konfluensi arah yang cukup**, kemudian membuang kandidat yang struktur dan arah antar-timeframe terlalu bertentangan.

## Catatan pengembangan

Versi saat ini adalah fondasi scanner.

Tahap kalibrasi berikutnya yang disarankan:

1. Simpan seluruh kandidat, termasuk yang gagal threshold.
2. Simpan skor masing-masing timeframe.
3. Catat kondisi indikator ketika kandidat ditemukan.
4. Bandingkan kandidat dengan hasil postingan historis.
5. Kalibrasi threshold volume, perubahan 24H, scoring, dan timeframe.
6. Tambahkan filter chase/pullback yang lebih spesifik.
7. Backtest sebelum menggunakan hasil scanner sebagai dasar keputusan trading.

Jangan menganggap parameter saat ini sebagai parameter optimal. Parameter tersebut adalah starting point yang sengaja dibuat cukup fleksibel untuk dikalibrasi dengan data nyata.
