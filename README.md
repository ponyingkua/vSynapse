vSynapse
vSynapse adalah scanner kandidat Binance Futures berbasis price action + konfirmasi indikator. Sistem ini dirancang untuk menyaring koin yang sedang aktif, menentukan bias LONG/SHORT, menghitung level Entry/TP/SL/Invalidation, lalu membuat visual chart melalui vSch.py.
Untuk riset/edukasi. Bukan financial advice. Hasil scanner bukan jaminan profit.
Struktur
.
├── Synaptic.py
├── vSch.py
├── vSynapse.yml
├── synaptic_candidates.json
└── charts/
Synaptic.py
Mesin utama.
Tugasnya:
Mengambil data Binance Futures USDT perpetual.
Menyaring kandidat berdasarkan aktivitas dan likuiditas 24H.
Menganalisis timeframe:
15m
1h
4h
Menggunakan kombinasi indikator:
EMA200
Volume
MACD
Supertrend 10 / 2.50
Memberikan scoring LONG vs SHORT.
Memerlukan keselarasan arah minimal 2 timeframe.
Memilih timeframe eksekusi terbaik.
Menghasilkan:
Direction
Score
Entry
TP1
TP2
TP3
SL
Invalidation
Key points
Menyimpan hasil ke synaptic_candidates.json.
vSch.py
Lapisan visualisasi.
Tugasnya:
Membaca hasil Synaptic.py.
Memilih kandidat.
Mengambil data candle timeframe yang dipilih.
Membuat chart candlestick custom bergaya terminal Binance.
Menampilkan:
Candle
Volume
EMA200
Entry
TP1/TP2/TP3
SL
Invalidation
Key points
Menyimpan PNG ke folder kerja.
vSynapse.yml
GitHub Actions workflow.
Workflow dijalankan:
manual melalui workflow_dispatch
otomatis setiap 15 menit melalui cron
Alurnya:
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
Logika kandidat
vSynapse tidak mencoba memindai seluruh pasar secara sama rata.
Tahap pertama mencari koin yang:
memiliki volume futures 24H memadai,
mengalami perubahan harga yang cukup signifikan,
memiliki aktivitas relatif tinggi.
Tujuannya adalah menemukan koin yang sedang "hidup" terlebih dahulu.
Kandidat kemudian dianalisis pada 15m, 1h, dan 4h.
Kombinasi indikator
EMA200
EMA200 digunakan sebagai filter regime.
Secara umum:
Price > EMA200 → bias bullish
Price < EMA200 → bias bearish
EMA200 bukan sinyal entry tunggal.
Volume
Volume digunakan untuk melihat apakah pergerakan harga memiliki dukungan aktivitas.
Default:
Volume baseline = MA20
Confirmation = Volume >= 1.5x baseline
Volume bullish dan bearish dibedakan berdasarkan arah candle.
MACD
Default:
Fast   = 12
Slow   = 26
Signal = 9
MACD digunakan sebagai konfirmasi momentum, bukan sebagai trigger tunggal.
Bullish:
MACD > Signal
dan histogram menguat
Bearish:
MACD < Signal
dan histogram melemah
Supertrend
Parameter:
Period     = 10
Multiplier = 2.50
Supertrend digunakan sebagai konfirmasi arah/trend.
RSI
RSI tersedia di engine tetapi default:
enabled: false
Alasannya adalah menghindari terlalu banyak konfirmasi yang sebenarnya membawa informasi momentum yang mirip.
Jika pengujian historis menunjukkan RSI meningkatkan kualitas kandidat, RSI dapat diaktifkan kembali.
Scoring
Scoring menggunakan bobot indikator dan price action.
Bobot dasar:
EMA200       2.0
Supertrend   2.0
MACD         1.5
Volume       1.5
Breakout     2.0
Structure    1.0
Scanner menghitung skor LONG dan SHORT secara terpisah.
Contoh konseptual:
LONG
EMA200 bullish       +2
Supertrend bullish   +2
MACD bullish         +1.5
Volume bullish       +1.5
Breakout             +2
Structure            +1
-------------------------
Total                10
SHORT menggunakan logika kebalikan.
Scanner tidak mengharuskan seluruh indikator memberikan sinyal yang sama. Tujuannya adalah mendapatkan konfluensi, bukan membuat filter yang terlalu ketat.
Multi-timeframe
Bobot timeframe:
15m = 25%
1h  = 35%
4h  = 40%
Interpretasi:
4H memberikan konteks struktur terbesar.
1H menjadi timeframe struktur utama.
15m membantu membaca momentum/timing yang lebih dekat.
Minimal dua dari tiga timeframe harus mendukung arah yang sama.
Contoh:
4H  → LONG
1H  → LONG
15m → SHORT
Masih dapat menghasilkan kandidat LONG karena mayoritas timeframe mendukung LONG.
Sebaliknya:
4H  → LONG
1H  → SHORT
15m → SHORT
cenderung menghasilkan SHORT.
Entry / TP / SL
Scanner saat ini menggunakan harga terakhir sebagai basis Entry.
Stop loss menggunakan kombinasi recent swing dan ATR.
Risk-reward default:
TP1 = 1.50R
TP2 = 2.25R
TP3 = 3.00R
Untuk LONG:
Risk = Entry - SL

TP1 = Entry + 1.50 × Risk
TP2 = Entry + 2.25 × Risk
TP3 = Entry + 3.00 × Risk
Untuk SHORT:
Risk = SL - Entry

TP1 = Entry - 1.50 × Risk
TP2 = Entry - 2.25 × Risk
TP3 = Entry - 3.00 × Risk
Invalidation
Invalidation bukan sekadar "SL kena".
Scanner juga memberikan kondisi struktural yang membatalkan bias.
Contoh LONG:
Close below recent swing low
Contoh SHORT:
Close above recent swing high
Artinya pengguna dapat membedakan antara level risiko numerik dan kondisi chart yang membuat tesis arah tidak lagi valid.
Konfigurasi utama
Parameter utama berada di Synaptic.py dan workflow vSynapse.yml.
Contoh:
scanner:
  min_quote_volume_24h: 5000000
  min_abs_change_24h: 4.0
  universe_size: 80
  max_results: 15
  min_score: 6.0
Timeframe:
timeframes:
  primary:
    - 15m
    - 1h
    - 4h
Indikator:
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
Menjalankan lokal
Install dependency:
pip install requests pandas numpy matplotlib pyyaml
Jalankan scanner:
python Synaptic.py --out synaptic_candidates.json
Kemudian buat chart:
python vSch.py --input synaptic_candidates.json
Untuk kandidat tertentu:
python vSch.py --input synaptic_candidates.json --symbol BTCUSDT
Menjalankan di GitHub
Repository minimal berisi:
Synaptic.py
vSch.py
vSynapse.yml
Pastikan workflow berada pada:
.github/workflows/vSynapse.yml
GitHub Actions kemudian dapat dijalankan manual atau mengikuti jadwal cron.
Workflow membutuhkan:
permissions:
  contents: write
karena workflow melakukan commit hasil scan ke repository.
Output JSON
Contoh struktur:
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
Filosofi scanner
vSynapse dibuat dengan prinsip:
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
Scanner tidak dirancang untuk menghasilkan sebanyak mungkin sinyal.
Targetnya adalah menemukan kandidat yang sedang aktif dan memiliki konfluensi arah yang cukup, kemudian membuang kandidat yang struktur dan arah antar-timeframe terlalu bertentangan.
Catatan pengembangan
Versi saat ini adalah fondasi scanner.
Tahap kalibrasi berikutnya yang disarankan:
Simpan seluruh kandidat, termasuk yang gagal threshold.
Simpan skor masing-masing timeframe.
Catat kondisi indikator ketika kandidat ditemukan.
Bandingkan kandidat dengan hasil postingan historis.
Kalibrasi threshold volume, perubahan 24H, scoring, dan timeframe.
Tambahkan filter chase/pullback yang lebih spesifik.
Backtest sebelum menggunakan hasil scanner sebagai dasar keputusan trading.
Jangan menganggap parameter saat ini sebagai parameter optimal. Parameter tersebut adalah starting point yang sengaja dibuat cukup fleksibel untuk dikalibrasi dengan data nyata.