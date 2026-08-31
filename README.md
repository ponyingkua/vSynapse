vSynapse

vSynapse adalah scanner kandidat Binance Futures berbasis price action, indikator, dan multi-timeframe analysis.

Scanner mencari koin yang sedang aktif, menentukan bias LONG/SHORT, lalu menghasilkan Entry, TP, SL, Invalidation, dan chart.

«For research and educational purposes only. Not financial advice.»

How It Works

Binance Futures
      ↓
Active Coins
      ↓
15m / 1h / 4h
      ↓
Price Action + Indicators
      ↓
LONG / SHORT
      ↓
Entry / TP / SL / Invalidation
      ↓
Chart

Indikator utama:

- EMA200
- Volume
- MACD
- Supertrend

Run Locally

pip install requests pandas numpy matplotlib pyyaml
python Synaptic.py --out synaptic_candidates.json
python vSch.py --input synaptic_candidates.json

GitHub Actions

Workflow dapat dijalankan secara manual maupun otomatis menggunakan cron.

Hasil scan disimpan sebagai:

synaptic_candidates.json

Chart disimpan di:

charts/

vSynapse adalah proyek riset untuk membantu menemukan kandidat market yang aktif dan memiliki konfluensi arah.

Not financial advice.
