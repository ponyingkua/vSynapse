# 🧠 vSynapse

**Binance Futures scanner untuk kandidat LONG / SHORT yang aktif.**

vSynapse memadukan *price action*, indikator teknikal, dan analisis multi-timeframe untuk menemukan kandidat pasar yang aktif — lengkap dengan Entry, TP, SL, level invalidasi, dan chart otomatis.

> ⚠️ **Disclaimer**: Untuk riset & edukasi saja. **Not Financial Advice (NFA).**

---

## ⚙️ Cara Kerja

```
Binance Futures
      ↓
Filter Active Coins
      ↓
Analisis 15m / 1h / 4h
      ↓
Price Action + Indicators
      ↓
Sinyal LONG / SHORT
      ↓
Entry / TP / SL / Invalidation
      ↓
Chart & Ringkasan
```

**Indikator yang dipakai:** `EMA200` · `Volume` · `MACD` · `Supertrend`

---

## 🚀 Menjalankan Secara Lokal

**1. Install dependencies**
```bash
pip install requests pandas numpy matplotlib pyyaml
```

**2. Jalankan scanner**
```bash
python Synaptic.py --out synaptic_candidates.json
```

**3. Generate chart & ringkasan**
```bash
python vSch.py --input synaptic_candidates.json
```

---

## 🤖 Otomatisasi via GitHub Actions

Scanner ini bisa dijalankan otomatis lewat GitHub Actions — tidak perlu server aktif 24 jam. Cukup atur jadwal (cron) di workflow, dan vSynapse akan scan pasar sesuai interval yang ditentukan.

---

## 📁 Struktur Proyek

```
├── Synaptic.py     # Scanner utama
├── vSch.py         # Generator chart & ringkasan
└── .github/
    └── workflows/  # Workflow otomatisasi
```

---

## ⚠️ Disclaimer

Proyek ini dibuat untuk **tujuan riset dan edukasi**. Semua output (sinyal, entry, TP, SL) **bukan merupakan nasihat finansial**. Selalu lakukan riset sendiri (DYOR) sebelum mengambil keputusan trading.

---

<p align="center">Made with 🧠 by <b>vSynapse</b></p>
