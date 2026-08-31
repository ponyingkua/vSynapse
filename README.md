vSynapse
Binance Futures scanner for active LONG / SHORT candidates.
vSynapse combines price action, technical indicators, and multi-timeframe analysis to identify active market candidates and generate Entry, TP, SL, Invalidation, and charts.
For research and educational purposes only. Not financial advice.
How It Works
Binance Futures
      ↓
Active Coins
      ↓
15m / 1h / 4h Analysis
      ↓
Price Action + Indicators
      ↓
LONG / SHORT
      ↓
Entry / TP / SL / Invalidation
      ↓
Chart
Indicators
EMA200
Volume
MACD
Supertrend
Run Locally
pip install requests pandas numpy matplotlib pyyaml

python Synaptic.py --out synaptic_candidates.json

python vSch.py --input synaptic_candidates.json
GitHub Actions
The scanner can run manually or automatically using GitHub Actions.

NFA