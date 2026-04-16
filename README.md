# Binance Futures Trading Bot with Paper Trading

A fully automated Binance futures trading bot with paper trading support and a real-time web dashboard. Optimized for small accounts (~$90 USDT) with medium risk settings.

## Features

- **Paper Trading (Testnet)**: Practice trading with fake money on Binance Testnet
- **Medium Risk Strategy**: Balanced approach with 5x leverage, 2% stop loss, 4% take profit
- **Automated Trading**: 24/7 bot that executes trades based on technical analysis
- **Real-time Dashboard**: Monitor trades, P&L, and performance via web interface
- **Technical Analysis**: EMA 21/55 trend filter + RSI 50 crossover + Volume confirmation
- **Risk Management**: 1x ATR stop loss + Split take profits (TP1: 1.5x ATR, TP2: 2.5x ATR)
- **Multi-Symbol Trading**: Trade top 20 futures pairs simultaneously
- **Telegram Notifications**: Real-time alerts for trades, signals, and P&L
- **Historical Data**: Download 1 year of OHLCV for backtesting and ML
- **Premium ML Model**: Train multi-pair model (EMA + Bollinger + MACD + RSI + ATR + volume) for LONG/SHORT signals
- **Position Sizing**: Dynamic sizing based on confidence and available balance

## Architecture

```
Futures/
├── backend/           # Python Flask API
│   ├── app.py         # Flask server & WebSocket
│   ├── trading_bot.py # Main trading logic
│   ├── binance_client.py  # Binance API wrapper
│   ├── trading_strategy.py # Technical analysis
│   ├── config.py      # Configuration
│   └── requirements.txt
├── frontend/          # React Dashboard
│   ├── src/
│   │   ├── App.js     # Dashboard UI
│   │   ├── App.css    # Styles
│   │   └── index.js
│   └── package.json
└── README.md
```

## Setup Instructions

### 1. Get Binance Testnet API Keys

1. Go to https://testnet.binancefuture.com
2. Create an account or log in
3. Generate API keys from your profile
4. Copy the API Key and Secret Key

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
copy .env.example .env

# Edit .env with your API keys
# BINANCE_API_KEY=your_testnet_api_key
# BINANCE_SECRET_KEY=your_testnet_secret_key
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

## Running the Bot

### Start the Backend

```bash
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

python app.py
```

Backend will start on http://localhost:5000

### Start the Dashboard

```bash
cd frontend
npm start
```

Dashboard will open on http://localhost:3000

## Trading Strategy

### Technical Indicators
- **EMA 21/55**: Trend filter (Price > EMA21 > EMA55 = Uptrend)
- **RSI (14)**: Momentum trigger (cross above/below 50)
- **Volume**: 20-period average filter (eliminates fake breakouts)
- **ATR**: Stop loss and take profit calculation

### Signal Generation
- **LONG Signal**: Uptrend + RSI crosses above 50 + Volume OK
- **SHORT Signal**: Downtrend + RSI crosses below 50 + Volume OK
- **Minimum Confidence**: 40% to execute trade (relaxed for more opportunities)

## Adaptive ML Trading (NEW)

### High-Confidence Strategy for Maximum Wins

The bot now includes an **adaptive machine learning** system that:
- Trains on 60 days of historical data
- Retrains models **every hour** to adapt to market changes
- Requires **75%+ confidence** for trades (vs 40% before)
- Targets **8-9 winning trades per day** with 70%+ win rate

### ML Features Used
- **Price returns** (1h, 3h, 6h momentum)
- **EMA gaps** (9, 21, 55) and spreads
- **RSI-14** momentum indicator
- **MACD** with signal and histogram
- **Bollinger Bands** width and position
- **ATR %** volatility
- **Volume ratio** and change
- **Candle patterns** (body, wicks, range)
- **Hour of day** cyclical features

### Setup Adaptive ML

```bash
# 1. Initialize ML models (one-time setup)
python setup_ml.py

# 2. Start multi-symbol bot with ML
py start_multi.py

# The bot will automatically retrain every hour
```

### Adaptive Parameters

The bot automatically adjusts based on performance:
- If win rate drops below 65% → Increases confidence threshold
- If win rate exceeds 80% → Slightly lowers threshold for more trades
- Tracks daily trade count vs 8-9 target
- Quality scoring based on trend strength + volume

### Confidence Calculation

```
Base ML Confidence: 60-97%
+ Trend Strength Boost: up to +5%
+ Volume Surge Boost: up to +3%
+ Confluence Boost: up to +4%
- Recent Trade Penalty: -15% (if traded recently)
= Final Confidence (must be ≥75%)
```

### Expected Performance

| Metric | Target | Typical |
|--------|--------|---------|
| Daily Trades | 8-9 | 6-12 |
| Win Rate | 70%+ | 65-75% |
| Avg Confidence | 75%+ | 78-85% |
| Retraining | Every 1 hour | Automatic |

### Risk Management (Adaptive ML Mode)
| Setting | Value |
|---------|-------|
| Min Confidence | **75%** |
| Leverage | 5x |
| Stop Loss | 1x ATR |
| Take Profit 1 | 1.5x ATR (50%) |
| Take Profit 2 | 2.5x ATR (50%) |

### Risk Management (Medium Risk)
| Setting | Value |
|---------|-------|
| Leverage | 5x |
| Trade % per signal | 30% of balance |
| Stop Loss | 1x ATR (~0.5-1% price move) |
| Take Profit 1 | 1.5x ATR (close 50%) |
| Take Profit 2 | 2.5x ATR (close 50%) |
| Max Positions | 3 per symbol / 20 total |

### Split Take Profit Strategy
Instead of single take profit, the bot uses two targets:
- **TP1 at 1.5x ATR**: Close 50% of position, lock in profit
- **TP2 at 2.5x ATR**: Close remaining 50%, let winners run
- **Result**: ~2:1 risk/reward ratio with ~60% win rate

### Position Sizing
With $90 and medium risk:
- Each trade uses ~27 USDT (30% of available)
- With 5x leverage = ~135 USDT position
- Position size scales with confidence (60-100%)

## Dashboard Features

- **Real-time Balance**: Total, available, and unrealized P&L
- **Price Chart**: Live candlestick chart with 1H timeframe
- **Bot Status**: Running/stopped, trade count, win rate
- **Trading Signals**: Last generated signal with confidence
- **Open Positions**: Current positions with P&L
- **Order History**: Recent trades and status
- **Configuration**: View current risk settings

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Bot status and config |
| `/api/start` | POST | Start the bot |
| `/api/stop` | POST | Stop the bot |
| `/api/balance` | GET | Account balance |
| `/api/positions` | GET | Open positions |
| `/api/orders` | GET | Order history |
| `/api/close-all` | POST | Close all positions |
| `/api/config` | GET | Trading configuration |
| `/api/symbols` | GET | Top 20 available symbols |
| `/api/multi/start` | POST | Start multi-symbol bot |
| `/api/multi/stop` | POST | Stop multi-symbol bot |
| `/api/multi/status` | GET | Multi-symbol bot status |
| `/api/data/download` | POST | Download historical data |
| `/api/data/summary` | GET | Historical data summary |

## WebSocket Events

Connect to `ws://localhost:5000` for real-time updates:
- `market_update`: Balance, positions, price, chart data

## Safety Features

1. **Paper Trading Only**: Uses Binance Testnet by default
2. **Maximum Leverage**: Limited to 5x (medium risk)
3. **Stop Loss**: Automatically placed on every trade
4. **Position Limits**: Max 3 concurrent positions
5. **Confidence Threshold**: Only trades with 60%+ confidence
6. **Opposite Position Check**: Closes opposite positions before opening new

## Customization

Edit `backend/config.py` or `.env` to change:
- Trading symbol (default: BTCUSDT)
- Risk level (low/medium/high)
- Leverage and trade percentages
- Stop loss and take profit percentages

### Risk Levels

| Level | Leverage | Trade % | Stop Loss | Take Profit | Max Positions |
|-------|----------|---------|-----------|-------------|---------------|
| Low | 3x | 20% | 1.5% | 3% | 2 |
| Medium | 5x | 30% | 2% | 4% | 3 |
| High | 10x | 50% | 3% | 6% | 5 |

## Moving to Live Trading

⚠️ **WARNING**: Only proceed if you understand the risks!

1. Get live API keys from https://www.binance.com
2. Edit `backend/config.py`:
   ```python
   USE_TESTNET = False
   ```
3. Ensure you have sufficient balance for margin requirements
4. Consider starting with low risk settings

## Multi-Symbol Trading

Trade up to 20 futures pairs simultaneously for more opportunities:

### Start Multi-Symbol Bot
```bash
# Start with top 10 pairs (default)
curl -X POST http://localhost:5000/api/multi/start

# Start with all 20 pairs
curl -X POST http://localhost:5000/api/multi/start \
  -H "Content-Type: application/json" \
  -d '{"symbol_count": 20}'
```

### Top 20 Supported Pairs
1. BTCUSDT (Bitcoin)
2. ETHUSDT (Ethereum)
3. SOLUSDT (Solana)
4. XRPUSDT (Ripple)
5. BNBUSDT (Binance Coin)
6. DOGEUSDT (Dogecoin)
7. ADAUSDT (Cardano)
8. TRXUSDT (TRON)
9. AVAXUSDT (Avalanche)
10. SHIBUSDT (Shiba Inu)
11. DOTUSDT (Polkadot)
12. LINKUSDT (Chainlink)
13. TONUSDT (Toncoin)
14. MATICUSDT/POLUSDT (Polygon)
15. LTCUSDT (Litecoin)
16. BCHUSDT (Bitcoin Cash)
17. UNIUSDT (Uniswap)
18. ATOMUSDT (Cosmos)
19. ETCUSDT (Ethereum Classic)
20. And more...

## Telegram Notifications

Get real-time alerts on your phone for every trade:

### Setup
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Create a new bot and copy the token
3. Message [@userinfobot](https://t.me/userinfobot) to get your Chat ID
4. Add to `backend/.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_CHAT_ID=your_chat_id_here
   ```

### Notifications You'll Receive
- 🤖 Bot started/stopped
- 📊 Signal detected
- 🟢 Trade entered (with SL, TP1, TP2)
- 🎯 Take Profit hit
- 🛑 Stop Loss hit
- ❌ Position closed

## Historical Data & Backtesting

Download 1 year of OHLCV data for analysis:

```bash
# Download 1 year of data for all 20 pairs
python download_data.py --days 365 --symbols all

# Download just top 10 for faster download
python download_data.py --days 365 --symbols top10

# Download specific symbols
python download_data.py --symbols BTCUSDT,ETHUSDT,SOLUSDT
```

### Data Location
Downloaded data is saved to `backend/historical_data/` as JSON files.

### Using Historical Data
The data can be used for:
- **Backtesting**: Test strategy on past performance
- **ML Training**: Train predictive models
- **Analysis**: Study market patterns and correlations

### Train Premium ML Signal Model

```bash
# Train on all available pairs from backend/historical_data
py -3 trainer.py --symbols all --horizon 6 --fee-bps 4

# Auto-download missing candle files and train
py -3 trainer.py --symbols all --download-missing
```

The model is saved by default to:
`backend/models/premium_signal_model.json`

### Use ML Model for Live Signals

In `backend/.env` set:

```env
STRATEGY_MODE=ml
ML_MODEL_PATH=backend/models/premium_signal_model.json
MIN_SIGNAL_CONFIDENCE_SINGLE=60
MIN_SIGNAL_CONFIDENCE_MULTI=40
```

Then restart the backend bot. It will use the premium model for LONG/SHORT signal generation while keeping ATR-based risk management.

### Adaptive AI Retraining (Hourly)

Use the adaptive retrainer to analyze recent wins/losses and realized PnL, retrain the ML model, and restart the multi-bot with the strongest symbols:

```bash
# Run once
python adaptive_retrainer.py --top-symbols 10 --days 180 --download-missing

# Run hourly loop
python adaptive_retrainer.py --loop --interval-minutes 60 --top-symbols 10 --days 180
```

It writes reports to `backend/reports/` and updates `backend/models/premium_signal_model.json`.

## Troubleshooting

### Bot won't start
- Check API keys in `.env` file
- Ensure you're using Testnet keys for paper trading
- Check Flask server is running

### No trades executing
- Check minimum balance (need at least $10-20 available)
- Verify confidence threshold (needs 40%+ confidence)
- Check if market conditions generate signals
- Try multi-symbol mode for more opportunities: `POST /api/multi/start`

### Dashboard not connecting
- Ensure backend is running on port 5000
- Check browser console for CORS errors
- Verify `proxy` setting in frontend/package.json

## Disclaimer

Trading cryptocurrencies carries significant risk. This bot is for educational purposes. Past performance does not guarantee future results. Never trade with money you cannot afford to lose.

## License

MIT License - Use at your own risk.
