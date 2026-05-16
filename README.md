# FutureTestBot 📈

FutureTestBot is a sophisticated automated cryptocurrency trading system designed specifically for **Binance Futures**. It combines machine learning (ML) models with traditional technical analysis to generate high-probability trade signals across multiple trading pairs.

The bot is engineered to move beyond simple indicator-based strategies by utilizing an ensemble approach: combining ML predictions (RandomForest, GradientBoosting, LSTM) with technical confirmation filters to reduce false positives and optimize profit factors.

## 🎯 Project Purpose & Problem Solved
Trading cryptocurrency futures is high-risk due to extreme volatility. Most bots rely on static indicators (like EMA crossovers) which fail in changing market regimes. 

**FutureTestBot solves this by:**
- **Adaptive Learning:** Using ML models that can be retrained as market conditions evolve.
- **Signal Filtering:** Requiring both an ML "edge" and technical confirmation before entering a trade.
- **Multi-Symbol Scalability:** Managing multiple pairs (BTC, ETH, SOL, ADA, etc.) simultaneously with pair-specific optimizations.
- **Rigorous Validation:** Providing a full pipeline from historical data download to backtesting and optimization before going live.

## ✨ Core Features
- **ML-Driven Signals:** Integration of `RandomForest`, `GradientBoosting`, and `LSTM` models for price direction prediction.
- **Technical Analysis Suite:** Built-in indicators including RSI, MACD, Bollinger Bands, ATR, and EMA.
- **Multi-Symbol Bot:** Capability to monitor and trade multiple Binance Futures pairs concurrently.
- **Adaptive Retraining:** An automated pipeline to retrain models based on recent performance losses (`adaptive_retrainer.py`).
- **Backtesting Engine:** Comprehensive backtesting framework to evaluate strategies against 6+ months of hourly data.
- **Real-time Monitoring:** WebSocket integration for live price streaming and a Flask-based backend for status tracking.
- **Alerting System:** Integrated Telegram notifications for trade entries, exits, and system health.

## ⚙️ Internal Architecture & Workflow

### High-Level Logic Flow
`Data Acquisition` $\rightarrow$ `Feature Engineering` $\rightarrow$ `ML Training/Optimization` $\rightarrow$ `Signal Generation` $\rightarrow$ `Execution`

1. **Data Pipeline:** `download_data.py` fetches historical K-line data from Binance.
2. **Feature Engineering:** Raw price data is transformed into features (Returns, EMA ratios, RSI, MACD, Volume) used by the ML models.
3. **Model Ensemble:** 
   - The bot uses a "Confirmation" architecture.
   - A trade is only triggered if the **ML Model** predicts a direction AND the **Technical Strategy** (e.g., Pullback strategy with RSI/ADX filters) confirms the trend.
4. **Execution:** `multi_symbol_bot.py` manages the state of multiple pairs, handling order placement and risk management via the `binance_client.py`.

### Project Structure
```text
FutureTestBot/
├── backend/                # Core Trading Logic
│   ├── ml_models/          # Saved trained model binaries (.pkl)
│   ├── pair_models/        # Pair-specific configuration and models
│   ├── binance_client.py    # Wrapper for Binance API interactions
│   ├── multi_symbol_bot.py  # Main execution engine for multiple pairs
│   ├── core_strategy.py     # Base trading logic and signal rules
│   ├── adaptive_trainer.py  # Logic for retraining models on the fly
│   └── performance_analyzer.py # Metrics calculation (Win%, Profit Factor)
├── frontend/               # Dashboard UI for monitoring bot status
├── trainer.py              # Entry point for training new ML models
├── adaptive_retrainer.py    # Script to trigger model updates based on loss
├── download_data.py         # Utility to fetch historical data
└── setup.bat / start.bat    # Windows convenience scripts for environment setup
```

## 🚀 Setup & Usage

### Prerequisites
- Python 3.9+
- Binance API Key and Secret (with Futures enabled)
- Telegram Bot Token (for notifications)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Maddyssjg2/FutureTestBot.git
   cd FutureTestBot
   ```
2. **Configure Environment:**
   Create a `.env` file in the `backend/` directory based on `.env.example`:
   ```env
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   TELEGRAM_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

### Running the Bot
- **For Windows users:** Run `setup.bat` followed by `start.bat`.
- **Manual Start:** 
  ```bash
  python start_multi.py
  ```

## 🛠 APIs & Dependencies
- **Binance API:** For market data and futures order execution.
- **Scikit-Learn / Pandas / NumPy:** For ML model training, feature engineering, and data manipulation.
- **Flask / Flask-SocketIO:** Powers the backend server and real-time communication with the frontend.
- **Telegram Bot API:** Used for remote monitoring and alerts.

## 📈 Example Use Case
**Scenario: Trading ADAUSDT**
1. The bot downloads the last 6 months of hourly data.
2. A `GradientBoosting` model is trained to predict if the next candle will be bullish.
3. **Live Trade:** 
   - ML Model predicts "UP".
   - RSI indicates a pullback from oversold levels.
   - ADX confirms a strong trend.
   - **Action:** Bot opens a Long position with a 2:1 Take Profit/Stop Loss ratio.

## 🗺 Roadmap & Future Improvements
- [ ] **Sentiment Analysis:** Integrate Twitter/News API to filter trades based on market sentiment.
- [ ] **Advanced Risk Management:** Implement dynamic position sizing based on account equity (Kelly Criterion).
- [ ] **Hyperparameter Tuning:** Automate model optimization using Optuna or GridSearch.
- [ ] **Enhanced UI:** Add real-time PnL charts and trade history to the frontend dashboard.
