# FutureTestBot: Advanced AI-Driven Cryptocurrency Futures Trading Bot

## Project Purpose and Problem It Solves

The FutureTestBot project aims to provide a sophisticated, automated solution for cryptocurrency futures trading on the Binance platform. In the highly volatile and fast-paced cryptocurrency market, manual trading is often inefficient, emotionally driven, and prone to human error. This project addresses these challenges by:

*   **Automating Trading Decisions:** Eliminating the need for constant manual monitoring and execution, allowing for 24/7 operation.
*   **Leveraging Machine Learning:** Utilizing data-driven models to identify potential trading opportunities and predict market movements, aiming for more objective and potentially profitable strategies.
*   **Enabling Backtesting and Optimization:** Providing tools to rigorously test trading strategies against historical data and optimize parameters before live deployment, minimizing risk.
*   **Facilitating Multi-Symbol Trading:** Managing and executing strategies across multiple cryptocurrency pairs simultaneously, diversifying opportunities.
*   **Providing Real-time Insights:** Offering real-time data processing and alerts to keep users informed of market conditions and bot performance.

The core problem solved is the need for a robust, intelligent, and automated system that can navigate the complexities of crypto futures trading with a focus on data-driven decision-making and continuous improvement.

## Core Features and Functionality

FutureTestBot is designed with a comprehensive set of features to support advanced automated trading:

*   **Machine Learning Model Training:**
    *   Supports various models, including LSTM (`simple_lstm_train.py`), RandomForestClassifier, and custom `TalonMLModel` and `EnhancedCryptoModel`.
    *   `adaptive_retrainer.py` enables continuous model improvement and adaptation to changing market conditions.
    *   `reset_and_train_advanced.py` and `reset_and_retrain_talon.py` provide utilities for re-initializing and retraining models.
*   **Predictive Signal Generation:** Utilizes trained ML models to generate buy/sell signals for specified cryptocurrency futures pairs (`predict_signals.py`).
*   **Historical Data Management:** Includes a `download_data.py` script to fetch and manage historical candlestick data from Binance.
*   **Strategy Backtesting:** Allows for comprehensive evaluation of trading strategies against historical data, generating performance metrics and insights (`BACKTESTING_GUIDE.md`, `backtest_results_*.json`).
*   **Real-time Data Streaming:** Integrates with Binance WebSockets for real-time market data updates, crucial for timely signal generation and trade execution.
*   **Automated Trade Execution:** Connects with the Binance Futures API to place and manage orders based on generated signals.
*   **Multi-Symbol Trading:** The `MultiSymbolBot` architecture enables the bot to simultaneously monitor and trade multiple cryptocurrency pairs (e.g., ADAUSDT, ATOMUSDT, AVAXUSDT, BCHUSDT, BNBUSDT).
*   **Web Server/API Backend:** A Flask-based backend (`backend/`) with Socket.IO integration for real-time communication and a potential frontend interface.
*   **Automated Notifications and Alerts:** Designed to provide timely updates on trading activities, performance, and critical market events.
*   **Strategy Optimization:** Tools and summaries (`STRATEGY_OPTIMIZATION_SUMMARY.md`, `optimization_results_*.json`) for fine-tuning trading parameters.

## How the Codebase Works Internally & Key Workflows

The project follows a modular architecture, separating concerns into distinct components and workflows:

1.  **Data Acquisition (`download_data.py`):**
    *   Connects to the Binance API to fetch historical candlestick data for specified symbols and timeframes.
    *   Stores data locally for training and backtesting.
2.  **Machine Learning Model Lifecycle:**
    *   **Setup (`setup_ml.py`):** Initializes the ML environment.
    *   **Training (`simple_lstm_train.py`, `trainer.py`, `adaptive_retrainer.py`, `reset_and_train_advanced.py`, `reset_and_retrain_talon.py`):**
        *   Historical data is preprocessed (e.g., feature engineering with technical indicators like RSI, EMA, OBV, ATR).
        *   Various ML models are trained on this data to learn patterns and predict future price movements or signal generation.
        *   Adaptive retraining ensures models remain relevant to evolving market conditions.
    *   **Prediction (`predict_signals.py`):**
        *   Trained models are used to generate real-time trading signals (buy/sell/hold) based on the latest market data.
3.  **Trading Strategy Execution (`start.bat`, `start_multi.py`):**
    *   The `MultiSymbolBot` orchestrates trading across multiple symbols.
    *   It continuously fetches real-time data, feeds it to the prediction models, and executes trades via the Binance Futures Client based on the generated signals and predefined trading strategies.
    *   Risk management and position sizing are integrated into the trading logic.
4.  **Backtesting and Performance Analysis (`BACKTESTING_GUIDE.md`, `performance_analyzer`):**
    *   Historical data is replayed, and trading strategies are simulated to evaluate their performance without real capital risk.
    *   Metrics like profit/loss, drawdown, win rate, etc., are calculated and analyzed.
5.  **Web Interface (Backend/Frontend):**
    *   The `backend/` directory likely contains a Flask application that serves as an API for the frontend and handles real-time communication via Flask-SocketIO.
    *   The `frontend/` directory would house the user interface for monitoring the bot, viewing performance, and potentially configuring strategies.

## Folder/Project Structure

The repository is organized to separate different aspects of the project:

*   **`backend/`**: Contains the Flask web server, API endpoints, and Socket.IO implementation for real-time communication.
*   **`frontend/`**: Intended for the web-based user interface (e.g., React, Vue, Angular) that interacts with the backend.
*   **`.gitattributes`, `.gitignore`, `LICENSE`**: Standard repository configuration files.
*   **`README.md`**: This documentation file.
*   **`BACKTESTING_GUIDE.md`**: Detailed guide on how to perform backtesting.
*   **`COMPLETE_SUMMARY.md`**: A comprehensive summary of the project or specific runs.
*   **`STRATEGY_OPTIMIZATION_SUMMARY.md`**: Documentation related to strategy optimization processes and results.
*   **`adaptive_retrainer.py`**: Script for adaptively retraining machine learning models.
*   **`backtest_results_*.json`**: JSON files storing the results of backtesting runs.
*   **`download_data.py`**: Script to download historical cryptocurrency data.
*   **`optimization_results_*.json`**: JSON files storing the results of strategy optimization runs.
*   **`predict_signals.py`**: Script responsible for generating trading signals using trained models.
*   **`render.yaml`**: Configuration file for deploying the application on Render.com.
*   **`reset_all_data.py`**: Utility to clear all stored data.
*   **`reset_and_retrain_talon.py`**: Script to reset and retrain the Talon ML model.
*   **`reset_and_train_advanced.py`**: Script to reset and train advanced ML models.
*   **`setup.bat`**: Windows batch script for initial project setup.
*   **`setup_ml.py`**: Script for setting up the machine learning environment.
*   **`simple_lstm_train.py`**: Script for training a simple LSTM model.
*   **`start.bat`**: Windows batch script to start the bot.
*   **`start_debug.bat`**: Windows batch script to start the bot in debug mode.
*   **`start_multi.bat`**: Windows batch script to start the multi-symbol bot.
*   **`start_multi.py`**: Python script for running the multi-symbol trading bot.
*   **`start_multi_simple.bat`**: Simplified Windows batch script for starting the multi-symbol bot.
*   **`test_minimal.py`**: Minimal test script.
*   **`trainer.py`**: General script for training machine learning models.
*   **Other Python files (e.g., `talon_ml_model`, `enhanced_crypto_model`, `trading_strategy`, `technical`, `binance_client`, `config`, `performance_analyzer`):** These likely contain modular classes and functions supporting the core logic, such as ML model definitions, trading strategy implementations, technical indicator calculations, Binance API wrappers, and configuration management.

## Setup and Usage Instructions

To get the FutureTestBot up and running, follow these general steps:

### Prerequisites

*   **Python 3.8+**: Ensure Python is installed on your system.
*   **Git**: For cloning the repository.
*   **Binance API Key and Secret**: You will need an API key and secret from your Binance Futures account with appropriate permissions (read and trade).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Maddyssjg2/FutureTestBot.git
    cd FutureTestBot
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    The project uses a variety of libraries listed in the "Tech Stack" section. You will need to install them. A `requirements.txt` file is typically used for this, but if not present, you might need to install them manually or infer from the `Tech Stack`.
    ```bash
    # Example (you might need to create a requirements.txt first)
    pip install -r requirements.txt
    ```
    *Common dependencies include:* `Flask`, `Flask-SocketIO`, `Flask-CORS`, `python-binance`, `numpy`, `pandas`, `scikit-learn`, `tensorflow` (for LSTM), `python-dotenv`, `ta` (for technical indicators).
4.  **Configuration:**
    *   Create a `.env` file in the root directory based on a `.env.example` (if available) or manually.
    *   Add your Binance API key and secret:
        ```
        BINANCE_API_KEY=YOUR_BINANCE_API_KEY
        BINANCE_API_SECRET=YOUR_BINANCE_API_SECRET
        # Other configuration parameters like symbols, timeframes, etc.
        ```
    *   Review and adjust any configuration files (e.g., `config.py` if present) for trading parameters, model settings, etc.

### Running the Bot

1.  **Initial Setup (Windows):**
    ```bash
    setup.bat
    ```
    This script might handle initial data downloads or environment setup.
2.  **Download Historical Data:**
    ```bash
    python download_data.py
    ```
3.  **Train Machine Learning Models:**
    Choose the appropriate training script based on your desired model:
    ```bash
    python simple_lstm_train.py
    # or
    python trainer.py
    # or for advanced models
    python reset_and_train_advanced.py
    ```
4.  **Start the Trading Bot:**
    *   For single-symbol operation (if applicable):
        ```bash
        start.bat # On Windows
        python start_multi.py # On macOS/Linux (if start.bat just calls this)
        ```
    *   For multi-symbol operation:
        ```bash
        start_multi.bat # On Windows
        python start_multi.py # On macOS/Linux
        ```
5.  **Access the Web Interface (if running):**
    If the `backend/` and `frontend/` are set up, you would typically start the backend server and then the frontend development server.
    ```bash
    # In backend/ directory
    python app.py # or similar entry point
    # In frontend/ directory
    npm start # or yarn start (depending on frontend framework)
    ```

## APIs, Integrations, and Dependencies Used

*   **Binance API:** Primary integration for market data (REST and WebSockets) and trade execution on Binance Futures.
*   **Flask:** Python web framework for building the backend API.
*   **Flask-SocketIO:** Enables real-time, bidirectional communication between the backend and frontend.
*   **Flask-CORS:** Handles Cross-Origin Resource Sharing for frontend-backend communication.
*   **NumPy & Pandas:** Fundamental libraries for numerical operations and data manipulation, especially for handling time-series market data.
*   **Scikit-learn:** Provides various machine learning algorithms (e.g., RandomForestClassifier) and utilities for data preprocessing (e.g., StandardScaler, train_test_split).
*   **TensorFlow/Keras (implied by LSTM):** For building and training deep learning models like LSTMs.
*   **`python-dotenv`:** For managing environment variables (API keys, secrets).
*   **`ta` (Technical Analysis Library - implied):** Likely used for calculating technical indicators like RSI, EMA, OBV, ATR.
*   **`pickle`:** For serializing and deserializing Python objects, often used to save and load trained ML models.
*   **`threading`:** For managing concurrent operations, especially in multi-symbol or real-time data processing.

## Example Inputs/Outputs or Use Cases

### Example Inputs

*   **Historical Data:** CSV files or database entries containing OHLCV (Open, High, Low, Close, Volume) data for cryptocurrency pairs (e.g., `ADAUSDT_1h.csv`).
*   **Configuration:** `.env` file specifying `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `SYMBOLS_TO_TRADE`, `TIMEFRAME`, `STRATEGY_PARAMETERS` (e.g., RSI thresholds, stop-loss percentages).
*   **User Commands:** Running scripts like `python download_data.py` or `python start_multi.py`.

### Example Outputs

*   **Trading Signals:** Log messages or API responses indicating `BUY ADAUSDT`, `SELL BNBUSDT`, `HOLD`.
*   **Executed Trades:** Records of placed orders, fills, and current positions on Binance.
*   **Backtest Reports:** `backtest_results_20260417_033838.json` containing metrics like:
    ```json
    {
        "total_profit": 1500.25,
        "max_drawdown": -10.5,
        "win_rate": 0.65,
        "total_trades": 120,
        "performance_chart_data": [...]
    }
    ```
*   **Optimization Results:** `optimization_results_20260417_035021.json` showing optimal parameters for a strategy.
*   **Real-time Alerts:** Notifications (e.g., via Telegram, email, or web UI) for significant events like large price movements, trade executions, or model retraining completion.

### Use Cases

*   **Automated Portfolio Management:** Users can set up the bot to manage a portion of their crypto portfolio on Binance Futures, executing trades based on predefined or ML-driven strategies.
*   **Strategy Development and Testing:** Traders can develop new strategies, backtest them extensively, and optimize parameters using the provided tools before deploying them live.
*   **Market Research:** The data downloading and analysis capabilities can be used by researchers to study market behavior and develop new predictive models.
*   **Educational Tool:** A platform for learning about algorithmic trading, machine learning in finance, and interacting with cryptocurrency APIs.

## Future Improvements / Roadmap

*   **Expanded Exchange Support:** Integrate with other major cryptocurrency exchanges (e.g., Bybit, OKX) to diversify trading opportunities.
*   **Advanced ML Models:** Explore and implement more sophisticated deep learning architectures (e.g., Transformers, Reinforcement Learning) for signal generation.
*   **Enhanced Risk Management:** Implement more granular risk management features, including dynamic position sizing, portfolio-level risk controls, and advanced stop-loss/take-profit mechanisms.
*   **User Interface (Frontend):** Develop a comprehensive and intuitive web-based dashboard for:
    *   Real-time monitoring of bot performance and open positions.
    *   Configuration of trading strategies and ML models.
    *   Visualization of backtest results and market data.
    *   Management of API keys and account settings.
*   **Strategy Marketplace:** Potentially allow users to share, discover, and subscribe to community-contributed trading strategies.
*   **Cloud Deployment Automation:** Further streamline deployment to cloud platforms (e.g., AWS, GCP, Azure) beyond Render.yaml.
*   **Robust Error Handling & Logging:** Improve error handling, alerting, and logging for production environments.
*   **Paper Trading Mode:** Implement a dedicated paper trading mode for risk-free testing of live strategies without using real funds.
*   **Community Contributions:** Encourage and integrate contributions from the open-source community.
