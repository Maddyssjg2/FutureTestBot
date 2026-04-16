import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Binance API
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # Trading Settings
    TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
    RISK_LEVEL = os.getenv('RISK_LEVEL', 'medium')
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', 3))
    LEVERAGE = int(os.getenv('LEVERAGE', 5))
    
    # Risk Management
    STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 2.0))
    TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 4.0))
    TRADE_PERCENTAGE = float(os.getenv('TRADE_PERCENTAGE', 30))
    
    # Server
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Paper Trading (Testnet)
    USE_TESTNET = True
    TESTNET_BASE_URL = 'https://testnet.binancefuture.com'
    
    # Telegram Notifications
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    
    # Top 20 Futures Trading Pairs (High volume, good liquidity)
    TOP_20_SYMBOLS = [
        'BTCUSDT',   # Bitcoin
        'ETHUSDT',   # Ethereum
        'SOLUSDT',   # Solana
        'XRPUSDT',   # Ripple
        'BNBUSDT',   # Binance Coin
        'DOGEUSDT',  # Dogecoin
        'ADAUSDT',   # Cardano
        'TRXUSDT',   # TRON
        'AVAXUSDT',  # Avalanche
        'DOTUSDT',   # Polkadot
        'LINKUSDT',  # Chainlink
        'TONUSDT',   # Toncoin
        'MATICUSDT', # Polygon (deprecated but still active)
        'POLUSDT',   # Polygon (new)
        'LTCUSDT',   # Litecoin
        'BCHUSDT',   # Bitcoin Cash
        'UNIUSDT',   # Uniswap
        'ATOMUSDT',  # Cosmos
        'ETCUSDT',   # Ethereum Classic
    ]
    
    # Enable multi-symbol trading (trade all 20 or single symbol)
    TRADE_ALL_SYMBOLS = os.getenv('TRADE_ALL_SYMBOLS', 'False').lower() == 'true'

    # Multi-bot defaults
    DEFAULT_MULTI_SYMBOL_COUNT = int(os.getenv('DEFAULT_MULTI_SYMBOL_COUNT', 10))
    FORCE_INCLUDE_SYMBOLS = [
        symbol.strip().upper()
        for symbol in os.getenv('FORCE_INCLUDE_SYMBOLS', 'BTCUSDT').split(',')
        if symbol.strip()
    ]

    # Strategy mode: "rule" or "ml"
    STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'rule').lower()
    _ml_model_path_env = os.getenv('ML_MODEL_PATH', 'models/premium_signal_model.json')
    if os.path.isabs(_ml_model_path_env):
        ML_MODEL_PATH = _ml_model_path_env
    else:
        normalized_ml_path = _ml_model_path_env.replace('\\', '/').lstrip('./')
        if normalized_ml_path.startswith('backend/'):
            normalized_ml_path = normalized_ml_path[len('backend/'):]
        ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), normalized_ml_path)

    # Signal confidence gates - 80% for 80%+ WR target
    # Target: Quality trades with good win rate
    MIN_SIGNAL_CONFIDENCE_SINGLE = int(os.getenv('MIN_SIGNAL_CONFIDENCE_SINGLE', 80))
    MIN_SIGNAL_CONFIDENCE_MULTI = int(os.getenv('MIN_SIGNAL_CONFIDENCE_MULTI', 80))
    ENABLE_EXCHANGE_STOP_LOSS = os.getenv('ENABLE_EXCHANGE_STOP_LOSS', 'False').lower() == 'true'

    @classmethod
    def get_default_multi_symbols(cls, symbol_count=None):
        count = symbol_count if symbol_count is not None else cls.DEFAULT_MULTI_SYMBOL_COUNT
        count = max(1, min(int(count), len(cls.TOP_20_SYMBOLS)))

        selected = []
        for forced_symbol in cls.FORCE_INCLUDE_SYMBOLS:
            if forced_symbol in cls.TOP_20_SYMBOLS and forced_symbol not in selected:
                selected.append(forced_symbol)

        for symbol in cls.TOP_20_SYMBOLS:
            if symbol not in selected:
                selected.append(symbol)
            if len(selected) >= count:
                break

        return selected
    
    @classmethod
    def get_risk_config(cls):
        risk_configs = {
            'low': {
                'leverage': 3,
                'stop_loss': 1.5,
                'take_profit': 3.0,
                'trade_percentage': 20,
                'max_positions': 2
            },
            'medium': {
                'leverage': 5,
                'stop_loss': 2.0,
                'take_profit': 4.0,
                'trade_percentage': 30,
                'max_positions': 3
            },
            'high': {
                'leverage': 10,
                'stop_loss': 3.0,
                'take_profit': 6.0,
                'trade_percentage': 50,
                'max_positions': 5
            }
        }
        return risk_configs.get(cls.RISK_LEVEL, risk_configs['medium'])
