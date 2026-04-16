"""
Enhanced ML Strategy using RandomForest + GradientBoosting
Optimized for crypto with 75%+ confidence targets
"""

import logging
import os
import warnings
from typing import Optional
from datetime import datetime, timedelta

# Suppress warnings from technical analysis library
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")

from enhanced_crypto_model import EnhancedCryptoModel, get_or_train_model, predict_with_enhanced_model
from trading_strategy import TradingStrategy
from config import Config
from data_downloader import DataDownloader

logger = logging.getLogger(__name__)

# High-confidence threshold
MIN_CONFIDENCE = 75.0
TARGET_DAILY_TRADES = 10  # 9-10 total across ALL symbols combined


class _GlobalTradeCounter:
    """Shared trade counter across all symbol strategies - 9-10 total trades per day"""
    def __init__(self):
        self.trades_today = 0
        self.last_trade_date = None
        self.trade_log = []  # Track which symbols traded

    def can_trade(self, symbol: str) -> bool:
        today = datetime.utcnow().date()
        if self.last_trade_date != today:
            self.trades_today = 0
            self.last_trade_date = today
            self.trade_log = []
        return self.trades_today < TARGET_DAILY_TRADES

    def record_trade(self, symbol: str):
        today = datetime.utcnow().date()
        if self.last_trade_date != today:
            self.trades_today = 0
            self.last_trade_date = today
            self.trade_log = []
        self.trades_today += 1
        self.trade_log.append(symbol)

    def get_status(self) -> str:
        return f"{self.trades_today}/{TARGET_DAILY_TRADES}"


# Single global instance shared by all EnhancedMLStrategy instances
_global_trade_counter = _GlobalTradeCounter()


class EnhancedMLStrategy(TradingStrategy):
    """
    Advanced ML strategy with RandomForest + GradientBoosting ensemble
    Optimized for BTC, ETH, and major cryptos
    Uses global trade counter: 9-10 total trades/day across ALL symbols
    """
    
    def __init__(self, risk_config, symbol: str, auto_train: bool = True):
        super().__init__(risk_config=risk_config)
        self.symbol = symbol
        self.model: Optional[EnhancedCryptoModel] = None
        self.auto_train = auto_train
        self.last_training = None
        
        # Try to load existing model or train
        self._ensure_model()
    
    def _ensure_model(self):
        """Ensure model is loaded or trained"""
        model_path = f"backend/ml_models/enhanced_{self.symbol}.pkl"
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.model = EnhancedCryptoModel.load(model_path)
                self.last_training = self.model.training_date
                logger.info(f"[EnhancedML] Loaded model for {self.symbol} (trained: {self.last_training})")
                return
            except Exception as e:
                logger.warning(f"[EnhancedML] Failed to load model: {e}")
        
        # Auto-train if enabled and no model exists
        if self.auto_train:
            self._train_model()
    
    def _train_model(self):
        """Train model on historical data"""
        logger.info(f"[EnhancedML] Training model for {self.symbol}...")
        
        try:
            downloader = DataDownloader()
            downloader.client.symbol = self.symbol
            
            # Get 60 days of data (Binance limit is 1000 candles per request)
            # Use multiple requests to get more data
            limit = 1000
            candles = downloader.client.get_klines(interval='1h', limit=limit)
            
            if not candles or len(candles) < 500:
                logger.warning(f"[EnhancedML] Insufficient data for {self.symbol}: {len(candles) if candles else 0}")
                return
            
            self.model = EnhancedCryptoModel()
            metrics = self.model.train(candles, self.symbol)
            
            # Save model
            model_path = f"backend/ml_models/enhanced_{self.symbol}.pkl"
            self.model.save(model_path)
            
            self.last_training = datetime.utcnow()
            
            logger.info(f"[EnhancedML] Model trained for {self.symbol}:")
            logger.info(f"  Samples: {metrics['samples']}")
            logger.info(f"  RF Accuracy: {metrics['rf_accuracy']:.1%}")
            logger.info(f"  GB Accuracy: {metrics['gb_accuracy']:.1%}")
            logger.info(f"  Features: {metrics['features']}")
            
        except Exception as e:
            logger.error(f"[EnhancedML] Training failed for {self.symbol}: {e}")
    
    def generate_signal(self, klines):
        """Generate high-confidence signal using enhanced ML"""
        
        # Check global daily trade limit (9-10 total across ALL symbols)
        if not _global_trade_counter.can_trade(self.symbol):
            logger.debug(f"[EnhancedML] Global daily trade limit reached ({_global_trade_counter.get_status()})")
            return None, 0, None
        
        # Ensure model is available
        if not self.model:
            logger.warning(f"[EnhancedML] No model for {self.symbol}, using fallback")
            # Fallback to base strategy with high threshold
            signal, confidence, details = super().generate_signal(klines)
            if signal and confidence >= MIN_CONFIDENCE:
                _global_trade_counter.record_trade(self.symbol)
                return signal, confidence, details
            return None, 0, None
        
        # Get prediction
        prediction = self.model.predict(klines)
        
        if not prediction or prediction.get('signal') is None:
            reason = prediction.get('reason', 'no_signal') if prediction else 'no_prediction'
            logger.debug(f"[EnhancedML] {self.symbol} no signal: {reason}")
            return None, 0, None
        
        confidence = prediction['confidence']
        
        # Require 75% confidence
        if confidence < MIN_CONFIDENCE:
            logger.debug(f"[EnhancedML] {self.symbol} confidence {confidence:.1f}% < {MIN_CONFIDENCE}%")
            return None, 0, None
        
        # Signal accepted - record in global counter
        _global_trade_counter.record_trade(self.symbol)
        
        signal = prediction['signal']
        
        logger.info(
            f"[EnhancedML-HIGHCONF] {self.symbol} {signal} "
            f"CONF={confidence:.1f}% (target: 75%+) "
            f"Global trades today: {_global_trade_counter.get_status()}"
        )
        
        # Build details
        details = {
            'stop_loss': prediction['stop_loss'],
            'take_profit_1': prediction['take_profit_1'],
            'take_profit_2': prediction['take_profit_2'],
            'entry_price': prediction['entry_price'],
            'atr': prediction['atr'],
            'ml_confidence': confidence,
            'ml_probas': prediction.get('probas', {}),
            'model_agreement': prediction.get('model_agreement', False),
        }
        
        return signal, confidence, details
    
    def should_retrain(self) -> bool:
        """Check if model needs retraining (older than 24 hours)"""
        if not self.last_training:
            return True
        
        age = datetime.utcnow() - self.last_training
        return age > timedelta(hours=24)


# Factory function
def create_enhanced_strategy(risk_config, symbol: str) -> EnhancedMLStrategy:
    """Create enhanced ML strategy for a symbol"""
    return EnhancedMLStrategy(risk_config, symbol, auto_train=True)
