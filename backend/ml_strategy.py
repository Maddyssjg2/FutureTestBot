import logging
import os

from enhanced_crypto_model import EnhancedCryptoModel
from trading_strategy import TradingStrategy
from adaptive_trainer import adaptive_trainer
from config import Config

logger = logging.getLogger(__name__)

# High-confidence threshold for quality trades
MIN_CONFIDENCE_THRESHOLD = 75.0
TARGET_DAILY_TRADES = 10  # 9-10 total across ALL symbols


class MLTradingStrategy(TradingStrategy):
    def __init__(self, risk_config, symbol: str, model_path: str = None):
        super().__init__(risk_config=risk_config)
        self.symbol = symbol
        self.model_path = model_path
        self.model = None
        self.recent_signals = []  # Track recent signals for frequency control
        
        # Try to get enhanced model from adaptive trainer first
        self.model = adaptive_trainer.get_model(symbol)
        if self.model:
            logger.info(f"[MLStrategy] Loaded Enhanced model for {symbol} (RF+GB ensemble)")
        else:
            logger.warning(f"[MLStrategy] No Enhanced model for {symbol} - will use fallback strategy")

    def _risk_levels_from_atr(self, signal: str, entry_price: float, atr_value: float):
        if signal == "LONG":
            stop_loss = entry_price - (atr_value * 1.0)
            take_profit_1 = entry_price + (atr_value * 1.5)
            take_profit_2 = entry_price + (atr_value * 2.5)
        else:
            stop_loss = entry_price + (atr_value * 1.0)
            take_profit_1 = entry_price - (atr_value * 1.5)
            take_profit_2 = entry_price - (atr_value * 2.5)
        return stop_loss, take_profit_1, take_profit_2

    def generate_signal(self, klines):
        """Generate high-confidence signal (75%+) using Enhanced ML model"""
        base_df = self.calculate_indicators(klines)
        if len(base_df) < 80:
            return None, 0, None

        # Try Enhanced ML prediction (with 75% confidence requirement)
        ml_pred = None
        
        if self.model:
            # Use enhanced model prediction directly with klines
            raw_pred = self.model.predict(klines)
            
            if raw_pred and raw_pred.get("signal") and raw_pred["confidence"] >= MIN_CONFIDENCE_THRESHOLD:
                latest = base_df.iloc[-1]
                
                # Additional quality checks for maximum win rate
                trend_strength = abs(latest["ema_21"] - latest["ema_55"]) / latest["close"] * 100
                volume_ratio = float(latest["volume"] / latest["volume_ma"]) if float(latest["volume_ma"]) > 0 else 1.0
                
                # Quality gate: require decent trend or volume
                if trend_strength > 0.2 or volume_ratio > 1.1:
                    ml_pred = {
                        **raw_pred,
                        "trend_strength": trend_strength,
                        "volume_ratio": volume_ratio,
                    }
        
        # If no ML model or ML didn't produce high-confidence signal, use fallback
        if not ml_pred:
            # Use base strategy but require 75% confidence
            signal, confidence, details = super().generate_signal(klines)
            if signal and confidence >= MIN_CONFIDENCE_THRESHOLD:
                logger.info(f"[ML-Fallback] {self.symbol} {signal} at {confidence:.1f}% confidence")
                return signal, confidence, details
            return None, 0, None
        
        # Extract prediction data
        signal = ml_pred["signal"]
        confidence = int(round(ml_pred["confidence"]))
        
        # Final 75% confidence check
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            logger.debug(f"[ML] {self.symbol} signal below threshold: {confidence}% < {MIN_CONFIDENCE_THRESHOLD}%")
            return None, 0, None
        
        # Use risk levels from model or calculate
        if ml_pred.get("stop_loss"):
            stop_loss = ml_pred["stop_loss"]
            take_profit_1 = ml_pred["take_profit_1"]
            take_profit_2 = ml_pred["take_profit_2"]
        else:
            latest = base_df.iloc[-1]
            stop_loss, take_profit_1, take_profit_2 = self._risk_levels_from_atr(
                signal=signal, 
                entry_price=float(latest["close"]), 
                atr_value=float(latest["atr"])
            )
        
        trend_strength = ml_pred.get("trend_strength", 0)
        volume_ratio = ml_pred.get("volume_ratio", 1.0)
        probas = ml_pred.get("probas", {})
        
        logger.info(
            f"[ML-HighConf] {self.symbol} {signal} CONF={confidence}% (target: 75%+) "
            f"RF={probas.get('long', 0):.1f}/{probas.get('short', 0):.1f}/{probas.get('neutral', 0):.1f} "
            f"trend={trend_strength:.2f} vol={volume_ratio:.2f}x"
        )

        return signal, confidence, {
            "stop_loss": round(stop_loss, 2),
            "take_profit_1": round(take_profit_1, 2),
            "take_profit_2": round(take_profit_2, 2),
            "entry_price": ml_pred.get("entry_price") or round(float(base_df.iloc[-1]["close"]), 2),
            "atr": ml_pred.get("atr", round(float(base_df.iloc[-1]["atr"]), 2)),
            "rsi": round(float(base_df.iloc[-1]["rsi"]), 2),
            "ema_21": round(float(base_df.iloc[-1]["ema_21"]), 2),
            "ema_55": round(float(base_df.iloc[-1]["ema_55"]), 2),
            "volume_ratio": round(volume_ratio, 2),
            "trend_strength": round(float(trend_strength), 2),
            "ml_confidence": confidence,
            "ml_probas": probas,
            "model_agreement": ml_pred.get("model_agreement", False),
        }
