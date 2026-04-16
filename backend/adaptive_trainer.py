"""
Adaptive Hourly ML Training for Talon Sniper v1 Strategy
Targets: 75%+ confidence, 8-9 winning trades per day with Heikin Ashi
"""

import os
import json
import logging
import threading
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Suppress warnings from technical analysis library
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")

from data_downloader import DataDownloader
from talon_ml_model import TalonMLModel, predict_with_talon_model
from premium_model import candles_to_dataframe, build_symbol_training_frame
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks trade performance for adaptive learning"""
    
    def __init__(self):
        self.records: List[Dict] = []
        self.last_trade_date = None
        self.trades_today = 0
    
    def record_trade(self, symbol: str, signal: str, confidence: float, 
                     entry_price: float = None, exit_price: float = None,
                     pnl: float = None, klines: list = None, exit_reason: str = None):
        """Record a trade result for tracking"""
        trade = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'success': pnl > 0 if pnl is not None else None,
        }
        self.records.append(trade)
        
        # Update daily counter
        today = datetime.utcnow().date()
        if self.last_trade_date != today:
            self.last_trade_date = today
            self.trades_today = 1
        else:
            self.trades_today += 1
        
        # Keep only last 200 records
        if len(self.records) > 200:
            self.records = self.records[-200:]
    
    def rolling_win_rate(self, lookback: int = 10) -> float:
        """Calculate win rate over last N trades"""
        if len(self.records) < lookback:
            return 0.0
        
        recent = self.records[-lookback:]
        wins = sum(1 for r in recent if r.get('success') is True)
        return wins / lookback
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.records:
            return {'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
        
        total = len(self.records)
        wins = sum(1 for r in self.records if r.get('success') is True)
        losses = sum(1 for r in self.records if r.get('success') is False)
        win_rate = wins / total if total > 0 else 0.0
        
        return {
            'total': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'trades_today': self.trades_today,
        }


@dataclass
class TrainingConfig:
    """Configuration optimized for 8-9 daily wins with 75%+ confidence"""
    target_daily_trades: int = 10  # 9-10 total across ALL symbols
    target_win_rate: float = 0.70  # 70% win rate target
    min_confidence: float = 75.0   # Increased from default
    max_confidence: float = 97.0
    training_interval_hours: int = 1
    lookback_days: int = 60        # Use 60 days of history
    min_samples_per_symbol: int = 500
    
    # Confidence boost factors for high-quality signals
    strong_trend_boost: float = 5.0
    volume_surge_boost: float = 3.0
    confluence_boost: float = 4.0


class AdaptiveMLTrainer:
    """
    Adaptive trainer that adjusts strategy hourly to maximize high-confidence wins
    """
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.data_dir = os.path.join(os.path.dirname(__file__), 'historical_data')
        self.models_dir = os.path.join(os.path.dirname(__file__), 'ml_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.symbols = Config.TOP_20_SYMBOLS[:10]  # Focus on top 10 for quality
        self.models: Dict[str, TalonMLModel] = {}
        self.last_training_time: Optional[datetime] = None
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
        self.performance_history: List[dict] = []
        
        # Performance tracker for win/loss recording
        self.tracker = PerformanceTracker()
        
        # Adaptive parameters adjusted based on performance
        self.adaptive_params = {
            'confidence_threshold': 75.0,
            'trend_strength_min': 0.3,  # Minimum trend strength
            'volume_threshold': 1.2,     # Volume surge threshold
            'max_positions_per_hour': 1,  # Limit trades per hour per symbol
        }
    
    def download_training_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Download fresh data for all symbols"""
        if symbols is None:
            symbols = self.symbols
        
        downloader = DataDownloader()
        symbol_data = {}
        
        logger.info(f"Downloading {self.config.lookback_days} days of data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            try:
                # Download fresh data - set symbol on client first
                downloader.client.symbol = symbol
                candles = downloader.client.get_klines(
                    interval='1h',
                    limit=self.config.lookback_days * 24
                )
                
                if candles and len(candles) >= self.config.min_samples_per_symbol:
                    df = build_symbol_training_frame(candles, symbol)
                    if not df.empty and len(df) >= 100:
                        symbol_data[symbol] = df
                        logger.info(f"✓ {symbol}: {len(df)} training samples")
                    else:
                        logger.warning(f"⚠ {symbol}: Insufficient data after feature engineering")
                else:
                    logger.warning(f"⚠ {symbol}: Downloaded {len(candles) if candles else 0} candles")
                    
                time.sleep(0.5)  # Rate limit
                
            except Exception as e:
                logger.error(f"✗ {symbol}: Error downloading - {e}")
                continue
        
        return symbol_data
    
    def train_models(self, symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Train enhanced ML models for each symbol with RandomForest+GB"""
        model_paths = {}
        
        for symbol, df in symbol_data.items():
            try:
                logger.info(f"\nTraining Talon model for {symbol}...")
                
                # Convert DataFrame to list of dicts for the model
                candles = df.to_dict('records')
                
                # Create and train Talon model
                model = TalonMLModel()
                metrics = model.train(candles, symbol)
                
                # Save model
                model_path = os.path.join(self.models_dir, f"talon_{symbol}.pkl")
                model.save(model_path)
                
                # Cache model
                self.models[symbol] = model
                model_paths[symbol] = model_path
                
                # Log training metrics
                logger.info(f"  Samples: {metrics['samples']}")
                logger.info(f"  RF Accuracy: {metrics['rf_accuracy']:.1%}")
                logger.info(f"  GB Accuracy: {metrics['gb_accuracy']:.1%}")
                logger.info(f"  Features: {metrics['features']}")
                logger.info(f"  ✓ Model saved to {model_path}")
                
            except Exception as e:
                logger.error(f"  ✗ Training failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        return model_paths
    
    def adjust_confidence_for_targets(self, base_confidence: float, 
                                       trend_strength: float,
                                       volume_ratio: float,
                                       recent_trades: int) -> float:
        """
        Boost confidence based on signal quality to hit 75%+ target
        Also penalize if we're hitting trade frequency limits
        """
        confidence = base_confidence
        
        # Boost for strong trend
        if trend_strength > self.adaptive_params['trend_strength_min']:
            confidence += self.config.strong_trend_boost
        
        # Boost for volume surge
        if volume_ratio > self.adaptive_params['volume_threshold']:
            confidence += self.config.volume_surge_boost
        
        # Penalize if we've traded too recently on this symbol
        if recent_trades >= self.adaptive_params['max_positions_per_hour']:
            confidence -= 15.0
        
        # Cap at max
        confidence = min(confidence, self.config.max_confidence)
        
        return confidence
    
    def should_take_signal(self, prediction: dict, trend_strength: float, 
                          volume_ratio: float, recent_trades: int) -> tuple:
        """
        Determine if signal should be taken based on 75% confidence target
        Returns: (should_trade: bool, adjusted_confidence: float)
        """
        base_conf = prediction['confidence']
        
        # Apply adaptive adjustments
        adjusted_conf = self.adjust_confidence_for_targets(
            base_conf, trend_strength, volume_ratio, recent_trades
        )
        
        # Require 75% minimum confidence
        should_trade = (
            prediction['signal'] is not None and
            adjusted_conf >= self.config.min_confidence and
            prediction.get('expected_return', 0) > 0.001  # Positive expected return
        )
        
        return should_trade, adjusted_conf
    
    def log_performance(self, trade_result: dict):
        """Log trade results for adaptive adjustments"""
        self.performance_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': trade_result.get('symbol'),
            'signal': trade_result.get('signal'),
            'confidence': trade_result.get('confidence'),
            'pnl': trade_result.get('pnl'),
            'success': trade_result.get('pnl', 0) > 0
        })
        
        # Keep only last 100 trades
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Adjust parameters based on recent performance
        self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adjust trading parameters based on recent performance"""
        if len(self.performance_history) < 10:
            return
        
        recent = self.performance_history[-20:]
        win_rate = sum(1 for t in recent if t['success']) / len(recent)
        avg_confidence = sum(t['confidence'] for t in recent) / len(recent)
        
        # Adjust confidence threshold to maintain 70%+ win rate
        if win_rate < 0.65 and self.adaptive_params['confidence_threshold'] < 85:
            self.adaptive_params['confidence_threshold'] += 2.0
            logger.info(f"Adaptive: Increased confidence threshold to {self.adaptive_params['confidence_threshold']}%")
        elif win_rate > 0.80 and self.adaptive_params['confidence_threshold'] > 70:
            self.adaptive_params['confidence_threshold'] -= 1.0
            logger.info(f"Adaptive: Decreased confidence threshold to {self.adaptive_params['confidence_threshold']}%")
        
        # Log current stats
        daily_trades = len([t for t in self.performance_history 
                           if datetime.fromisoformat(t['timestamp']) > datetime.utcnow() - timedelta(days=1)])
        logger.info(f"Performance: Win Rate={win_rate:.1%}, Avg Conf={avg_confidence:.1f}%, Daily Trades={daily_trades}")
    
    def run_training_cycle(self, force_retrain: bool = False):
        """Complete training cycle: download data, train models, update
        
        Args:
            force_retrain: If True, retrain even if models exist. If False, only train missing models.
        """
        logger.info("\n" + "="*60)
        logger.info("Starting Adaptive ML Training Cycle")
        logger.info(f"Target: {self.config.target_daily_trades} daily wins at 75%+ confidence")
        logger.info("="*60)
        
        self.is_training = True
        start_time = time.time()
        
        try:
            # Step 1: Check for existing models first
            existing_models = 0
            missing_symbols = []
            
            for symbol in self.symbols:
                model_path = os.path.join(self.models_dir, f"talon_{symbol}.pkl")
                if os.path.exists(model_path):
                    try:
                        model = TalonMLModel.load(model_path)
                        self.models[symbol] = model
                        logger.info(f"✓ {symbol}: Loaded existing model (trained: {model.training_date.strftime('%Y-%m-%d %H:%M')})")
                        existing_models += 1
                    except Exception as e:
                        logger.warning(f"⚠ {symbol}: Failed to load existing model: {e}")
                        missing_symbols.append(symbol)
                else:
                    missing_symbols.append(symbol)
            
            if existing_models > 0 and not force_retrain:
                logger.info(f"\n✓ {existing_models} models already loaded. Skipping training for existing models.")
                if not missing_symbols:
                    logger.info("All models ready! Training complete.")
                    self.is_training = False
                    return True
            
            # Step 2: Download fresh data only for missing models
            if missing_symbols:
                logger.info(f"\nTraining {len(missing_symbols)} new models: {missing_symbols}")
                symbol_data = self.download_training_data()
                
                if not symbol_data:
                    logger.error("No training data available!")
                    return False
                
                # Only train missing symbols
                filtered_data = {k: v for k, v in symbol_data.items() if k in missing_symbols}
                model_paths = self.train_models(filtered_data)
            else:
                model_paths = {}
            
            # Step 3: Save training metadata
            metadata = {
                'trained_at': datetime.utcnow().isoformat(),
                'symbols': self.symbols,
                'model_paths': model_paths,
                'adaptive_params': self.adaptive_params,
                'existing_models': existing_models
            }
            
            with open(os.path.join(self.models_dir, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_training_time = datetime.utcnow()
            
            elapsed = time.time() - start_time
            if model_paths:
                logger.info(f"\n✓ Training complete in {elapsed:.1f}s")
                logger.info(f"  Models trained for {len(model_paths)} symbols")
            else:
                logger.info(f"\n✓ All {existing_models} models ready and loaded in {elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Training cycle failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            self.is_training = False
    
    def schedule_hourly_training(self):
        """Start background thread for hourly retraining"""
        def training_loop():
            # First run - check and load existing models (don't force retrain)
            try:
                self.run_training_cycle(force_retrain=False)
            except Exception as e:
                logger.error(f"Initial training error: {e}")
            
            while True:
                # Sleep until next training
                time.sleep(self.config.training_interval_hours * 3600)
                
                try:
                    # Only retrain missing models, not all models
                    self.run_training_cycle(force_retrain=False)
                    logger.info(f"Next training check in {self.config.training_interval_hours} hours...")
                except Exception as e:
                    logger.error(f"Training error: {e}")
        
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        logger.info(f"✓ Scheduled hourly training every {self.config.training_interval_hours} hours")
    
    def get_model(self, symbol: str) -> Optional[TalonMLModel]:
        """Get cached or load Talon model for symbol"""
        if symbol in self.models:
            return self.models[symbol]
        
        # Try to load Talon model from disk
        model_path = os.path.join(self.models_dir, f"talon_{symbol}.pkl")
        if os.path.exists(model_path):
            try:
                self.models[symbol] = TalonMLModel.load(model_path)
                return self.models[symbol]
            except Exception as e:
                logger.error(f"Failed to load Talon model for {symbol}: {e}")
        
        return None
    
    def predict_with_high_confidence(self, symbol: str, klines: list) -> Optional[dict]:
        """
        Get prediction with 75%+ confidence requirement using Talon model
        Returns None if confidence too low
        """
        model = self.get_model(symbol)
        if not model:
            logger.warning(f"No Talon model available for {symbol}")
            return None
        
        # Predict with Talon model
        pred = model.predict(klines)
        if not pred:
            return None
        
        # Check trend strength and volume from klines
        df = candles_to_dataframe(klines)
        if len(df) < 55:
            return None
        
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate trend strength
        from ta.trend import EMAIndicator
        ema_21 = EMAIndicator(close=df['close'], window=21).ema_indicator()
        ema_55 = EMAIndicator(close=df['close'], window=55).ema_indicator()
        trend_strength = abs(ema_21.iloc[-1] - ema_55.iloc[-1]) / df['close'].iloc[-1] * 100
        
        # Calculate volume ratio
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
        
        # Apply adaptive confidence adjustment
        should_trade, adjusted_conf = self.should_take_signal(
            pred, trend_strength, volume_ratio, 0  # recent_trades would need to be tracked externally
        )
        
        if not should_trade:
            return None
        
        # Return enriched prediction
        return {
            **pred,
            'adjusted_confidence': adjusted_conf,
            'trend_strength': trend_strength,
            'volume_ratio': volume_ratio,
            'quality_score': adjusted_conf * (1 + trend_strength/100) * min(volume_ratio, 2.0)
        }


# Global instance
adaptive_trainer = AdaptiveMLTrainer()


def initialize_ml_models():
    """Initial training run to create models before trading"""
    logger.info("="*60)
    logger.info("Initializing ML Models - First Training Run")
    logger.info("="*60)
    return adaptive_trainer.run_training_cycle()


if __name__ == '__main__':
    # Run initial training
    success = initialize_ml_models()
    if success:
        # Start hourly retraining
        adaptive_trainer.schedule_hourly_training()
        logger.info("\nAdaptive ML trainer running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
    else:
        logger.error("Failed to initialize models!")
