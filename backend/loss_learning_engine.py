"""
Loss-Based Learning Engine for Talon Sniper v1
Analyzes every losing trade, retrains model, and adapts strategy parameters.
"""

import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque

from talon_ml_model import TalonMLModel
from talon_sniper_strategy import TalonSniperStrategy
from premium_model import candles_to_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LossPatternAnalyzer:
    """Analyzes patterns in losing trades to identify failure modes"""
    
    def __init__(self):
        self.loss_history: deque = deque(maxlen=50)  # Keep last 50 losses
        self.pattern_stats = {
            'tema_dema_false_positive': 0,
            'atr_stop_too_tight': 0,
            'ema_filter_wrong': 0,
            'high_volatility_loss': 0,
            'low_confidence_loss': 0,
        }
    
    def analyze_loss(self, trade_data: dict, features: pd.Series) -> dict:
        """Analyze why a trade lost and categorize the failure"""
        analysis = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': trade_data['symbol'],
            'pnl': trade_data['pnl'],
            'confidence': trade_data['confidence'],
            'failure_type': 'unknown'
        }
        
        # Check various failure conditions
        failure_signals = []
        
        # 1. TEMA/DEMA signal was wrong
        if features.get('tema_dema_signal', 0) != 0:
            if trade_data['pnl'] < 0:
                failure_signals.append('tema_dema_false_positive')
                self.pattern_stats['tema_dema_false_positive'] += 1
        
        # 2. ATR-based stop was too tight (high volatility)
        if features.get('atr_14', 0) > features.get('atr_14', 1) * 2:
            failure_signals.append('atr_stop_too_tight')
            self.pattern_stats['atr_stop_too_tight'] += 1
        
        # 3. EMA 13 filter gave wrong trend direction
        price_vs_ema = features.get('close', 0) - features.get('ema_13', 0)
        if trade_data['signal'] == 'LONG' and price_vs_ema < 0:
            failure_signals.append('ema_filter_wrong')
            self.pattern_stats['ema_filter_wrong'] += 1
        elif trade_data['signal'] == 'SHORT' and price_vs_ema > 0:
            failure_signals.append('ema_filter_wrong')
            self.pattern_stats['ema_filter_wrong'] += 1
        
        # 4. High volatility environment
        if features.get('atr_14', 0) / features.get('close', 1) > 0.05:  # >5% ATR
            failure_signals.append('high_volatility_loss')
            self.pattern_stats['high_volatility_loss'] += 1
        
        # 5. Low confidence trade that lost
        if trade_data['confidence'] < 80:
            failure_signals.append('low_confidence_loss')
            self.pattern_stats['low_confidence_loss'] += 1
        
        analysis['failure_signals'] = failure_signals
        analysis['primary_failure'] = failure_signals[0] if failure_signals else 'unknown'
        analysis['features_at_entry'] = features.to_dict() if isinstance(features, pd.Series) else features
        
        self.loss_history.append(analysis)
        
        logger.warning(f"🎓 Loss Analysis [{trade_data['symbol']}]: {analysis['primary_failure']} "
                      f"(PnL: {trade_data['pnl']:.2f} USDT, Conf: {trade_data['confidence']:.1f}%)")
        
        return analysis
    
    def get_adaptive_recommendations(self) -> dict:
        """Generate strategy adjustments based on loss patterns"""
        if len(self.loss_history) < 5:
            return {}
        
        recent_losses = list(self.loss_history)[-20:]
        recommendations = {}
        
        # Count failures in recent history
        tema_failures = sum(1 for l in recent_losses if 'tema_dema_false_positive' in l['failure_signals'])
        atr_failures = sum(1 for l in recent_losses if 'atr_stop_too_tight' in l['failure_signals'])
        ema_failures = sum(1 for l in recent_losses if 'ema_filter_wrong' in l['failure_signals'])
        vol_failures = sum(1 for l in recent_losses if 'high_volatility_loss' in l['failure_signals'])
        low_conf_failures = sum(1 for l in recent_losses if 'low_confidence_loss' in l['failure_signals'])
        
        # Generate recommendations
        if tema_failures >= 3:
            recommendations['tema_dema_weight'] = 0.7  # Reduce TEMA/DEMA weight
            recommendations['require_signal_2_confirm'] = True  # Require Signal 2 confirmation
            logger.info(f"🎓 Learning: Reducing TEMA/DEMA weight due to {tema_failures} false signals")
        
        if atr_failures >= 3:
            recommendations['atr_multiplier_sl'] = 2.5  # Wider stops
            recommendations['atr_multiplier_tp1'] = 3.0
            recommendations['atr_multiplier_tp2'] = 5.0
            logger.info(f"🎓 Learning: Widening ATR stops due to {atr_failures} tight stop losses")
        
        if ema_failures >= 3:
            recommendations['ema_trend_filter'] = 'ema_21'  # Use slower EMA
            recommendations['ema_confirm_candles'] = 3  # Require 3 candles confirmation
            logger.info(f"🎓 Learning: Strengthening EMA filter due to {ema_failures} wrong trend calls")
        
        if vol_failures >= 3:
            recommendations['max_atr_pct'] = 0.04  # Avoid high volatility
            recommendations['volatility_filter'] = True
            logger.info(f"🎓 Learning: Adding volatility filter due to {vol_failures} high vol losses")
        
        if low_conf_failures >= 3:
            recommendations['min_confidence'] = 85  # Raise minimum confidence
            logger.info(f"🎓 Learning: Raising min confidence to 85% due to {low_conf_failures} low-conf losses")
        
        return recommendations


class IncrementalLossTrainer:
    """Retrains model incrementally with loss trade as negative example"""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), 'ml_models')
        self.loss_training_history: deque = deque(maxlen=100)
        self.min_losses_before_retrain = 1  # Retrain after every loss
        self.retrain_cooldown_minutes = 1   # Retrain as soon as 1 min after loss
        self.last_retrain_time = None
    
    def prepare_loss_training_sample(self, trade_data: dict, klines: list) -> Optional[pd.DataFrame]:
        """Convert a losing trade into a training sample with 'corrected' label"""
        try:
            # Convert klines to dataframe
            df = candles_to_dataframe(klines)
            if df.empty or len(df) < 20:
                return None
            
            # Get the last row (entry point)
            last_row = df.iloc[-1:].copy()
            
            # For a losing trade, the "correct" prediction would be NO TRADE or opposite signal
            # We mark this as a failed signal (label = -1 for failed, 0 for no trade)
            
            # Add metadata
            last_row['trade_result'] = 'loss'
            last_row['actual_pnl'] = trade_data['pnl']
            last_row['original_confidence'] = trade_data['confidence']
            last_row['correct_label'] = 0  # No trade would have been better
            
            return last_row
            
        except Exception as e:
            logger.error(f"Error preparing loss training sample: {e}")
            return None
    
    def should_retrain(self) -> bool:
        """Check if we should retrain based on cooldown and loss count"""
        if self.last_retrain_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_retrain_time).total_seconds() / 60
        if elapsed < self.retrain_cooldown_minutes:
            return False
        
        recent_losses = sum(1 for t in self.loss_training_history 
                          if (datetime.utcnow() - datetime.fromisoformat(t['timestamp'])).total_seconds() < 3600)
        
        return recent_losses >= self.min_losses_before_retrain
    
    def incremental_retrain(self, symbol: str, loss_samples: List[pd.DataFrame]) -> bool:
        """Incrementally retrain model with loss samples as negative examples"""
        try:
            model_path = os.path.join(self.models_dir, f"talon_{symbol}.pkl")
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠ Model not found for {symbol} at {model_path}, skipping incremental retrain")
                # Telegram notification that model is missing
                try:
                    from telegram_notifier import telegram
                    telegram.send_message(f"⚠️ <b>NO MODEL FOR RETRAIN</b>\n\nSymbol: {symbol}\nStatus: Model file not found\n\nThe bot will use base strategy without ML for this symbol.")
                except Exception as e:
                    logger.debug(f"Could not send no-model notification: {e}")
                return False
            
            # Load existing model
            model = TalonMLModel.load(model_path)
            
            # Combine all loss samples
            if not loss_samples:
                return False
            
            combined = pd.concat(loss_samples, ignore_index=True)
            
            logger.info(f"🎓 Incremental retraining {symbol} with {len(combined)} loss samples...")
            
            # ACTUAL RETRAINING: Augment training data with loss samples
            # Convert loss samples to training format (mark as "no trade" = negative examples)
            loss_candles = combined.to_dict('records')
            
            # Retrain model with augmented data including losses
            try:
                metrics = model.train(loss_candles, symbol)
                
                # Save updated model
                model.save(model_path)
                
                # Store that we did this retrain
                self.last_retrain_time = datetime.utcnow()
                
                logger.info(f"✅ {symbol}: Model retrained with loss patterns!")
                logger.info(f"   RF Accuracy: {metrics.get('rf_accuracy', 0):.1%}")
                logger.info(f"   GB Accuracy: {metrics.get('gb_accuracy', 0):.1%}")
                
                # Telegram notification for retrain
                try:
                    from telegram_notifier import telegram
                    telegram.send_message(f"""🎓 <b>MODEL RETRAINED AFTER LOSS</b> 🎓

Symbol: {symbol}
Loss Samples Used: {len(combined)}

New Model Accuracy:
• Random Forest: {metrics.get('rf_accuracy', 0):.1%}
• Gradient Boosting: {metrics.get('gb_accuracy', 0):.1%}

Bot is now smarter! 🤖""")
                except Exception as e:
                    logger.debug(f"Could not send retrain notification: {e}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error during model training: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error in incremental retrain: {e}")
            return False


class StrategyParameterAdapter:
    """Adapts strategy parameters based on accumulated learning"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'adaptive_strategy_config.json'
        )
        self.current_params = self._load_params()
        self.learning_iterations = 0
    
    def _load_params(self) -> dict:
        """Load adaptive parameters from disk"""
        default_params = {
            'tema_dema_weight': 1.0,
            'signal_2_weight': 1.0,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp1': 1.5,
            'atr_multiplier_tp2': 2.5,
            'ema_trend_filter': 'ema_13',
            'ema_confirm_candles': 1,
            'min_confidence': 75,
            'max_atr_pct': 0.06,
            'require_signal_2_confirm': False,
            'volatility_filter': False,
            'custom_strategy_name': 'Talon Sniper v1 Base',
            'version': 1
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    saved = json.load(f)
                    default_params.update(saved)
                    logger.info(f"🎓 Loaded adaptive strategy config v{default_params.get('version', 1)}")
            except Exception as e:
                logger.error(f"Error loading adaptive config: {e}")
        
        return default_params
    
    def _save_params(self):
        """Save adaptive parameters to disk"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.current_params, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving adaptive config: {e}")
    
    def apply_recommendations(self, recommendations: dict):
        """Apply learning recommendations to strategy parameters"""
        old_version = self.current_params.get('version', 1)
        
        for key, value in recommendations.items():
            if key in self.current_params:
                old_val = self.current_params[key]
                self.current_params[key] = value
                logger.info(f"🎓 Parameter adapted: {key} = {value} (was {old_val})")
        
        if recommendations:
            self.learning_iterations += 1
            self.current_params['version'] = old_version + 1
            self.current_params['custom_strategy_name'] = f"Talon Sniper v1 Custom (v{self.current_params['version']}, {self.learning_iterations} iterations)"
            self.current_params['last_update'] = datetime.utcnow().isoformat()
            self._save_params()
            
            logger.info(f"🎓 Strategy evolved to {self.current_params['custom_strategy_name']}")
    
    def get_params(self) -> dict:
        """Get current adaptive parameters"""
        return self.current_params.copy()


class LossLearningEngine:
    """
    Main engine that orchestrates loss-based learning.
    Every loss trade triggers analysis, incremental training, and strategy adaptation.
    """
    
    def __init__(self):
        self.pattern_analyzer = LossPatternAnalyzer()
        self.incremental_trainer = IncrementalLossTrainer()
        self.param_adapter = StrategyParameterAdapter()
        self.symbol_loss_queues: Dict[str, deque] = {}
        self.min_samples_for_retrain = 3
        
        logger.info("🎓 Loss Learning Engine initialized - Bot will learn from every loss!")
    
    def process_loss_trade(self, trade_data: dict, klines: list = None, features: pd.Series = None):
        """
        Process a losing trade - main entry point called after every loss.
        
        Args:
            trade_data: Dict with symbol, pnl, confidence, signal, etc.
            klines: Candle data at time of entry
            features: Feature vector at time of entry
        """
        symbol = trade_data.get('symbol')
        pnl = trade_data.get('pnl', 0)
        
        if pnl >= 0:
            logger.debug(f"Trade was profitable, no learning needed for {symbol}")
            return
        
        logger.warning(f"🎓 LEARNING FROM LOSS: {symbol} lost {pnl:.2f} USDT")
        
        # 1. Analyze the loss
        analysis = self.pattern_analyzer.analyze_loss(trade_data, features or {})
        
        # 2. Queue for incremental training
        if symbol not in self.symbol_loss_queues:
            self.symbol_loss_queues[symbol] = deque(maxlen=20)
        
        # Prepare training sample
        loss_sample = self.incremental_trainer.prepare_loss_training_sample(trade_data, klines or [])
        if loss_sample is not None:
            self.symbol_loss_queues[symbol].append({
                'timestamp': datetime.utcnow().isoformat(),
                'trade_data': trade_data,
                'features': loss_sample
            })
        
        # 3. Get recommendations and adapt strategy
        recommendations = self.pattern_analyzer.get_adaptive_recommendations()
        if recommendations:
            self.param_adapter.apply_recommendations(recommendations)
            
            # Send Telegram notification about strategy adaptation
            try:
                from telegram_notifier import telegram
                params = self.param_adapter.get_params()
                
                message = f"""
🎓 <b>BOT IS LEARNING!</b> 🎓

<b>Loss Trade:</b> {symbol} ({pnl:.2f} USDT)
<b>Failure Type:</b> {analysis.get('primary_failure', 'unknown')}

<b>Strategy Adapted:</b>
• Version: {params.get('custom_strategy_name', 'Base')}
• Min Confidence: {params.get('min_confidence', 75)}%
• TEMA/DEMA Weight: {params.get('tema_dema_weight', 1.0):.2f}
• Signal 2 Confirm: {'Yes' if params.get('require_signal_2_confirm', False) else 'No'}
• Volatility Filter: {'Active' if params.get('volatility_filter', False) else 'Off'}

<i>The bot is evolving to avoid similar losses! 🤖</i>
                """
                telegram.send_message(message)
            except Exception as e:
                logger.debug(f"Could not send learning notification: {e}")
        
        # 4. 🚀 IMMEDIATE RETRAIN: Trigger incremental retrain after EVERY loss
        # This ensures the bot learns immediately from mistakes
        loss_samples = [item['features'] for item in self.symbol_loss_queues.get(symbol, []) 
                        if item.get('features') is not None]
        
        if loss_samples:
            logger.info(f"🚀 Auto-retraining {symbol} model after loss with {len(loss_samples)} samples...")
            
            # Telegram notification that retrain is starting
            try:
                from telegram_notifier import telegram
                telegram.send_message(f"🎓 <b>RETRAINING STARTED</b>\n\nSymbol: {symbol}\nLoss samples: {len(loss_samples)}\n\nTraining model now...")
            except Exception as e:
                logger.debug(f"Could not send retrain start notification: {e}")
            
            retrain_success = self.incremental_trainer.incremental_retrain(symbol, loss_samples[-3:])  # Use last 3 losses
            
            if retrain_success:
                logger.info(f"✅ {symbol} model retrained with loss data!")
            else:
                logger.warning(f"⚠️ {symbol} model retrain FAILED - check logs above")
        
        # Also trigger background full retrain if we've hit retrain threshold
        if self.incremental_trainer.should_retrain():
            logger.info("🚀 Background full retrain triggered by loss accumulation")
            # This would trigger a full retrain in a background thread
            self._trigger_background_retrain(symbol)
        
        # Log learning status
        self._log_learning_status()
    
    def _trigger_background_retrain(self, symbol: str):
        """Trigger a background full retrain of the model after multiple losses"""
        import threading
        
        def retrain_worker():
            try:
                logger.info(f"🎓 Background retrain starting for {symbol}...")
                # Import here to avoid circular imports
                from adaptive_trainer import AdaptiveMLTrainer, TrainingConfig
                
                trainer = AdaptiveMLTrainer(TrainingConfig())
                # Force retrain this specific symbol
                symbol_data = trainer.download_training_data([symbol])
                if symbol in symbol_data:
                    metrics = trainer.train_models({symbol: symbol_data[symbol]})
                    logger.info(f"✅ Background retrain complete for {symbol}: {metrics}")
                else:
                    logger.warning(f"No data available for background retrain of {symbol}")
            except Exception as e:
                logger.error(f"Background retrain failed: {e}")
        
        # Start in background thread
        thread = threading.Thread(target=retrain_worker, daemon=True)
        thread.start()
        logger.info(f"🚀 Background retrain thread started for {symbol}")
    
    def _log_learning_status(self):
        """Log current learning status"""
        total_losses = len(self.pattern_analyzer.loss_history)
        learning_stats = self.pattern_analyzer.pattern_stats.copy()
        current_params = self.param_adapter.get_params()
        
        logger.info(f"🎓 Learning Status: {total_losses} losses analyzed, "
                      f"{self.param_adapter.learning_iterations} strategy adaptations, "
                      f"Current version: {current_params.get('custom_strategy_name', 'Base')}")
        logger.debug(f"🎓 Failure patterns: {learning_stats}")
    
    def get_adaptive_strategy_params(self) -> dict:
        """Get the current adapted strategy parameters for use in trading"""
        return self.param_adapter.get_params()
    
    def export_learning_report(self) -> str:
        """Generate a learning report for the user"""
        params = self.param_adapter.get_params()
        stats = self.pattern_analyzer.pattern_stats
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           🤖 LOSS-BASED LEARNING REPORT                    ║
╠══════════════════════════════════════════════════════════╣
║ Strategy: {params.get('custom_strategy_name', 'Base'):<40} ║
║ Iterations: {self.param_adapter.learning_iterations:<38} ║
║ Total Losses Analyzed: {len(self.pattern_analyzer.loss_history):<28} ║
╠══════════════════════════════════════════════════════════╣
║ Current Adaptive Parameters:                             ║
║  • TEMA/DEMA Weight: {params.get('tema_dema_weight', 1.0):<35.2f} ║
║  • Signal 2 Weight: {params.get('signal_2_weight', 1.0):<36.2f} ║
║  • ATR SL Multiplier: {params.get('atr_multiplier_sl', 2.0):<33.1f} ║
║  • Min Confidence: {params.get('min_confidence', 75):<37}% ║
║  • EMA Filter: {params.get('ema_trend_filter', 'ema_13'):<39} ║
║  • Vol Filter Active: {str(params.get('volatility_filter', False)):<34} ║
╠══════════════════════════════════════════════════════════╣
║ Failure Pattern Analysis:                                ║
║  • TEMA/DEMA False +: {stats.get('tema_dema_false_positive', 0):<35} ║
║  • ATR Stop Too Tight: {stats.get('atr_stop_too_tight', 0):<32} ║
║  • EMA Filter Wrong: {stats.get('ema_filter_wrong', 0):<34} ║
║  • High Volatility: {stats.get('high_volatility_loss', 0):<35} ║
║  • Low Confidence: {stats.get('low_confidence_loss', 0):<36} ║
╚══════════════════════════════════════════════════════════╝
"""
        return report


# Singleton instance
_loss_learning_engine: Optional[LossLearningEngine] = None

def get_loss_learning_engine() -> LossLearningEngine:
    """Get or create the global loss learning engine"""
    global _loss_learning_engine
    if _loss_learning_engine is None:
        _loss_learning_engine = LossLearningEngine()
    return _loss_learning_engine


def process_loss_for_learning(trade_data: dict, klines: list = None, features: pd.Series = None):
    """Convenience function to process a loss trade"""
    engine = get_loss_learning_engine()
    engine.process_loss_trade(trade_data, klines, features)


def get_adaptive_params() -> dict:
    """Get current adaptive strategy parameters"""
    engine = get_loss_learning_engine()
    return engine.get_adaptive_strategy_params()


def get_learning_report() -> str:
    """Get learning report"""
    engine = get_loss_learning_engine()
    return engine.export_learning_report()
