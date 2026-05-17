"""
Talon Sniper ML Model
Enhanced ML model trained on Talon Sniper v1 signals
"""

import json
import os
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered")

from talon_sniper_strategy import TalonSniperStrategy, ema, tema, dema, atr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TalonMLModel:
    """
    ML Model trained on Talon Sniper v1 features
    Uses both signal components
    """
    
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_date = None
        self.symbol_performance = {}
    
    def extract_talon_features(self, df: pd.DataFrame, ha_df: pd.DataFrame) -> pd.DataFrame:
        """Extract Talon Sniper specific features"""
        features = pd.DataFrame(index=ha_df.index)
        
        close = ha_df['close']
        high = ha_df['high']
        low = ha_df['low']
        
        # --- TEMA/DEMA Signal 1 Features ---
        # TEMA(1) components
        features['tema'] = tema(close, 1)
        
        # DEMA(8,5) components
        e_e1 = ema(close, 8)
        e_e2 = ema(e_e1, 5)
        features['dema'] = 2 * e_e1 - e_e2
        
        # TEMA/DEMA crossover
        features['tema_above_dema'] = (features['tema'] > features['dema']).astype(int)
        features['tema_dema_distance'] = (features['tema'] - features['dema']) / close
        
        # vh1/vl1 adaptive bounds
        avg_low_close = (low + close) / 2
        highest_avg = avg_low_close.rolling(window=5).max()
        features['vh1'] = ema(highest_avg, 5)
        
        avg_high_close = (high + close) / 2
        lowest_avg = avg_high_close.rolling(window=8).min()
        features['vl1'] = ema(lowest_avg, 8)
        
        # Signal line momentum
        signal_line = np.where(
            features['tema'] > features['dema'],
            features['vh1'],
            features['vl1']
        )
        features['signal_line'] = pd.Series(signal_line, index=features.index)
        features['signal_momentum'] = features['signal_line'].diff(1)
        features['signal_accel'] = features['signal_momentum'].diff(1)
        
        # Signal 1 trigger conditions
        features['s1_call'] = (
            (features['tema'] > features['dema']) & 
            (features['signal_line'] > low) & 
            (features['signal_momentum'] > features['signal_momentum'].shift(1))
        ).astype(int)
        
        features['s1_put'] = (
            (features['tema'] < features['dema']) & 
            (features['signal_line'] < high) & 
            (features['signal_momentum'].shift(1) > features['signal_momentum'])
        ).astype(int)
        
        # --- Signal 2 ATR Trend Features ---
        hl2 = (high + low) / 2
        atr_val = atr(ha_df, 14)
        
        features['atr'] = atr_val
        features['atr_normalized'] = atr_val / close
        
        # ATR bands
        factor = 1
        features['atr_up'] = hl2 - factor * atr_val
        features['atr_down'] = hl2 + factor * atr_val
        
        # Trend calculation (simplified)
        features['price_vs_up'] = (close - features['atr_up']) / close
        features['price_vs_down'] = (close - features['atr_down']) / close
        
        # --- Trend Filter EMA 13 ---
        ema13 = ema(close, 13)
        features['ema13'] = ema13
        features['ema13_slope'] = ema13.diff(2) / ema13
        features['price_vs_ema13'] = (close - ema13) / ema13
        
        # --- Price Action Features ---
        features['returns_1h'] = close.pct_change(1)
        features['returns_4h'] = close.pct_change(4)
        features['volatility_12h'] = features['returns_1h'].rolling(12).std()
        
        # --- Volume Features ---
        if 'volume' in df.columns:
            features['volume'] = df['volume']
            features['volume_sma20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = features['volume'] / features['volume_sma20']
        
        return features
    
    def create_labels(self, df: pd.DataFrame, ha_df: pd.DataFrame, lookahead: int = 6) -> pd.Series:
        """
        Create labels based on future returns
        1 = profitable LONG after fees
        -1 = profitable SHORT after fees
        0 = no trade / not profitable
        """
        fee = 0.0004  # 0.04% taker fee round trip
        
        future_returns = ha_df['close'].pct_change(lookahead).shift(-lookahead)
        future_high = ha_df['high'].rolling(lookahead).max().shift(-lookahead)
        future_low = ha_df['low'].rolling(lookahead).min().shift(-lookahead)
        
        entry_price = ha_df['close']
        
        # LONG potential: buy at close, sell at future high
        long_potential = (future_high - entry_price) / entry_price - fee
        # SHORT potential: sell at close, buy at future low
        short_potential = (entry_price - future_low) / entry_price - fee
        
        labels = pd.Series(0, index=df.index)
        
        # Label as 1 (LONG) if long potential > 0.4% and better than short
        long_signals = (long_potential > 0.004) & (long_potential > short_potential)
        labels[long_signals] = 1
        
        # Label as -1 (SHORT) if short potential > 0.4% and better than long
        short_signals = (short_potential > 0.004) & (short_potential > long_potential)
        labels[short_signals] = -1
        
        return labels
    
    def train(self, candles, symbol: str) -> Dict:
        """Train model on Talon Sniper features"""
        logger.info(f"Training Talon ML model for {symbol}...")
        
        # Convert candles to DataFrame
        if isinstance(candles, pd.DataFrame):
            candles = candles[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
        
        if not candles or len(candles) == 0:
            raise ValueError("No candles provided")
        
        if isinstance(candles[0], dict):
            needed = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            rows = [{col: c.get(col, 0) for col in needed} for c in candles]
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        if len(df) < 200:
            raise ValueError(f"Insufficient data for {symbol}: {len(df)} candles")
        
        # Convert to Heikin Ashi
        ha_df = heikin_ashi(df)
        
        # Extract Talon features
        features = self.extract_talon_features(df, ha_df)
        
        # Create labels
        labels = self.create_labels(df, ha_df)
        
        # Align and clean
        valid_idx = features.dropna().index.intersection(labels.dropna().index)
        X = features.loc[valid_idx].values
        y = labels.loc[valid_idx].values
        
        if len(X) < 100:
            raise ValueError(f"Too few valid samples: {len(X)}")
        
        # Sample weights - more recent = higher weight
        n = len(X)
        recent_cutoff = int(n * 0.80)
        sample_weights = np.ones(n)
        sample_weights[recent_cutoff:] = 2.0
        
        self.feature_names = list(features.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RandomForest
        logger.info(f"Training RandomForest on {len(X)} samples...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_scaled, y, sample_weight=sample_weights)
        
        # Train GradientBoosting
        logger.info(f"Training GradientBoosting on {len(X)} samples...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.gb_model.fit(X_scaled, y, sample_weight=sample_weights)
        
        # Evaluate
        rf_score = self.rf_model.score(X_scaled, y)
        gb_score = self.gb_model.score(X_scaled, y)
        
        logger.info(f"  RandomForest accuracy: {rf_score:.2%}")
        logger.info(f"  GradientBoosting accuracy: {gb_score:.2%}")
        
        self.is_trained = True
        self.training_date = datetime.utcnow()
        
        return {
            'symbol': symbol,
            'samples': len(X),
            'rf_accuracy': rf_score,
            'gb_accuracy': gb_score,
            'features': len(self.feature_names),
            'training_date': self.training_date.isoformat()
        }
    
    def predict(self, klines: List) -> Optional[Dict]:
        """Predict signal with confidence"""
        if not self.is_trained:
            return None
        
        try:
            # Convert klines
            if isinstance(klines[0], dict):
                needed = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                rows = [{col: k.get(col, 0) for col in needed} for k in klines]
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if len(df) < 20:
                return None
            
            # Convert to Heikin Ashi
            ha_df = heikin_ashi(df)
            
            # Extract features
            features = self.extract_talon_features(df, ha_df)
            features = features.dropna()
            
            if len(features) == 0:
                return None
            
            latest = features.iloc[-1].values.reshape(1, -1)
            latest_scaled = self.scaler.transform(latest)
            
            # Get predictions
            rf_proba = self.rf_model.predict_proba(latest_scaled)[0]
            gb_proba = self.gb_model.predict_proba(latest_scaled)[0]
            
            classes = self.rf_model.classes_
            avg_proba = (rf_proba + gb_proba) / 2
            
            long_idx = list(classes).index(1) if 1 in classes else None
            short_idx = list(classes).index(-1) if -1 in classes else None
            neutral_idx = list(classes).index(0) if 0 in classes else None
            
            long_conf = avg_proba[long_idx] * 100 if long_idx is not None else 0
            short_conf = avg_proba[short_idx] * 100 if short_idx is not None else 0
            neutral_conf = avg_proba[neutral_idx] * 100 if neutral_idx is not None else 0
            
            # Determine signal
            if long_conf > short_conf and long_conf > neutral_conf:
                signal = "LONG"
                confidence = long_conf
            elif short_conf > long_conf and short_conf > neutral_conf:
                signal = "SHORT"
                confidence = short_conf
            else:
                return {
                    'signal': None,
                    'confidence': 0,
                    'probas': {'long': long_conf, 'short': short_conf, 'neutral': neutral_conf}
                }
            
            probas = {'long': long_conf, 'short': short_conf, 'neutral': neutral_conf}
            
            if confidence < 75:
                return {
                    'signal': None,
                    'confidence': confidence,
                    'probas': probas,
                    'reason': 'confidence_too_low'
                }
            
            # Calculate stops and targets
            latest_price = float(ha_df['close'].iloc[-1])
            atr_val = atr(ha_df, 14).iloc[-1]
            
            if signal == "LONG":
                stop_loss = latest_price - (atr_val * 2.0)
                take_profit_1 = latest_price + (atr_val * 3.0)
                take_profit_2 = latest_price + (atr_val * 5.0)
            else:
                stop_loss = latest_price + (atr_val * 2.0)
                take_profit_1 = latest_price - (atr_val * 3.0)
                take_profit_2 = latest_price - (atr_val * 5.0)
            
            return {
                'signal': signal,
                'confidence': round(confidence, 1),
                'probas': probas,
                'entry_price': round(latest_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_1': round(take_profit_1, 2),
                'take_profit_2': round(take_profit_2, 2),
                'atr': round(atr_val, 2),
                'model_agreement': abs(rf_proba.max() - gb_proba.max()) < 0.1
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def save(self, path: str):
        """Save model to disk"""
        import joblib
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'symbol_performance': self.symbol_performance
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Talon model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TalonMLModel":
        """Load model from disk"""
        import joblib
        
        model_data = joblib.load(path)
        
        model = cls()
        model.rf_model = model_data['rf_model']
        model.gb_model = model_data['gb_model']
        model.scaler = model_data['scaler']
        model.feature_names = model_data['feature_names']
        model.is_trained = model_data['is_trained']
        model.training_date = datetime.fromisoformat(model_data['training_date']) if model_data['training_date'] else None
        model.symbol_performance = model_data.get('symbol_performance', {})
        
        logger.info(f"Talon model loaded from {path}")
        return model


# Global model cache
_model_cache: Dict[str, TalonMLModel] = {}


def get_or_train_model(symbol: str, candles: List) -> TalonMLModel:
    """Get cached model or train new one"""
    if symbol in _model_cache:
        return _model_cache[symbol]
    
    model = TalonMLModel()
    model.train(candles, symbol)
    _model_cache[symbol] = model
    
    return model


def predict_with_talon_model(symbol: str, klines: List) -> Optional[Dict]:
    """Convenience function for prediction"""
    model = _model_cache.get(symbol)
    if not model:
        logger.warning(f"No Talon model available for {symbol}")
        return None
    
    return model.predict(klines)
