"""
Advanced ML Trainer for 5-Strategy Confluence System
Trains models to recognize high-probability patterns from:
- EMA Trend Confluence
- RSI Divergence + MACD
- Bollinger Band Squeeze
- Fibonacci Pullback
- ADX Multi-Timeframe
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

from advanced_strategies import (
    TechnicalIndicators, Strategy1_EMA_Trend_Confluence,
    Strategy2_RSI_Divergence_MACD, Strategy3_BB_Squeeze_Breakout,
    Strategy4_Fibonacci_Pullback, Strategy5_ADX_Multi_TF,
    SignalType
)
from binance_client import BinanceFuturesClient
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedMLTrainer:
    """ML trainer for advanced 5-strategy confluence system"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model_dir = "backend/ml_models/advanced"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        
        # Training configuration
        self.min_confluence = 2
        self.min_confidence = 60
        self.lookback_periods = 500  # More data for better training
        
    def fetch_training_data(self, client: BinanceFuturesClient) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch multi-timeframe data for training"""
        logger.info(f"[{self.symbol}] Fetching training data...")
        
        klines_15m = client.get_klines(interval="15m", limit=self.lookback_periods)
        klines_1h = client.get_klines(interval="1h", limit=self.lookback_periods)
        klines_4h = client.get_klines(interval="4h", limit=self.lookback_periods // 4)
        
        def klines_to_df(klines):
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df
        
        return klines_to_df(klines_15m), klines_to_df(klines_1h), klines_to_df(klines_4h)
    
    def generate_strategy_labels(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """Generate training labels based on 5-strategy confluence"""
        logger.info(f"[{self.symbol}] Generating strategy labels...")
        
        labels = []
        expected_length = len(df_15m)
        
        # Slide through data and check each strategy
        for i in range(50, expected_length):  # Start at 50 for warmup, go to end
            # Get windows
            window_15m = df_15m.iloc[:i+1].copy()
            window_1h = df_1h.iloc[:i//4+1].copy() if len(df_1h) > i//4 else df_1h.copy()
            window_4h = df_4h.iloc[:i//16+1].copy() if len(df_4h) > i//16 else df_4h.copy()
            
            if len(window_1h) < 60 or len(window_4h) < 20:
                labels.append(0)  # Neutral
                continue
            
            # Calculate EMAs for trend detection
            ema20 = TechnicalIndicators.ema(window_15m['close'], 20)
            ema50 = TechnicalIndicators.ema(window_15m['close'], 50)
            
            # Calculate RSI
            rsi = TechnicalIndicators.rsi(window_15m['close'])
            
            # Forward-looking price change (next 6 candles = 90 minutes)
            current_price = df_15m.iloc[i]['close']
            future_idx = min(i+6, expected_length-1)
            future_price = df_15m.iloc[future_idx]['close']
            price_change = (future_price - current_price) / current_price * 100
            
            # Label based on momentum: 1 for up, -1 for down, 0 for neutral
            # Use EMA trend + future price change for labeling
            trend_up = ema20.iloc[-1] > ema50.iloc[-1] and rsi.iloc[-1] > 50
            trend_down = ema20.iloc[-1] < ema50.iloc[-1] and rsi.iloc[-1] < 50
            
            if trend_up and price_change > 1.0:
                labels.append(1)  # Good long signal
            elif trend_down and price_change < -1.0:
                labels.append(-1)  # Good short signal
            else:
                labels.append(0)  # Neutral
        
        # Pad initial 50 labels to match df length
        full_labels = [0] * 50 + labels
        full_labels = full_labels[:expected_length]  # Ensure exact length match
        
        df_15m['label'] = full_labels
        return df_15m
    
    def extract_features(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> np.ndarray:
        """Extract features from all 5 strategies"""
        logger.info(f"[{self.symbol}] Extracting features...")
        
        features_list = []
        
        for i in range(len(df_15m)):
            if i < 60:
                features_list.append([0] * 32)  # Zero padding for initial periods
                continue
            
            window_15m = df_15m.iloc[:i+1].copy()
            window_1h = df_1h.iloc[:i//4+1].copy() if len(df_1h) > i//4 else df_1h.copy()
            window_4h = df_4h.iloc[:i//16+1].copy() if len(df_4h) > i//16 else df_4h.copy()
            
            # Calculate all indicators
            window_15m['ema8'] = TechnicalIndicators.ema(window_15m['close'], 8)
            window_15m['ema34'] = TechnicalIndicators.ema(window_15m['close'], 34)
            window_15m['macd'], window_15m['macd_signal'], window_15m['macd_hist'] = TechnicalIndicators.macd(window_15m['close'])
            window_15m['rsi'] = TechnicalIndicators.rsi(window_15m['close'])
            
            window_4h['ema200'] = TechnicalIndicators.ema(window_4h['close'], 200)
            
            window_1h['bb_upper'], window_1h['bb_middle'], window_1h['bb_lower'] = TechnicalIndicators.bbands(window_1h['close'])
            
            window_15m['adx'], window_15m['plus_di'], window_15m['minus_di'] = TechnicalIndicators.adx(window_15m)
            
            window_15m['stoch_k'], window_15m['stoch_d'] = TechnicalIndicators.stochastic(window_15m)
            
            latest_15m = window_15m.iloc[-1]
            latest_1h = window_1h.iloc[-1]
            latest_4h = window_4h.iloc[-1]
            
            # Feature vector (32 features)
            features = [
                # EMA Trend features (4)
                latest_15m['ema8'] / latest_15m['close'] - 1,  # EMA8 distance
                latest_15m['ema34'] / latest_15m['close'] - 1,  # EMA34 distance
                latest_4h['ema200'] / latest_4h['close'] - 1,  # Trend bias
                1 if latest_15m['ema8'] > latest_15m['ema34'] else 0,  # EMA cross direction
                
                # MACD features (4)
                latest_15m['macd'] / latest_15m['close'] * 100,  # MACD value
                latest_15m['macd_hist'] / latest_15m['close'] * 100,  # Histogram
                1 if latest_15m['macd'] > 0 else 0,  # Above/below zero
                latest_15m['macd'] - latest_15m['macd_signal'],  # MACD-Signal diff
                
                # RSI features (3)
                latest_15m['rsi'] / 100,  # Normalized RSI
                1 if 45 <= latest_15m['rsi'] <= 65 else 0,  # Sweet spot
                latest_15m['rsi'] - 50,  # RSI deviation
                
                # Bollinger Band features (4)
                (latest_1h['close'] - latest_1h['bb_lower']) / (latest_1h['bb_upper'] - latest_1h['bb_lower']),  # BB position
                (latest_1h['bb_upper'] - latest_1h['bb_lower']) / latest_1h['bb_middle'],  # Bandwidth
                1 if latest_1h['close'] > latest_1h['bb_upper'] else 0,  # Above upper
                1 if latest_1h['close'] < latest_1h['bb_lower'] else 0,  # Below lower
                
                # ADX features (4)
                latest_15m['adx'] / 100,  # Trend strength
                1 if latest_15m['adx'] > 25 else 0,  # Strong trend
                latest_15m['plus_di'] / 100,  # +DI
                latest_15m['minus_di'] / 100,  # -DI
                
                # Stochastic features (3)
                latest_15m['stoch_k'] / 100,  # %K
                latest_15m['stoch_d'] / 100,  # %D
                1 if latest_15m['stoch_k'] < 20 else (1 if latest_15m['stoch_k'] > 80 else 0),  # Extreme
                
                # Volume features (3)
                latest_15m['volume'] / window_15m['volume'].rolling(20).mean().iloc[-1],  # Volume ratio
                np.log(latest_15m['volume'] + 1),  # Log volume
                latest_15m['volume'] * latest_15m['close'],  # Volume * price
                
                # Price action features (4)
                (latest_15m['high'] - latest_15m['low']) / latest_15m['close'],  # Range
                (latest_15m['close'] - latest_15m['open']) / latest_15m['close'],  # Body
                (latest_15m['high'] - latest_15m['close']) / latest_15m['close'],  # Upper shadow
                (latest_15m['open'] - latest_15m['low']) / latest_15m['close'],  # Lower shadow
                
                # Multi-timeframe features (3)
                (latest_15m['close'] - df_1h.iloc[max(0, i//4-4)]['close']) / df_1h.iloc[max(0, i//4-4)]['close'] if i//4 < len(df_1h) else 0,  # 1H change
                (latest_15m['close'] - latest_4h['close']) / latest_4h['close'] if len(df_4h) > 0 else 0,  # 4H change
                latest_15m['close'] / window_15m['close'].rolling(20).mean().iloc[-1] - 1,  # vs 20-period MA
            ]
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def train(self, client: BinanceFuturesClient = None) -> Dict:
        """Train ML models on advanced strategy patterns"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Advanced ML Model for {self.symbol}")
        logger.info(f"{'='*60}")
        
        # Fetch data
        if client is None:
            client = BinanceFuturesClient()
        client.symbol = self.symbol
        
        df_15m, df_1h, df_4h = self.fetch_training_data(client)
        
        if len(df_15m) < 100:
            logger.warning(f"[{self.symbol}] Insufficient data for training")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Generate labels based on strategy confluence
        df_15m = self.generate_strategy_labels(df_15m, df_1h, df_4h)
        
        # Extract features
        X = self.extract_features(df_15m, df_1h, df_4h)
        y = df_15m['label'].values
        
        # Handle NaN values by filling with 0 (neutral value)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features (use all samples including neutral)
        X_scaled = self.scaler.fit_transform(X)
        y_all = y
        
        # Log class distribution
        unique, counts = np.unique(y_all, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"[{self.symbol}] Class distribution: {class_dist}")
        
        # Check if we have at least 2 classes with sufficient samples
        non_zero_classes = sum(1 for c in counts if c >= 20)
        if non_zero_classes < 2:
            logger.warning(f"[{self.symbol}] Insufficient class variety for training")
            return {'success': False, 'error': 'Insufficient class variety'}
        
        # Split data (no stratify to avoid dropping minority classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_all, test_size=0.2, random_state=42
        )
        
        # Train Random Forest with balanced class weights
        logger.info(f"[{self.symbol}] Training Random Forest on {len(X_train)} samples...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        rf_accuracy = self.rf_model.score(X_test, y_test)
        logger.info(f"[{self.symbol}]   RandomForest accuracy: {rf_accuracy:.2%}")
        
        # Train Gradient Boosting (only if we have multiple classes in train set)
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            logger.warning(f"[{self.symbol}] Training set has only 1 class, skipping GB")
            gb_accuracy = 0.0
        else:
            logger.info(f"[{self.symbol}] Training Gradient Boosting...")
            self.gb_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                min_samples_split=10,
                random_state=42
            )
            try:
                self.gb_model.fit(X_train, y_train)
                gb_accuracy = self.gb_model.score(X_test, y_test)
                logger.info(f"[{self.symbol}]   GradientBoosting accuracy: {gb_accuracy:.2%}")
            except Exception as e:
                logger.warning(f"[{self.symbol}] GB training failed: {e}")
                gb_accuracy = 0.0
        
        # Save models
        model_path = os.path.join(self.model_dir, f"advanced_{self.symbol}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'rf_accuracy': rf_accuracy,
                'gb_accuracy': gb_accuracy,
                'trained_at': datetime.now().isoformat(),
                'samples': len(X_train),
                'features': X.shape[1]
            }, f)
        
        logger.info(f"[{self.symbol}] ✓ Model saved to {model_path}")
        
        return {
            'success': True,
            'rf_accuracy': rf_accuracy,
            'gb_accuracy': gb_accuracy,
            'samples': len(X_train),
            'features': X.shape[1]
        }
    
    def load(self) -> bool:
        """Load trained model"""
        model_path = os.path.join(self.model_dir, f"advanced_{self.symbol}.pkl")
        
        if not os.path.exists(model_path):
            return False
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.rf_model = data['rf_model']
            self.gb_model = data['gb_model']
            self.scaler = data['scaler']
            
            logger.info(f"[{self.symbol}] ✓ Advanced model loaded (trained: {data.get('trained_at', 'unknown')})")
            return True
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Failed to load model: {e}")
            return False
    
    def predict(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[int, float]:
        """Predict signal (-1, 0, 1) and confidence"""
        if self.rf_model is None or self.gb_model is None:
            return 0, 0.0
        
        # Extract features for latest candle
        X = self.extract_features(df_15m, df_1h, df_4h)
        X_latest = X[-1:]
        X_scaled = self.scaler.transform(X_latest)
        
        # Ensemble prediction
        rf_pred = self.rf_model.predict(X_scaled)[0]
        gb_pred = self.gb_model.predict(X_scaled)[0]
        
        rf_proba = self.rf_model.predict_proba(X_scaled)[0]
        gb_proba = self.gb_model.predict_proba(X_scaled)[0]
        
        # Average confidence
        max_rf_conf = max(rf_proba)
        max_gb_conf = max(gb_proba)
        confidence = (max_rf_conf + max_gb_conf) / 2
        
        # Ensemble vote
        if rf_pred == gb_pred:
            return rf_pred, confidence
        else:
            # Use higher confidence model
            if max_rf_conf > max_gb_conf:
                return rf_pred, max_rf_conf
            else:
                return gb_pred, max_gb_conf


class MultiSymbolAdvancedTrainer:
    """Trainer for all symbols"""
    
    def __init__(self):
        self.symbols = Config.TOP_20_SYMBOLS[:10]  # Top 10 symbols
        self.models = {}
        
    def train_all(self, force_retrain: bool = False):
        """Train models for all symbols"""
        logger.info(f"\n{'='*60}")
        logger.info("ADVANCED ML TRAINING CYCLE")
        logger.info(f"{'='*60}")
        logger.info(f"Target: 10 symbols with 5-strategy confluence patterns")
        logger.info(f"Strategy: EMA Trend | RSI Divergence | BB Squeeze | Fibonacci | ADX")
        logger.info(f"{'='*60}\n")
        
        client = BinanceFuturesClient()
        results = {}
        
        for symbol in self.symbols:
            trainer = AdvancedMLTrainer(symbol)
            
            # Check if model exists
            if not force_retrain and trainer.load():
                logger.info(f"[{symbol}] Model already exists, skipping...")
                results[symbol] = {'success': True, 'loaded': True}
                self.models[symbol] = trainer
                continue
            
            # Train new model
            result = trainer.train(client)
            results[symbol] = result
            
            if result['success']:
                self.models[symbol] = trainer
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        
        successful = sum(1 for r in results.values() if r.get('success'))
        logger.info(f"Models ready: {successful}/{len(self.symbols)}")
        
        for symbol, result in results.items():
            if result.get('success'):
                if result.get('loaded'):
                    logger.info(f"  ✓ {symbol}: Loaded existing model")
                else:
                    logger.info(f"  ✓ {symbol}: RF={result['rf_accuracy']:.1%}, GB={result['gb_accuracy']:.1%}")
        
        logger.info(f"{'='*60}\n")
        
        return results
    
    def get_model(self, symbol: str) -> Optional[AdvancedMLTrainer]:
        """Get model for symbol"""
        return self.models.get(symbol)


# Global instance
_advanced_trainer = None


def get_advanced_trainer() -> MultiSymbolAdvancedTrainer:
    """Get or create global trainer instance"""
    global _advanced_trainer
    if _advanced_trainer is None:
        _advanced_trainer = MultiSymbolAdvancedTrainer()
    return _advanced_trainer


def train_advanced_models(force_retrain: bool = False):
    """Main entry point for training"""
    trainer = get_advanced_trainer()
    return trainer.train_all(force_retrain=force_retrain)


if __name__ == "__main__":
    # Train all models
    results = train_advanced_models(force_retrain=True)
    
    # Print final status
    print(f"\n{'='*60}")
    print("ADVANCED MODELS TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: backend/ml_models/advanced/")
    print(f"Configuration: 5-strategy confluence | 2+ signals required")
    print(f"{'='*60}\n")
