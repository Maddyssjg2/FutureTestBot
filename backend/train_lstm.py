"""
LSTM Model Training for Crypto Price Prediction
============================================
Trains LSTM models to predict price direction for all 10 pairs.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("TensorFlow not available. LSTM training disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

# Best performing pairs (based on backtest)
GOOD_PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]

TIMEFRAME = "1h"
LOOKBACK = 60  # 60 candles lookback
EPOCHS = 50
BATCH_SIZE = 32
TEST_SPLIT = 0.2

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'lstm_models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    def __init__(self):
        self.base_url = "https://data-api.binance.vision/api/v3"
    
    def get_klines(self, symbol, interval, limit=1000, start_time=None):
        import requests
        url = f"{self.base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"Download error: {e}")
        return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for LSTM"""
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Moving averages
    for period in [7, 14, 21, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    return df


def create_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        # Predict direction: 1 if next candle up, 0 if down
        y.append(1 if data[i, 0] > data[i-1, 0] else 0)
    return np.array(X), np.array(y)


# ============================================================================
# LSTM MODEL
# ============================================================================

if HAS_TENSORFLOW:

    def build_model(input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def train_model(symbol: str, df: pd.DataFrame) -> Dict:
        """Train LSTM model for a symbol"""
        logger.info(f"Training LSTM for {symbol}...")
        
        # Create features
        df = create_features(df)
        df = df.dropna()
        
        # Select features
        feature_cols = ['returns', 'ema_7_ratio', 'ema_14_ratio', 'ema_21_ratio', 
                       'ema_50_ratio', 'rsi', 'macd_hist', 'bb_position', 
                       'volume_ratio', 'atr_ratio']
        
        # Normalize
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df[feature_cols].values)
        
        # Create sequences
        X, y = create_sequences(data, LOOKBACK)
        
        # Split
        split = int(len(X) * (1 - TEST_SPLIT))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        logger.info(f"  Samples: {len(X_train)} train, {len(X_test)} test")
        
        if len(X_train) < 100:
            return {'error': 'insufficient_data'}
        
        # Build and train model
        model = build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        logger.info(f"  Train Accuracy: {train_acc:.2%} | Test Accuracy: {test_acc:.2%}")
        
        # Save
        model_path = os.path.join(MODELS_DIR, f'lstm_{symbol}.h5')
        model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(MODELS_DIR, f'scaler_{symbol}.json')
        with open(scaler_path, 'w') as f:
            json.dump({
                'feature_cols': feature_cols,
                'data_min': scaler.data_min_.tolist(),
                'data_max': scaler.data_max_.tolist()
            }, f)
        
        return {
            'symbol': symbol,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'model_path': model_path
        }
    
    def predict(model, scaler_params, df_current: pd.DataFrame) -> Dict:
        """Make prediction using trained model"""
        if not HAS_TENSORFLOW:
            return {'direction': 'NEUTRAL', 'confidence': 50}
        
        try:
            df = create_features(df_current.tail(LOOKBACK + 10))
            df = df.dropna()
            
            scaler = MinMaxScaler()
            scaler.data_min_ = np.array(scaler_params['data_min'])
            scaler.data_max_ = np.array(scaler_params['data_max'])
            
            feature_cols = scaler_params['feature_cols']
            data = scaler.transform(df[feature_cols].values[-LOOKBACK:])
            X = data.reshape(1, LOOKBACK, len(feature_cols))
            
            prob = model.predict(X, verbose=0)[0][0]
            
            if prob > 0.55:
                direction = 'LONG'
                confidence = prob * 100
            elif prob < 0.45:
                direction = 'SHORT'
                confidence = (1 - prob) * 100
            else:
                direction = 'NEUTRAL'
                confidence = 50
            
            return {'direction': direction, 'confidence': float(confidence), 'probability': float(prob)}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 50}


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    if not HAS_TENSORFLOW:
        logger.error("TensorFlow required for LSTM training")
        print("\nTo install TensorFlow:")
        print("  pip install tensorflow")
        return
    
    print("=" * 70)
    print("LSTM MODEL TRAINING")
    print("=" * 70)
    print(f"Training on: {', '.join(PAIRS)}")
    print(f"Lookback: {LOOKBACK} candles")
    print(f"Epochs: {EPOCHS}")
    print("=" * 70)
    
    downloader = DataDownloader()
    results = []
    
    for pair in PAIRS:
        print(f"\n>>> {pair}")
        
        # Download 6 months of data
        start_time = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
        klines = downloader.get_klines(pair, TIMEFRAME, limit=2000, start_time=start_time)
        
        if not klines:
            print(f"   Failed to download data")
            continue
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"   Data: {len(df)} candles")
        
        # Train
        result = train_model(pair, df)
        results.append(result)
        
        if 'error' in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"   Test Accuracy: {result['test_accuracy']:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Symbol':<12} {'Train Acc':<12} {'Test Acc':<12} {'Samples':<12}")
    print("-" * 70)
    
    for r in results:
        if 'error' not in r:
            print(f"{r['symbol']:<12} {r['train_accuracy']:<12.2%} {r['test_accuracy']:<12.2%} {r['train_samples']:<12}")
    
    # Save results
    results_file = os.path.join(MODELS_DIR, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    main()
