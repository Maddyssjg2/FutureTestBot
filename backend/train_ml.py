"""
ML Model Training (sklearn) for Crypto Trading
==============================================
Uses RandomForest and GradientBoosting instead of TensorFlow.
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

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

# Best pairs (exclude volatile ones)
GOOD_PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "BNBUSDT"]

TIMEFRAME = "1h"
LOOKBACK = 20
TEST_SPLIT = 0.2

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'ml_models')
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
            logger.error(f"Error: {e}")
        return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML model"""
    df = df.copy()
    
    # Price returns
    for i in [1, 2, 3, 5, 10]:
        df[f'return_{i}'] = df['close'].pct_change(i)
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        df[f'ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
    
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
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - (df['bb_middle'] - 2*df['bb_std'])) / (4*df['bb_std'] + 0.0001)
    
    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14).mean()
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Target: 1 if price up in next candle, 0 if down
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df


# ============================================================================
# ML TRAINING
# ============================================================================

def prepare_data(df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features with lookback"""
    df = df.dropna()
    
    # Features to use
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target', 'ignore']]
    
    X = []
    y = df['target'].values
    
    for i in range(lookback, len(df)):
        # Flatten lookback window
        features = []
        for col in feature_cols[:15]:  # Limit features
            features.extend(df[col].iloc[i-lookback:i].values.tolist())
        X.append(features)
    
    return np.array(X), y[lookback:]


def train_model(symbol: str, df: pd.DataFrame) -> Dict:
    """Train ML model"""
    logger.info(f"Training ML for {symbol}...")
    
    df = create_features(df)
    
    X, y = prepare_data(df, LOOKBACK)
    
    if len(X) < 500:
        return {'error': 'insufficient_data', 'symbol': symbol}
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT, shuffle=False)
    
    logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # Gradient Boosting (smaller for speed)
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    # Use best model
    if rf_acc >= gb_acc:
        best_model = rf
        best_acc = rf_acc
        best_name = 'RandomForest'
    else:
        best_model = gb
        best_acc = gb_acc
        best_name = 'GradientBoosting'
    
    logger.info(f"  {best_name}: {best_acc:.2%}")
    
    # Save model
    import pickle
    model_path = os.path.join(MODELS_DIR, f'ml_{symbol}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    return {
        'symbol': symbol,
        'model_type': best_name,
        'accuracy': float(best_acc),
        'rf_accuracy': float(rf_acc),
        'gb_accuracy': float(gb_acc),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': LOOKBACK * 15,
        'model_path': model_path
    }


def predict(symbol: str, df_current: pd.DataFrame) -> Dict:
    """Make prediction"""
    if not HAS_SKLEARN:
        return {'direction': 'NEUTRAL', 'confidence': 50}
    
    import pickle
    
    model_path = os.path.join(MODELS_DIR, f'ml_{symbol}.pkl')
    if not os.path.exists(model_path):
        return {'direction': 'NEUTRAL', 'confidence': 50}
    
    try:
        df = create_features(df_current.tail(LOOKBACK + 100))
        X, _ = prepare_data(df, LOOKBACK)
        
        if len(X) == 0:
            return {'direction': 'NEUTRAL', 'confidence': 50}
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        prob = model.predict_proba(X[-1:])[0]
        
        if prob[1] > 0.55:
            direction = 'LONG'
            confidence = prob[1] * 100
        elif prob[0] > 0.55:
            direction = 'SHORT'
            confidence = prob[0] * 100
        else:
            direction = 'NEUTRAL'
            confidence = 50
        
        return {'direction': direction, 'confidence': float(confidence), 'probability': float(max(prob))}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'direction': 'NEUTRAL', 'confidence': 50}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ML MODEL TRAINING (RandomForest + GradientBoosting)")
    print("=" * 70)
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Lookback: {LOOKBACK} candles")
    print("=" * 70)
    
    downloader = DataDownloader()
    results = []
    
    for pair in PAIRS:
        print(f"\n>>> {pair}")
        
        start_time = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
        klines = downloader.get_klines(pair, TIMEFRAME, limit=2000, start_time=start_time)
        
        if not klines:
            print(f"   Failed to download")
            continue
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"   {len(df)} candles")
        
        result = train_model(pair, df)
        results.append(result)
        
        if 'error' in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"   Accuracy: {result['accuracy']:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Symbol':<12} {'Model':<18} {'Accuracy':<12} {'Samples':<10}")
    print("-" * 70)
    
    total_acc = 0
    count = 0
    
    for r in results:
        if 'error' not in r:
            print(f"{r['symbol']:<12} {r['model_type']:<18} {r['accuracy']:<12.2%} {r['test_samples']:<10}")
            total_acc += r['accuracy']
            count += 1
    
    if count > 0:
        avg_acc = total_acc / count
        print("-" * 70)
        print(f"{'AVERAGE':<12} {'':<18} {avg_acc:<12.2%}")
    
    # Save
    results_file = os.path.join(MODELS_DIR, 'ml_training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    main()
