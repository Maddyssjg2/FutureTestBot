"""
LSTM/ML Training with Pair-Specific Optimized Strategies
======================================================
Trains models to predict signals based on optimized pair-specific parameters.

Each pair uses its best strategy:
- BTCUSDT: Breakout Pullback
- ETHUSDT: Stochastic (oversold 15)
- SOLUSDT: RSI Mean Reversion (RSI 25, ADX 25)
- XRPUSDT: Stochastic (oversold 15)
- BNBUSDT: Stochastic (oversold 15)
- DOGEUSDT: Stochastic (oversold 20)
- ADAUSDT: Stochastic (oversold 20)
- TRXUSDT: RSI Mean Reversion (RSI 35, ADX 20)
- AVAXUSDT: RSI Mean Reversion (RSI 25, ADX 20)
- DOTUSDT: RSI Mean Reversion (RSI 30, ADX 25)
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# CONFIGURATION
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

# Optimized strategies per pair
PAIR_STRATEGIES = {
    "BTCUSDT": {"strategy": "Breakout_PW", "params": {"lookback": 20}},
    "ETHUSDT": {"strategy": "Stochastic", "params": {"oversold": 15}},
    "SOLUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 25}},
    "XRPUSDT": {"strategy": "Stochastic", "params": {"oversold": 15}},
    "BNBUSDT": {"strategy": "Stochastic", "params": {"oversold": 15}},
    "DOGEUSDT": {"strategy": "Stochastic", "params": {"oversold": 20}},
    "ADAUSDT": {"strategy": "Stochastic", "params": {"oversold": 20}},
    "TRXUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 35, "adx_min": 20}},
    "AVAXUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 20}},
    "DOTUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 25}},
}

TIMEFRAME = "1h"
LOOKBACK = 30
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'pair_models')


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
# INDICATORS
# ============================================================================

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = calc_atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(alpha=1/period).mean()

def calc_stochastic(high, low, close, k_period=14, d_period=3):
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 0.0001)
    d = k.rolling(d_period).mean()
    return k, d


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Create features for ML model based on pair-specific strategy"""
    df = df.copy()
    
    # Basic returns
    for i in [1, 2, 3, 5, 10, 15]:
        df[f'return_{i}'] = df['close'].pct_change(i)
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = calc_ema(df['close'], period)
        df[f'ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
    
    # RSI
    df['rsi'] = calc_rsi(df['close'], 14)
    
    # ADX
    df['adx'] = calc_adx(df['high'], df['low'], df['close'], 14)
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = calc_stochastic(df['high'], df['low'], df['close'])
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_position'] = (df['close'] - (df['bb_middle'] - 2*df['bb_std'])) / (4*df['bb_std'] + 0.0001)
    
    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # ATR
    df['atr'] = calc_atr(df)
    df['atr_ratio'] = df['atr'] / df['close']
    
    # Strategy-specific features
    strategy = PAIR_STRATEGIES.get(pair, {}).get('strategy', '')
    
    if strategy == 'Breakout_PW':
        lookback = PAIR_STRATEGIES[pair]['params'].get('lookback', 20)
        df['resistance'] = df['high'].rolling(lookback).max().shift(1)
        df['support'] = df['low'].rolling(lookback).min().shift(1)
        df['breakout_distance'] = (df['close'] - df['resistance']) / df['close']
        df['near_support'] = (df['close'] - df['support']) / df['close']
    
    elif strategy == 'RSI_MeanRev':
        df['rsi_oversold_level'] = PAIR_STRATEGIES[pair]['params'].get('rsi_oversold', 30)
        df['rsi_zone'] = (df['rsi'] <= df['rsi_oversold_level']).astype(int)
        df['adx_min_level'] = PAIR_STRATEGIES[pair]['params'].get('adx_min', 20)
        df['adx_strong'] = (df['adx'] >= df['adx_min_level']).astype(int)
    
    elif strategy == 'Stochastic':
        oversold = PAIR_STRATEGIES[pair]['params'].get('oversold', 20)
        df['stoch_oversold'] = oversold
        df['stoch_cross_up'] = ((df['stoch_k'] <= oversold) & (df['stoch_k'].shift(1) > oversold)).astype(int)
        df['stoch_k_d'] = df['stoch_k'] - df['stoch_d']
    
    return df


def create_labels(df: pd.DataFrame, pair: str) -> pd.Series:
    """Create labels based on pair-specific strategy"""
    df = df.copy()
    strategy = PAIR_STRATEGIES.get(pair, {}).get('strategy', '')
    
    # Target: 1 if strategy signal would trigger in next candle
    future_returns = df['close'].shift(-1) / df['close'] - 1
    
    if strategy == 'Breakout_PW':
        lookback = PAIR_STRATEGIES[pair]['params'].get('lookback', 20)
        resistance = df['high'].rolling(lookback).max().shift(1)
        breakout = df['close'] > resistance
        signal = breakout & (future_returns > 0.005)  # Signal + positive return
    
    elif strategy == 'RSI_MeanRev':
        rsi_oversold = PAIR_STRATEGIES[pair]['params'].get('rsi_oversold', 30)
        adx_min = PAIR_STRATEGIES[pair]['params'].get('adx_min', 20)
        rsi = calc_rsi(df['close'], 14)
        adx = calc_adx(df['high'], df['low'], df['close'], 14)
        
        signal = (rsi <= rsi_oversold) & (adx >= adx_min) & (future_returns > 0.005)
    
    elif strategy == 'Stochastic':
        oversold = PAIR_STRATEGIES[pair]['params'].get('oversold', 20)
        stoch_k, stoch_d = calc_stochastic(df['high'], df['low'], df['close'])
        
        k_cross = (stoch_k <= oversold) & (stoch_k.shift(1) > oversold)
        signal = k_cross & (stoch_k < stoch_d) & (future_returns > 0.005)
    
    else:
        # Default: use any positive return
        signal = future_returns > 0.005
    
    # 1 if positive return follows signal, 0 otherwise
    return (signal & (future_returns > 0)).astype(int)


# ============================================================================
# ML TRAINING
# ============================================================================

def prepare_data(df: pd.DataFrame, pair: str, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features with lookback window"""
    df = create_features(df, pair)
    df = df.dropna()
    
    # Feature columns
    feature_cols = [
        'return_1', 'return_2', 'return_3', 'return_5', 'return_10',
        'ema_ratio_5', 'ema_ratio_10', 'ema_ratio_20', 'ema_ratio_50',
        'rsi', 'adx', 'stoch_k', 'stoch_d', 'bb_position',
        'volume_ratio', 'atr_ratio'
    ]
    
    # Add strategy-specific features
    strategy = PAIR_STRATEGIES.get(pair, {}).get('strategy', '')
    if strategy == 'Breakout_PW':
        feature_cols.extend(['breakout_distance', 'near_support'])
    elif strategy == 'RSI_MeanRev':
        feature_cols.extend(['rsi_zone', 'adx_strong'])
    elif strategy == 'Stochastic':
        feature_cols.extend(['stoch_cross_up', 'stoch_k_d'])
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = []
    y = []
    
    for i in range(lookback, len(df)):
        features = df[feature_cols].iloc[i-lookback:i].values.flatten()
        X.append(features)
        y.append(create_labels(df, pair).iloc[i])
    
    return np.array(X), np.array(y)


def train_model(pair: str, df: pd.DataFrame) -> Dict:
    """Train model for a pair"""
    logger.info(f"Training for {pair}...")
    
    X, y = prepare_data(df, pair, LOOKBACK)
    
    if len(X) < 500:
        return {'error': 'insufficient_data', 'symbol': pair}
    
    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    logger.info(f"  Samples: {len(X_train)} train, {len(X_test)} test")
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = (rf.predict(X_test) == y_test).mean()
    
    # Train Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_acc = (gb.predict(X_test) == y_test).mean()
    
    # Use best model
    if rf_acc >= gb_acc:
        best_model = rf
        best_name = 'RandomForest'
        best_acc = rf_acc
    else:
        best_model = gb
        best_name = 'GradientBoosting'
        best_acc = gb_acc
    
    logger.info(f"  {best_name}: {best_acc:.2%} accuracy")
    
    # Save model
    import pickle
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'model_{pair}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    return {
        'symbol': pair,
        'strategy': PAIR_STRATEGIES[pair]['strategy'],
        'params': PAIR_STRATEGIES[pair]['params'],
        'model_type': best_name,
        'accuracy': float(best_acc),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X.shape[1],
        'model_path': model_path
    }


def predict(pair: str, df: pd.DataFrame) -> Dict:
    """Make prediction using trained model"""
    if not HAS_SKLEARN:
        return {'direction': 'NEUTRAL', 'confidence': 50}
    
    import pickle
    
    model_path = os.path.join(MODELS_DIR, f'model_{pair}.pkl')
    if not os.path.exists(model_path):
        return {'direction': 'NEUTRAL', 'confidence': 50}
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        X, _ = prepare_data(df.tail(LOOKBACK + 100), pair, LOOKBACK)
        if len(X) == 0:
            return {'direction': 'NEUTRAL', 'confidence': 50}
        
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
        
        return {
            'direction': direction,
            'confidence': float(confidence),
            'probability': float(max(prob))
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'direction': 'NEUTRAL', 'confidence': 50}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PAIR-SPECIFIC LSTM/ML TRAINING")
    print("=" * 70)
    print(f"Training with optimized strategy parameters for each pair")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    downloader = DataDownloader()
    results = []
    
    for pair in PAIRS:
        print(f"\n>>> {pair}")
        print(f"    Strategy: {PAIR_STRATEGIES[pair]['strategy']}")
        print(f"    Params: {PAIR_STRATEGIES[pair]['params']}")
        
        # Download data
        start_time = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
        klines = downloader.get_klines(pair, TIMEFRAME, limit=2000, start_time=start_time)
        
        if not klines:
            print(f"    Failed to download data")
            continue
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"    {len(df)} candles loaded")
        
        # Train
        result = train_model(pair, df)
        results.append(result)
        
        if 'error' in result:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Accuracy: {result['accuracy']:.2%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Symbol':<12} {'Strategy':<18} {'Accuracy':<12} {'Samples'}")
    print("-" * 70)
    
    total_acc = 0
    count = 0
    
    for r in results:
        if 'error' not in r:
            print(f"{r['symbol']:<12} {r['strategy']:<18} {r['accuracy']:<12.2%} {r['test_samples']}")
            total_acc += r['accuracy']
            count += 1
    
    if count > 0:
        avg_acc = total_acc / count
        print("-" * 70)
        print(f"{'AVERAGE':<12} {'':<18} {avg_acc:<12.2%}")
    
    # Save results
    results_file = os.path.join(MODELS_DIR, 'pair_training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {MODELS_DIR}")
    
    return results


if __name__ == '__main__':
    main()
