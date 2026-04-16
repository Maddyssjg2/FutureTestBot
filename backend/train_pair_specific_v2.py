"""
Pair-Specific ML Training V2
============================
Trains models using V2 optimized parameters for high-frequency trading.

Each pair uses its V2 optimized strategy:
- BTCUSDT: RSI Mean Reversion (RSI 25)
- ETHUSDT: Bollinger Bands (period 20)
- SOLUSDT: RSI Mean Reversion (RSI 25)
- XRPUSDT: RSI Mean Reversion (RSI 25)
- BNBUSDT: RSI Mean Reversion (RSI 30)
- DOGEUSDT: RSI Mean Reversion (RSI 35)
- ADAUSDT: Bollinger Bands (period 20)
- TRXUSDT: RSI Mean Reversion (RSI 30)
- AVAXUSDT: RSI Mean Reversion (RSI 30)
- DOTUSDT: Bollinger Bands (period 20)
"""

import os
import sys
import json
import logging
import warnings
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# CONFIGURATION - V2
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

PAIR_STRATEGIES_V2 = {
    "BTCUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 0}},
    "ETHUSDT": {"strategy": "Bollinger", "params": {"period": 20}},
    "SOLUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 0}},
    "XRPUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 0}},
    "BNBUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 0}},
    "DOGEUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 35, "adx_min": 0}},
    "ADAUSDT": {"strategy": "Bollinger", "params": {"period": 20}},
    "TRXUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 0}},
    "AVAXUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 0}},
    "DOTUSDT": {"strategy": "Bollinger", "params": {"period": 20}},
}

TIMEFRAME = "1h"
LOOKBACK = 30
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'pair_models_v2')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), 'pair_models_backup')

# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    def __init__(self):
        self.base_url = "https://data-api.binance.vision/api/v3"
    
    def get_klines(self, symbol, interval, limit=1000, start_time=None):
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

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_features(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    df = df.copy()
    
    for i in [1, 2, 3, 5, 10]:
        df[f'return_{i}'] = df['close'].pct_change(i)
    
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = calc_ema(df['close'], period)
        df[f'ema_ratio_{period}'] = df['close'] / df[f'ema_{period}']
    
    df['rsi'] = calc_rsi(df['close'], 14)
    df['adx'] = calc_adx(df['high'], df['low'], df['close'], 14)
    
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['atr'] = calc_atr(df)
    df['atr_ratio'] = df['atr'] / df['close']
    
    strategy = PAIR_STRATEGIES_V2.get(pair, {}).get('strategy', '')
    
    if strategy == 'RSI_MeanRev':
        rsi_oversold = PAIR_STRATEGIES_V2[pair]['params'].get('rsi_oversold', 30)
        df['rsi_zone'] = (df['rsi'] <= rsi_oversold).astype(int)
        df['rsi_oversold_level'] = rsi_oversold
    
    elif strategy == 'Bollinger':
        period = PAIR_STRATEGIES_V2[pair]['params'].get('period', 20)
        bb_middle = df['close'].rolling(period).mean()
        bb_std = df['close'].rolling(period).std()
        df['bb_lower'] = bb_middle - (2 * bb_std)
        df['bb_touched'] = (df['close'] <= df['bb_lower']).astype(int)
    
    return df


def create_labels(df: pd.DataFrame, pair: str) -> pd.Series:
    df = df.copy()
    strategy = PAIR_STRATEGIES_V2.get(pair, {}).get('strategy', '')
    
    future_returns = df['close'].shift(-1) / df['close'] - 1
    
    if strategy == 'RSI_MeanRev':
        rsi_oversold = PAIR_STRATEGIES_V2[pair]['params'].get('rsi_oversold', 30)
        rsi = calc_rsi(df['close'], 14)
        signal = (rsi <= rsi_oversold) & (future_returns > 0.005)
    
    elif strategy == 'Bollinger':
        period = PAIR_STRATEGIES_V2[pair]['params'].get('period', 20)
        bb_middle = df['close'].rolling(period).mean()
        bb_std = df['close'].rolling(period).std()
        bb_lower = bb_middle - (2 * bb_std)
        signal = (df['close'] <= bb_lower) & (future_returns > 0.005)
    
    else:
        signal = future_returns > 0.005
    
    return (signal & (future_returns > 0)).astype(int)


# ============================================================================
# ML TRAINING
# ============================================================================

def prepare_data(df: pd.DataFrame, pair: str, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    df = create_features(df, pair)
    df = df.dropna()
    
    feature_cols = [
        'return_1', 'return_2', 'return_3', 'return_5', 'return_10',
        'ema_ratio_5', 'ema_ratio_10', 'ema_ratio_20', 'ema_ratio_50',
        'rsi', 'adx', 'volume_ratio', 'atr_ratio'
    ]
    
    strategy = PAIR_STRATEGIES_V2.get(pair, {}).get('strategy', '')
    if strategy == 'RSI_MeanRev':
        feature_cols.extend(['rsi_zone'])
    elif strategy == 'Bollinger':
        feature_cols.extend(['bb_touched'])
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = []
    y = []
    
    for i in range(lookback, len(df)):
        features = df[feature_cols].iloc[i-lookback:i].values.flatten()
        X.append(features)
        y.append(create_labels(df, pair).iloc[i])
    
    return np.array(X), np.array(y)


def train_model(pair: str, df: pd.DataFrame) -> Dict:
    logger.info(f"Training for {pair}...")
    
    X, y = prepare_data(df, pair, LOOKBACK)
    
    if len(X) < 500:
        return {'error': 'insufficient_data', 'symbol': pair}
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    logger.info(f"  Samples: {len(X_train)} train, {len(X_test)} test")
    
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
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_acc = (gb.predict(X_test) == y_test).mean()
    
    if rf_acc >= gb_acc:
        best_model = rf
        best_name = 'RandomForest'
        best_acc = rf_acc
    else:
        best_model = gb
        best_name = 'GradientBoosting'
        best_acc = gb_acc
    
    logger.info(f"  {best_name}: {best_acc:.2%} accuracy")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f'model_{pair}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    return {
        'symbol': pair,
        'strategy': PAIR_STRATEGIES_V2[pair]['strategy'],
        'params': PAIR_STRATEGIES_V2[pair]['params'],
        'model_type': best_name,
        'accuracy': float(best_acc),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X.shape[1],
        'model_path': model_path
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PAIR-SPECIFIC ML TRAINING V2 - HIGH FREQUENCY")
    print("=" * 70)
    print(f"Training with V2 optimized strategy parameters")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    downloader = DataDownloader()
    results = []
    
    for pair in PAIRS:
        print(f"\n>>> {pair}")
        print(f"    Strategy: {PAIR_STRATEGIES_V2[pair]['strategy']}")
        print(f"    Params: {PAIR_STRATEGIES_V2[pair]['params']}")
        
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
        
        result = train_model(pair, df)
        results.append(result)
        
        if 'error' in result:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Accuracy: {result['accuracy']:.2%}")
    
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
    
    results_file = os.path.join(MODELS_DIR, 'pair_training_results_v2.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {MODELS_DIR}")
    
    return results


if __name__ == '__main__':
    main()
