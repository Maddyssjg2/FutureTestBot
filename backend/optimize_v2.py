"""
RE-OPTIMIZATION FOR HIGHER SIGNAL FREQUENCY
==========================================
Target: 20-40+ trades per pair over 6 months with 90%+ win rate

Strategies tested:
- RSI Mean Reversion (multiple oversold levels)
- Stochastic (multiple oversold levels)
- MACD Crossover
- Bollinger Band Bounce
- EMA Crossover
- Volume Spike
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
import requests

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

TIMEFRAME = "1h"
LOOKBACK_DAYS = 180
MIN_TRADES = 20
TARGET_WR = 90.0
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'pair_models')

# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    def __init__(self):
        self.base_url = "https://data-api.binance.vision/api/v3"
    
    def get_klines(self, symbol, interval, limit=2000, start_time=None):
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

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = calc_ema(macd, signal)
    return macd, macd_signal

def calc_bb(close, period=20, std_dev=2):
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest_rsi_meanrev(df, rsi_oversold, rsi_overbought, adx_min=0, tp_pct=0.01, sl_pct=0.02):
    """RSI Mean Reversion strategy"""
    rsi = calc_rsi(df['close'])
    adx = calc_adx(df['high'], df['low'], df['close'])
    
    trades = []
    position = None
    
    for i in range(50, len(df) - 1):
        if position is not None:
            continue
            
        rsi_val = float(rsi.iloc[i])
        adx_val = float(adx.iloc[i])
        
        if adx_min > 0 and adx_val < adx_min:
            continue
            
        if rsi_val <= rsi_oversold:
            entry_price = float(df['close'].iloc[i])
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            
            for j in range(i + 1, min(i + 72, len(df))):
                close_j = float(df['close'].iloc[j])
                
                if close_j <= stop_loss:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'LOSS', 'pnl': -sl_pct})
                    break
                elif close_j >= take_profit:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'WIN', 'pnl': tp_pct})
                    break
            else:
                close_j = float(df['close'].iloc[i + 72 if i + 72 < len(df) else -1])
                trades.append({'entry': entry_price, 'exit': close_j, 'result': 'EXPIRED', 'pnl': (close_j - entry_price) / entry_price})
    
    return trades

def backtest_stochastic(df, oversold, overbought, tp_pct=0.01, sl_pct=0.02):
    """Stochastic Oscillator strategy"""
    k, d = calc_stochastic(df['high'], df['low'], df['close'])
    
    trades = []
    
    for i in range(20, len(df) - 1):
        k_val = float(k.iloc[i])
        k_prev = float(k.iloc[i-1])
        d_val = float(d.iloc[i])
        
        if k_prev <= oversold and k_val > oversold and k_val < d_val:
            entry_price = float(df['close'].iloc[i])
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            
            for j in range(i + 1, min(i + 72, len(df))):
                close_j = float(df['close'].iloc[j])
                
                if close_j <= stop_loss:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'LOSS', 'pnl': -sl_pct})
                    break
                elif close_j >= take_profit:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'WIN', 'pnl': tp_pct})
                    break
            else:
                close_j = float(df['close'].iloc[i + 72 if i + 72 < len(df) else -1])
                trades.append({'entry': entry_price, 'exit': close_j, 'result': 'EXPIRED', 'pnl': (close_j - entry_price) / entry_price})
    
    return trades

def backtest_macd(df, tp_pct=0.01, sl_pct=0.02):
    """MACD Crossover strategy"""
    macd, signal = calc_macd(df['close'])
    
    trades = []
    
    for i in range(30, len(df) - 1):
        macd_val = float(macd.iloc[i])
        macd_prev = float(macd.iloc[i-1])
        signal_val = float(signal.iloc[i])
        signal_prev = float(signal.iloc[i-1])
        
        if macd_prev <= signal_prev and macd_val > signal_val:
            entry_price = float(df['close'].iloc[i])
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            
            for j in range(i + 1, min(i + 72, len(df))):
                close_j = float(df['close'].iloc[j])
                
                if close_j <= stop_loss:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'LOSS', 'pnl': -sl_pct})
                    break
                elif close_j >= take_profit:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'WIN', 'pnl': tp_pct})
                    break
            else:
                close_j = float(df['close'].iloc[i + 72 if i + 72 < len(df) else -1])
                trades.append({'entry': entry_price, 'exit': close_j, 'result': 'EXPIRED', 'pnl': (close_j - entry_price) / entry_price})
    
    return trades

def backtest_bollinger(df, bb_period=20, tp_pct=0.01, sl_pct=0.02):
    """Bollinger Band Bounce strategy"""
    upper, middle, lower = calc_bb(df['close'], bb_period)
    
    trades = []
    
    for i in range(bb_period + 1, len(df) - 1):
        close_i = float(df['close'].iloc[i])
        lower_i = float(lower.iloc[i])
        
        if close_i <= lower_i:
            entry_price = close_i
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            
            for j in range(i + 1, min(i + 72, len(df))):
                close_j = float(df['close'].iloc[j])
                
                if close_j <= stop_loss:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'LOSS', 'pnl': -sl_pct})
                    break
                elif close_j >= take_profit:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'WIN', 'pnl': tp_pct})
                    break
            else:
                close_j = float(df['close'].iloc[i + 72 if i + 72 < len(df) else -1])
                trades.append({'entry': entry_price, 'exit': close_j, 'result': 'EXPIRED', 'pnl': (close_j - entry_price) / entry_price})
    
    return trades

def backtest_ema_cross(df, fast_period=9, slow_period=21, tp_pct=0.01, sl_pct=0.02):
    """EMA Crossover strategy"""
    ema_fast = calc_ema(df['close'], fast_period)
    ema_slow = calc_ema(df['close'], slow_period)
    
    trades = []
    
    for i in range(slow_period + 1, len(df) - 1):
        fast_val = float(ema_fast.iloc[i])
        fast_prev = float(ema_fast.iloc[i-1])
        slow_val = float(ema_slow.iloc[i])
        slow_prev = float(ema_slow.iloc[i-1])
        
        if fast_prev <= slow_prev and fast_val > slow_val:
            entry_price = float(df['close'].iloc[i])
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            
            for j in range(i + 1, min(i + 72, len(df))):
                close_j = float(df['close'].iloc[j])
                
                if close_j <= stop_loss:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'LOSS', 'pnl': -sl_pct})
                    break
                elif close_j >= take_profit:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'WIN', 'pnl': tp_pct})
                    break
            else:
                close_j = float(df['close'].iloc[i + 72 if i + 72 < len(df) else -1])
                trades.append({'entry': entry_price, 'exit': close_j, 'result': 'EXPIRED', 'pnl': (close_j - entry_price) / entry_price})
    
    return trades

def backtest_volume_spike(df, volume_mult=2.0, tp_pct=0.01, sl_pct=0.02):
    """Volume Spike strategy"""
    vol_ma = df['volume'].rolling(20).mean()
    
    trades = []
    
    for i in range(25, len(df) - 1):
        vol_i = float(df['volume'].iloc[i])
        vol_ma_i = float(vol_ma.iloc[i])
        close_i = float(df['close'].iloc[i])
        close_prev = float(df['close'].iloc[i-1])
        
        if vol_i > vol_ma_i * volume_mult and close_i > close_prev:
            entry_price = close_i
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
            
            for j in range(i + 1, min(i + 72, len(df))):
                close_j = float(df['close'].iloc[j])
                
                if close_j <= stop_loss:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'LOSS', 'pnl': -sl_pct})
                    break
                elif close_j >= take_profit:
                    trades.append({'entry': entry_price, 'exit': close_j, 'result': 'WIN', 'pnl': tp_pct})
                    break
            else:
                close_j = float(df['close'].iloc[i + 72 if i + 72 < len(df) else -1])
                trades.append({'entry': entry_price, 'exit': close_j, 'result': 'EXPIRED', 'pnl': (close_j - entry_price) / entry_price})
    
    return trades

def evaluate_trades(trades):
    """Evaluate trading results"""
    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'avg_pnl': 0}
    
    wins = sum(1 for t in trades if t['result'] == 'WIN')
    losses = sum(1 for t in trades if t['result'] == 'LOSS')
    total = wins + losses
    
    if total == 0:
        return {'trades': len(trades), 'wins': wins, 'losses': losses, 'win_rate': 0, 'avg_pnl': 0}
    
    win_rate = (wins / total) * 100
    avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
    
    return {
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl
    }

# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_pair(df, pair):
    """Find best parameters for a pair with high trade count and win rate"""
    results = []
    
    # RSI Mean Reversion - test various oversold levels
    for rsi_ov in [25, 30, 35, 40, 45]:
        for adx_min in [0, 15, 20, 25]:
            trades = backtest_rsi_meanrev(df, rsi_oversold=rsi_ov, rsi_overbought=70, adx_min=adx_min)
            eval_result = evaluate_trades(trades)
            if eval_result['trades'] >= MIN_TRADES:
                results.append({
                    'strategy': 'RSI_MeanRev',
                    'params': {'rsi_oversold': rsi_ov, 'adx_min': adx_min},
                    **eval_result
                })
    
    # Stochastic - test various oversold levels
    for stoch_ov in [15, 20, 25, 30, 35]:
        trades = backtest_stochastic(df, oversold=stoch_ov, overbought=85)
        eval_result = evaluate_trades(trades)
        if eval_result['trades'] >= MIN_TRADES:
            results.append({
                'strategy': 'Stochastic',
                'params': {'oversold': stoch_ov},
                **eval_result
            })
    
    # MACD Crossover
    trades = backtest_macd(df)
    eval_result = evaluate_trades(trades)
    if eval_result['trades'] >= MIN_TRADES:
        results.append({
            'strategy': 'MACD_Cross',
            'params': {},
            **eval_result
        })
    
    # Bollinger Bands
    for period in [15, 20, 25]:
        trades = backtest_bollinger(df, bb_period=period)
        eval_result = evaluate_trades(trades)
        if eval_result['trades'] >= MIN_TRADES:
            results.append({
                'strategy': 'Bollinger',
                'params': {'period': period},
                **eval_result
            })
    
    # EMA Crossover
    for fast, slow in [(9, 21), (12, 26), (5, 20)]:
        trades = backtest_ema_cross(df, fast_period=fast, slow_period=slow)
        eval_result = evaluate_trades(trades)
        if eval_result['trades'] >= MIN_TRADES:
            results.append({
                'strategy': 'EMA_Cross',
                'params': {'fast': fast, 'slow': slow},
                **eval_result
            })
    
    # Volume Spike
    for vol_mult in [1.5, 2.0, 2.5]:
        trades = backtest_volume_spike(df, volume_mult=vol_mult)
        eval_result = evaluate_trades(trades)
        if eval_result['trades'] >= MIN_TRADES:
            results.append({
                'strategy': 'Volume_Spike',
                'params': {'volume_mult': vol_mult},
                **eval_result
            })
    
    # Sort by win rate (descending), then by trades (descending)
    results.sort(key=lambda x: (x['win_rate'], x['trades']), reverse=True)
    
    return results

def run_optimization():
    """Run full optimization for all pairs"""
    print("=" * 80)
    print("RE-OPTIMIZATION: TARGET 20-40+ TRADES | 90%+ WIN RATE")
    print("=" * 80)
    
    downloader = DataDownloader()
    all_results = {}
    
    for pair in PAIRS:
        print(f"\n>>> {pair}")
        print("-" * 50)
        
        start_time = int((datetime.now() - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
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
        
        results = optimize_pair(df, pair)
        
        if not results:
            print(f"    No valid combinations found (need {MIN_TRADES}+ trades)")
            # Try with lower trade requirement
            print(f"    Trying with relaxed criteria...")
            for rsi_ov in [30, 35, 40]:
                trades = backtest_rsi_meanrev(df, rsi_oversold=rsi_ov, rsi_overbought=70, adx_min=0)
                eval_result = evaluate_trades(trades)
                print(f"    RSI {rsi_ov}: {eval_result['trades']} trades, {eval_result['win_rate']:.1f}% WR")
            
            for stoch_ov in [25, 30, 35]:
                trades = backtest_stochastic(df, oversold=stoch_ov, overbought=85)
                eval_result = evaluate_trades(trades)
                print(f"    Stochastic {stoch_ov}: {eval_result['trades']} trades, {eval_result['win_rate']:.1f}% WR")
            continue
        
        print(f"\n    TOP RESULTS (sorted by win rate):")
        print(f"    {'Strategy':<20} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win Rate':<10} {'Avg PnL':<10}")
        print(f"    {'-'*62}")
        
        for i, r in enumerate(results[:10]):
            print(f"    {r['strategy']:<20} {r['trades']:<8} {r['wins']:<6} {r['losses']:<8} {r['win_rate']:.1f}%{'':<5} {r['avg_pnl']*100:.2f}%")
        
        # Best result
        best = results[0]
        print(f"\n    BEST: {best['strategy']} with {best['trades']} trades, {best['win_rate']:.1f}% WR")
        
        all_results[pair] = {
            'best': best,
            'all_results': results[:10]
        }
    
    return all_results

def save_results(all_results):
    """Save optimization results"""
    output = {}
    for pair, data in all_results.items():
        if data.get('best'):
            output[pair] = {
                'strategy': data['best']['strategy'],
                'params': data['best']['params'],
                'win_rate': data['best']['win_rate'],
                'trades': data['best']['trades'],
                'wins': data['best']['wins'],
                'losses': data['best']['losses'],
                'avg_pnl': data['best']['avg_pnl']
            }
    
    filename = f'optimization_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FINAL OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Pair':<12} {'Strategy':<20} {'Trades':<8} {'WR':<8} {'Avg PnL':<10}")
    print("-"*60)
    
    for pair, data in output.items():
        print(f"{pair:<12} {data['strategy']:<20} {data['trades']:<8} {data['win_rate']:.1f}%{'':<3} {data['avg_pnl']*100:.2f}%")
    
    print("="*80)
    print(f"Results saved to: {filename}")
    
    return output

if __name__ == '__main__':
    results = run_optimization()
    optimized_config = save_results(results)
