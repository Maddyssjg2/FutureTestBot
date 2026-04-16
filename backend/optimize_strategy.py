"""
COMPREHENSIVE STRATEGY OPTIMIZER
================================
Tests multiple strategies and parameter combinations to find 75%+ win rate setups.

Strategies Tested:
1. Mean Reversion (RSI extremes)
2. Trend Following (EMA crossovers)
3. Breakout Trading (support/resistance)
4. VWAP Reversion
5. Bollinger Band Bounce
6. MACD Histogram Reversal

Target: 75%+ Win Rate
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
from itertools import product

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

TIMEFRAME = "1h"
INITIAL_BALANCE = 10000
MIN_WIN_RATE = 75  # Target


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

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def macd(close, f=12, s=26, sig=9):
    ema_fast = ema(close, f)
    ema_slow = ema(close, s)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, sig)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger(close, period=20, std=2):
    mid = close.rolling(period).mean()
    std_dev = close.rolling(period).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    return upper, mid, lower

def vwap(df):
    tp = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = (tp * df['volume']).cumsum()
    cum_vol_total = df['volume'].cumsum()
    return cum_vol / cum_vol_total

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(alpha=1/period).mean()


# ============================================================================
# STRATEGIES
# ============================================================================

class StrategyTester:
    """Test different strategies with various parameters"""
    
    def __init__(self, pair, df):
        self.pair = pair
        self.df = df
    
    def test_mean_reversion_rsi(self, rsi_oversold=30, rsi_overbought=70, rsi_exit=50,
                                tp_pct=0.01, sl_pct=0.02, min_adx=20) -> Dict:
        """RSI Mean Reversion - Buy oversold, sell overbought"""
        df = self.df.copy()
        df['rsi'] = rsi(df['close'])
        df['adx'] = adx(df['high'], df['low'], df['close'])
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(50, len(df)):
            close = float(df['close'].iloc[i])
            rsi_val = float(df['rsi'].iloc[i])
            adx_val = float(df['adx'].iloc[i])
            
            if position is None:
                # Buy when RSI oversold AND ADX shows trend
                if rsi_val <= rsi_oversold and adx_val >= min_adx:
                    sl = close * (1 - sl_pct)
                    tp = close * (1 + tp_pct)
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                # Exit on RSI overbought or TP/SL
                if rsi_val >= rsi_exit or close <= position['sl'] or close >= position['tp']:
                    pnl = close - position['entry']
                    if close <= position['sl']:
                        reason = 'SL'
                    elif close >= position['tp']:
                        reason = 'TP'
                    else:
                        reason = 'RSI'
                    trades.append({'pnl': pnl, 'reason': reason})
                    balance += pnl
                    position = None
        
        if position:
            close = float(df['close'].iloc[-1])
            pnl = close - position['entry']
            trades.append({'pnl': pnl, 'reason': 'END'})
            balance += pnl
        
        return self._calc_stats(trades, balance, 'RSI_MeanRev')
    
    def test_bollinger_bounce(self, bb_period=20, bb_std=2, 
                              tp_pct=0.01, sl_pct=0.015, min_atr_pct=0.5) -> Dict:
        """Bollinger Band Bounce - Buy at lower band, sell at middle/upper"""
        df = self.df.copy()
        upper, mid, lower = bollinger(df['close'], bb_period, bb_std)
        df['atr'] = atr(df)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(bb_period + 5, len(df)):
            close = float(df['close'].iloc[i])
            lower_val = float(lower.iloc[i])
            mid_val = float(mid.iloc[i])
            atr_pct = float(df['atr_pct'].iloc[i])
            
            if position is None:
                # Buy when price touches lower band
                if close <= lower_val and atr_pct >= min_atr_pct:
                    sl = lower_val * 0.99  # Just below lower band
                    tp = mid_val  # Target middle band
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                if close <= position['sl']:
                    trades.append({'pnl': position['sl'] - position['entry'], 'reason': 'SL'})
                    balance += position['sl'] - position['entry']
                    position = None
                elif close >= position['tp']:
                    trades.append({'pnl': position['tp'] - position['entry'], 'reason': 'TP'})
                    balance += position['tp'] - position['entry']
                    position = None
        
        return self._calc_stats(trades, balance, 'BB_Bounce')
    
    def test_macd_histogram_reversal(self, hist_threshold=-0.5, 
                                    tp_pct=0.008, sl_pct=0.015, 
                                    min_adx=25) -> Dict:
        """MACD Histogram Reversal - Buy when histogram turns positive"""
        df = self.df.copy()
        _, _, df['hist'] = macd(df['close'])
        df['adx'] = adx(df['high'], df['low'], df['close'])
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(35, len(df)):
            close = float(df['close'].iloc[i])
            hist = float(df['hist'].iloc[i])
            hist_prev = float(df['hist'].iloc[i-1])
            adx_val = float(df['adx'].iloc[i])
            
            if position is None:
                # Buy when histogram crosses above threshold (negative to positive)
                if hist_prev < hist_threshold and hist > 0 and adx_val >= min_adx:
                    sl = close * (1 - sl_pct)
                    tp = close * (1 + tp_pct)
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                if close <= position['sl'] or close >= position['tp']:
                    pnl = close - position['entry'] if close >= position['tp'] else position['sl'] - position['entry']
                    reason = 'TP' if close >= position['tp'] else 'SL'
                    trades.append({'pnl': pnl, 'reason': reason})
                    balance += pnl
                    position = None
        
        return self._calc_stats(trades, balance, 'MACD_Hist')
    
    def test_vwap_reversion(self, vwap_dev=0.005, 
                           tp_pct=0.01, sl_pct=0.02,
                           min_adx=15) -> Dict:
        """VWAP Reversion - Buy when price deviates below VWAP"""
        df = self.df.copy()
        df['vwap'] = vwap(df)
        df['adx'] = adx(df['high'], df['low'], df['close'])
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(20, len(df)):
            close = float(df['close'].iloc[i])
            vwap_val = float(df['vwap'].iloc[i])
            adx_val = float(df['adx'].iloc[i])
            deviation = (close - vwap_val) / vwap_val
            
            if position is None:
                # Buy when price is below VWAP by threshold
                if deviation < -vwap_dev and adx_val >= min_adx:
                    sl = close * (1 - sl_pct)
                    tp = close * (1 + tp_pct)
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                if close <= position['sl'] or close >= position['tp']:
                    pnl = close - position['entry'] if close >= position['tp'] else position['sl'] - position['entry']
                    reason = 'TP' if close >= position['tp'] else 'SL'
                    trades.append({'pnl': pnl, 'reason': reason})
                    balance += pnl
                    position = None
        
        return self._calc_stats(trades, balance, 'VWAP_Rev')
    
    def test_trendContinuation_ema(self, fast_ema=21, slow_ema=50,
                                 tp_pct=0.015, sl_pct=0.01,
                                 min_adx=25) -> Dict:
        """Trend Continuation - Buy when EMA fast crosses above slow in uptrend"""
        df = self.df.copy()
        df['ema_fast'] = ema(df['close'], fast_ema)
        df['ema_slow'] = ema(df['close'], slow_ema)
        df['adx'] = adx(df['high'], df['low'], df['close'])
        df['rsi'] = rsi(df['close'])
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(slow_ema + 5, len(df)):
            close = float(df['close'].iloc[i])
            ema_f = float(df['ema_fast'].iloc[i])
            ema_s = float(df['ema_slow'].iloc[i])
            ema_f_prev = float(df['ema_fast'].iloc[i-1])
            adx_val = float(df['adx'].iloc[i])
            rsi_val = float(df['rsi'].iloc[i])
            
            if position is None:
                # Buy on EMA crossover with ADX and RSI filter
                if ema_f_prev <= ema_s and ema_f > ema_s and adx_val >= min_adx and 40 <= rsi_val <= 70:
                    sl = close * (1 - sl_pct)
                    tp = close * (1 + tp_pct)
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                if close <= position['sl'] or close >= position['tp']:
                    pnl = close - position['entry'] if close >= position['tp'] else position['sl'] - position['entry']
                    reason = 'TP' if close >= position['tp'] else 'SL'
                    trades.append({'pnl': pnl, 'reason': reason})
                    balance += pnl
                    position = None
        
        return self._calc_stats(trades, balance, 'EMA_Trend')
    
    def test_breakout_pullback(self, lookback=20, atr_mult=2,
                              tp_pct=0.02, sl_pct=0.01) -> Dict:
        """Breakout with Pullback - Buy after resistance breakout with pullback"""
        df = self.df.copy()
        df['atr'] = atr(df)
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(lookback + 5, len(df)):
            close = float(df['close'].iloc[i])
            atr_val = float(df['atr'].iloc[i])
            
            # Calculate resistance (recent high)
            recent_high = float(df['high'].iloc[i-lookback:i].max())
            atr_distance = atr_val * atr_mult
            
            if position is None:
                # Buy when price breaks resistance
                if close > recent_high + atr_distance * 0.5:
                    sl = recent_high  # Stop at breakout level
                    tp = close + (close - sl) * 2  # 2:1 reward
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                if close <= position['sl'] or close >= position['tp']:
                    pnl = close - position['entry'] if close >= position['tp'] else position['sl'] - position['entry']
                    reason = 'TP' if close >= position['tp'] else 'SL'
                    trades.append({'pnl': pnl, 'reason': reason})
                    balance += pnl
                    position = None
        
        return self._calc_stats(trades, balance, 'Breakout_PW')
    
    def test_stochastic_reversion(self, k_period=14, d_period=3, 
                                  oversold=20, overbought=80,
                                  tp_pct=0.01, sl_pct=0.02) -> Dict:
        """Stochastic Oscillator Reversion"""
        df = self.df.copy()
        
        # Calculate Stochastic
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        df['k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
        df['d'] = df['k'].rolling(d_period).mean()
        
        trades = []
        balance = INITIAL_BALANCE
        position = None
        
        for i in range(k_period + d_period + 5, len(df)):
            close = float(df['close'].iloc[i])
            k = float(df['k'].iloc[i])
            k_prev = float(df['k'].iloc[i-1])
            d = float(df['d'].iloc[i])
            
            if position is None:
                # Buy when Stochastic crosses up from oversold
                if k_prev <= oversold and k > oversold and k < d:
                    sl = close * (1 - sl_pct)
                    tp = close * (1 + tp_pct)
                    position = {'entry': close, 'sl': sl, 'tp': tp}
            else:
                if close <= position['sl'] or close >= position['tp']:
                    pnl = close - position['entry'] if close >= position['tp'] else position['sl'] - position['entry']
                    reason = 'TP' if close >= position['tp'] else 'SL'
                    trades.append({'pnl': pnl, 'reason': reason})
                    balance += pnl
                    position = None
        
        return self._calc_stats(trades, balance, 'Stochastic')
    
    def _calc_stats(self, trades, balance, strategy_name) -> Dict:
        if not trades:
            return {'strategy': strategy_name, 'trades': 0, 'win_rate': 0}
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades) * 100
        total_pnl = balance - INITIAL_BALANCE
        
        return {
            'strategy': strategy_name,
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'pnl_pct': round((total_pnl / INITIAL_BALANCE) * 100, 2),
            'final_balance': round(balance, 2)
        }


# ============================================================================
# OPTIMIZER
# ============================================================================

class StrategyOptimizer:
    """Optimize strategies to find 75%+ win rate setups"""
    
    def __init__(self, pair, df):
        self.pair = pair
        self.df = df
        self.tester = StrategyTester(pair, df)
    
    def run_all_strategies(self) -> List[Dict]:
        """Run all strategies and return results"""
        results = []
        
        # Strategy 1: RSI Mean Reversion (multiple parameter sets)
        logger.info(f"  Testing RSI Mean Reversion...")
        for rsi_buy in [25, 30, 35]:
            for rsi_sell in [65, 70, 75]:
                for adx in [15, 20, 25]:
                    result = self.tester.test_mean_reversion_rsi(
                        rsi_oversold=rsi_buy, rsi_overbought=rsi_sell,
                        min_adx=adx, tp_pct=0.01, sl_pct=0.02
                    )
                    result['params'] = f'rsi({rsi_buy}/{rsi_sell})_adx{adx}'
                    results.append(result)
        
        # Strategy 2: Bollinger Bounce
        logger.info(f"  Testing Bollinger Bounce...")
        for bb_std in [1.5, 2.0, 2.5]:
            result = self.tester.test_bollinger_bounce(bb_std=bb_std)
            result['params'] = f'bb_std{bb_std}'
            results.append(result)
        
        # Strategy 3: MACD Histogram
        logger.info(f"  Testing MACD Histogram...")
        for hist_thresh in [-1.0, -0.5, -0.2]:
            for adx in [20, 25, 30]:
                result = self.tester.test_macd_histogram_reversal(
                    hist_threshold=hist_thresh, min_adx=adx
                )
                result['params'] = f'hist{hist_thresh}_adx{adx}'
                results.append(result)
        
        # Strategy 4: VWAP Reversion
        logger.info(f"  Testing VWAP Reversion...")
        for vwap_dev in [0.003, 0.005, 0.008]:
            result = self.tester.test_vwap_reversion(vwap_dev=vwap_dev)
            result['params'] = f'vwap_dev{vwap_dev}'
            results.append(result)
        
        # Strategy 5: EMA Trend
        logger.info(f"  Testing EMA Trend...")
        for fast in [9, 21]:
            for slow in [50, 100]:
                result = self.tester.test_trendContinuation_ema(
                    fast_ema=fast, slow_ema=slow
                )
                result['params'] = f'ema{fast}/{slow}'
                results.append(result)
        
        # Strategy 6: Stochastic
        logger.info(f"  Testing Stochastic...")
        for oversold in [15, 20, 25]:
            result = self.tester.test_stochastic_reversion(oversold=oversold)
            result['params'] = f'stoch_ov{oversold}'
            results.append(result)
        
        # Strategy 7: Breakout Pullback
        logger.info(f"  Testing Breakout Pullback...")
        for lookback in [10, 20, 30]:
            result = self.tester.test_breakout_pullback(lookback=lookback)
            result['params'] = f'lbp{lookback}'
            results.append(result)
        
        return sorted(results, key=lambda x: x.get('win_rate', 0), reverse=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE STRATEGY OPTIMIZER")
    print("=" * 80)
    print(f"Target: {MIN_WIN_RATE}%+ Win Rate")
    print(f"Pairs: {', '.join(PAIRS)}")
    print("=" * 80)
    
    downloader = DataDownloader()
    all_results = {}
    
    for pair in PAIRS:
        print(f"\n>>> {pair}")
        
        # Download data
        start_time = int((datetime.now() - timedelta(days=180)).timestamp() * 1000)
        klines = downloader.get_klines(pair, TIMEFRAME, limit=2000, start_time=start_time)
        
        if not klines:
            print(f"   No data")
            continue
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"   {len(df)} candles loaded")
        
        # Run optimization
        optimizer = StrategyOptimizer(pair, df)
        results = optimizer.run_all_strategies()
        
        # Filter for 75%+ win rate
        high_wr_results = [r for r in results if r.get('win_rate', 0) >= MIN_WIN_RATE]
        
        print(f"   Total strategies tested: {len(results)}")
        print(f"   75%+ Win Rate strategies found: {len(high_wr_results)}")
        
        if high_wr_results:
            print(f"\n   TOP STRATEGIES FOR {pair}:")
            print(f"   {'Strategy':<20} {'Win%':<8} {'Trades':<8} {'PnL':<12} {'Params'}")
            print(f"   " + "-" * 70)
            for r in high_wr_results[:5]:
                print(f"   {r['strategy']:<20} {r['win_rate']:<8} {r['trades']:<8} ${r['total_pnl']:<11,.0f} {r.get('params', '')}")
        else:
            # Show best results
            print(f"\n   BEST RESULTS (Highest Win Rate):")
            print(f"   {'Strategy':<20} {'Win%':<8} {'Trades':<8} {'PnL':<12}")
            print(f"   " + "-" * 60)
            for r in results[:3]:
                print(f"   {r['strategy']:<20} {r['win_rate']:<8} {r['trades']:<8} ${r['total_pnl']:<11,.0f}")
        
        all_results[pair] = {
            'all_results': results,
            'high_wr_results': high_wr_results,
            'best_result': results[0] if results else None
        }
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - PAIRS WITH 75%+ WIN RATE")
    print("=" * 80)
    
    high_wr_pairs = []
    for pair, data in all_results.items():
        if data['high_wr_results']:
            high_wr_pairs.append(pair)
            print(f"\n{pair}: {len(data['high_wr_results'])} strategies with 75%+ WR")
            for r in data['high_wr_results'][:3]:
                print(f"  - {r['strategy']}: {r['win_rate']}% WR, {r['trades']} trades, ${r['total_pnl']}")
    
    if not high_wr_pairs:
        print("\nNo pairs achieved 75%+ win rate.")
        print("Showing best results per pair:")
        for pair, data in all_results.items():
            best = data.get('best_result')
            if best:
                print(f"\n{pair}: {best['win_rate']}% WR ({best['strategy']})")
    
    # Save results
    output_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(os.path.dirname(__file__), '..', output_file), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    main()
