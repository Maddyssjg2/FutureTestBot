"""
IMPROVED Backtesting Script with Better Parameters
================================================
Based on backtest results, we adjust parameters for better win rates.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

# Only trade pairs with positive backtest results
GOOD_PAIRS = ["ADAUSDT", "AVAXUSDT", "DOTUSDT", "XRPUSDT"]

TIMEFRAME = "1h"
INITIAL_BALANCE = 10000

# IMPROVED PARAMETERS based on results
# - Wait for pullback (RSI 40-50 for longs, 50-60 for shorts)
# - Higher TP/SL ratio
# - Tighter stop loss
# - Only trade in clear trends
RISK_PER_TRADE = 0.01  # 1% risk (lower)
TP_PCT = 0.02  # 2% take profit
SL_PCT = 0.01  # 1% stop loss (1:2 ratio)


# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    def __init__(self):
        self.base_url = "https://data-api.binance.vision/api/v3"
    
    def get_klines(self, symbol: str, interval: str, limit: int = 1000, start_time: int = None):
        import requests
        url = f"{self.base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Download failed: {e}")
        return None


# ============================================================================
# IMPROVED INDICATORS
# ============================================================================

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr = calculate_atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), period)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.ewm(alpha=1/period).mean()


# ============================================================================
# IMPROVED BACKTEST ENGINE
# ============================================================================

class ImprovedBacktest:
    """Improved backtest with better entry/exit rules"""
    
    def __init__(self, pair: str, data: pd.DataFrame):
        self.pair = pair
        self.data = data
        self.trades = []
        self.balance = INITIAL_BALANCE
        self.equity = []
        
    def run(self) -> Dict:
        logger.info(f"Running improved backtest for {self.pair}")
        
        position = None
        
        for i in range(200, len(self.data)):
            lookback = self.data.iloc[max(0, i-200):i+1].copy()
            current = self.data.iloc[i]
            
            close = float(current['close'])
            high = float(current['high'])
            low = float(current['low'])
            
            # Calculate all indicators
            ema_9 = float(calculate_ema(lookback['close'], 9).iloc[-1])
            ema_21 = float(calculate_ema(lookback['close'], 21).iloc[-1])
            ema_50 = float(calculate_ema(lookback['close'], 50).iloc[-1])
            ema_200 = float(calculate_ema(lookback['close'], 200).iloc[-1])
            rsi = float(calculate_rsi(lookback['close'], 14).iloc[-1])
            atr = float(calculate_atr(lookback, 14).iloc[-1])
            adx = float(calculate_adx(lookback['high'], lookback['low'], lookback['close'], 14).iloc[-1])
            
            # ====== IMPROVED ENTRY LOGIC ======
            if position is None:
                signal = self._check_signal_improved(
                    close, ema_9, ema_21, ema_50, ema_200, rsi, atr, adx, high, low, lookback
                )
                
                if signal:
                    risk_amount = self.balance * RISK_PER_TRADE
                    sl_distance = abs(close - signal['stop_loss'])
                    position_size = risk_amount / sl_distance if sl_distance > 0 else 0
                    
                    position = {
                        'side': signal['direction'],
                        'entry_price': close,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'size': position_size,
                        'entry_idx': i,
                        'reason': signal['reason']
                    }
            
            # ====== EXIT LOGIC ======
            else:
                should_exit, reason = self._check_exit(position, close, high, low, i)
                
                if should_exit:
                    if position['side'] == 'LONG':
                        pnl_pct = (close - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - close) / position['entry_price']
                    
                    pnl = position['size'] * pnl_pct
                    
                    self.trades.append({
                        'pair': self.pair,
                        'side': position['side'],
                        'entry': position['entry_price'],
                        'exit': close,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'duration': i - position['entry_idx'],
                        'reason': reason,
                        'trade_reason': position['reason']
                    })
                    
                    self.balance += pnl
                    self.equity.append(self.balance)
                    position = None
            
            # Emergency stop
            if self.balance < INITIAL_BALANCE * 0.5:
                logger.warning(f"{self.pair}: Emergency stop at {self.balance:.2f}")
                break
        
        return self._calculate_stats()
    
    def _check_signal_improved(self, close, ema_9, ema_21, ema_50, ema_200, rsi, atr, adx, high, low, lookback) -> Dict:
        """IMPROVED signal detection with multiple confirmations"""
        
        # Trend: EMA 9 > EMA 21 AND both above EMA 50 (for long)
        long_trend = ema_9 > ema_21 > ema_50 and ema_50 > ema_200
        short_trend = ema_9 < ema_21 < ema_50 and ema_50 < ema_200
        
        # ADX filter - only trade if trend is strong
        if adx < 20:
            return None
        
        # ====== LONG ENTRY ======
        if long_trend:
            # RSI pullback (between 35-50 = oversold bounce zone)
            if 35 <= rsi <= 50:
                # Price near EMA 21 (pullback)
                distance_from_ema21 = abs(close - ema_21) / close * 100
                if distance_from_ema21 < 1.0:  # Within 1% of EMA 21
                    return {
                        'direction': 'LONG',
                        'stop_loss': round(close - (atr * 1.5), 2),  # Tighter SL
                        'take_profit': round(close + (atr * 3), 2),  # 1:2 ratio
                        'reason': 'ema_pullback'
                    }
            
            # Alternative: RSI oversold bounce
            elif rsi < 35 and rsi > 25:
                return {
                    'direction': 'LONG',
                    'stop_loss': round(close - (atr * 2), 2),
                    'take_profit': round(close + (atr * 4), 2),
                    'reason': 'rsi_oversold'
                }
        
        # ====== SHORT ENTRY ======
        if short_trend:
            # RSI pullback (between 50-65 = overbought drop zone)
            if 50 <= rsi <= 65:
                distance_from_ema21 = abs(close - ema_21) / close * 100
                if distance_from_ema21 < 1.0:
                    return {
                        'direction': 'SHORT',
                        'stop_loss': round(close + (atr * 1.5), 2),
                        'take_profit': round(close - (atr * 3), 2),
                        'reason': 'ema_pullback'
                    }
            
            elif rsi > 65 and rsi < 75:
                return {
                    'direction': 'SHORT',
                    'stop_loss': round(close + (atr * 2), 2),
                    'take_profit': round(close - (atr * 4), 2),
                    'reason': 'rsi_overbought'
                }
        
        return None
    
    def _check_exit(self, position, close, high, low, idx) -> Tuple[bool, str]:
        """Check exit conditions"""
        if position['side'] == 'LONG':
            if low <= position['stop_loss']:
                return True, 'STOP_LOSS'
            if high >= position['take_profit']:
                return True, 'TAKE_PROFIT'
        else:
            if high >= position['stop_loss']:
                return True, 'STOP_LOSS'
            if low <= position['take_profit']:
                return True, 'TAKE_PROFIT'
        
        # Time exit (24 hours for 1h timeframe)
        if idx - position['entry_idx'] >= 24:
            return True, 'TIME_EXIT'
        
        return False, ''
    
    def _calculate_stats(self) -> Dict:
        if not self.trades:
            return {'pair': self.pair, 'total_trades': 0, 'win_rate': 0}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        
        win_rate = len(wins) / len(self.trades) * 100
        pf = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        equity = INITIAL_BALANCE
        max_eq = INITIAL_BALANCE
        max_dd = 0
        for t in self.trades:
            equity += t['pnl']
            max_eq = max(max_eq, equity)
            dd = (max_eq - equity) / max_eq * 100
            max_dd = max(max_dd, dd)
        
        return {
            'pair': self.pair,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(pf, 2) if pf != float('inf') else 'inf',
            'total_pnl': round(self.balance - INITIAL_BALANCE, 2),
            'pnl_pct': round((self.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
            'max_drawdown': round(max_dd, 2),
            'final_balance': round(self.balance, 2),
            'avg_win': round(total_wins / len(wins), 2) if wins else 0,
            'avg_loss': round(total_losses / len(losses), 2) if losses else 0
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("IMPROVED BACKTEST - Pullback Strategy with RSI Confirmation")
    print("=" * 80)
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Strategy: EMA Pullback + RSI Confirmation + ADX Filter")
    print(f"TP/SL Ratio: 2:1")
    print(f"Risk per Trade: {RISK_PER_TRADE*100}%")
    print("=" * 80)
    
    downloader = DataDownloader()
    all_results = []
    
    for pair in PAIRS:
        print(f"\n>>> {pair}...")
        
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
        
        print(f"   {len(df)} candles")
        
        engine = ImprovedBacktest(pair, df)
        stats = engine.run()
        all_results.append(stats)
        
        print(f"   Trades: {stats['total_trades']} | Win: {stats['win_rate']}% | PF: {stats['profit_factor']} | PnL: ${stats['total_pnl']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Pair':<12} {'Trades':<8} {'Win%':<8} {'PF':<8} {'PnL':<12} {'DD%':<8}")
    print("-" * 80)
    
    total = {'trades': 0, 'wins': 0, 'pnl': 0}
    
    for r in all_results:
        print(f"{r['pair']:<12} {r['total_trades']:<8} {r['win_rate']:<8} {r['profit_factor']:<8} ${r['total_pnl']:<11,.0f} {r['max_drawdown']:<8}")
        total['trades'] += r['total_trades']
        total['wins'] += r['wins']
        total['pnl'] += r['total_pnl']
    
    overall_wr = total['wins'] / total['trades'] * 100 if total['trades'] > 0 else 0
    print("-" * 80)
    print(f"{'TOTAL':<12} {total['trades']:<8} {overall_wr:<8.1f} {'':<8} ${total['pnl']:<11,.0f}")
    print("=" * 80)
    
    return all_results


if __name__ == '__main__':
    main()
