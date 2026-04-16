"""
Backtesting Script for Core Strategy
===================================
Downloads 6 months of data and runs backtest on all 10 pairs.
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

# Import strategy components
sys.path.insert(0, os.path.dirname(__file__))
from core_strategy import (
    CoreStrategy, StrategySettings, calculate_ema, calculate_rsi, 
    calculate_atr, calculate_adx, calculate_vwap, calculate_bollinger_width
)


# ============================================================================
# CONFIGURATION
# ============================================================================

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "DOTUSDT"
]

TIMEFRAME = "1h"  # Use 1h for faster backtest
MONTHS_BACK = 6
INITIAL_BALANCE = 10000  # USDT

# Trading Parameters
RISK_PER_TRADE = 0.02  # 2% risk
TP_PCT = 0.015  # 1.5% take profit
SL_PCT = 0.01   # 1.0% stop loss


# ============================================================================
# DATA DOWNLOADER
# ============================================================================

class DataDownloader:
    """Download historical data from Binance"""
    
    def __init__(self):
        self.base_url = "https://data-api.binance.vision/api/v3"
    
    def get_klines(self, symbol: str, interval: str, limit: int = 1000, start_time: int = None):
        """Get klines from Binance"""
        import requests
        
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Download failed for {symbol}: {e}")
        return None


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Backtesting engine for Core Strategy"""
    
    def __init__(self, pair: str, data: pd.DataFrame):
        self.pair = pair
        self.data = data
        self.strategy = CoreStrategy()
        self.strategy.settings.TP_PCT = TP_PCT
        self.strategy.settings.SL_PCT = SL_PCT
        
        # Results
        self.trades = []
        self.balance = INITIAL_BALANCE
        self.equity_curve = []
        
    def run(self) -> Dict:
        """Run backtest"""
        logger.info(f"Running backtest for {self.pair} with {len(self.data)} candles")
        
        position = None
        entry_price = 0
        entry_time = None
        
        for i in range(100, len(self.data)):  # Start from bar 100 for indicators
            current_bar = self.data.iloc[i]
            lookback = self.data.iloc[max(0, i-100):i+1].copy()
            
            close = float(current_bar['close'])
            high = float(current_bar['high'])
            low = float(current_bar['low'])
            
            # Check if we have a position
            if position is None:
                # Generate signal
                signal = self._check_signal(lookback)
                
                if signal:
                    # Calculate position size
                    risk_amount = self.balance * RISK_PER_TRADE
                    sl_distance = abs(close - signal['stop_loss'])
                    position_size = risk_amount / sl_distance if sl_distance > 0 else 0
                    
                    position = {
                        'side': signal['direction'],
                        'entry_price': close,
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'size': position_size,
                        'entry_time': current_bar.name if hasattr(current_bar, 'name') else i,
                        'confidence': signal['confidence']
                    }
                    entry_price = close
                    entry_time = i
                    
            else:
                # Check exit conditions
                should_exit, reason = self._check_exit(position, close, high, low, i)
                
                if should_exit:
                    # Calculate PnL
                    if position['side'] == 'LONG':
                        pnl_pct = (close - position['entry_price']) / position['entry_price']
                    else:
                        pnl_pct = (position['entry_price'] - close) / position['entry_price']
                    
                    pnl = position['size'] * pnl_pct
                    
                    # Record trade
                    trade = {
                        'pair': self.pair,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': close,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'duration': i - entry_time,
                        'reason': reason,
                        'confidence': position['confidence']
                    }
                    self.trades.append(trade)
                    
                    # Update balance
                    self.balance += pnl
                    
                    # Record equity
                    self.equity_curve.append({
                        'bar': i,
                        'balance': self.balance,
                        'drawdown': (INITIAL_BALANCE - self.balance) / INITIAL_BALANCE * 100
                    })
                    
                    position = None
                    entry_price = 0
            
            # Emergency exit if balance drops too low
            if self.balance < INITIAL_BALANCE * 0.5:
                logger.warning(f"{self.pair}: Balance dropped below 50%, stopping")
                break
        
        # Close any open position at end
        if position:
            close = float(self.data.iloc[-1]['close'])
            if position['side'] == 'LONG':
                pnl_pct = (close - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - close) / position['entry_price']
            pnl = position['size'] * pnl_pct
            self.balance += pnl
        
        return self._calculate_stats()
    
    def _check_signal(self, df: pd.DataFrame) -> Dict:
        """Check for trading signal"""
        if len(df) < 50:
            return None
        
        close = float(df['close'].iloc[-1])
        
        # Calculate indicators
        ema_50 = float(calculate_ema(df['close'], 50).iloc[-1])
        ema_200 = float(calculate_ema(df['close'], 200).iloc[-1])
        rsi = float(calculate_rsi(df['close'], 14).iloc[-1])
        
        # ATR for stops
        atr = float(calculate_atr(df, 14).iloc[-1])
        
        # Determine direction based on EMA
        if ema_50 > ema_200:
            direction = 'LONG'
            stop_loss = close - (atr * 2)
            take_profit = close + (atr * 3)
        else:
            direction = 'SHORT'
            stop_loss = close + (atr * 2)
            take_profit = close - (atr * 3)
        
        # Calculate confidence based on EMA gap and RSI
        ema_gap = abs(ema_50 - ema_200) / close * 100
        
        # Long conditions
        if direction == 'LONG':
            if rsi < 70 and rsi > 30:  # Not overbought/oversold
                confidence = min(ema_gap * 10 + 50, 95)
                return {
                    'direction': direction,
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2),
                    'confidence': confidence
                }
        else:
            if rsi > 30 and rsi < 70:
                confidence = min(ema_gap * 10 + 50, 95)
                return {
                    'direction': direction,
                    'stop_loss': round(stop_loss, 2),
                    'take_profit': round(take_profit, 2),
                    'confidence': confidence
                }
        
        return None
    
    def _check_exit(self, position: Dict, close: float, high: float, low: float, bar_idx: int) -> Tuple[bool, str]:
        """Check if position should be exited"""
        if position['side'] == 'LONG':
            # Stop loss hit
            if low <= position['stop_loss']:
                return True, 'STOP_LOSS'
            # Take profit hit
            if high >= position['take_profit']:
                return True, 'TAKE_PROFIT'
        else:
            if high >= position['stop_loss']:
                return True, 'STOP_LOSS'
            if low <= position['take_profit']:
                return True, 'TAKE_PROFIT'
        
        # Max hold time (48 hours = 48 candles for 1h timeframe)
        if bar_idx > 0:
            duration = bar_idx - (position.get('entry_idx', 0))
            if duration >= 48:
                return True, 'TIME_EXIT'
        
        return False, ''
    
    def _calculate_stats(self) -> Dict:
        """Calculate backtest statistics"""
        if not self.trades:
            return {
                'pair': self.pair,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_pnl': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'expectancy': 0
            }
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_wins = sum(t['pnl'] for t in wins)
        total_losses = abs(sum(t['pnl'] for t in losses))
        
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        equity = INITIAL_BALANCE
        max_equity = INITIAL_BALANCE
        max_drawdown = 0
        for t in self.trades:
            equity += t['pnl']
            max_equity = max(max_equity, equity)
            dd = (max_equity - equity) / max_equity * 100
            max_drawdown = max(max_drawdown, dd)
        
        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = [t['pnl_pct'] for t in self.trades]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        
        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
        
        return {
            'pair': self.pair,
            'total_trades': len(self.trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'total_pnl': round(self.balance - INITIAL_BALANCE, 2),
            'total_pnl_pct': round((self.balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 2),
            'final_balance': round(self.balance, 2)
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("CORE STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Pairs: {', '.join(PAIRS)}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Period: Last {MONTHS_BACK} months")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Risk per Trade: {RISK_PER_TRADE*100}%")
    print(f"Take Profit: {TP_PCT*100}%")
    print(f"Stop Loss: {SL_PCT*100}%")
    print("=" * 80)
    
    downloader = DataDownloader()
    all_results = []
    
    for pair in PAIRS:
        print(f"\n>>> Downloading data for {pair}...")
        
        # Calculate start time (6 months ago)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=MONTHS_BACK * 30)).timestamp() * 1000)
        
        # Download data
        klines = downloader.get_klines(pair, TIMEFRAME, limit=2000, start_time=start_time)
        
        if not klines:
            print(f"   No data for {pair}, skipping...")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"   Loaded {len(df)} candles from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        
        # Run backtest
        print(f"   Running backtest...")
        engine = BacktestEngine(pair, df)
        stats = engine.run()
        all_results.append(stats)
        
        # Print results for this pair
        print(f"\n{'='*60}")
        print(f"  {pair} RESULTS")
        print(f"{'='*60}")
        print(f"  Total Trades:      {stats['total_trades']}")
        print(f"  Win Rate:          {stats['win_rate']}%")
        print(f"  Profit Factor:     {stats['profit_factor']}")
        print(f"  Total PnL:         ${stats['total_pnl']:,.2f} ({stats['total_pnl_pct']}%)")
        print(f"  Max Drawdown:      {stats['max_drawdown']}%")
        print(f"  Sharpe Ratio:      {stats['sharpe_ratio']}")
        print(f"  Expectancy:        ${stats['expectancy']}")
        print(f"  Final Balance:     ${stats['final_balance']:,.2f}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY - ALL PAIRS")
    print("=" * 80)
    print(f"{'Pair':<12} {'Trades':<8} {'Win%':<8} {'PF':<8} {'PnL':<12} {'DD%':<8} {'Sharpe':<8}")
    print("-" * 80)
    
    total_trades = 0
    total_wins = 0
    total_pnl = 0
    max_dd = 0
    
    for r in all_results:
        print(f"{r['pair']:<12} {r['total_trades']:<8} {r['win_rate']:<8} {r['profit_factor']:<8} ${r['total_pnl']:<11,.0f} {r['max_drawdown']:<8} {r['sharpe_ratio']:<8}")
        total_trades += r['total_trades']
        total_wins += r['winning_trades']
        total_pnl += r['total_pnl']
        max_dd = max(max_dd, r['max_drawdown'])
    
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print("-" * 80)
    print(f"{'TOTAL':<12} {total_trades:<8} {overall_win_rate:<8.1f} {'':<8} ${total_pnl:<11,.0f} {max_dd:<8}")
    print("=" * 80)
    
    # Save results
    results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(os.path.dirname(__file__), '..', results_file), 'w') as f:
        json.dump({
            'settings': {
                'pairs': PAIRS,
                'timeframe': TIMEFRAME,
                'months': MONTHS_BACK,
                'initial_balance': INITIAL_BALANCE,
                'risk_per_trade': RISK_PER_TRADE,
                'tp_pct': TP_PCT,
                'sl_pct': SL_PCT
            },
            'results': all_results,
            'summary': {
                'total_trades': total_trades,
                'overall_win_rate': overall_win_rate,
                'total_pnl': total_pnl,
                'max_drawdown': max_dd
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


if __name__ == '__main__':
    main()
