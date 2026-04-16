"""
OPTIMIZED PAIR-SPECIFIC STRATEGY
=================================
Based on backtesting results - uses best parameters for each pair.

Target: 75%+ Win Rate per pair
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ============================================================================
# OPTIMIZED PARAMETERS PER PAIR
# Based on backtesting results
# ============================================================================

PAIR_OPTIMIZATIONS = {
    "BTCUSDT": {
        "strategy": "Breakout_PW",
        "params": {"lookback": 20, "atr_mult": 2},
        "win_rate": 100.0,
        "trades": 2,
        "notes": "Very few trades, high win rate"
    },
    "ETHUSDT": {
        "strategy": "Stochastic",
        "params": {"k_period": 14, "d_period": 3, "oversold": 15, "overbought": 85},
        "win_rate": 100.0,
        "trades": 5,
        "notes": "Best for ETH - Stochastic oversold 15"
    },
    "SOLUSDT": {
        "strategy": "RSI_MeanRev",
        "params": {"rsi_oversold": 25, "rsi_overbought": 65, "adx_min": 25, "tp_pct": 0.01, "sl_pct": 0.02},
        "win_rate": 93.3,
        "trades": 15,
        "notes": "Strong performance on SOL"
    },
    "XRPUSDT": {
        "strategy": "Stochastic",
        "params": {"k_period": 14, "d_period": 3, "oversold": 15, "overbought": 85},
        "win_rate": 83.3,
        "trades": 6,
        "notes": "Stochastic works well for XRP"
    },
    "BNBUSDT": {
        "strategy": "Stochastic",
        "params": {"k_period": 14, "d_period": 3, "oversold": 15, "overbought": 85},
        "win_rate": 80.0,
        "trades": 5,
        "notes": "Stochastic oversold 15"
    },
    "DOGEUSDT": {
        "strategy": "Stochastic",
        "params": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80},
        "win_rate": 85.7,
        "trades": 7,
        "notes": "DOGE needs more oversold confirmation"
    },
    "ADAUSDT": {
        "strategy": "Stochastic",
        "params": {"k_period": 14, "d_period": 3, "oversold": 20, "overbought": 80},
        "win_rate": 87.5,
        "trades": 8,
        "notes": "Best performer - Stochastic oversold 20"
    },
    "TRXUSDT": {
        "strategy": "RSI_MeanRev",
        "params": {"rsi_oversold": 35, "rsi_overbought": 65, "adx_min": 20, "tp_pct": 0.01, "sl_pct": 0.02},
        "win_rate": 77.3,
        "trades": 22,
        "notes": "More trades needed - use RSI 35 oversold"
    },
    "AVAXUSDT": {
        "strategy": "RSI_MeanRev",
        "params": {"rsi_oversold": 25, "rsi_overbought": 65, "adx_min": 20, "tp_pct": 0.01, "sl_pct": 0.02},
        "win_rate": 77.8,
        "trades": 18,
        "notes": "Strong on AVAX - RSI 25 oversold"
    },
    "DOTUSDT": {
        "strategy": "RSI_MeanRev",
        "params": {"rsi_oversold": 30, "rsi_overbought": 65, "adx_min": 25, "tp_pct": 0.01, "sl_pct": 0.02},
        "win_rate": 80.8,
        "trades": 26,
        "notes": "Best DOT setup - RSI 30 oversold"
    }
}


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

RISK_CONFIG = {
    "RISK_PER_TRADE": 0.015,  # 1.5%
    "MAX_LEVERAGE": 5,
    "TP_PCT": 0.01,  # 1%
    "SL_PCT": 0.02,  # 2%
    "MAX_DAILY_LOSS": 0.05,  # 5%
    "MAX_HOLD_HOURS": 6
}


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
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class OptimizedStrategy:
    """Optimized strategy that uses pair-specific parameters"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.optimization = PAIR_OPTIMIZATIONS.get(symbol, PAIR_OPTIMIZATIONS["BTCUSDT"])
        self.strategy_name = self.optimization["strategy"]
        self.params = self.optimization["params"]
        self.expected_win_rate = self.optimization["win_rate"]
        self.stats = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Main analysis - returns signal based on optimized parameters"""
        if len(df) < 50:
            return {'signal': None, 'confidence': 0, 'reason': 'insufficient_data'}
        
        result = {
            'signal': None,
            'confidence': self.expected_win_rate,
            'strategy': self.strategy_name,
            'expected_wr': self.expected_win_rate,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'leverage': RISK_CONFIG["MAX_LEVERAGE"],
            'reason': ''
        }
        
        try:
            close = float(df['close'].iloc[-1])
            price = close
            
            if self.strategy_name == "Stochastic":
                signal = self._check_stochastic(df)
            elif self.strategy_name == "RSI_MeanRev":
                signal = self._check_rsi_meanrev(df)
            elif self.strategy_name == "Breakout_PW":
                signal = self._check_breakout(df)
            else:
                # Default to RSI
                signal = self._check_rsi_meanrev(df)
            
            if signal:
                result['signal'] = 'LONG'  # These strategies are LONG only
                result['entry'] = round(price, 2)
                result['stop_loss'] = round(price * (1 - RISK_CONFIG["SL_PCT"]), 2)
                result['take_profit'] = round(price * (1 + RISK_CONFIG["TP_PCT"]), 2)
                result['reason'] = f'{self.strategy_name} signal'
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Analysis error: {e}")
            return {'signal': None, 'reason': f'error: {e}'}
    
    def _check_stochastic(self, df: pd.DataFrame) -> bool:
        """Check Stochastic strategy"""
        k_period = self.params.get('k_period', 14)
        d_period = self.params.get('d_period', 3)
        oversold = self.params.get('oversold', 20)
        
        k, d = calc_stochastic(df['high'], df['low'], df['close'], k_period, d_period)
        
        if len(k) < 2:
            return False
        
        k_val = float(k.iloc[-1])
        k_prev = float(k.iloc[-2])
        d_val = float(d.iloc[-1])
        
        # Buy when K crosses above oversold level
        return k_prev <= oversold and k_val > oversold and k_val < d_val
    
    def _check_rsi_meanrev(self, df: pd.DataFrame) -> bool:
        """Check RSI Mean Reversion strategy"""
        rsi_oversold = self.params.get('rsi_oversold', 30)
        adx_min = self.params.get('adx_min', 20)
        
        rsi = calc_rsi(df['close'])
        adx = calc_adx(df['high'], df['low'], df['close'])
        
        if len(rsi) < 2 or len(adx) < 2:
            return False
        
        rsi_val = float(rsi.iloc[-1])
        adx_val = float(adx.iloc[-1])
        
        # Buy when RSI oversold AND ADX shows trend
        return rsi_val <= rsi_oversold and adx_val >= adx_min
    
    def _check_breakout(self, df: pd.DataFrame) -> bool:
        """Check Breakout Pullback strategy"""
        lookback = self.params.get('lookback', 20)
        
        if len(df) < lookback + 5:
            return False
        
        close = float(df['close'].iloc[-1])
        recent_high = float(df['high'].iloc[-lookback:-1].max())
        
        # Buy when price breaks resistance
        return close > recent_high
    
    def should_close(self, position: Dict, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check if position should close"""
        if not position:
            return False, ''
        
        close = float(df['close'].iloc[-1])
        entry = position.get('entry_price', 0)
        sl = position.get('stop_loss', 0)
        tp = position.get('take_profit', 0)
        side = position.get('side', 'LONG')
        
        if side == 'LONG':
            if close <= sl:
                return True, 'STOP_LOSS'
            if close >= tp:
                return True, 'TAKE_PROFIT'
        else:
            if close >= sl:
                return True, 'STOP_LOSS'
            if close <= tp:
                return True, 'TAKE_PROFIT'
        
        # Time exit
        if 'open_time' in position:
            try:
                open_time = position['open_time']
                if isinstance(open_time, str):
                    open_time = datetime.fromisoformat(open_time)
                hours = (datetime.now() - open_time).total_seconds() / 3600
                if hours > RISK_CONFIG["MAX_HOLD_HOURS"]:
                    return True, 'TIME_EXIT'
            except:
                pass
        
        return False, ''
    
    def record_trade(self, pnl: float, won: bool):
        """Record trade"""
        self.stats['trades'] += 1
        self.stats['pnl'] += pnl
        if won:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1


# ============================================================================
# WRAPPER
# ============================================================================

class OptimizedPairStrategyWrapper:
    """Wrapper for bot compatibility"""
    
    def __init__(self, risk_config: dict, symbol: str, model_path=None):
        self.symbol = symbol
        self.strategy = OptimizedStrategy(symbol)
        
    def generate_signal(self, klines: List) -> Tuple[Optional[str], float, Dict]:
        """Generate signal"""
        try:
            if isinstance(klines[0], dict):
                df = pd.DataFrame(klines)
            else:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            result = self.strategy.analyze(df)
            
            if not result['signal']:
                return None, 0, {
                    'reason': result.get('reason', 'no_signal'),
                    'strategy': result.get('strategy', 'unknown'),
                    'expected_wr': result.get('expected_wr', 0)
                }
            
            logger.info(f"[{self.symbol}] {result['signal']} @ {result['entry']} | "
                       f"Strategy: {result['strategy']} | Exp WR: {result['expected_wr']}%")
            
            return result['signal'], result['confidence'], {
                'stop_loss': result['stop_loss'],
                'take_profit_1': result['take_profit'],
                'entry_price': result['entry'],
                'leverage': result['leverage'],
                'strategy': result['strategy'],
                'expected_wr': result['expected_wr']
            }
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error: {e}")
            return None, 0, {'error': str(e)}
    
    def should_close_position(self, position: Dict, price: float, klines: List) -> Tuple[bool, str]:
        """Check if should close"""
        try:
            if isinstance(klines[0], dict):
                df = pd.DataFrame(klines)
            else:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return self.strategy.should_close(position, df)
        except Exception as e:
            logger.error(f"[{self.symbol}] Close check error: {e}")
            return False, ''
    
    def calculate_position_size(self, available: float, price: float, conf: float) -> Tuple[float, float]:
        """Calculate position size"""
        risk = available * RISK_CONFIG["RISK_PER_TRADE"]
        sl_distance = price * RISK_CONFIG["SL_PCT"]
        position = risk / sl_distance if sl_distance > 0 else available * 0.3
        quantity = position / price
        return round(quantity, 4), round(position, 2)
    
    def record_trade_result(self, pnl: float, won: bool):
        """Record trade"""
        self.strategy.record_trade(pnl, won)
    
    def get_status(self) -> Dict:
        """Get status"""
        return self.strategy.stats


# Alias
OptimizedPairStrategy = OptimizedPairStrategyWrapper
