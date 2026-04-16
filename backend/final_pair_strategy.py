"""
FINAL STRATEGY V2 - High Frequency Optimized Pair Rules + ML Predictions
========================================================================
Combines:
- Pair-specific optimized rules (75-80% win rate backtested)
- Higher trade frequency (20-200 trades per pair over 6 months)
- ML model predictions (55%+ confidence threshold)
- Risk management (1% TP, 2% SL, 5x max leverage)

Target: 75-80% win rate with 20-200+ trades per pair
"""

import os
import sys
import json
import logging
import warnings
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - V2 OPTIMIZED
# ============================================================================

PAIR_OPTIMIZATIONS = {
    "BTCUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 0}, "win_rate": 73.8, "trades": 42},
    "ETHUSDT": {"strategy": "Bollinger", "params": {"period": 20}, "win_rate": 71.2, "trades": 61},
    "SOLUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 0}, "win_rate": 80.3, "trades": 68},
    "XRPUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 25, "adx_min": 0}, "win_rate": 70.4, "trades": 27},
    "BNBUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 0}, "win_rate": 70.7, "trades": 196},
    "DOGEUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 35, "adx_min": 0}, "win_rate": 74.3, "trades": 35},
    "ADAUSDT": {"strategy": "Bollinger", "params": {"period": 20}, "win_rate": 80.6, "trades": 68},
    "TRXUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 0}, "win_rate": 70.2, "trades": 174},
    "AVAXUSDT": {"strategy": "RSI_MeanRev", "params": {"rsi_oversold": 30, "adx_min": 0}, "win_rate": 75.4, "trades": 70},
    "DOTUSDT": {"strategy": "Bollinger", "params": {"period": 20}, "win_rate": 76.8, "trades": 56},
}

RISK_CONFIG = {
    "RISK_PER_TRADE": 0.015,
    "MAX_LEVERAGE": 5,
    "TP_PCT": 0.01,
    "SL_PCT": 0.02,
    "MAX_DAILY_LOSS": 0.05,
    "MAX_HOLD_HOURS": 6,
    "ML_CONFIDENCE_THRESHOLD": 0.55,
}

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'pair_models_v2')
LOOKBACK = 30

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
# FEATURE ENGINEERING FOR ML
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
    
    strategy = PAIR_OPTIMIZATIONS.get(pair, {}).get('strategy', '')
    
    if strategy == 'RSI_MeanRev':
        df['rsi_zone'] = (df['rsi'] <= PAIR_OPTIMIZATIONS[pair]['params'].get('rsi_oversold', 30)).astype(int)
    
    elif strategy == 'Bollinger':
        period = PAIR_OPTIMIZATIONS[pair]['params'].get('period', 20)
        bb_middle = df['close'].rolling(period).mean()
        bb_std = df['close'].rolling(period).std()
        df['bb_lower'] = bb_middle - (2 * bb_std)
        df['bb_touched'] = (df['close'] <= df['bb_lower']).astype(int)
    
    return df


def prepare_ml_features(df: pd.DataFrame, pair: str) -> np.ndarray:
    df = create_features(df, pair)
    df = df.dropna()
    
    feature_cols = [
        'return_1', 'return_2', 'return_3', 'return_5', 'return_10',
        'ema_ratio_5', 'ema_ratio_10', 'ema_ratio_20', 'ema_ratio_50',
        'rsi', 'adx', 'volume_ratio', 'atr_ratio'
    ]
    
    strategy = PAIR_OPTIMIZATIONS.get(pair, {}).get('strategy', '')
    if strategy == 'RSI_MeanRev':
        feature_cols.extend(['rsi_zone'])
    elif strategy == 'Bollinger':
        feature_cols.extend(['bb_touched'])
    
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if len(df) < LOOKBACK:
        return np.array([])
    
    features = df[feature_cols].iloc[-LOOKBACK:].values.flatten()
    return features


# ============================================================================
# ML PREDICTOR
# ============================================================================

class MLPredictor:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.feature_cols = None
        self._load_model()
    
    def _load_model(self):
        model_path = os.path.join(MODELS_DIR, f'model_{self.symbol}.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"[{self.symbol}] ML model loaded")
            except Exception as e:
                logger.warning(f"[{self.symbol}] Failed to load ML model: {e}")
    
    def predict(self, df: pd.DataFrame) -> Dict:
        if self.model is None:
            return {'direction': 'NEUTRAL', 'confidence': 0, 'source': 'no_model'}
        
        try:
            features = prepare_ml_features(df, self.symbol)
            if len(features) == 0:
                return {'direction': 'NEUTRAL', 'confidence': 0, 'source': 'no_features'}
            
            prob = self.model.predict_proba(features.reshape(1, -1))[0]
            
            if prob[1] >= RISK_CONFIG["ML_CONFIDENCE_THRESHOLD"]:
                direction = 'LONG'
                confidence = prob[1] * 100
            elif prob[0] >= RISK_CONFIG["ML_CONFIDENCE_THRESHOLD"]:
                direction = 'SHORT'
                confidence = prob[0] * 100
            else:
                direction = 'NEUTRAL'
                confidence = max(prob) * 100
            
            return {
                'direction': direction,
                'confidence': float(confidence),
                'probability': float(max(prob)),
                'source': 'ml_model'
            }
        except Exception as e:
            logger.error(f"[{self.symbol}] ML prediction error: {e}")
            return {'direction': 'NEUTRAL', 'confidence': 0, 'source': 'error'}


# ============================================================================
# STRATEGY SIGNAL GENERATOR
# ============================================================================

class FinalStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.optimization = PAIR_OPTIMIZATIONS.get(symbol, PAIR_OPTIMIZATIONS["BTCUSDT"])
        self.strategy_name = self.optimization["strategy"]
        self.params = self.optimization["params"]
        self.expected_win_rate = self.optimization["win_rate"]
        self.ml_predictor = MLPredictor(symbol)
        self.stats = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
    
    def _check_stochastic(self, df: pd.DataFrame) -> bool:
        k_period = self.params.get('k_period', 14)
        d_period = self.params.get('d_period', 3)
        oversold = self.params.get('oversold', 20)
        
        k, d = calc_stochastic(df['high'], df['low'], df['close'], k_period, d_period)
        
        if len(k) < 2:
            return False
        
        k_val = float(k.iloc[-1])
        k_prev = float(k.iloc[-2])
        d_val = float(d.iloc[-1])
        
        return k_prev <= oversold and k_val > oversold and k_val < d_val
    
    def _check_rsi_meanrev(self, df: pd.DataFrame) -> bool:
        rsi_oversold = self.params.get('rsi_oversold', 30)
        adx_min = self.params.get('adx_min', 20)
        
        rsi = calc_rsi(df['close'])
        adx = calc_adx(df['high'], df['low'], df['close'])
        
        if len(rsi) < 2 or len(adx) < 2:
            return False
        
        return float(rsi.iloc[-1]) <= rsi_oversold and float(adx.iloc[-1]) >= adx_min
    
    def _check_breakout(self, df: pd.DataFrame) -> bool:
        lookback = self.params.get('lookback', 20)
        
        if len(df) < lookback + 5:
            return False
        
        close = float(df['close'].iloc[-1])
        recent_high = float(df['high'].iloc[-lookback:-1].max())
        
        return close > recent_high
    
    def _check_bollinger(self, df: pd.DataFrame) -> bool:
        period = self.params.get('period', 20)
        
        if len(df) < period + 1:
            return False
        
        middle = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        lower = middle - (2 * std)
        
        close = float(df['close'].iloc[-1])
        lower_val = float(lower.iloc[-1])
        
        return close <= lower_val
    
    def _rule_signal(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if self.strategy_name == "Stochastic":
            return self._check_stochastic(df), "Stochastic"
        elif self.strategy_name == "RSI_MeanRev":
            return self._check_rsi_meanrev(df), "RSI_MeanRev"
        elif self.strategy_name == "Breakout_PW":
            return self._check_breakout(df), "Breakout_PW"
        elif self.strategy_name == "Bollinger":
            return self._check_bollinger(df), "Bollinger"
        return False, "Unknown"
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        if len(df) < 50:
            return {'signal': None, 'confidence': 0, 'reason': 'insufficient_data'}
        
        result = {
            'signal': None,
            'confidence': 0,
            'strategy': self.strategy_name,
            'expected_wr': self.expected_win_rate,
            'entry': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'leverage': RISK_CONFIG["MAX_LEVERAGE"],
            'reason': '',
            'ml_direction': 'NEUTRAL',
            'ml_confidence': 0,
            'rule_signal': False,
            'ml_signal': False,
        }
        
        try:
            close = float(df['close'].iloc[-1])
            price = close
            
            rule_triggered, rule_name = self._rule_signal(df)
            result['rule_signal'] = rule_triggered
            
            ml_pred = self.ml_predictor.predict(df)
            result['ml_direction'] = ml_pred['direction']
            result['ml_confidence'] = ml_pred['confidence']
            result['ml_signal'] = ml_pred['direction'] == 'LONG'
            
            if rule_triggered and ml_pred['direction'] == 'LONG':
                result['signal'] = 'LONG'
                result['confidence'] = (self.expected_win_rate + ml_pred['confidence']) / 2
                result['entry'] = round(price, 2)
                result['stop_loss'] = round(price * (1 - RISK_CONFIG["SL_PCT"]), 2)
                result['take_profit'] = round(price * (1 + RISK_CONFIG["TP_PCT"]), 2)
                result['reason'] = f'{rule_name} + ML_confirm'
                logger.info(f"[{self.symbol}] LONG signal: {rule_name} + ML({ml_pred['confidence']:.1f}%)")
            
            elif rule_triggered and ml_pred['direction'] == 'NEUTRAL':
                result['signal'] = 'LONG'
                result['confidence'] = self.expected_win_rate * 0.9
                result['entry'] = round(price, 2)
                result['stop_loss'] = round(price * (1 - RISK_CONFIG["SL_PCT"]), 2)
                result['take_profit'] = round(price * (1 + RISK_CONFIG["TP_PCT"]), 2)
                result['reason'] = f'{rule_name}_only'
            
            elif not rule_triggered and ml_pred['direction'] == 'LONG' and ml_pred['confidence'] >= 70:
                result['signal'] = 'LONG'
                result['confidence'] = ml_pred['confidence'] * 0.9
                result['entry'] = round(price, 2)
                result['stop_loss'] = round(price * (1 - RISK_CONFIG["SL_PCT"]), 2)
                result['take_profit'] = round(price * (1 + RISK_CONFIG["TP_PCT"]), 2)
                result['reason'] = f'ML_only({ml_pred["confidence"]:.1f}%)'
            
            return result
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Analysis error: {e}")
            return {'signal': None, 'reason': f'error: {e}'}
    
    def should_close(self, position: Dict, df: pd.DataFrame) -> Tuple[bool, str]:
        if not position:
            return False, ''
        
        close = float(df['close'].iloc[-1])
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
        self.stats['trades'] += 1
        self.stats['pnl'] += pnl
        if won:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1


# ============================================================================
# WRAPPER FOR BOT
# ============================================================================

class FinalStrategyWrapper:
    def __init__(self, risk_config: dict, symbol: str, model_path=None):
        self.symbol = symbol
        self.strategy = FinalStrategy(symbol)
        self.risk_config = risk_config
    
    def generate_signal(self, klines_15m=None, klines_1h=None, klines_4h=None) -> Dict:
        try:
            if klines_1h is not None and len(klines_1h) >= 50:
                klines = klines_1h
            elif klines_15m is not None and len(klines_15m) >= 50:
                klines = klines_15m
            else:
                return {'signal': None, 'confidence': 0}
            
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
                return {
                    'signal': None,
                    'confidence': 0,
                    'reason': result.get('reason', 'no_signal'),
                }
            
            logger.info(f"[{self.symbol}] {result['signal']} @ {result['entry']} | "
                       f"WR: {result['expected_wr']}% | ML: {result['ml_confidence']:.1f}%")
            
            return {
                'signal': result['signal'],
                'confidence': result['confidence'],
                'entry_price': result['entry'],
                'stop_loss': result['stop_loss'],
                'take_profit': result['take_profit'],
                'take_profit_1': result['take_profit'],
                'take_profit_2': result['entry'] * 1.02,
                'leverage': result['leverage'],
                'risk_reward': 0.5,
                'confluence_count': 2 if result['rule_signal'] and result['ml_signal'] else 1,
                'strategies': [result['reason']],
                'expected_wr': result['expected_wr'],
                'ml_direction': result['ml_direction'],
                'ml_confidence': result['ml_confidence'],
            }
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Signal generation error: {e}")
            return {'signal': None, 'confidence': 0, 'error': str(e)}
    
    def should_close_position(self, position: Dict, price: float, klines=None) -> Tuple[bool, str]:
        try:
            if klines is None:
                return False, ''
            
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
    
    def calculate_position_size(self, available: float, price: float, stop_loss: float) -> float:
        risk = available * RISK_CONFIG["RISK_PER_TRADE"]
        sl_distance = abs(price - stop_loss)
        position = risk / sl_distance if sl_distance > 0 else available * 0.3
        return position / price
    
    def record_trade_result(self, pnl: float, won: bool):
        self.strategy.record_trade(pnl, won)
    
    def get_status(self) -> Dict:
        return self.strategy.stats


FinalStrategyWrapper.__init__.__doc__ = """
Final Strategy Wrapper for Multi-Symbol Bot
Combines:
- Optimized pair-specific rules (75%+ backtested win rate)
- ML model predictions (55%+ confidence threshold)
- Risk management (1% TP, 2% SL, 5x max leverage)
"""
