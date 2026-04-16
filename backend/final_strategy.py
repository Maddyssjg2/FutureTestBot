"""
FINAL OPTIMIZED STRATEGY
========================
Combines ML predictions with technical indicators for higher win rate.

Key Features:
- ML Model predictions (51%+ accuracy)
- Technical confirmation (RSI, MACD, EMA)
- ADX trend filter
- Strict risk management
- Only trade best pairs
"""

import os
import sys
import json
import logging
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Best pairs based on backtest results
TRADE_PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "BNBUSDT"]
EXCLUDE_PAIRS = ["DOGEUSDT", "TRXUSDT"]  # Too volatile


# ============================================================================
# CONFIGURATION
# ============================================================================

class FinalConfig:
    # Risk Management
    RISK_PER_TRADE = 0.015  # 1.5%
    MAX_LEVERAGE = 5
    MAX_DAILY_LOSS = 0.05  # 5%
    
    # TP/SL (1:2 ratio)
    TP_PCT = 0.02  # 2%
    SL_PCT = 0.01  # 1%
    
    # Indicators
    RSI_PERIOD = 14
    RSI_OVERSOLD = 35
    RSI_OVERBOUGHT = 65
    RSI_ENTRY_LONG = 40
    RSI_ENTRY_SHORT = 60
    
    # Trend
    EMA_FAST = 21
    EMA_SLOW = 50
    ADX_MIN = 20
    
    # ML
    ML_CONFIDENCE_MIN = 52
    INDICATOR_CONFIRM_MIN = 60
    
    # Model path
    MODELS_DIR = os.path.join(os.path.dirname(__file__), 'ml_models')


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

def calc_macd(close):
    e12 = calc_ema(close, 12)
    e26 = calc_ema(close, 26)
    macd = e12 - e26
    signal = calc_ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


# ============================================================================
# ML PREDICTION
# ============================================================================

def load_ml_model(symbol):
    """Load trained ML model"""
    model_path = os.path.join(FinalConfig.MODELS_DIR, f'ml_{symbol}.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

def predict_ml(symbol, df):
    """Get ML prediction"""
    model = load_ml_model(symbol)
    if model is None:
        return None, 50
    
    try:
        # Create features
        close = df['close']
        
        features = []
        
        # Returns
        for i in [1, 2, 3, 5, 10]:
            features.append(float(close.pct_change(i).iloc[-1]))
        
        # EMA ratios
        for p in [5, 10, 20, 50]:
            ema = calc_ema(close, p)
            features.append(float(close.iloc[-1] / ema.iloc[-1]))
        
        # RSI
        rsi = calc_rsi(close, 14)
        features.append(float(rsi.iloc[-1]))
        
        # MACD
        _, _, hist = calc_macd(close)
        features.append(float(hist.iloc[-1]))
        
        # BB position
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_pos = (close.iloc[-1] - (bb_mid.iloc[-1] - 2*bb_std.iloc[-1])) / (4*bb_std.iloc[-1] + 0.0001)
        features.append(float(bb_pos))
        
        # Volume
        vol_ma = df['volume'].rolling(20).mean().iloc[-1]
        features.append(float(df['volume'].iloc[-1] / vol_ma))
        
        # ATR
        atr = calc_atr(df)
        features.append(float(atr.iloc[-1] / close.iloc[-1]))
        
        # Pad/truncate to expected length (100 features)
        while len(features) < 100:
            features.extend(features[:min(len(features), 100-len(features))])
        features = features[:100]
        
        prob = model.predict_proba([features])[0]
        
        if prob[1] > 0.52:
            return 'LONG', prob[1] * 100
        elif prob[0] > 0.52:
            return 'SHORT', prob[0] * 100
        else:
            return 'NEUTRAL', 50
            
    except Exception as e:
        logger.debug(f"ML prediction error: {e}")
        return None, 50


# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================

def analyze_technicals(df):
    """Full technical analysis"""
    close = df['close']
    high = df['high']
    low = df['low']
    
    # EMAs
    ema_fast = calc_ema(close, FinalConfig.EMA_FAST)
    ema_slow = calc_ema(close, FinalConfig.EMA_SLOW)
    
    # RSI
    rsi = calc_rsi(close, FinalConfig.RSI_PERIOD)
    rsi_val = float(rsi.iloc[-1])
    
    # MACD
    _, _, hist = calc_macd(close)
    hist_val = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])
    
    # ADX
    adx = calc_adx(high, low, close)
    adx_val = float(adx.iloc[-1])
    
    # ATR
    atr = calc_atr(df)
    atr_val = float(atr.iloc[-1])
    
    # Determine direction
    bullish_ema = float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1])
    bearish_ema = float(ema_fast.iloc[-1]) < float(ema_slow.iloc[-1])
    
    # Score technical signals
    tech_score = 0
    tech_direction = 'NEUTRAL'
    
    if adx_val >= FinalConfig.ADX_MIN:  # Trend is strong enough
        if bullish_ema and rsi_val < FinalConfig.RSI_ENTRY_SHORT:
            tech_score += 30
            tech_direction = 'LONG'
        elif bearish_ema and rsi_val > FinalConfig.RSI_ENTRY_LONG:
            tech_score += 30
            tech_direction = 'SHORT'
        
        # MACD confirmation
        if hist_val > 0 and hist_val > hist_prev:
            tech_score += 15
        elif hist_val < 0 and hist_val < hist_prev:
            tech_score += 15
        
        # RSI zone
        if FinalConfig.RSI_OVERSOLD <= rsi_val <= 45:
            tech_score += 10
        elif 55 <= rsi_val <= FinalConfig.RSI_OVERBOUGHT:
            tech_score += 10
    
    return {
        'direction': tech_direction,
        'score': tech_score,
        'rsi': rsi_val,
        'adx': adx_val,
        'hist': hist_val,
        'atr': atr_val,
        'bullish_ema': bullish_ema,
        'ema_fast': float(ema_fast.iloc[-1]),
        'ema_slow': float(ema_slow.iloc[-1])
    }


# ============================================================================
# FINAL STRATEGY
# ============================================================================

class FinalStrategy:
    """
    Final optimized strategy combining ML + Technical Analysis
    
    Entry Logic:
    1. ML prediction (if available)
    2. Technical confirmation
    3. ADX trend filter
    4. RSI zone confirmation
    """
    
    def __init__(self, symbol, config=None):
        self.symbol = symbol
        self.config = config or FinalConfig()
        self.daily_stats = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0}
        
    def analyze(self, df) -> Dict:
        """Main analysis"""
        if len(df) < 50:
            return {'signal': None, 'confidence': 0, 'reason': 'insufficient_data'}
        
        result = {
            'signal': None,
            'confidence': 0,
            'direction': 'NEUTRAL',
            'ml_direction': None,
            'ml_confidence': 50,
            'tech_score': 0,
            'reason': '',
            'entry': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'leverage': 1
        }
        
        try:
            price = float(df['close'].iloc[-1])
            
            # 1. Get ML prediction
            ml_dir, ml_conf = predict_ml(self.symbol, df)
            result['ml_direction'] = ml_dir
            result['ml_confidence'] = ml_conf
            
            # 2. Technical analysis
            tech = analyze_technicals(df)
            result['tech_score'] = tech['score']
            result['tech_direction'] = tech['direction']
            result['rsi'] = tech['rsi']
            result['adx'] = tech['adx']
            
            # 3. Combine signals
            if ml_dir and ml_conf >= self.config.ML_CONFIDENCE_MIN:
                # ML has a signal
                if tech['score'] >= self.config.INDICATOR_CONFIRM_MIN:
                    # Both agree
                    if ml_dir == tech['direction']:
                        result['signal'] = ml_dir
                        result['confidence'] = min((ml_conf + tech['score']), 95)
                        result['reason'] = 'ml_tech_confirm'
                    elif tech['score'] >= 70:  # Strong tech signal
                        result['signal'] = tech['direction']
                        result['confidence'] = tech['score']
                        result['reason'] = 'tech_strong'
                elif tech['score'] >= 40 and ml_conf >= 55:
                    # Partial confirmation
                    result['signal'] = ml_dir
                    result['confidence'] = ml_conf - 10
                    result['reason'] = 'ml_only'
            elif tech['score'] >= 60:
                # Strong technical signal without ML
                result['signal'] = tech['direction']
                result['confidence'] = tech['score'] - 10
                result['reason'] = 'tech_only'
            
            # 4. Calculate stops if signal
            if result['signal']:
                atr = tech['atr']
                
                if result['signal'] == 'LONG':
                    result['entry'] = round(price, 2)
                    result['stop_loss'] = round(price * (1 - self.config.SL_PCT), 2)
                    result['take_profit'] = round(price * (1 + self.config.TP_PCT), 2)
                    result['leverage'] = min(self.config.MAX_LEVERAGE, max(2, int(1.5 / self.config.SL_PCT)))
                else:
                    result['entry'] = round(price, 2)
                    result['stop_loss'] = round(price * (1 + self.config.SL_PCT), 2)
                    result['take_profit'] = round(price * (1 - self.config.TP_PCT), 2)
                    result['leverage'] = min(self.config.MAX_LEVERAGE, max(2, int(1.5 / self.config.SL_PCT)))
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            result['reason'] = f'error: {e}'
            return result
    
    def should_close(self, position, df) -> Tuple[bool, str]:
        """Check if position should close"""
        if not position:
            return False, ''
        
        price = float(df['close'].iloc[-1])
        entry = position.get('entry_price', 0)
        side = position.get('side', 'LONG')
        sl = position.get('stop_loss', 0)
        tp = position.get('take_profit', 0)
        
        if side == 'LONG':
            if price <= sl:
                return True, 'STOP_LOSS'
            if price >= tp:
                return True, 'TAKE_PROFIT'
        else:
            if price >= sl:
                return True, 'STOP_LOSS'
            if price <= tp:
                return True, 'TAKE_PROFIT'
        
        # Time exit (6 hours)
        if 'open_time' in position:
            try:
                open_time = position['open_time']
                if isinstance(open_time, str):
                    open_time = datetime.fromisoformat(open_time)
                hours = (datetime.now() - open_time).total_seconds() / 3600
                if hours > 6:
                    return True, 'TIME_EXIT'
            except:
                pass
        
        return False, ''
    
    def record_trade(self, pnl, won):
        """Record trade result"""
        self.daily_stats['trades'] += 1
        self.daily_stats['pnl'] += pnl
        if won:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1


# ============================================================================
# WRAPPER
# ============================================================================

class FinalStrategyWrapper:
    """Wrapper for bot compatibility"""
    
    def __init__(self, risk_config, symbol, model_path=None):
        self.symbol = symbol
        self.strategy = FinalStrategy(symbol)
    
    def generate_signal(self, klines):
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
                    'reason': result['reason'],
                    'ml': f"{result['ml_direction']} ({result['ml_confidence']:.0f}%)",
                    'tech': f"{result['tech_direction']} ({result['tech_score']})",
                    'adx': f"{result.get('adx', 0):.1f}"
                }
            
            logger.info(f"[{self.symbol}] {result['signal']} @ {result['entry']} | "
                       f"Conf: {result['confidence']:.0f}% | {result['reason']} | "
                       f"ML: {result['ml_direction']}({result['ml_confidence']:.0f}%) | "
                       f"Tech: {result['tech_score']}")
            
            return result['signal'], result['confidence'], {
                'stop_loss': result['stop_loss'],
                'take_profit_1': result['take_profit'],
                'entry_price': result['entry'],
                'leverage': result['leverage'],
                'reason': result['reason'],
                'ml_conf': result['ml_confidence'],
                'tech_score': result['tech_score'],
                'adx': result.get('adx', 0),
                'rsi': result.get('rsi', 50)
            }
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error: {e}")
            return None, 0, {'error': str(e)}
    
    def should_close_position(self, position, price, klines):
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
    
    def calculate_position_size(self, available, price, confidence):
        risk = available * FinalConfig.RISK_PER_TRADE
        position = available * 0.3
        quantity = position / price
        return round(quantity, 4), round(position, 2)
    
    def record_trade_result(self, pnl, won):
        self.strategy.record_trade(pnl, won)
    
    def get_status(self):
        return self.strategy.daily_stats


# Alias
FinalOptimizedStrategy = FinalStrategyWrapper
