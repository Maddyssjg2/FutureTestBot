"""
Advanced Binance Futures Trading Strategies
5 High-Probability Strategies with Confluence Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


@dataclass
class SignalResult:
    signal: SignalType
    confidence: float  # 0-100
    strategy_signals: Dict[str, bool]  # Which strategies triggered
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confluence_count: int
    timeframe_analysis: Dict[str, str]
    metadata: Dict


class TechnicalIndicators:
    """Technical indicator calculations"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        macd_line = TechnicalIndicators.ema(series, fast) - TechnicalIndicators.ema(series, slow)
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bbands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            '0': high,
            '0.236': high - 0.236 * diff,
            '0.382': high - 0.382 * diff,
            '0.5': high - 0.5 * diff,
            '0.618': high - 0.618 * diff,
            '0.65': high - 0.65 * diff,
            '0.786': high - 0.786 * diff,
            '1': low,
            '1.272': low - 0.272 * diff,
            '1.618': low - 0.618 * diff
        }


class Strategy1_EMA_Trend_Confluence:
    """
    Strategy 1: EMA Trend Confluence
    Timeframe: 15m entry / 1H / 4H bias
    """
    
    @staticmethod
    def analyze(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        signals = {'long': [], 'short': []}
        
        # 15m Analysis (Entry)
        df_15m['ema8'] = TechnicalIndicators.ema(df_15m['close'], 8)
        df_15m['ema34'] = TechnicalIndicators.ema(df_15m['close'], 34)
        df_15m['macd'], df_15m['macd_signal'], _ = TechnicalIndicators.macd(df_15m['close'])
        df_15m['rsi'] = TechnicalIndicators.rsi(df_15m['close'])
        
        # 4H Analysis (Trend Bias)
        df_4h['ema200'] = TechnicalIndicators.ema(df_4h['close'], 200)
        
        latest_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        latest_4h = df_4h.iloc[-1]
        
        # LONG Conditions
        ema_cross_long = prev_15m['ema8'] <= prev_15m['ema34'] and latest_15m['ema8'] > latest_15m['ema34']
        above_ema200 = latest_4h['close'] > latest_4h['ema200']
        macd_bullish = latest_15m['macd'] > latest_15m['macd_signal']
        rsi_range_long = 45 <= latest_15m['rsi'] <= 65
        
        if ema_cross_long:
            signals['long'].append('EMA8 crossed above EMA34')
        if above_ema200:
            signals['long'].append('Price above EMA200 (4H)')
        if macd_bullish:
            signals['long'].append('MACD bullish')
        if rsi_range_long:
            signals['long'].append('RSI in 45-65 range')
        
        # SHORT Conditions
        ema_cross_short = prev_15m['ema8'] >= prev_15m['ema34'] and latest_15m['ema8'] < latest_15m['ema34']
        below_ema200 = latest_4h['close'] < latest_4h['ema200']
        macd_bearish = latest_15m['macd'] < latest_15m['macd_signal']
        rsi_range_short = 35 <= latest_15m['rsi'] <= 55
        
        if ema_cross_short:
            signals['short'].append('EMA8 crossed below EMA34')
        if below_ema200:
            signals['short'].append('Price below EMA200 (4H)')
        if macd_bearish:
            signals['short'].append('MACD bearish')
        if rsi_range_short:
            signals['short'].append('RSI in 35-55 range')
        
        # Determine signal
        long_count = len(signals['long'])
        short_count = len(signals['short'])
        
        if long_count >= 3:
            confidence = min(40 + long_count * 15, 95)
            return SignalType.LONG, confidence, {'signals': signals['long'], 'ema34': latest_15m['ema34']}
        elif short_count >= 3:
            confidence = min(40 + short_count * 15, 95)
            return SignalType.SHORT, confidence, {'signals': signals['short'], 'ema34': latest_15m['ema34']}
        
        return SignalType.NONE, 0, {}


class Strategy2_RSI_Divergence_MACD:
    """
    Strategy 2: RSI Divergence + MACD Zero Cross
    Timeframe: 1H / 4H
    """
    
    @staticmethod
    def find_divergence(df: pd.DataFrame, lookback: int = 5) -> Tuple[Optional[str], Optional[float]]:
        """Find RSI divergence"""
        df['rsi'] = TechnicalIndicators.rsi(df['close'])
        
        # Look for swing highs/lows
        highs = df['high'].rolling(window=3, center=True).max()
        lows = df['low'].rolling(window=3, center=True).min()
        
        swing_highs = df['high'] == highs
        swing_lows = df['low'] == lows
        
        # Bullish divergence: price lower low, RSI higher low
        for i in range(2, min(lookback + 2, len(df))):
            if swing_lows.iloc[-i]:
                prev_price_low = df['low'].iloc[-i]
                prev_rsi_low = df['rsi'].iloc[-i]
                
                for j in range(i + 1, min(i + lookback, len(df))):
                    if swing_lows.iloc[-j]:
                        curr_price_low = df['low'].iloc[-j]
                        curr_rsi_low = df['rsi'].iloc[-j]
                        
                        if curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low:
                            return 'bullish', prev_price_low
        
        # Bearish divergence: price higher high, RSI lower high
        for i in range(2, min(lookback + 2, len(df))):
            if swing_highs.iloc[-i]:
                prev_price_high = df['high'].iloc[-i]
                prev_rsi_high = df['rsi'].iloc[-i]
                
                for j in range(i + 1, min(i + lookback, len(df))):
                    if swing_highs.iloc[-j]:
                        curr_price_high = df['high'].iloc[-j]
                        curr_rsi_high = df['rsi'].iloc[-j]
                        
                        if curr_price_high > prev_price_high and curr_rsi_high < prev_rsi_high:
                            return 'bearish', prev_price_high
        
        return None, None
    
    @staticmethod
    def analyze(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        signals = {'long': [], 'short': []}
        
        # MACD Analysis
        df_1h['macd'], df_1h['macd_signal'], df_1h['macd_hist'] = TechnicalIndicators.macd(df_1h['close'])
        df_1h['volume_ma'] = df_1h['volume'].rolling(20).mean()
        
        latest = df_1h.iloc[-1]
        prev = df_1h.iloc[-2]
        
        # Volume spike
        volume_spike = latest['volume'] > latest['volume_ma'] * 1.2
        
        # Divergence detection
        divergence_type, divergence_price = Strategy2_RSI_Divergence_MACD.find_divergence(df_4h)
        
        # MACD zero cross
        macd_above_zero_cross = prev['macd'] < 0 and latest['macd'] > 0
        macd_below_zero_cross = prev['macd'] > 0 and latest['macd'] < 0
        
        # LONG Conditions
        if divergence_type == 'bullish':
            signals['long'].append('Bullish RSI divergence')
        if macd_above_zero_cross:
            signals['long'].append('MACD crossed above zero')
        if volume_spike:
            signals['long'].append('Volume spike confirmed')
        
        # SHORT Conditions
        if divergence_type == 'bearish':
            signals['short'].append('Bearish RSI divergence')
        if macd_below_zero_cross:
            signals['short'].append('MACD crossed below zero')
        if volume_spike:
            signals['short'].append('Volume confirmed')
        
        long_count = len(signals['long'])
        short_count = len(signals['short'])
        
        if long_count >= 2 and divergence_type == 'bullish':
            confidence = min(50 + long_count * 15, 90)
            return SignalType.LONG, confidence, {'signals': signals['long'], 'divergence_price': divergence_price}
        elif short_count >= 2 and divergence_type == 'bearish':
            confidence = min(50 + short_count * 15, 90)
            return SignalType.SHORT, confidence, {'signals': signals['short'], 'divergence_price': divergence_price}
        
        return SignalType.NONE, 0, {}


class Strategy3_BB_Squeeze_Breakout:
    """
    Strategy 3: Bollinger Band Squeeze Breakout
    Timeframe: 1H entry / 4H confirm
    """
    
    @staticmethod
    def calculate_bandwidth(df: pd.DataFrame, period: int = 20) -> pd.Series:
        upper, middle, lower = TechnicalIndicators.bbands(df['close'], period)
        bandwidth = (upper - lower) / middle
        return bandwidth
    
    @staticmethod
    def analyze(df_1h: pd.DataFrame, df_4h: pd.DataFrame, lookback: int = 120) -> Tuple[SignalType, float, Dict]:
        signals = {'long': [], 'short': []}
        
        # Calculate Bollinger Bands
        df_1h['bb_upper'], df_1h['bb_middle'], df_1h['bb_lower'] = TechnicalIndicators.bbands(df_1h['close'])
        df_1h['bandwidth'] = Strategy3_BB_Squeeze_Breakout.calculate_bandwidth(df_1h)
        df_1h['macd'], _, _ = TechnicalIndicators.macd(df_1h['close'])
        df_1h['rsi'] = TechnicalIndicators.rsi(df_1h['close'])
        
        latest = df_1h.iloc[-1]
        prev = df_1h.iloc[-2]
        
        # Squeeze detection (bandwidth at 6-month low equivalent)
        min_bandwidth = df_1h['bandwidth'].tail(lookback).min()
        is_squeeze = latest['bandwidth'] <= min_bandwidth * 1.05  # Within 5% of lowest
        
        # Breakout conditions
        close_above_upper = latest['close'] > latest['bb_upper'] and prev['close'] <= prev['bb_upper']
        close_below_lower = latest['close'] < latest['bb_lower'] and prev['close'] >= prev['bb_lower']
        
        macd_positive = latest['macd'] > 0
        macd_negative = latest['macd'] < 0
        
        rsi_above_50 = latest['rsi'] > 50
        rsi_below_50 = latest['rsi'] < 50
        
        # LONG Conditions
        if is_squeeze:
            signals['long'].append('BB squeeze detected')
            signals['short'].append('BB squeeze detected')
        if close_above_upper:
            signals['long'].append('Close above upper BB')
        if macd_positive:
            signals['long'].append('MACD positive')
        if rsi_above_50:
            signals['long'].append('RSI > 50')
        
        # SHORT Conditions
        if close_below_lower:
            signals['short'].append('Close below lower BB')
        if macd_negative:
            signals['short'].append('MACD negative')
        if rsi_below_50:
            signals['short'].append('RSI < 50')
        
        band_width_at_entry = latest['bb_upper'] - latest['bb_lower']
        
        long_count = len(signals['long'])
        short_count = len(signals['short'])
        
        if long_count >= 3 and is_squeeze:
            confidence = min(45 + long_count * 12, 85)
            return SignalType.LONG, confidence, {
                'signals': signals['long'], 
                'bb_middle': latest['bb_middle'],
                'band_width': band_width_at_entry
            }
        elif short_count >= 3 and is_squeeze:
            confidence = min(45 + short_count * 12, 85)
            return SignalType.SHORT, confidence, {
                'signals': signals['short'],
                'bb_middle': latest['bb_middle'],
                'band_width': band_width_at_entry
            }
        
        return SignalType.NONE, 0, {}


class Strategy4_Fibonacci_Pullback:
    """
    Strategy 4: Fibonacci Golden Pocket Pullback
    Timeframe: 15m entry / 1H / 4H trend
    """
    
    @staticmethod
    def analyze(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        signals = {'long': [], 'short': []}
        
        # 4H Trend
        df_4h['ema200'] = TechnicalIndicators.ema(df_4h['close'], 200)
        latest_4h = df_4h.iloc[-1]
        
        uptrend = latest_4h['close'] > latest_4h['ema200']
        downtrend = latest_4h['close'] < latest_4h['ema200']
        
        # Find swing high/low for Fibonacci
        lookback = 50
        recent_high = df_1h['high'].tail(lookback).max()
        recent_low = df_1h['low'].tail(lookback).min()
        
        fib_levels = TechnicalIndicators.fibonacci_retracement(recent_high, recent_low)
        
        current_price = df_15m.iloc[-1]['close']
        
        # Check if price is in golden pocket (50-61.8%)
        in_golden_pocket_long = fib_levels['0.5'] <= current_price <= fib_levels['0.618']
        in_golden_pocket_short = fib_levels['0.382'] <= current_price <= fib_levels['0.5']
        
        # Candle patterns - ensure 'open' column exists
        if 'open' not in df_15m.columns:
            df_15m['open'] = df_15m['close'].shift(1)
            df_15m['open'] = df_15m['open'].fillna(df_15m['close'])
        
        df_15m['body'] = df_15m['close'] - df_15m['open']
        df_15m['upper_shadow'] = df_15m['high'] - df_15m[['close', 'open']].max(axis=1)
        df_15m['lower_shadow'] = df_15m[['close', 'open']].min(axis=1) - df_15m['low']
        
        # Stochastic
        df_15m['stoch_k'], df_15m['stoch_d'] = TechnicalIndicators.stochastic(df_15m)
        latest_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        
        stoch_cross_up = prev_15m['stoch_k'] <= prev_15m['stoch_d'] and latest_15m['stoch_k'] > latest_15m['stoch_d']
        stoch_cross_down = prev_15m['stoch_k'] >= prev_15m['stoch_d'] and latest_15m['stoch_k'] < latest_15m['stoch_d']
        stoch_near_20 = latest_15m['stoch_k'] < 25
        stoch_near_80 = latest_15m['stoch_k'] > 75
        
        bullish_engulfing = prev_15m['body'] < 0 and latest_15m['body'] > abs(prev_15m['body']) * 1.5
        pin_bar_bullish = latest_15m['lower_shadow'] > abs(latest_15m['body']) * 2 and latest_15m['upper_shadow'] < abs(latest_15m['body']) * 0.5
        
        bearish_engulfing = prev_15m['body'] > 0 and latest_15m['body'] < -abs(prev_15m['body']) * 1.5
        pin_bar_bearish = latest_15m['upper_shadow'] > abs(latest_15m['body']) * 2 and latest_15m['lower_shadow'] < abs(latest_15m['body']) * 0.5
        
        # LONG Conditions
        if uptrend:
            signals['long'].append('4H above EMA200 (uptrend)')
        if in_golden_pocket_long:
            signals['long'].append('Price in 50-61.8% Fib zone')
        if stoch_cross_up and stoch_near_20:
            signals['long'].append('Stochastic cross-up near 20')
        if bullish_engulfing or pin_bar_bullish:
            signals['long'].append('Bullish candle pattern')
        
        # SHORT Conditions
        if downtrend:
            signals['short'].append('4H below EMA200 (downtrend)')
        if in_golden_pocket_short:
            signals['short'].append('Price retrace to 50-61.8% Fib')
        if stoch_cross_down and stoch_near_80:
            signals['short'].append('Stochastic cross-down near 80')
        if bearish_engulfing or pin_bar_bearish:
            signals['short'].append('Bearish rejection candle')
        
        long_count = len(signals['long'])
        short_count = len(signals['short'])
        
        if long_count >= 3:
            confidence = min(45 + long_count * 12, 90)
            return SignalType.LONG, confidence, {
                'signals': signals['long'],
                'fib_65': fib_levels['0.65'],
                'fib_1272': fib_levels['1.272'],
                'swing_high': recent_high
            }
        elif short_count >= 3:
            confidence = min(45 + short_count * 12, 90)
            return SignalType.SHORT, confidence, {
                'signals': signals['short'],
                'fib_65': fib_levels['0.65'],
                'fib_1272': fib_levels['1.272'],
                'swing_low': recent_low
            }
        
        return SignalType.NONE, 0, {}


class Strategy5_ADX_Multi_TF:
    """
    Strategy 5: ADX Multi-Timeframe Trend Filter
    Timeframe: 15m / 1H / 4H
    """
    
    @staticmethod
    def analyze(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Tuple[SignalType, float, Dict]:
        signals = {'long': [], 'short': []}
        
        # Calculate ADX for all timeframes
        for df in [df_15m, df_1h, df_4h]:
            df['adx'], df['plus_di'], df['minus_di'] = TechnicalIndicators.adx(df)
        
        # RSI
        df_15m['rsi'] = TechnicalIndicators.rsi(df_15m['close'])
        
        # MACD Histogram
        df_15m['macd'], df_15m['macd_signal'], df_15m['macd_hist'] = TechnicalIndicators.macd(df_15m['close'])
        
        latest_15m = df_15m.iloc[-1]
        prev_15m = df_15m.iloc[-2]
        
        # Strong trend condition
        strong_trend_15m = latest_15m['adx'] > 25
        strong_trend_1h = df_1h.iloc[-1]['adx'] > 25
        strong_trend_4h = df_4h.iloc[-1]['adx'] > 25
        
        # DI Cross conditions
        di_cross_up = prev_15m['plus_di'] <= prev_15m['minus_di'] and latest_15m['plus_di'] > latest_15m['minus_di']
        di_cross_down = prev_15m['plus_di'] >= prev_15m['minus_di'] and latest_15m['plus_di'] < latest_15m['minus_di']
        
        # RSI conditions
        rsi_cross_up = prev_15m['rsi'] < 50 and latest_15m['rsi'] > 50
        rsi_cross_down = prev_15m['rsi'] > 50 and latest_15m['rsi'] < 50
        
        # MACD histogram expanding
        hist_expanding_up = latest_15m['macd_hist'] > prev_15m['macd_hist'] and latest_15m['macd_hist'] > 0
        hist_expanding_down = latest_15m['macd_hist'] < prev_15m['macd_hist'] and latest_15m['macd_hist'] < 0
        
        # LONG Conditions
        if strong_trend_15m:
            signals['long'].append(f'ADX > 25 (15m: {latest_15m["adx"]:.1f})')
        if strong_trend_1h:
            signals['long'].append('ADX > 25 (1H)')
        if di_cross_up:
            signals['long'].append('+DI crossed above -DI')
        if rsi_cross_up:
            signals['long'].append('RSI crossed above 50')
        if hist_expanding_up:
            signals['long'].append('MACD histogram expanding positive')
        
        # SHORT Conditions
        if strong_trend_15m:
            signals['short'].append(f'ADX > 25 (15m: {latest_15m["adx"]:.1f})')
        if strong_trend_1h:
            signals['short'].append('ADX > 25 (1H)')
        if di_cross_down:
            signals['short'].append('-DI crossed above +DI')
        if rsi_cross_down:
            signals['short'].append('RSI crossed below 50')
        if hist_expanding_down:
            signals['short'].append('MACD histogram expanding negative')
        
        # Recent -DI swing for SL reference
        recent_swing_low = df_15m['low'].tail(20).min()
        recent_swing_high = df_15m['high'].tail(20).max()
        
        long_count = len(signals['long'])
        short_count = len(signals['short'])
        
        # Require at least 3 confluence signals including strong trend
        if long_count >= 3 and (strong_trend_15m or strong_trend_1h):
            confidence = min(45 + long_count * 12, 90)
            return SignalType.LONG, confidence, {
                'signals': signals['long'],
                'recent_swing_low': recent_swing_low,
                'adx_value': latest_15m['adx']
            }
        elif short_count >= 3 and (strong_trend_15m or strong_trend_1h):
            confidence = min(45 + short_count * 12, 90)
            return SignalType.SHORT, confidence, {
                'signals': signals['short'],
                'recent_swing_high': recent_swing_high,
                'adx_value': latest_15m['adx']
            }
        
        return SignalType.NONE, 0, {}


class AdvancedStrategyEngine:
    """
    Main engine that combines all 5 strategies with confluence detection
    """
    
    def __init__(self):
        self.strategies = {
            'ema_trend': Strategy1_EMA_Trend_Confluence,
            'rsi_divergence': Strategy2_RSI_Divergence_MACD,
            'bb_squeeze': Strategy3_BB_Squeeze_Breakout,
            'fibonacci': Strategy4_Fibonacci_Pullback,
            'adx_trend': Strategy5_ADX_Multi_TF
        }
        self.min_confluence = 2  # Minimum 2 strategies must agree
        self.max_leverage = 10
        self.max_risk_percent = 2.0
        self.max_open_trades = 3
    
    def analyze_all(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> SignalResult:
        """
        Run all strategies and combine signals
        """
        results = {}
        
        # Strategy 1: EMA Trend
        try:
            sig, conf, meta = self.strategies['ema_trend'].analyze(df_15m, df_1h, df_4h)
            results['ema_trend'] = {'signal': sig, 'confidence': conf, 'meta': meta}
        except Exception as e:
            logger.error(f"EMA Trend strategy error: {e}")
            results['ema_trend'] = {'signal': SignalType.NONE, 'confidence': 0, 'meta': {}}
        
        # Strategy 2: RSI Divergence
        try:
            sig, conf, meta = self.strategies['rsi_divergence'].analyze(df_1h, df_4h)
            results['rsi_divergence'] = {'signal': sig, 'confidence': conf, 'meta': meta}
        except Exception as e:
            logger.error(f"RSI Divergence strategy error: {e}")
            results['rsi_divergence'] = {'signal': SignalType.NONE, 'confidence': 0, 'meta': {}}
        
        # Strategy 3: BB Squeeze
        try:
            sig, conf, meta = self.strategies['bb_squeeze'].analyze(df_1h, df_4h, lookback=120)
            results['bb_squeeze'] = {'signal': sig, 'confidence': conf, 'meta': meta}
        except Exception as e:
            logger.error(f"BB Squeeze strategy error: {e}")
            results['bb_squeeze'] = {'signal': SignalType.NONE, 'confidence': 0, 'meta': {}}
        
        # Strategy 4: Fibonacci
        try:
            sig, conf, meta = self.strategies['fibonacci'].analyze(df_15m, df_1h, df_4h)
            results['fibonacci'] = {'signal': sig, 'confidence': conf, 'meta': meta}
        except Exception as e:
            logger.error(f"Fibonacci strategy error: {e}")
            results['fibonacci'] = {'signal': SignalType.NONE, 'confidence': 0, 'meta': {}}
        
        # Strategy 5: ADX
        try:
            sig, conf, meta = self.strategies['adx_trend'].analyze(df_15m, df_1h, df_4h)
            results['adx_trend'] = {'signal': sig, 'confidence': conf, 'meta': meta}
        except Exception as e:
            logger.error(f"ADX Trend strategy error: {e}")
            results['adx_trend'] = {'signal': SignalType.NONE, 'confidence': 0, 'meta': {}}
        
        # Count confluence
        long_count = sum(1 for r in results.values() if r['signal'] == SignalType.LONG)
        short_count = sum(1 for r in results.values() if r['signal'] == SignalType.SHORT)
        
        # Determine final signal based on confluence
        final_signal = SignalType.NONE
        final_confidence = 0
        active_strategies = {}
        metadata = {}
        
        if long_count >= self.min_confluence and long_count > short_count:
            final_signal = SignalType.LONG
            final_confidence = min(50 + long_count * 10, 95)
            active_strategies = {k: True for k, r in results.items() if r['signal'] == SignalType.LONG}
            metadata = {k: r['meta'] for k, r in results.items() if r['signal'] == SignalType.LONG}
        elif short_count >= self.min_confluence and short_count > long_count:
            final_signal = SignalType.SHORT
            final_confidence = min(50 + short_count * 10, 95)
            active_strategies = {k: True for k, r in results.items() if r['signal'] == SignalType.SHORT}
            metadata = {k: r['meta'] for k, r in results.items() if r['signal'] == SignalType.SHORT}
        
        # Calculate entry, SL, TP
        entry_price = df_15m.iloc[-1]['close']
        stop_loss = self._calculate_stop_loss(final_signal, metadata, entry_price, df_15m)
        take_profit = self._calculate_take_profit(final_signal, entry_price, stop_loss)
        
        risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 0
        
        return SignalResult(
            signal=final_signal,
            confidence=final_confidence,
            strategy_signals=active_strategies,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            confluence_count=max(long_count, short_count),
            timeframe_analysis={
                '15m': 'active',
                '1h': 'active',
                '4h': 'active'
            },
            metadata=metadata
        )
    
    def _calculate_stop_loss(self, signal: SignalType, metadata: Dict, entry: float, df: pd.DataFrame) -> float:
        """Calculate appropriate stop loss based on strategy"""
        if signal == SignalType.NONE:
            return entry
        
        atr_value = TechnicalIndicators.atr(df).iloc[-1]
        
        # Strategy-specific SL calculation
        sl_candidates = []
        
        # EMA34 based SL (Strategy 1)
        if 'ema_trend' in metadata and 'ema34' in metadata['ema_trend']:
            ema34 = metadata['ema_trend']['ema34']
            sl_candidates.append(ema34)
        
        # Divergence extreme (Strategy 2)
        if 'rsi_divergence' in metadata and 'divergence_price' in metadata['rsi_divergence']:
            div_price = metadata['rsi_divergence']['divergence_price']
            if div_price:
                sl_candidates.append(div_price)
        
        # BB Middle (Strategy 3)
        if 'bb_squeeze' in metadata and 'bb_middle' in metadata['bb_squeeze']:
            bb_middle = metadata['bb_squeeze']['bb_middle']
            sl_candidates.append(bb_middle)
        
        # Fib 65% level (Strategy 4)
        if 'fibonacci' in metadata and 'fib_65' in metadata['fibonacci']:
            fib_65 = metadata['fibonacci']['fib_65']
            sl_candidates.append(fib_65)
        
        # Recent DI swing (Strategy 5)
        if 'adx_trend' in metadata:
            if signal == SignalType.LONG and 'recent_swing_low' in metadata['adx_trend']:
                sl_candidates.append(metadata['adx_trend']['recent_swing_low'])
            elif signal == SignalType.SHORT and 'recent_swing_high' in metadata['adx_trend']:
                sl_candidates.append(metadata['adx_trend']['recent_swing_high'])
        
        # Default ATR-based SL if no specific SL
        if not sl_candidates:
            if signal == SignalType.LONG:
                sl_candidates.append(entry - atr_value * 1.5)
            else:
                sl_candidates.append(entry + atr_value * 1.5)
        
        # Use the most conservative (closest to entry) SL
        if signal == SignalType.LONG:
            stop_loss = max(sl_candidates)  # Highest price for LONG
            # Ensure minimum distance (1% from entry)
            min_sl = entry * 0.99
            stop_loss = max(stop_loss, min_sl)
        else:
            stop_loss = min(sl_candidates)  # Lowest price for SHORT
            # Ensure minimum distance (1% from entry)
            max_sl = entry * 1.01
            stop_loss = min(stop_loss, max_sl)
        
        return stop_loss
    
    def _calculate_take_profit(self, signal: SignalType, entry: float, stop_loss: float) -> float:
        """Calculate take profit based on minimum 1.5R target"""
        if signal == SignalType.NONE:
            return entry
        
        risk = abs(entry - stop_loss)
        min_reward = risk * 1.5  # Minimum 1.5R
        
        if signal == SignalType.LONG:
            return entry + min_reward
        else:
            return entry - min_reward
    
    def get_position_size(self, account_balance: float, entry: float, stop_loss: float, leverage: int = 10) -> float:
        """Calculate position size based on 1-2% risk"""
        risk_percent = min(self.max_risk_percent, 1.5)  # Cap at 1.5% per trade
        risk_amount = account_balance * (risk_percent / 100)
        
        price_risk = abs(entry - stop_loss)
        if price_risk == 0:
            price_risk = entry * 0.01  # Default 1% if no SL calculated
        
        # Position size in base asset
        position_size = (risk_amount / price_risk) * entry
        
        # Apply leverage
        position_size = position_size * leverage
        
        return position_size


# Global strategy engine instance
strategy_engine = AdvancedStrategyEngine()


def generate_signal(klines_15m: List, klines_1h: List, klines_4h: List) -> SignalResult:
    """
    Main entry point for signal generation
    """
    # Convert klines to DataFrames
    def klines_to_df(klines):
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                           'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                                           'taker_buy_quote', 'ignore'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    
    df_15m = klines_to_df(klines_15m)
    df_1h = klines_to_df(klines_1h)
    df_4h = klines_to_df(klines_4h)
    
    # Run strategy analysis
    result = strategy_engine.analyze_all(df_15m, df_1h, df_4h)
    
    return result
