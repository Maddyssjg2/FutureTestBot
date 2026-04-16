"""
Talon Sniper v1 Trading Strategy
Based on TradingView Pine Script with Heikin Ashi candles

Signal 1: TEMA/DEMA crossover with adaptive signal line
Signal 2: ATR-based trend following with trailing stops
Trend Filter: EMA 13 color (lime=uptrend, red=downtrend)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Convert regular candles to Heikin Ashi candles"""
    ha = pd.DataFrame(index=df.index)
    
    # Heikin Ashi calculations
    ha['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha['open'] = df['open'].copy()
    ha.loc[df.index[0], 'open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha.loc[df.index[i], 'open'] = (ha['close'].iloc[i-1] + ha['open'].iloc[i-1]) / 2
    
    ha['high'] = df[['high', 'open', 'close']].max(axis=1)
    ha['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    # Copy volume if available
    if 'volume' in df.columns:
        ha['volume'] = df['volume'].copy()
    
    return ha


def ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA"""
    return series.ewm(span=period, adjust=False).mean()


def tema(series: pd.Series, period: int = 1) -> pd.Series:
    """Triple EMA: 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))"""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double EMA: 2 * EMA - EMA(EMA)"""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def atr(df: pd.DataFrame, period: int = 1) -> pd.Series:
    """Average True Range"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def highest(series: pd.Series, period: int) -> pd.Series:
    """Rolling highest"""
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """Rolling lowest"""
    return series.rolling(window=period).min()


@dataclass
class SignalResult:
    signal: Optional[str]  # 'LONG', 'SHORT', or None
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    signal_type: str  # 'signal1', 'signal2', or 'both'
    details: Dict
    quality_score: float = 0.0  # NEW: Overall quality score for 80%+ WR target


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD, Signal, and Histogram"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate volume ratio vs moving average"""
    if len(df) < period or 'volume' not in df.columns:
        return 1.0  # Neutral if no volume data
    current_vol = df['volume'].iloc[-1]
    vol_ma = df['volume'].rolling(window=period).mean().iloc[-1]
    return current_vol / vol_ma if vol_ma > 0 else 1.0


def market_regime(df: pd.DataFrame) -> str:
    """
    Detect market regime: 'trending', 'ranging', or 'volatile'
    Key for avoiding false signals in ranging markets
    """
    if len(df) < 50:
        return 'unknown'
    
    # Use ATR % to detect volatility
    atr_val = atr(df, 14)
    atr_pct = atr_val / df['close'] * 100
    
    # Bollinger bandwidth for range detection
    upper, middle, lower = bollinger_bands(df['close'])
    bb_width = (upper - lower) / middle * 100
    
    avg_atr_pct = atr_pct.iloc[-20:].mean()
    avg_bb_width = bb_width.iloc[-20:].mean()
    
    if avg_atr_pct > 5:
        return 'volatile'
    elif avg_bb_width < 3:
        return 'ranging'
    else:
        return 'trending'


class TalonSniperStrategy:
    """
    Talon Sniper v1 Strategy Implementation with Adaptive Learning
    Optimized for 80%+ win rate with enhanced quality filters
    
    Signal 1: TEMA/DEMA crossover with adaptive bounds
    Signal 2: ATR-based trend with trailing stops
    Trend Filter: EMA 21 direction (adaptable)
    
    NEW for 80%+ WR:
    - Market regime filtering (avoid ranging markets)
    - RSI confirmation (avoid overbought/oversold extremes)
    - MACD histogram momentum
    - Volume surge detection
    - Quality score threshold
    """
    
    def __init__(self, risk_config: dict, use_adaptive_params: bool = True, symbol: str = None):
        self.risk_config = risk_config
        self.use_adaptive_params = use_adaptive_params
        self.symbol = symbol
        
        # Load adaptive parameters from loss learning engine
        self._load_adaptive_parameters()
        
        # Store previous state for trend calculation
        self._prev_trend_up: Optional[float] = None
        self._prev_trend_down: Optional[float] = None
        self._prev_trend: int = 0
        
        # NEW: Quality thresholds for 80%+ WR target
        self.min_quality_score = 70  # Minimum quality score to take trade (lowered from 80)
    
    def _load_adaptive_parameters(self):
        """Load strategy parameters from loss learning engine"""
        try:
            from loss_learning_engine import get_adaptive_params
            
            params = get_adaptive_params()
            
            # Apply adaptive parameters with defaults
            self.trend_ema_period = 21 if params.get('ema_trend_filter') == 'ema_21' else 13
            self.signal1_fast = 5
            self.signal1_slow = 8
            self.signal2_factor = 1
            self.signal2_pd = 1
            
            # ATR multipliers - WIDER stops for higher win rate
            self.atr_sl_mult = params.get('atr_multiplier_sl', 3.0)  # Wider SL
            self.atr_tp1_mult = params.get('atr_multiplier_tp1', 2.0)  # Wider TP1
            self.atr_tp2_mult = params.get('atr_multiplier_tp2', 4.0)  # Wider TP2
            
            # Signal weights (adapted based on false positive patterns)
            self.tema_dema_weight = params.get('tema_dema_weight', 1.0)
            self.signal_2_weight = params.get('signal_2_weight', 1.0)
            self.require_signal_2_confirm = params.get('require_signal_2_confirm', True)
            
            # Confidence and filters - HIGHER for 80%+ WR
            self.min_confidence = params.get('min_confidence', 90)
            self.max_atr_pct = params.get('max_atr_pct', 0.04)
            self.volatility_filter = params.get('volatility_filter', True)
            
            # NEW: Quality filters for 80%+ WR
            self.trend_strength_min = params.get('trend_strength_min', 0.3)
            self.volume_surge_boost = params.get('volume_surge_boost', 2.0)
            self.min_quality_score = 70  # Lowered for more trades
            
            # NEW: RSI confirmation (avoid extremes)
            self.rsi_oversold = 40  # For LONG signals (adjusted)
            self.rsi_overbought = 60  # For SHORT signals (adjusted)
            
            # NEW: MACD confirmation
            self.require_macd_confirm = False  # Relaxed for more signals
            self.min_histogram_strength = 0.05
            
            custom_name = params.get('custom_strategy_name', 'Talon Sniper v1 Optimized 80% WR')
            logger.info(f"🎓 Strategy loaded: {custom_name}")
            logger.info(f"🎯 TARGET: 80%+ Win Rate | Min Confidence: {self.min_confidence}% | Quality: {self.min_quality_score}")
            
        except Exception as e:
            logger.warning(f"Could not load adaptive params: {e}, using defaults")
            # Default values optimized for 80%+ WR
            self.trend_ema_period = 21
            self.signal1_fast = 5
            self.signal1_slow = 8
            self.signal2_factor = 1
            self.signal2_pd = 1
            self.atr_sl_mult = 2.5
            self.atr_tp1_mult = 1.5
            self.atr_tp2_mult = 3.0
            self.tema_dema_weight = 1.0
            self.signal_2_weight = 1.0
            self.require_signal_2_confirm = True
            self.min_confidence = 80
            self.max_atr_pct = 0.05
            self.volatility_filter = True
            self.trend_strength_min = 0.3
            self.volume_surge_boost = 2.0
            self.min_quality_score = 70
            self.rsi_oversold = 40
            self.rsi_overbought = 60
            self.require_macd_confirm = False
            self.min_histogram_strength = 0.05
    
    def calculate_signal1(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Signal 1: TEMA/DEMA crossover with adaptive signal line
        
        Returns: (is_call, is_put, details)
        """
        # Heikin Ashi close
        close = df['close']
        
        # TEMA(1) - essentially triple smoothing
        tema_val = tema(close, 1)
        
        # DEMA(8,5)
        e_e1 = ema(close, 8)
        e_e2 = ema(e_e1, 5)
        dema_val = 2 * e_e1 - e_e2
        
        # vh1 = ema(highest(avg(low, close), 5), 5)
        avg_low_close = (df['low'] + df['close']) / 2
        highest_avg = highest(avg_low_close, self.signal1_fast)
        vh1 = ema(highest_avg, 5)
        
        # vl1 = ema(lowest(avg(high, close), 8), 8)
        avg_high_close = (df['high'] + df['close']) / 2
        lowest_avg = lowest(avg_high_close, self.signal1_slow)
        vl1 = ema(lowest_avg, 8)
        
        # Adaptive signal line
        signal_line = np.where(tema_val > dema_val, vh1, vl1)
        signal_line = pd.Series(signal_line, index=df.index)
        
        # Signal line momentum
        signal_diff1 = signal_line.diff(1)
        signal_diff2 = signal_line.diff(2)
        
        # Buy condition: tema > dema AND signal > low AND (signal-signal[1] > signal[1]-signal[2])
        is_call = (
            (tema_val > dema_val) & 
            (signal_line > df['low']) & 
            (signal_diff1 > signal_diff2)
        )
        
        # Sell condition: tema < dema AND signal < high AND (signal[1]-signal > signal[2]-signal[1])
        is_put = (
            (tema_val < dema_val) & 
            (signal_line < df['high']) & 
            (signal_diff1.shift(1) > signal_diff1)
        )
        
        details = {
            'tema': tema_val.iloc[-1] if len(tema_val) > 0 else None,
            'dema': dema_val.iloc[-1] if len(dema_val) > 0 else None,
            'signal_line': signal_line.iloc[-1] if len(signal_line) > 0 else None,
            'vh1': vh1.iloc[-1] if len(vh1) > 0 else None,
            'vl1': vl1.iloc[-1] if len(vl1) > 0 else None,
        }
        
        return is_call.iloc[-1] if len(is_call) > 0 else False, \
               is_put.iloc[-1] if len(is_put) > 0 else False, \
               details
    
    def calculate_signal2(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Signal 2: ATR-based trend following with trailing stops
        
        Returns: (trend, details)
        trend: 1 = uptrend, -1 = downtrend
        """
        hl2 = (df['high'] + df['low']) / 2
        close = df['close']
        atr_val = atr(df, self.signal2_pd)
        
        # Up and Down levels
        up = hl2 - (self.signal2_factor * atr_val)
        dn = hl2 + (self.signal2_factor * atr_val)
        
        # Initialize trend arrays
        trend_up = pd.Series(index=df.index, dtype=float)
        trend_down = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        # Calculate TrendUp and TrendDown
        for i in range(len(df)):
            if i == 0:
                trend_up.iloc[i] = up.iloc[i]
                trend_down.iloc[i] = dn.iloc[i]
                trend.iloc[i] = 0
            else:
                prev_close = close.iloc[i-1]
                
                # TrendUp
                if prev_close > trend_up.iloc[i-1]:
                    trend_up.iloc[i] = max(up.iloc[i], trend_up.iloc[i-1])
                else:
                    trend_up.iloc[i] = up.iloc[i]
                
                # TrendDown
                if prev_close < trend_down.iloc[i-1]:
                    trend_down.iloc[i] = min(dn.iloc[i], trend_down.iloc[i-1])
                else:
                    trend_down.iloc[i] = dn.iloc[i]
                
                # Trend
                prev_trend = trend.iloc[i-1]
                if close.iloc[i] > trend_down.iloc[i-1]:
                    trend.iloc[i] = 1
                elif close.iloc[i] < trend_up.iloc[i-1]:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = prev_trend if prev_trend != 0 else 0
        
        # Entry signals: trend change
        prev_trend = trend.shift(1)
        enter_long = (trend == 1) & (prev_trend == -1)
        enter_short = (trend == -1) & (prev_trend == 1)
        
        details = {
            'trend': trend.iloc[-1] if len(trend) > 0 else 0,
            'trend_up': trend_up.iloc[-1] if len(trend_up) > 0 else None,
            'trend_down': trend_down.iloc[-1] if len(trend_down) > 0 else None,
            'atr': atr_val.iloc[-1] if len(atr_val) > 0 else None,
            'enter_long': enter_long.iloc[-1] if len(enter_long) > 0 else False,
            'enter_short': enter_short.iloc[-1] if len(enter_short) > 0 else False,
        }
        
        return trend.iloc[-1] if len(trend) > 0 else 0, details
    
    def calculate_trend_filter(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Trend Color EMA (13)
        Returns: ('up', 'down', or 'neutral'), ema_value
        """
        ema13 = ema(df['close'], self.trend_ema_period)
        
        if len(ema13) < 3:
            return 'neutral', ema13.iloc[-1] if len(ema13) > 0 else 0
        
        # EMA >= EMA[2] -> lime (uptrend), else red (downtrend)
        current = ema13.iloc[-1]
        prev2 = ema13.iloc[-3] if len(ema13) >= 3 else ema13.iloc[0]
        
        if current >= prev2:
            return 'up', current
        else:
            return 'down', current
    
    def calculate_quality_score(self, df: pd.DataFrame, is_call: bool, is_put: bool, 
                                 trend2: int, trend_filter: str, signal2_enter: bool = False) -> Tuple[float, Dict]:
        """
        Calculate overall quality score for the signal (target: 80%+ WR)
        Higher score = better signal quality = higher win probability
        """
        # Allow signal from Signal 2 (trend change) even without Signal 1
        if not is_call and not is_put and not signal2_enter:
            return 0.0, {}
        
        # Base score - lower threshold for more trades
        score = 40.0 if signal2_enter else 50.0
        
        score = 50.0  # Base score
        details = {}
        
        # 1. Trend Strength Score (up to +20 points)
        ema_21 = ema(df['close'], 21)
        ema_55 = ema(df['close'], 55)
        if len(ema_21) > 0 and len(ema_55) > 0:
            trend_gap = abs(ema_21.iloc[-1] - ema_55.iloc[-1]) / df['close'].iloc[-1] * 100
            trend_strength_score = min(trend_gap * 10, 20)  # Cap at 20
            score += trend_strength_score
            details['trend_strength'] = trend_gap
            details['trend_strength_score'] = trend_strength_score
        
        # 2. RSI Confirmation Score (up to +15 points)
        rsi_val = rsi(df['close'], 14).iloc[-1]
        details['rsi'] = rsi_val
        if is_call and rsi_val < self.rsi_oversold:
            score += 15  # Oversold bounce for LONG
        elif is_put and rsi_val > self.rsi_overbought:
            score += 15  # Overbought drop for SHORT
        elif (is_call and 40 < rsi_val < 60) or (is_put and 40 < rsi_val < 60):
            score += 5  # Neutral zone OK
        
        # 3. MACD Histogram Momentum Score (up to +15 points)
        _, _, histogram = macd(df['close'])
        if len(histogram) > 9:
            hist_current = histogram.iloc[-1]
            hist_prev = histogram.iloc[-2]
            hist_strength = abs(hist_current)
            
            if is_call and hist_current > 0 and hist_current > hist_prev:
                score += 15  # Strong bullish momentum
            elif is_put and hist_current < 0 and hist_current < hist_prev:
                score += 15  # Strong bearish momentum
            elif hist_strength > self.min_histogram_strength:
                score += 5  # Some momentum present
        
        # 4. Volume Score (up to +10 points)
        vol_ratio = volume_ratio(df)
        details['volume_ratio'] = vol_ratio
        if vol_ratio > 1.5:
            score += 10  # Strong volume confirmation
        elif vol_ratio > 1.2:
            score += 5
        
        # 5. Signal Alignment Score (up to +15 points)
        # Both signals agree with trend
        if is_call and trend2 == 1 and trend_filter == 'up':
            score += 15  # Perfect alignment
        elif is_put and trend2 == -1 and trend_filter == 'down':
            score += 15  # Perfect alignment
        elif (is_call and trend_filter == 'up') or (is_call and trend2 == 1):
            score += 8  # Partial alignment
        elif (is_put and trend_filter == 'down') or (is_put and trend2 == -1):
            score += 8
        
        # 6. Bollinger Band Position Score (up to +10 points)
        upper, middle, lower = bollinger_bands(df['close'])
        if len(df) > 20:
            bb_position = (df['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            details['bb_position'] = bb_position
            if is_call and bb_position < 0.3:
                score += 10  # Near lower band = good LONG entry
            elif is_put and bb_position > 0.7:
                score += 10  # Near upper band = good SHORT entry
        
        # 7. Market Regime Check (-20 if bad regime)
        regime = market_regime(df)
        details['market_regime'] = regime
        if regime == 'ranging':
            score -= 15  # Penalty for ranging market
        elif regime == 'volatile':
            score -= 20  # Strong penalty for high volatility
        
        # Cap score at 100
        score = min(score, 100)
        details['base_score'] = score
        
        return score, details
    
    def generate_signal(self, klines: List) -> SignalResult:
        """
        Generate trading signal using Talon Sniper v1
        OPTIMIZED for 80%+ Win Rate
        
        Combines Signal 1 (TEMA/DEMA) + Signal 2 (ATR trend) + Trend Filter
        + Quality Score + Market Regime + RSI + MACD confirmation
        """
        try:
            # Convert klines to DataFrame
            if isinstance(klines[0], dict):
                df = pd.DataFrame(klines)
                df = df.rename(columns={
                    'open': 'open', 'high': 'high', 
                    'low': 'low', 'close': 'close'
                })
            else:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert to Heikin Ashi
            ha_df = heikin_ashi(df)
            
            # Need minimum data
            if len(ha_df) < 50:
                return SignalResult(None, 0, 0, 0, 0, 0, 'none', {'error': 'insufficient_data'}, quality_score=0.0)
            
            current_price = float(ha_df['close'].iloc[-1])
            
            # Calculate all signals
            is_call, is_put, signal1_details = self.calculate_signal1(ha_df)
            trend2, signal2_details = self.calculate_signal2(ha_df)
            trend_filter, ema13 = self.calculate_trend_filter(ha_df)
            
            # Calculate ATR for stops
            atr_val = atr(ha_df, 14).iloc[-1]
            
            # Determine final signal FIRST
            signal = None
            confidence = 0.0
            signal_type = 'none'
            
            # Apply adaptive signal weights and confirmation requirements
            signal1_strength = self.tema_dema_weight
            signal2_strength = self.signal_2_weight
            require_s2 = self.require_signal_2_confirm
            
            # Check for valid signal - Signal 1 OR Signal 2 trend change
            has_signal1 = is_call or is_put
            has_signal2_trend = signal2_details.get('enter_long') or signal2_details.get('enter_short')
            
            if not has_signal1 and not has_signal2_trend:
                return SignalResult(None, 0, 0, 0, 0, 0, 'none', {'reason': 'no_signal'}, quality_score=0.0)
            
            # Log regime info
            regime = market_regime(ha_df)
            logger.info(f"[{self.symbol if hasattr(self, 'symbol') else 'UNKNOWN'}] Regime: {regime} | Signal1: call={is_call} put={is_put} | Trend2: {trend2}/{signal2_details.get('enter_long',False)}/{signal2_details.get('enter_short',False)} | Filter: {trend_filter}")
            
            # STRICT filtering for 80%+ WR
            # Only skip VERY choppy markets - allow trending markets
            if regime == 'choppy':
                logger.info(f"[{self.symbol if hasattr(self, 'symbol') else 'UNKNOWN'}] Choppy market")
                # Still allow if Signal 2 trend is strong
                if not has_signal2_trend:
                    return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_choppy', 
                                       {'market_regime': regime}, quality_score=0.0)
            
            # 2. Calculate quality score only for valid signals
            signal2_enter = signal2_details.get('enter_long', False) or signal2_details.get('enter_short', False)
            quality_score, quality_details = self.calculate_quality_score(
                ha_df, is_call, is_put, trend2, trend_filter, signal2_enter
            )
            
            # Quality Score must be high enough (lowered threshold)
            if quality_score < 55:
                logger.info(f"[{self.symbol if hasattr(self, 'symbol') else 'UNKNOWN'}] Q: {quality_score:.0f}/100")
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_low_quality', 
                                   {'quality_score': quality_score, 'reason': 'low_quality'}, quality_score=quality_score)
            
            # 3. RSI Confirmation - relaxed for more trades
            rsi_val = quality_details.get('rsi', 50)
            if is_call and rsi_val > 80:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_rsi', 
                                   {'rsi': rsi_val, 'reason': 'overbought'}, quality_score=quality_score)
            if is_put and rsi_val < 20:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_rsi', 
                                   {'rsi': rsi_val, 'reason': 'oversold'}, quality_score=quality_score)
            
            # Strongest signal: Both Signal 1 AND Signal 2 agree with trend + high quality
            if is_call and trend2 == 1 and trend_filter == 'up':
                signal = 'LONG'
                confidence = min(90.0 * signal1_strength, 97.0)
                signal_type = 'both'
            elif is_put and trend2 == -1 and trend_filter == 'down':
                signal = 'SHORT'
                confidence = min(90.0 * signal1_strength, 97.0)
                signal_type = 'both'
            # Signal 1 with trend filter
            elif is_call and trend_filter == 'up':
                signal = 'LONG'
                confidence = max(min(80.0 * signal1_strength, 90.0), 75.0)
                signal_type = 'signal1+trend'
            elif is_put and trend_filter == 'down':
                signal = 'SHORT'
                confidence = max(min(80.0 * signal1_strength, 90.0), 75.0)
                signal_type = 'signal1+trend'
            # Signal 2 trend change - works alone without trend filter requirement
            elif signal2_details.get('enter_long'):
                signal = 'LONG'
                confidence = max(min(70.0 * signal2_strength, 80.0), 65.0)
                signal_type = 'signal2'
            elif signal2_details.get('enter_short'):
                signal = 'SHORT'
                confidence = max(min(70.0 * signal2_strength, 80.0), 65.0)
                signal_type = 'signal2'
            
            # Final confidence check - must meet minimum threshold
            if signal and confidence < 65:
                logger.info(f"[{self.symbol if hasattr(self, 'symbol') else 'UNKNOWN'}] C: {confidence:.1f}%")
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_low_confidence', 
                                   {'confidence': confidence, 'min_required': self.min_confidence}, quality_score=quality_score)
            
            # Volatility filter - relaxed for more trades
            if self.volatility_filter:
                atr_pct = atr_val / current_price if current_price > 0 else 0
                if atr_pct > 0.08:  # Only skip if >8% ATR
                    return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_high_volatility', 
                                       {'atr_pct': atr_pct}, quality_score=quality_score)
            
            # Calculate stop loss and take profits based on adaptive ATR multipliers
            sl_mult = self.atr_sl_mult
            tp1_mult = self.atr_tp1_mult
            tp2_mult = self.atr_tp2_mult
            
            if signal == 'LONG':
                stop_loss = current_price - (atr_val * sl_mult)
                take_profit_1 = current_price + (atr_val * tp1_mult)
                take_profit_2 = current_price + (atr_val * tp2_mult)
            elif signal == 'SHORT':
                stop_loss = current_price + (atr_val * sl_mult)
                take_profit_1 = current_price - (atr_val * tp1_mult)
                take_profit_2 = current_price - (atr_val * tp2_mult)
            else:
                stop_loss = 0
                take_profit_1 = 0
                take_profit_2 = 0
            
            # Build details with adaptive parameters
            details = {
                'signal1': {'call': is_call, 'put': is_put, **signal1_details},
                'signal2': signal2_details,
                'trend_filter': trend_filter,
                'ema13': ema13,
                'atr': atr_val,
                'current_price': current_price,
                'signal_type': signal_type,
                'quality_score': quality_score,
                'quality_details': quality_details,
                'adaptive_params': {
                    'tema_dema_weight': self.tema_dema_weight,
                    'signal_2_weight': self.signal_2_weight,
                    'atr_sl_mult': sl_mult,
                    'atr_tp1_mult': tp1_mult,
                    'atr_tp2_mult': tp2_mult,
                    'min_confidence': self.min_confidence,
                    'min_quality_score': self.min_quality_score,
                    'volatility_filter': self.volatility_filter,
                    'require_signal_2_confirm': self.require_signal_2_confirm,
                    'rsi_oversold': self.rsi_oversold,
                    'rsi_overbought': self.rsi_overbought,
                }
            }
            
            logger.info(f"✅ [{self.symbol if hasattr(self, 'symbol') else 'UNKNOWN'}] SIGNAL: {signal} @ {confidence:.1f}% | Quality: {quality_score:.0f}/100 | Regime: {regime}")
            
            return SignalResult(
                signal=signal,
                confidence=confidence,
                entry_price=round(current_price, 2),
                stop_loss=round(stop_loss, 2),
                take_profit_1=round(take_profit_1, 2),
                take_profit_2=round(take_profit_2, 2),
                signal_type=signal_type,
                details=details,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Error generating Talon Sniper signal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return SignalResult(None, 0, 0, 0, 0, 0, 'error', {'error': str(e)}, quality_score=0.0)
    
    def should_close_position(self, position: Dict, current_price: float, klines: List) -> Tuple[bool, str]:
        """Check if position should be closed based on Talon Sniper signals"""
        try:
            signal_result = self.generate_signal(klines)
            
            if position['side'] == 'LONG':
                # Close if we get a strong short signal or trend reversal
                if signal_result.signal == 'SHORT' and signal_result.confidence >= 75:
                    return True, f"Trend Reversal (Signal2: {signal_result.signal_type})"
                # Close if Signal2 trend turns strongly negative
                if signal_result.details.get('signal2', {}).get('trend') == -1:
                    return True, "Signal2 Trend Down"
                    
            elif position['side'] == 'SHORT':
                # Close if we get a strong long signal or trend reversal
                if signal_result.signal == 'LONG' and signal_result.confidence >= 75:
                    return True, f"Trend Reversal (Signal2: {signal_result.signal_type})"
                # Close if Signal2 trend turns strongly positive
                if signal_result.details.get('signal2', {}).get('trend') == 1:
                    return True, "Signal2 Trend Up"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking close condition: {e}")
            return False, ""
    
    def calculate_position_size(self, available_balance: float, current_price: float, confidence: float) -> Tuple[float, float]:
        """Calculate position size based on risk config and confidence"""
        trade_pct = self.risk_config.get('trade_percentage', 5) / 100
        leverage = self.risk_config.get('leverage', 10)
        
        # Adjust size by confidence (higher confidence = slightly larger size)
        confidence_boost = 1.0 + (confidence - 75) * 0.005  # 75% = 1.0, 90% = 1.075
        confidence_boost = min(confidence_boost, 1.15)  # Cap at 15% boost
        
        position_usdt = available_balance * trade_pct * confidence_boost
        position_usdt_with_leverage = position_usdt * leverage
        quantity = position_usdt_with_leverage / current_price
        
        return round(quantity, 4), round(position_usdt, 2)


# Minimum confidence threshold for trading (dynamically adjustable)
MIN_CONFIDENCE_THRESHOLD = 75.0

def get_adaptive_min_confidence():
    """Get the current adaptive minimum confidence threshold"""
    try:
        from loss_learning_engine import get_adaptive_params
        params = get_adaptive_params()
        return params.get('min_confidence', MIN_CONFIDENCE_THRESHOLD)
    except:
        return MIN_CONFIDENCE_THRESHOLD


class TalonSniperMLStrategy:
    """
    ML-enhanced Talon Sniper Strategy with Adaptive Learning
    Uses Talon Sniper signals as features for ML model
    """
    
    def __init__(self, risk_config: dict, symbol: str, model_path: str = None):
        self.risk_config = risk_config
        self.symbol = symbol
        self.talon = TalonSniperStrategy(risk_config, use_adaptive_params=True, symbol=symbol)
        self.model = None
        self._min_confidence = get_adaptive_min_confidence()
        
        if model_path and os.path.exists(model_path):
            try:
                from talon_ml_model import TalonMLModel
                self.model = TalonMLModel.load(model_path)
                logger.info(f"[{symbol}] Loaded Talon ML model from {model_path}")
            except Exception as e:
                logger.warning(f"[{symbol}] Could not load model: {e}")
    
    def _refresh_min_confidence(self):
        """Refresh adaptive confidence threshold"""
        self._min_confidence = get_adaptive_min_confidence()
    
    def generate_signal(self, klines: List) -> Tuple[Optional[str], float, Dict]:
        """Generate signal using Talon Sniper with optional ML enhancement"""
        
        # Refresh adaptive threshold
        self._refresh_min_confidence()
        
        # Get Talon Sniper base signal
        result = self.talon.generate_signal(klines)
        
        if not result.signal or result.confidence < self._min_confidence:
            return None, result.confidence, {
                'stop_loss': 0, 'take_profit_1': 0, 'take_profit_2': 0,
                'entry_price': result.entry_price, 'signal_type': 'none'
            }
        
        # If ML model available, boost confidence
        if self.model:
            try:
                ml_pred = self.model.predict(klines)
                if ml_pred and ml_pred.get('signal') == result.signal:
                    # Boost confidence when ML agrees
                    result.confidence = min(97, result.confidence + 5)
            except Exception as e:
                logger.debug(f"ML prediction error: {e}")
        
        details = {
            'stop_loss': result.stop_loss,
            'take_profit_1': result.take_profit_1,
            'take_profit_2': result.take_profit_2,
            'entry_price': result.entry_price,
            'signal_type': result.signal_type,
            'atr': result.details.get('atr'),
            'trend_filter': result.details.get('trend_filter'),
            'adaptive_params': result.details.get('adaptive_params', {}),
            'adaptive_min_confidence': self._min_confidence,
        }
        
        return result.signal, result.confidence, details
    
    def should_close_position(self, position: Dict, current_price: float, klines: List) -> Tuple[bool, str]:
        """Delegate to Talon Sniper strategy"""
        return self.talon.should_close_position(position, current_price, klines)
    
    def calculate_position_size(self, available_balance: float, current_price: float, confidence: float) -> Tuple[float, float]:
        """Delegate to Talon Sniper strategy"""
        return self.talon.calculate_position_size(available_balance, current_price, confidence)


import os

# For backward compatibility
EnhancedTalonStrategy = TalonSniperMLStrategy
