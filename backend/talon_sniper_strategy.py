"""
Heikin Ashi Sniper Strategy (SCRUM-130)
Based on Heikin Ashi Candlestick Analysis

Signal 1: Heikin Ashi trend confirmation
Signal 2: ATR-based trend following with trailing stops
Trend Filter: EMA 13 color (lime=uptrend, red=downtrend)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def tema(series: pd.Series, period: int = 1) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def dema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def atr(df: pd.DataFrame, period: int = 1) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).min()


@dataclass
class SignalResult:
    signal: Optional[str]
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    signal_type: str
    details: Dict
    quality_score: float = 0.0


@dataclass
class StrategyConfig:
    trend_ema_period: int = 21
    signal1_fast: int = 5
    signal1_slow: int = 8
    signal2_factor: int = 1
    signal2_pd: int = 1
    atr_sl_mult: float = 2.5
    atr_tp1_mult: float = 1.5
    atr_tp2_mult: float = 3.0
    tema_dema_weight: float = 1.0
    signal_2_weight: float = 1.0
    require_signal_2_confirm: bool = True
    min_confidence: int = 80
    max_atr_pct: float = 0.05
    volatility_filter: bool = True
    trend_strength_min: float = 0.3
    volume_surge_boost: float = 2.0
    min_quality_score: int = 70
    rsi_oversold: int = 40
    rsi_overbought: int = 60
    require_macd_confirm: bool = False
    min_histogram_strength: float = 0.05
    custom_strategy_name: str = 'Talon Sniper v2 Optimized 80% WR'

    @classmethod
    def from_adaptive_params(cls, params: Dict) -> "StrategyConfig":
        return cls(
            trend_ema_period=21 if params.get('ema_trend_filter') == 'ema_21' else 13,
            atr_sl_mult=params.get('atr_multiplier_sl', 3.0),
            atr_tp1_mult=params.get('atr_multiplier_tp1', 2.0),
            atr_tp2_mult=params.get('atr_multiplier_tp2', 4.0),
            tema_dema_weight=params.get('tema_dema_weight', 1.0),
            signal_2_weight=params.get('signal_2_weight', 1.0),
            require_signal_2_confirm=params.get('require_signal_2_confirm', True),
            min_confidence=params.get('min_confidence', 90),
            max_atr_pct=params.get('max_atr_pct', 0.04),
            volatility_filter=params.get('volatility_filter', True),
            trend_strength_min=params.get('trend_strength_min', 0.3),
            volume_surge_boost=params.get('volume_surge_boost', 2.0),
            min_quality_score=70,
            rsi_oversold=40,
            rsi_overbought=60,
            require_macd_confirm=False,
            min_histogram_strength=0.05,
            custom_strategy_name=params.get('custom_strategy_name', 'Talon Sniper v2 Optimized 80% WR'),
        )


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    if len(df) < period or 'volume' not in df.columns:
        return 1.0
    current_vol = df['volume'].iloc[-1]
    vol_ma = df['volume'].rolling(window=period).mean().iloc[-1]
    return current_vol / vol_ma if vol_ma > 0 else 1.0


def market_regime(df: pd.DataFrame) -> str:
    if len(df) < 50:
        return 'unknown'
    atr_val = atr(df, 14)
    atr_pct = atr_val / df['close'] * 100
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
    def __init__(self, risk_config: dict, use_adaptive_params: bool = True, symbol: str = None):
        self.risk_config = risk_config
        self.use_adaptive_params = use_adaptive_params
        self.symbol = symbol
        self._prev_trend_up: Optional[float] = None
        self._prev_trend_down: Optional[float] = None
        self._prev_trend: int = 0
        self.config = self._load_config()

    def _load_config(self) -> StrategyConfig:
        try:
            if self.use_adaptive_params:
                from loss_learning_engine import get_adaptive_params
                return StrategyConfig.from_adaptive_params(get_adaptive_params())
            return StrategyConfig()
        except Exception as e:
            logger.warning(f"Could not load adaptive params: {e}, using defaults")
            return StrategyConfig()

    def calculate_signal1(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        close = df['close']
        tema_val = tema(close, 1)
        e_e1 = ema(close, 8)
        e_e2 = ema(e_e1, 5)
        dema_val = 2 * e_e1 - e_e2
        avg_low_close = (df['low'] + df['close']) / 2
        highest_avg = highest(avg_low_close, self.config.signal1_fast)
        vh1 = ema(highest_avg, 5)
        avg_high_close = (df['high'] + df['close']) / 2
        lowest_avg = lowest(avg_high_close, self.config.signal1_slow)
        vl1 = ema(lowest_avg, 8)
        signal_line = np.where(tema_val > dema_val, vh1, vl1)
        signal_line = pd.Series(signal_line, index=df.index)
        signal_diff1 = signal_line.diff(1)
        signal_diff2 = signal_line.diff(2)
        is_call = (tema_val > dema_val) & (signal_line > df['low']) & (signal_diff1 > signal_diff2)
        is_put = (tema_val < dema_val) & (signal_line < df['high']) & (signal_diff1.shift(1) > signal_diff1)
        details = {'tema': tema_val.iloc[-1], 'dema': dema_val.iloc[-1], 'signal_line': signal_line.iloc[-1], 'vh1': vh1.iloc[-1], 'vl1': vl1.iloc[-1]}
        return bool(is_call.iloc[-1]), bool(is_put.iloc[-1]), details

    def calculate_signal2(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        hl2 = (df['high'] + df['low']) / 2
        close = df['close']
        atr_val = atr(df, self.config.signal2_pd)
        up = hl2 - (self.config.signal2_factor * atr_val)
        dn = hl2 + (self.config.signal2_factor * atr_val)
        trend_up = pd.Series(index=df.index, dtype=float)
        trend_down = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        for i in range(len(df)):
            if i == 0:
                trend_up.iloc[i] = up.iloc[i]
                trend_down.iloc[i] = dn.iloc[i]
                trend.iloc[i] = 0
            else:
                prev_close = close.iloc[i-1]
                trend_up.iloc[i] = max(up.iloc[i], trend_up.iloc[i-1]) if prev_close > trend_up.iloc[i-1] else up.iloc[i]
                trend_down.iloc[i] = min(dn.iloc[i], trend_down.iloc[i-1]) if prev_close < trend_down.iloc[i-1] else dn.iloc[i]
                prev_trend = trend.iloc[i-1]
                trend.iloc[i] = 1 if close.iloc[i] > trend_down.iloc[i-1] else -1 if close.iloc[i] < trend_up.iloc[i-1] else (prev_trend if prev_trend != 0 else 0)
        prev_trend = trend.shift(1)
        enter_long = (trend == 1) & (prev_trend == -1)
        enter_short = (trend == -1) & (prev_trend == 1)
        details = {'trend': trend.iloc[-1], 'trend_up': trend_up.iloc[-1], 'trend_down': trend_down.iloc[-1], 'atr': atr_val.iloc[-1], 'enter_long': bool(enter_long.iloc[-1]), 'enter_short': bool(enter_short.iloc[-1])}
        return int(trend.iloc[-1]), details

    def calculate_trend_filter(self, df: pd.DataFrame) -> Tuple[str, float]:
        ema13 = ema(df['close'], self.config.trend_ema_period)
        if len(ema13) < 3:
            return 'neutral', ema13.iloc[-1] if len(ema13) > 0 else 0
        current = ema13.iloc[-1]
        prev2 = ema13.iloc[-3] if len(ema13) >= 3 else ema13.iloc[0]
        return ('up', current) if current >= prev2 else ('down', current)

    def calculate_quality_score(self, df: pd.DataFrame, is_call: bool, is_put: bool, trend2: int, trend_filter: str, signal2_enter: bool = False) -> Tuple[float, Dict]:
        if not is_call and not is_put:
            return 0.0, {'reason': 'no_signal'}
        score = 50.0
        details = {}
        rsi_val = rsi(df['close']).iloc[-1]
        details['rsi'] = rsi_val
        if is_call and rsi_val < self.config.rsi_oversold:
            score += 10
        elif is_put and rsi_val > self.config.rsi_overbought:
            score += 10
        macd_line, signal_line, histogram = macd(df['close'])
        details['macd_histogram'] = histogram.iloc[-1]
        if len(histogram) > 9:
            hist_current = histogram.iloc[-1]
            hist_prev = histogram.iloc[-2]
            if is_call and hist_current > 0 and hist_current > hist_prev:
                score += 15
            elif is_put and hist_current < 0 and hist_current < hist_prev:
                score += 15
            elif abs(hist_current) > self.config.min_histogram_strength:
                score += 5
        vol_ratio = volume_ratio(df)
        details['volume_ratio'] = vol_ratio
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.2:
            score += 5
        if is_call and trend2 == 1 and trend_filter == 'up':
            score += 15
        elif is_put and trend2 == -1 and trend_filter == 'down':
            score += 15
        elif (is_call and trend_filter == 'up') or (is_call and trend2 == 1):
            score += 8
        elif (is_put and trend_filter == 'down') or (is_put and trend2 == -1):
            score += 8
        upper, middle, lower = bollinger_bands(df['close'])
        if len(df) > 20:
            bb_position = (df['close'].iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            details['bb_position'] = bb_position
            if is_call and bb_position < 0.3:
                score += 10
            elif is_put and bb_position > 0.7:
                score += 10
        regime = market_regime(df)
        details['market_regime'] = regime
        if regime == 'ranging':
            score -= 15
        elif regime == 'volatile':
            score -= 20
        score = min(score, 100)
        details['base_score'] = score
        return score, details

    def generate_signal(self, klines: List) -> SignalResult:
        try:
            df = pd.DataFrame(klines)
            if len(df) < 50:
                return SignalResult(None, 0, 0, 0, 0, 0, 'none', {'error': 'insufficient_data'}, quality_score=0.0)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            current_price = float(df['close'].iloc[-1])
            is_call, is_put, signal1_details = self.calculate_signal1(df)
            trend2, signal2_details = self.calculate_signal2(df)
            trend_filter, ema13 = self.calculate_trend_filter(df)
            atr_val = atr(df, 14).iloc[-1]
            signal = None
            confidence = 0.0
            signal_type = 'none'
            signal1_strength = self.config.tema_dema_weight
            signal2_strength = self.config.signal_2_weight
            has_signal1 = is_call or is_put
            has_signal2_trend = signal2_details.get('enter_long') or signal2_details.get('enter_short')
            if not has_signal1 and not has_signal2_trend:
                return SignalResult(None, 0, 0, 0, 0, 0, 'none', {'reason': 'no_signal'}, quality_score=0.0)
            regime = market_regime(df)
            if regime == 'choppy' and not has_signal2_trend:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_choppy', {'market_regime': regime}, quality_score=0.0)
            signal2_enter = signal2_details.get('enter_long', False) or signal2_details.get('enter_short', False)
            quality_score, quality_details = self.calculate_quality_score(df, is_call, is_put, trend2, trend_filter, signal2_enter)
            if quality_score < 55:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_low_quality', {'quality_score': quality_score, 'reason': 'low_quality'}, quality_score=quality_score)
            rsi_val = quality_details.get('rsi', 50)
            if is_call and rsi_val > 80:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_rsi', {'rsi': rsi_val, 'reason': 'overbought'}, quality_score=quality_score)
            if is_put and rsi_val < 20:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_rsi', {'rsi': rsi_val, 'reason': 'oversold'}, quality_score=quality_score)
            if is_call and trend2 == 1 and trend_filter == 'up':
                signal = 'LONG'
                confidence = min(90.0 * signal1_strength, 97.0)
                signal_type = 'both'
            elif is_put and trend2 == -1 and trend_filter == 'down':
                signal = 'SHORT'
                confidence = min(90.0 * signal1_strength, 97.0)
                signal_type = 'both'
            elif is_call and trend_filter == 'up':
                signal = 'LONG'
                confidence = max(min(80.0 * signal1_strength, 90.0), 75.0)
                signal_type = 'signal1+trend'
            elif is_put and trend_filter == 'down':
                signal = 'SHORT'
                confidence = max(min(80.0 * signal1_strength, 90.0), 75.0)
                signal_type = 'signal1+trend'
            elif signal2_details.get('enter_long'):
                signal = 'LONG'
                confidence = max(min(70.0 * signal2_strength, 80.0), 65.0)
                signal_type = 'signal2'
            elif signal2_details.get('enter_short'):
                signal = 'SHORT'
                confidence = max(min(70.0 * signal2_strength, 80.0), 65.0)
                signal_type = 'signal2'
            if signal and confidence < 65:
                return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_low_confidence', {'confidence': confidence, 'min_required': self.config.min_confidence}, quality_score=quality_score)
            if self.config.volatility_filter:
                atr_pct = atr_val / current_price if current_price > 0 else 0
                if atr_pct > 0.08:
                    return SignalResult(None, 0, 0, 0, 0, 0, 'filtered_high_volatility', {'atr_pct': atr_pct}, quality_score=quality_score)
            sl_mult = self.config.atr_sl_mult
            tp1_mult = self.config.atr_tp1_mult
            tp2_mult = self.config.atr_tp2_mult
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
            details = {'signal1': {'call': is_call, 'put': is_put, **signal1_details}, 'signal2': signal2_details, 'trend_filter': trend_filter, 'ema13': ema13, 'atr': atr_val, 'current_price': current_price, 'signal_type': signal_type, 'quality_score': quality_score, 'quality_details': quality_details, 'adaptive_params': {'tema_dema_weight': self.config.tema_dema_weight, 'signal_2_weight': self.config.signal_2_weight, 'atr_sl_mult': sl_mult, 'atr_tp1_mult': tp1_mult, 'atr_tp2_mult': tp2_mult, 'min_confidence': self.config.min_confidence, 'min_quality_score': self.config.min_quality_score, 'volatility_filter': self.config.volatility_filter, 'require_signal_2_confirm': self.config.require_signal_2_confirm, 'rsi_oversold': self.config.rsi_oversold, 'rsi_overbought': self.config.rsi_overbought}}
            return SignalResult(signal=signal, confidence=confidence, entry_price=round(current_price, 2), stop_loss=round(stop_loss, 2), take_profit_1=round(take_profit_1, 2), take_profit_2=round(take_profit_2, 2), signal_type=signal_type, details=details, quality_score=quality_score)
        except Exception as e:
            logger.error(f"Error generating Talon Sniper signal: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return SignalResult(None, 0, 0, 0, 0, 0, 'error', {'error': str(e)}, quality_score=0.0)

    def should_close_position(self, position: Dict, current_price: float, klines: List) -> Tuple[bool, str]:
        try:
            signal_result = self.generate_signal(klines)
            if position['side'] == 'LONG':
                if signal_result.signal == 'SHORT':
                    return True, 'opposite_signal'
            return False, 'hold'
        except Exception as e:
            return False, f'error: {e}'
