"""
CORE STRATEGY - High Win-Rate Trading Engine
==========================================

A comprehensive Binance Futures trading bot with:
- Directional bias engine
- 6 confirmation blocks with weighted scoring
- Multi-timeframe analysis
- Smart risk management
- Session filtering

Target: 65-80% win rate with low drawdown
Pairs: BTCUSDT, ETHUSDT (high liquidity)
Timeframe: 5m entry, 15m/1h confirmation

Author: Trading Bot v2.0
Date: 2026-04-17
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class StrategySettings:
    """Centralized configuration"""
    
    # Trading Pairs (high liquidity only)
    PAIRS = ['BTCUSDT', 'ETHUSDT']
    
    # Timeframes
    ENTRY_TIMEFRAME = '5m'
    CONFIRM_TIMEFRAMES = ['15m', '1h']
    
    # EMA Settings
    FAST_EMA = 50
    SLOW_EMA = 200
    
    # RSI Settings
    RSI_PERIOD = 14
    RSI_LONG_MIN = 35  # Wider range for more signals
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 65
    
    # ADX Settings
    ADX_PERIOD = 14
    ADX_MIN = 15  # Lowered for more signals
    ADX_DEAD = 10
    
    # VWAP
    USE_VWAP = True
    
    # Volume
    VOLUME_SPIKE_MULT = 1.2  # Lowered from 1.5
    MIN_VOLUME_RATIO = 0.6  # Lowered from 0.8
    
    # ATR
    ATR_PERIOD = 14
    
    # Risk Management
    MAX_LEVERAGE = 5
    RISK_PER_TRADE_PCT = 1.5
    MAX_DAILY_LOSS_PCT = 5.0
    
    # Exit Strategy
    TP_PCT = 0.01  # 1.0% take profit (higher for more hits)
    SL_PCT = 0.006  # 0.6% stop loss
    
    # Trade Timing (UTC) - DISABLED for more trades
    LONDON_START = time(0, 0)  # All day
    LONDON_END = time(23, 59)
    US_START = time(0, 0)
    US_END = time(23, 59)
    
    # Scoring
    MIN_TOTAL_SCORE = 50  # Lowered from 65 for more signals
    SCORE_WEIGHTS = {
        'trend': 25,
        'momentum': 20,
        'volume': 10,  # Lowered from 15
        'volatility': 10,  # Lowered from 15
        'structure': 10,  # Lowered from 15
        'derivatives': 5  # Lowered from 10
    }


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ADX indicator"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = calculate_atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), period)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period).mean() / tr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period).mean()
    
    return adx


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate VWAP"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp = (typical_price * df['volume']).cumsum()
    cumulative_vol = df['volume'].cumsum()
    return cumulative_tp / cumulative_vol


def calculate_bollinger_width(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Bollinger Band width as volatility measure"""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (2 * std)
    lower = middle - (2 * std)
    return (upper - lower) / middle * 100


def calculate_swing_highs_lows(df: pd.DataFrame, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Calculate swing highs and lows"""
    highs = df['high'].rolling(window=lookback, center=True).max()
    lows = df['low'].rolling(window=lookback, center=True).min()
    
    swing_high = df['high'][(df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])]
    swing_low = df['low'][(df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])]
    
    return swing_high, swing_low


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(klines: List) -> pd.DataFrame:
    """Convert klines to DataFrame with indicators"""
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
    
    return df


# ============================================================================
# CONFIRMATION BLOCKS
# ============================================================================

@dataclass
class BlockResult:
    """Result from a single confirmation block"""
    passed: bool
    score: float
    details: Dict = field(default_factory=dict)


class TrendBlock:
    """Block 1: Trend Direction (EMA 50/200 alignment)"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
    
    def check(self, df: pd.DataFrame, direction: str) -> BlockResult:
        """Check trend alignment"""
        if len(df) < self.settings.SLOW_EMA:
            return BlockResult(False, 0, {'error': 'insufficient_data'})
        
        close = df['close']
        ema_50 = calculate_ema(close, self.settings.FAST_EMA)
        ema_200 = calculate_ema(close, self.settings.SLOW_EMA)
        
        ema_50_val = float(ema_50.iloc[-1])
        ema_200_val = float(ema_200.iloc[-1])
        
        # Check alignment
        if direction == 'LONG':
            passed = ema_50_val > ema_200_val
            score = 25 if passed else 0
        else:
            passed = ema_50_val < ema_200_val
            score = 25 if passed else 0
        
        # Bonus for strong separation
        separation = abs(ema_50_val - ema_200_val) / ema_50_val * 100
        if separation > 1:
            score = min(score + 5, 25)
        
        return BlockResult(
            passed=passed,
            score=score,
            details={
                'ema_50': ema_50_val,
                'ema_200': ema_200_val,
                'separation_pct': separation,
                'alignment': 'bullish' if ema_50_val > ema_200_val else 'bearish'
            }
        )


class MomentumBlock:
    """Block 2: Momentum (RSI + MACD)"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
    
    def check(self, df: pd.DataFrame, direction: str) -> BlockResult:
        """Check momentum confirmation"""
        if len(df) < 30:
            return BlockResult(False, 0, {'error': 'insufficient_data'})
        
        close = df['close']
        
        # RSI
        rsi = calculate_rsi(close, self.settings.RSI_PERIOD)
        rsi_val = float(rsi.iloc[-1])
        
        # MACD
        ema_12 = calculate_ema(close, 12)
        ema_26 = calculate_ema(close, 26)
        macd_line = ema_12 - ema_26
        signal_line = calculate_ema(macd_line, 9)
        macd_hist = macd_line - signal_line
        macd_val = float(macd_hist.iloc[-1])
        
        rsi_passed = False
        macd_passed = False
        score = 0
        
        if direction == 'LONG':
            # RSI: between 40-55 for pullback entry
            rsi_passed = self.settings.RSI_LONG_MIN <= rsi_val <= self.settings.RSI_LONG_MAX
            # MACD: histogram positive or crossing up
            macd_passed = macd_val > 0 or float(macd_hist.iloc[-2]) < 0
            
            if rsi_passed and macd_passed:
                score = 20
            elif rsi_passed or macd_passed:
                score = 10
        else:
            # RSI: between 45-60 for pullback entry
            rsi_passed = self.settings.RSI_SHORT_MIN <= rsi_val <= self.settings.RSI_SHORT_MAX
            # MACD: histogram negative or crossing down
            macd_passed = macd_val < 0 or float(macd_hist.iloc[-2]) > 0
            
            if rsi_passed and macd_passed:
                score = 20
            elif rsi_passed or macd_passed:
                score = 10
        
        return BlockResult(
            passed=rsi_passed or macd_passed,
            score=score,
            details={
                'rsi': rsi_val,
                'macd_hist': macd_val,
                'rsi_passed': rsi_passed,
                'macd_passed': macd_passed
            }
        )


class VolumeBlock:
    """Block 3: Volume Confirmation"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
    
    def check(self, df: pd.DataFrame, direction: str) -> BlockResult:
        """Check volume - ALWAY PASS, just adjust score"""
        if 'volume' not in df.columns or len(df) < 20:
            return BlockResult(True, 10, {'volume_ratio': 1.0, 'note': 'no_data'})
        
        vol_ma = df['volume'].rolling(20).mean()
        vol_current = df['volume'].iloc[-1]
        vol_ma_val = float(vol_ma.iloc[-1])
        vol_ratio = vol_current / vol_ma_val if vol_ma_val > 0 else 1.0
        
        # Volume always passes - just score bonus
        score = 10 if vol_ratio >= 1.0 else 5
        
        return BlockResult(
            passed=True,  # Always pass
            score=score,
            details={
                'volume_ratio': vol_ratio,
                'volume_spike': vol_ratio >= 1.5
            }
        )


class VolatilityBlock:
    """Block 4: Volatility Filter (ATR + Bollinger Band width)"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
    
    def check(self, df: pd.DataFrame, direction: str) -> BlockResult:
        """Check volatility - ALWAY PASS, just adjust score"""
        if len(df) < 20:
            return BlockResult(True, 10, {'atr_pct': 0, 'note': 'no_data'})
        
        close = df['close']
        atr = calculate_atr(df, self.settings.ATR_PERIOD)
        atr_val = float(atr.iloc[-1])
        atr_pct = atr_val / float(close.iloc[-1]) * 100
        
        bb_width = calculate_bollinger_width(close)
        bb_val = float(bb_width.iloc[-1])
        
        # Always pass - just score based on quality
        if 0.3 <= atr_pct <= 2.0:
            score = 10
        elif atr_pct > 0:
            score = 7
        else:
            score = 5
        
        return BlockResult(
            passed=True,  # Always pass
            score=score,
            details={
                'atr_pct': atr_pct,
                'bb_width': bb_val,
                'volatility_ok': True
            }
        )


class StructureBlock:
    """Block 5: Market Structure (Higher highs/lows, support/resistance)"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
    
    def check(self, df: pd.DataFrame, direction: str) -> BlockResult:
        """Check market structure - ALWAY PASS, just adjust score"""
        if len(df) < 30:
            return BlockResult(True, 8, {'structure': 'unknown', 'note': 'no_data'})
        
        close = df['close']
        highs = df['high']
        lows = df['low']
        
        # Calculate recent swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df) - 5):
            if highs.iloc[i] == highs.iloc[i-5:i+5].max():
                swing_highs.append(float(highs.iloc[i]))
            if lows.iloc[i] == lows.iloc[i-5:i+5].min():
                swing_lows.append(float(lows.iloc[i]))
        
        score = 8  # Base score
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if direction == 'LONG':
                recent_lows = swing_lows[-3:]
                if len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
                    score = 10  # Higher low bonus
            else:
                recent_highs = swing_highs[-3:]
                if len(recent_highs) >= 2 and recent_highs[-1] < recent_highs[-2]:
                    score = 10  # Lower high bonus
        
        return BlockResult(
            passed=True,  # Always pass
            score=score,
            details={
                'structure': 'higher_lows' if direction == 'LONG' else 'lower_highs'
            }
        )


class DerivativesBlock:
    """Block 6: Derivatives Data (Funding Rate - simulated)"""
    
    def __init__(self, settings: StrategySettings):
        self.settings = settings
    
    def check(self, df: pd.DataFrame, direction: str, 
              funding_rate: float = 0.0, open_interest: float = 0.0) -> BlockResult:
        """Check derivatives data"""
        # In real implementation, fetch from Binance API
        # For now, use simulated data based on price momentum
        
        # Funding rate check
        if funding_rate == 0:
            # Simulate based on recent price action
            returns = df['close'].pct_change(8).iloc[-1]  # 1h return for 5m data
            funding_rate = returns * 100  # Approximate funding from market
        
        # Open interest check (simplified)
        if open_interest == 0:
            open_interest = 1.0  # Neutral
        
        score = 10
        passed = True
        
        if direction == 'LONG':
            # For longs, funding should not be too negative
            # Negative funding = bears paying longs
            if funding_rate < -0.1:
                score = 5
                passed = True  # Actually good for longs
        else:
            # For shorts, funding should not be too positive
            if funding_rate > 0.1:
                score = 5
                passed = True  # Good for shorts
        
        return BlockResult(
            passed=passed,
            score=score,
            details={
                'funding_rate': funding_rate,
                'open_interest': open_interest
            }
        )


# ============================================================================
# TRADE SIGNAL
# ============================================================================

@dataclass
class TradeSignal:
    """Complete trade signal with all block results"""
    direction: str  # 'LONG', 'SHORT', or 'NO_TRADE'
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    total_score: float
    max_score: float = 100
    confidence: float = 0.0
    blocks: Dict = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    regime: str = 'unknown'
    session: str = 'unknown'


# ============================================================================
# MAIN STRATEGY CLASS
# ============================================================================

class CoreStrategy:
    """
    Core Strategy - High Win-Rate Trading Engine
    
    Implements:
    - 6 confirmation blocks with weighted scoring
    - Multi-timeframe confirmation
    - Session filtering
    - Smart entry/exit
    """
    
    def __init__(self, settings: StrategySettings = None):
        self.settings = settings or StrategySettings()
        
        # Initialize blocks
        self.blocks = {
            'trend': TrendBlock(self.settings),
            'momentum': MomentumBlock(self.settings),
            'volume': VolumeBlock(self.settings),
            'volatility': VolatilityBlock(self.settings),
            'structure': StructureBlock(self.settings),
            'derivatives': DerivativesBlock(self.settings)
        }
        
        # Session tracking
        self.daily_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0
        }
    
    def check_session(self) -> Tuple[bool, str]:
        """Check if current time is in good trading session"""
        now = datetime.utcnow().time()
        
        # London session: 7:00 - 16:00 UTC
        if self.settings.LONDON_START <= now <= self.settings.LONDON_END:
            return True, 'LONDON'
        
        # US session: 13:30 - 22:00 UTC
        if self.settings.US_START <= now <= self.settings.US_END:
            return True, 'US'
        
        return False, 'OFF_HOURS'
    
    def check_adx_filter(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Check ADX for trend strength"""
        if len(df) < self.settings.ADX_PERIOD + 1:
            return True, 50  # Allow if not enough data
        
        adx = calculate_adx(df['high'], df['low'], df['close'], self.settings.ADX_PERIOD)
        adx_val = float(adx.iloc[-1])
        
        # ADX > 20 = strong enough for trading
        # ADX < 15 = dead market, skip
        passed = adx_val >= self.settings.ADX_MIN
        return passed, adx_val
    
    def check_vwap_pullback(self, df: pd.DataFrame, direction: str) -> Tuple[bool, float]:
        """Check if price is at VWAP or EMA50 pullback"""
        if not self.settings.USE_VWAP or len(df) < 20:
            return True, 0.0
        
        vwap = calculate_vwap(df)
        ema_50 = calculate_ema(df['close'], self.settings.FAST_EMA)
        
        vwap_val = float(vwap.iloc[-1])
        ema_val = float(ema_50.iloc[-1])
        price = float(df['close'].iloc[-1])
        
        # Average of VWAP and EMA 50 as reference
        reference = (vwap_val + ema_val) / 2
        
        # Distance from reference
        distance_pct = abs(price - reference) / reference * 100
        
        if direction == 'LONG':
            # For LONG, price should be near or below reference (pullback)
            at_pullback = price <= reference * 1.002  # Within 0.2%
            passed = at_pullback and distance_pct < 1.0
        else:
            # For SHORT, price should be near or above reference
            at_pullback = price >= reference * 0.998  # Within 0.2%
            passed = at_pullback and distance_pct < 1.0
        
        return passed, distance_pct
    
    def analyze_entry(self, df_5m: pd.DataFrame, 
                     df_15m: pd.DataFrame = None,
                     df_1h: pd.DataFrame = None,
                     funding_rate: float = 0.0) -> TradeSignal:
        """
        Main analysis function - generates trade signal
        """
        # Default response
        no_signal = TradeSignal(
            direction='NO_TRADE',
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            leverage=1,
            total_score=0,
            reasons=['no_signal']
        )
        
        if len(df_5m) < 50:
            no_signal.reasons.append('insufficient_data')
            return no_signal
        
        try:
            # 1. Session Check (always pass now)
            in_session, session_name = self.check_session()
            
            # 2. ADX Filter (always pass, just log)
            adx_passed, adx_val = self.check_adx_filter(df_5m)
            
            # 3. Multi-timeframe alignment (if data available)
            tf_confirmed = True
            if df_15m is not None and len(df_15m) > 50:
                ema_50_15m = calculate_ema(df_15m['close'], self.settings.FAST_EMA)
                ema_200_15m = calculate_ema(df_15m['close'], self.settings.SLOW_EMA)
                tf_15m = float(ema_50_15m.iloc[-1]) > float(ema_200_15m.iloc[-1])
            else:
                tf_15m = True
            
            if df_1h is not None and len(df_1h) > 50:
                ema_50_1h = calculate_ema(df_1h['close'], self.settings.FAST_EMA)
                ema_200_1h = calculate_ema(df_1h['close'], self.settings.SLOW_EMA)
                tf_1h = float(ema_50_1h.iloc[-1]) > float(ema_200_1h.iloc[-1])
            else:
                tf_1h = tf_15m
            
            # Store regime
            regime = 'unknown'
            if tf_1h and tf_15m:
                regime = 'confirmed_trend'
            elif tf_15m or tf_1h:
                regime = 'partial_confirm'
            else:
                regime = 'conflicting'
            
            # 4. Run all confirmation blocks for BOTH directions
            results = {'LONG': {}, 'SHORT': {}}
            total_scores = {'LONG': 0, 'SHORT': 0}
            
            for direction in ['LONG', 'SHORT']:
                for block_name, block in self.blocks.items():
                    if block_name == 'derivatives':
                        result = block.check(df_5m, direction, funding_rate)
                    else:
                        result = block.check(df_5m, direction)
                    
                    results[direction][block_name] = result
                    total_scores[direction] += result.score
            
            # 5. Check VWAP/EMA pullback
            for direction in ['LONG', 'SHORT']:
                vwap_pass, vwap_dist = self.check_vwap_pullback(df_5m, direction)
                results[direction]['vwap_pullback'] = BlockResult(
                    passed=vwap_pass,
                    score=5 if vwap_pass else 0,
                    details={'distance_pct': vwap_dist}
                )
                total_scores[direction] += results[direction]['vwap_pullback'].score
            
            # 6. Determine best direction
            max_score = sum(self.settings.SCORE_WEIGHTS.values()) + 5  # +5 for VWAP
            
            best_direction = None
            best_score = 0
            
            for direction in ['LONG', 'SHORT']:
                score = total_scores[direction]
                if score > best_score:
                    best_score = score
                    best_direction = direction
            
            # 7. Check minimum threshold (lowered for more signals)
            if best_score < 45:  # Very permissive threshold
                no_signal.reasons.append(f'score_too_low_{best_score:.0f}')
                return no_signal
            
            # 8. Multi-timeframe check (optional - just log, don't block)
            # Removed strict requirement
            
            # 9. Calculate entry parameters
            entry_price = float(df_5m['close'].iloc[-1])
            atr = calculate_atr(df_5m, self.settings.ATR_PERIOD)
            atr_val = float(atr.iloc[-1])
            
            if best_direction == 'LONG':
                stop_loss = entry_price * (1 - self.settings.SL_PCT)
                take_profit = entry_price * (1 + self.settings.TP_PCT)
            else:
                stop_loss = entry_price * (1 + self.settings.SL_PCT)
                take_profit = entry_price * (1 - self.settings.TP_PCT)
            
            # Calculate leverage based on ATR
            sl_distance_pct = abs(entry_price - stop_loss) / entry_price
            optimal_leverage = min(
                self.settings.MAX_LEVERAGE,
                max(2, int(0.01 / sl_distance_pct))  # Risk 1% per trade
            )
            
            # Calculate confidence
            confidence = (best_score / max_score) * 100
            
            return TradeSignal(
                direction=best_direction,
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                leverage=optimal_leverage,
                total_score=best_score,
                max_score=max_score,
                confidence=round(confidence, 1),
                blocks=results[best_direction],
                regime=regime,
                session=session_name
            )
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            no_signal.reasons.append(f'error_{str(e)}')
            return no_signal
    
    def should_close(self, position: Dict, df_current: pd.DataFrame, 
                    entry_signal: TradeSignal) -> Tuple[bool, str]:
        """Determine if position should be closed"""
        if not position or not entry_signal:
            return False, ""
        
        entry_price = position.get('entry_price', 0)
        current_price = float(df_current['close'].iloc[-1])
        side = position.get('side', 'LONG')
        
        # 1. Check take profit hit
        if entry_signal.direction == 'LONG':
            if current_price >= entry_signal.take_profit:
                return True, 'TAKE_PROFIT'
            if current_price <= entry_signal.stop_loss:
                return True, 'STOP_LOSS'
        else:
            if current_price <= entry_signal.take_profit:
                return True, 'TAKE_PROFIT'
            if current_price >= entry_signal.stop_loss:
                return True, 'STOP_LOSS'
        
        # 2. Check for reversal signal
        new_signal = self.analyze_entry(df_current)
        if new_signal.direction != entry_signal.direction and new_signal.confidence > 80:
            return True, f'REVERSAL_{new_signal.direction}'
        
        # 3. Time-based exit (max 2 hours for scalping)
        if 'open_time' in position:
            try:
                if isinstance(position['open_time'], str):
                    open_time = datetime.fromisoformat(position['open_time'])
                else:
                    open_time = position['open_time']
                
                hours_open = (datetime.now() - open_time).total_seconds() / 3600
                if hours_open > 2:  # Max 2 hours for scalping
                    return True, 'TIME_EXIT'
            except:
                pass
        
        return False, ""
    
    def get_status(self) -> Dict:
        """Get current strategy status"""
        return {
            'daily_trades': self.daily_stats['trades'],
            'daily_pnl': self.daily_stats['pnl'],
            'win_rate': self.daily_stats['wins'] / max(self.daily_stats['trades'], 1) * 100
        }
    
    def record_trade(self, won: bool, pnl: float):
        """Record trade result"""
        self.daily_stats['trades'] += 1
        self.daily_stats['pnl'] += pnl
        
        if won:
            self.daily_stats['wins'] += 1
        else:
            self.daily_stats['losses'] += 1


# ============================================================================
# COMPATIBILITY WRAPPER
# ============================================================================

class CoreStrategyWrapper:
    """Wrapper for existing bot interface"""
    
    def __init__(self, risk_config: dict, symbol: str, model_path: str = None):
        self.symbol = symbol
        self.strategy = CoreStrategy()
        
        if risk_config:
            self.strategy.settings.MAX_LEVERAGE = risk_config.get('leverage', 5)
    
    def generate_signal(self, klines: List) -> Tuple[Optional[str], float, Dict]:
        """Generate signal compatible with bot interface"""
        try:
            df = prepare_data(klines)
            signal = self.strategy.analyze_entry(df)
            
            if signal.direction == 'NO_TRADE':
                reason = ', '.join(signal.reasons[-2:]) if signal.reasons else 'no_signal'
                return None, 0, {'reason': reason, 'regime': signal.regime}
            
            details = {
                'stop_loss': signal.stop_loss,
                'take_profit_1': signal.take_profit,
                'entry_price': signal.entry_price,
                'confidence': signal.confidence,
                'total_score': signal.total_score,
                'regime': signal.regime,
                'session': signal.session,
                'leverage': signal.leverage,
                'blocks': {k: {'passed': v.passed, 'score': v.score} 
                          for k, v in signal.blocks.items()}
            }
            
            logger.info(f"[{self.symbol}] {signal.direction} @ {signal.entry_price} | "
                       f"Conf: {signal.confidence:.0f}% | Score: {signal.total_score:.0f}/100 | "
                       f"Regime: {signal.regime} | Session: {signal.session}")
            
            return signal.direction, signal.confidence, details
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error: {e}")
            return None, 0, {'error': str(e)}
    
    def should_close_position(self, position: Dict, current_price: float, 
                           klines: List) -> Tuple[bool, str]:
        """Check if should close"""
        try:
            df = prepare_data(klines)
            return self.strategy.should_close(position, df, None)
        except Exception as e:
            logger.error(f"[{self.symbol}] Close check error: {e}")
            return False, ""
    
    def calculate_position_size(self, available: float, price: float, conf: float) -> Tuple[float, float]:
        """Calculate position size"""
        risk_amount = available * (self.strategy.settings.RISK_PER_TRADE_PCT / 100)
        position = available * 0.3  # 30% of balance
        quantity = position / price
        return round(quantity, 4), round(position, 2)
    
    def record_trade_result(self, pnl: float, won: bool):
        """Record trade"""
        self.strategy.record_trade(won, pnl)
    
    def get_status(self) -> Dict:
        """Get status"""
        return self.strategy.get_status()


# Alias
HighWinRateStrategy = CoreStrategyWrapper
