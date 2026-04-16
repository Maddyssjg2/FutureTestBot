import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import logging

logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self, risk_config):
        self.risk_config = risk_config
        self.min_quantity = 0.001  # Minimum BTC quantity
        
    def calculate_indicators(self, klines):
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # EMA 21/55 Trend Filter
        ema_21 = EMAIndicator(close=df['close'], window=21)
        ema_55 = EMAIndicator(close=df['close'], window=55)
        df['ema_21'] = ema_21.ema_indicator()
        df['ema_55'] = ema_55.ema_indicator()
        
        # RSI 14 for Momentum
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # ATR for Stop Loss and Take Profit calculation
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        
        # Volume moving average for volume filter
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def generate_signal(self, klines):
        df = self.calculate_indicators(klines)
        
        if len(df) < 60:  # Need at least 55 periods for EMA 55
            return None, 0, None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3] if len(df) > 2 else prev
        
        signal = None
        confidence = 0
        
        # === TREND FILTER: EMA 21/55 ===
        # Bullish: Price > EMA 21 > EMA 55 (uptrend)
        # Bearish: Price < EMA 21 < EMA 55 (downtrend)
        price_above_ema21 = latest['close'] > latest['ema_21']
        ema21_above_ema55 = latest['ema_21'] > latest['ema_55']
        trend_bullish = price_above_ema21 and ema21_above_ema55
        
        price_below_ema21 = latest['close'] < latest['ema_21']
        ema21_below_ema55 = latest['ema_21'] < latest['ema_55']
        trend_bearish = price_below_ema21 and ema21_below_ema55
        
        # === MOMENTUM TRIGGER: RSI 50 Crossover ===
        # Long: RSI crosses above 50 (momentum confirms trend)
        # Short: RSI crosses below 50 (momentum confirms trend)
        rsi_above_50 = latest['rsi'] > 50
        rsi_below_50 = latest['rsi'] < 50
        prev_rsi_above_50 = prev['rsi'] > 50
        prev_rsi_below_50 = prev['rsi'] < 50
        
        rsi_bullish_cross = prev_rsi_below_50 and rsi_above_50
        rsi_bearish_cross = prev_rsi_above_50 and rsi_below_50
        
        # RSI momentum strength (distance from 50)
        rsi_momentum_strength = abs(latest['rsi'] - 50) / 50 * 100
        
        # === VOLUME FILTER: Above 20-period average ===
        volume_confirmed = latest['volume'] > latest['volume_ma'] * 1.1  # 10% above average
        
        # === SIGNAL GENERATION ===
        # Long: Uptrend + RSI crosses above 50 + Volume confirmation
        # Short: Downtrend + RSI crosses below 50 + Volume confirmation
        
        long_conditions = trend_bullish and rsi_bullish_cross and volume_confirmed
        short_conditions = trend_bearish and rsi_bearish_cross and volume_confirmed
        
        # Confidence scoring (60-100% range)
        if long_conditions or short_conditions:
            base_confidence = 60
            
            # Add confidence for strong trend
            trend_strength = abs(latest['ema_21'] - latest['ema_55']) / latest['close'] * 100
            if trend_strength > 1:
                base_confidence += 10
            if trend_strength > 2:
                base_confidence += 10
            
            # Add confidence for RSI momentum strength
            if rsi_momentum_strength > 20:
                base_confidence += 10
            if rsi_momentum_strength > 40:
                base_confidence += 10
            
            # Volume boost
            volume_ratio = latest['volume'] / latest['volume_ma']
            if volume_ratio > 1.5:
                base_confidence += 5
            
            confidence = min(base_confidence, 95)
        
        if long_conditions:
            signal = 'LONG'
            logger.info(f"LONG Signal: Trend={trend_bullish}, RSI Cross={rsi_bullish_cross}, Vol={volume_confirmed}, Conf={confidence}%")
        elif short_conditions:
            signal = 'SHORT'
            logger.info(f"SHORT Signal: Trend={trend_bearish}, RSI Cross={rsi_bearish_cross}, Vol={volume_confirmed}, Conf={confidence}%")
        
        # === RISK MANAGEMENT: Split Take Profits ===
        # Stop Loss: 1x ATR (tight stop)
        # TP1: 1.5x ATR (close 50% of position)
        # TP2: 2.5x ATR (close remaining 50%)
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None
        
        if signal:
            atr = latest['atr']
            
            if signal == 'LONG':
                stop_loss = latest['close'] - (atr * 1.0)  # 1x ATR stop
                take_profit_1 = latest['close'] + (atr * 1.5)  # 50% at 1.5x
                take_profit_2 = latest['close'] + (atr * 2.5)  # 50% at 2.5x
            else:
                stop_loss = latest['close'] + (atr * 1.0)  # 1x ATR stop
                take_profit_1 = latest['close'] - (atr * 1.5)  # 50% at 1.5x
                take_profit_2 = latest['close'] - (atr * 2.5)  # 50% at 2.5x
            
            # Risk/Reward calculation for logging
            risk = atr * 1.0
            reward_avg = (atr * 1.5 + atr * 2.5) / 2  # Average of both TPs
            rr_ratio = reward_avg / risk if risk > 0 else 0
            logger.info(f"R/R Ratio: {rr_ratio:.2f}:1 | SL: ${stop_loss:.2f} | TP1: ${take_profit_1:.2f} | TP2: ${take_profit_2:.2f}")
        
        return signal, confidence, {
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'take_profit_1': round(take_profit_1, 2) if take_profit_1 else None,
            'take_profit_2': round(take_profit_2, 2) if take_profit_2 else None,
            'entry_price': round(latest['close'], 2),
            'rsi': round(latest['rsi'], 2),
            'ema_21': round(latest['ema_21'], 2),
            'ema_55': round(latest['ema_55'], 2),
            'atr': round(latest['atr'], 2),
            'volume_ratio': round(latest['volume'] / latest['volume_ma'], 2) if latest['volume_ma'] > 0 else 0,
            'trend_strength': round(abs(latest['ema_21'] - latest['ema_55']) / latest['close'] * 100, 2)
        }
    
    def calculate_position_size(self, balance, current_price, confidence):
        risk_percentage = self.risk_config['trade_percentage']
        
        # Fixed 30% trade size for consistency (per your strategy)
        # Only adjust slightly based on confidence
        confidence_multiplier = 0.8 + (confidence / 100) * 0.4  # 0.8x to 1.2x
        adjusted_percentage = risk_percentage * confidence_multiplier
        
        # Calculate position size in USDT
        position_usdt = balance * (adjusted_percentage / 100)
        
        # Apply leverage
        leveraged_position = position_usdt * self.risk_config['leverage']
        
        # Calculate quantity in BTC
        quantity = leveraged_position / current_price
        
        # Round to appropriate precision
        quantity = round(quantity, 3)
        
        # Ensure minimum quantity
        if quantity < self.min_quantity:
            quantity = self.min_quantity
        
        return quantity, position_usdt
    
    def should_close_position(self, position, current_price, klines):
        df = self.calculate_indicators(klines)
        if len(df) < 10:
            return False, None
        
        latest = df.iloc[-1]
        
        # Calculate PnL %
        entry = position['entry_price']
        if position['side'] == 'LONG':
            price_change_pct = ((current_price - entry) / entry) * 100
            # Exit if trend reverses (EMA 21 crosses below EMA 55)
            if latest['ema_21'] < latest['ema_55']:
                return True, 'Trend Reversed'
        else:
            price_change_pct = ((entry - current_price) / entry) * 100
            # Exit if trend reverses (EMA 21 crosses above EMA 55)
            if latest['ema_21'] > latest['ema_55']:
                return True, 'Trend Reversed'
        
        # Check stop loss / take profit hit (calculated by effective leverage)
        effective_pnl = price_change_pct * position['leverage']
        
        # Stop Loss at -2% effective (keep same)
        if effective_pnl <= -2.0:
            return True, f'Stop Loss Hit ({effective_pnl:.1f}%)'
        
        # Take Profit levels - 2% and 4% effective
        if effective_pnl >= 4.0:  # TP2
            return True, f'TP2 Hit ({effective_pnl:.1f}%)'
        if effective_pnl >= 2.0:  # TP1
            return True, f'TP1 Hit ({effective_pnl:.1f}%)'
        
        return False, None
