"""
Multi-Symbol Trading Bot
Monitors and trades top 20 futures pairs simultaneously.

Strategy: Final Pair Strategy
- Pair-specific optimized rules (75%+ backtested win rate)
- ML model predictions (55%+ confidence threshold)
- Risk management: 1% TP, 2% SL, 5x max leverage
"""

import time
import threading
import logging
from datetime import datetime
from binance_client import BinanceFuturesClient
from config import Config
from telegram_notifier import telegram
from advanced_strategies import generate_signal, SignalType, strategy_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SymbolTrader:
    """Handles trading for a single symbol"""

    def __init__(self, symbol, client, risk_config):
        self.symbol = symbol
        self.client = client
        self.client.symbol = symbol
        self.risk_config = risk_config
        self.strategy_mode = Config.STRATEGY_MODE
        self.strategy = self._build_strategy()
        self.last_signal = None
        self.open_positions = []

        # Cache the last seen entry signal so we can report it on close
        self._pending_entry: dict = {}  # symbol → {signal, confidence, entry_price}
        
        # Track last known position for exchange TP/SL detection
        self._last_position = None
        self._trailing_stop_activated = False  # Track if trailing stop has been set

    def _build_strategy(self):
        """Using Final Strategy: Optimized Pair Rules + ML Predictions"""
        try:
            from final_pair_strategy import FinalStrategyWrapper
            logger.info(
                f"[{self.symbol}] Using FINAL PAIR STRATEGY "
                f"(Optimized Rules + ML | 75%+ WR Target | 1% TP | 2% SL | 5x)"
            )
            return FinalStrategyWrapper(
                risk_config=self.risk_config,
                symbol=self.symbol,
            )
        except Exception as e:
            logger.warning(f"[{self.symbol}] Final pair strategy failed: {e}")
            try:
                from optimized_pair_strategy import OptimizedPairStrategyWrapper
                logger.info(
                    f"[{self.symbol}] Falling back to OPTIMIZED PAIR STRATEGY "
                    f"(75%+ Win Rate Target | Stochastic/RSI | 1% TP | 2% SL)"
                )
                return OptimizedPairStrategyWrapper(
                    risk_config=self.risk_config,
                    symbol=self.symbol,
                )
            except Exception as e2:
                logger.warning(f"[{self.symbol}] Optimized pair strategy failed: {e2}")
                return TradingStrategy(risk_config=self.risk_config)

    # ──────────────────────────────────────────────────────────────────────────
    # Trade reporting helper
    # ──────────────────────────────────────────────────────────────────────────

    def _report_closed_trade(
        self,
        position: dict,
        exit_price: float,
        reason: str,
        klines=None,
        realized_pnl=None,
    ):
        """Log closed trade for tracking."""
        try:
            sym = position["symbol"]
            pnl = realized_pnl if realized_pnl is not None else position.get("unrealized_pnl", 0)
            self._pending_entry.pop(sym, None)
            logger.info(f"[{sym}] Trade closed: {reason} | PnL: {pnl:.2f}")
        except Exception as e:
            logger.debug(f"[{self.symbol}] Trade report error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main check-and-trade loop
    # ──────────────────────────────────────────────────────────────────────────

    def check_and_trade(self):
        """Check for signals and execute trades for this symbol"""
        try:
            self.client.symbol = self.symbol

            balance = self.client.get_account_balance()
            if not balance:
                return None

            available_balance = balance["available"]

            # Fetch multi-timeframe data for advanced strategies
            klines_15m = self.client.get_klines(interval="15m", limit=100)
            klines_1h = self.client.get_klines(interval="1h", limit=100)
            klines_4h = self.client.get_klines(interval="4h", limit=100)
            
            if not klines_15m or len(klines_15m) < 60:
                return None
            if not klines_1h or len(klines_1h) < 60:
                return None
            if not klines_4h or len(klines_4h) < 60:
                return None

            current_price = self.client.get_mark_price()
            if not current_price:
                return None

            all_positions = self.client.get_open_positions()
            symbol_positions = [p for p in all_positions if p["symbol"] == self.symbol]

            # ── Check existing positions for exits ────────────────────────
            for position in symbol_positions:
                # Use position's mark_price for calculations - more accurate than separate API call
                position_mark_price = position.get("mark_price", current_price)
                
                should_close, reason = self.strategy.should_close_position(
                    position, position_mark_price, klines
                )

                if position["side"] == "LONG":
                    price_change_pct = (
                        (position_mark_price - position["entry_price"])
                        / position["entry_price"]
                        * 100
                    )
                else:
                    price_change_pct = (
                        (position["entry_price"] - position_mark_price)
                        / position["entry_price"]
                        * 100
                    )

                effective_pnl = price_change_pct * position["leverage"]

                # Bot-managed TP/SL as fallback if exchange orders fail
                # Both testnet and mainnet try exchange TP/SL first
                # If exchange fails, bot manages with 1-sec polling
                if effective_pnl <= -2.0:
                    should_close = True
                    reason = f"Stop Loss Hit ({effective_pnl:.1f}%)"
                elif effective_pnl >= 4.0:
                    should_close = True
                    reason = f"TP2 Hit ({effective_pnl:.1f}%)"
                elif effective_pnl >= 2.0:
                    should_close = True
                    reason = f"TP1 Hit ({effective_pnl:.1f}%)"

                if should_close:
                    logger.info(f"[{self.symbol}] 🚨 URGENT {reason} DETECTED - CLOSING NOW!")
                    logger.info(f"[{self.symbol}] Entry: ${position['entry_price']:.4f}, Current: ${position_mark_price:.4f}, "
                               f"Unrealized PnL: ${position.get('unrealized_pnl', 0):.2f}")
                    logger.info(f"[{self.symbol}] Size: {abs(position['amount']):.2f} {self.symbol} - EXECUTING MARKET CLOSE!")
                    
                    close_order = self.client.close_position(position)
                    
                    if not close_order:
                        logger.error(f"[{self.symbol}] ❌ FAILED to close position! {reason} hit but close order failed!")
                        continue
                    
                    # Get actual fill price from close order if available
                    actual_exit_price = position_mark_price  # fallback to mark price
                    realized_pnl = position.get("unrealized_pnl", 0)  # fallback
                    
                    if close_order and 'avgPrice' in close_order and float(close_order['avgPrice']) > 0:
                        try:
                            actual_exit_price = float(close_order['avgPrice'])
                            logger.info(f"[{self.symbol}] ✅ Position closed at avg price: ${actual_exit_price:.4f}")
                        except:
                            logger.warning(f"[{self.symbol}] Could not parse avgPrice from close order")
                    elif close_order and 'price' in close_order and float(close_order['price']) > 0:
                        try:
                            actual_exit_price = float(close_order['price'])
                            logger.info(f"[{self.symbol}] ✅ Position closed at price: ${actual_exit_price:.4f}")
                        except:
                            pass
                    
                    # Get actual realized PnL from close order if available
                    if close_order and 'realizedPnl' in close_order:
                        try:
                            realized_pnl = float(close_order['realizedPnl'])
                            logger.info(f"[{self.symbol}] 💰 Realized PnL from Binance: ${realized_pnl:.2f} USDT")
                        except:
                            pass
                    
                    logger.info(f"[{self.symbol}] Order ID: {close_order.get('orderId')}, Status: {close_order.get('status')}")

                    # ── Report to adaptive tracker ─────────────────────────
                    self._report_closed_trade(
                        position=position,
                        exit_price=actual_exit_price,
                        reason=reason,
                        klines=klines,
                        realized_pnl=realized_pnl,
                    )

                    # Telegram - calculate margin used with ACTUAL exit price
                    position_value = abs(position["amount"]) * position["entry_price"]
                    margin_used = position_value / position["leverage"]

                    if "Stop Loss" in reason:
                        telegram.send_stop_loss_hit(
                            symbol=self.symbol,
                            side=position["side"],
                            entry_price=position["entry_price"],
                            exit_price=actual_exit_price,
                            quantity=abs(position["amount"]),
                            pnl=realized_pnl,
                            leverage=position["leverage"],
                            margin_used=margin_used,
                        )
                    elif "TP" in reason:
                        tp_level = "2" if "TP2" in reason else "1"
                        telegram.send_take_profit_hit(
                            symbol=self.symbol,
                            side=position["side"],
                            entry_price=position["entry_price"],
                            exit_price=actual_exit_price,
                            quantity=abs(position["amount"]),
                            pnl=realized_pnl,
                            tp_level=tp_level,
                            leverage=position["leverage"],
                            margin_used=margin_used,
                        )

                    return {
                        "action": "closed",
                        "symbol": self.symbol,
                        "reason": reason,
                        "pnl": realized_pnl,
                        "exit_price": actual_exit_price,
                    }

            # ── TRAILING STOP ACTIVATION AFTER TP1 ──────────────────────────
            # Check if TP1 hit and activate trailing stop on remaining position
            if symbol_positions and not self._trailing_stop_activated:
                # Check if TP1 order has filled
                if self.client.check_tp1_filled():
                    logger.info(f"[{self.symbol}] 🎯 TP1 filled! Activating trailing stop on remaining 50%...")
                    
                    # Cancel old SL order
                    self.client.cancel_stop_loss()
                    
                    # Set trailing stop on remaining position
                    position = symbol_positions[0]
                    trailing_success = self.client.set_trailing_stop(position, callback_rate=1.0)
                    
                    if trailing_success:
                        self._trailing_stop_activated = True
                        logger.info(f"[{self.symbol}] ✅ Trailing stop activated! Remaining 50% protected.")
                        
                        # Send Telegram notification
                        try:
                            telegram.send_message(
                                f"📈 <b>TRAILING STOP ACTIVATED</b>\n\n"
                                f"Symbol: {self.symbol}\n"
                                f"TP1 hit at 2% profit ✅\n"
                                f"Trailing stop: 1% callback\n"
                                f"Remaining 50% position protected!\n\n"
                                f"Bot will capture more upside if price continues, "
                                f"or protect profit if it reverses."
                            )
                        except Exception as e:
                            logger.debug(f"Could not send trailing stop notification: {e}")

            # ── EXCHANGE TP/SL DETECTION ────────────────────────────────────
            # Check if position closed via exchange TP/SL (not bot-managed)
            if self._last_position and not symbol_positions:
                # We had a position, now we don't = exchange TP/SL fired!
                last_pos = self._last_position
                side = last_pos.get("side", "LONG")
                entry_price = last_pos.get("entry_price", current_price)
                last_pnl = last_pos.get("unrealized_pnl", 0)
                
                # Reset trailing stop flag for next trade
                self._trailing_stop_activated = False
                
                # Determine if TP or SL based on PnL
                if last_pnl > 0:
                    reason = f"TP Hit (Exchange) (+{last_pnl:.2f} USDT)"
                    tp_level = "2" if last_pnl >= 4 else "1"
                    logger.info(f"[{self.symbol}] 🎯 Exchange TP executed! PnL: ${last_pnl:.2f}")
                    
                    # Send notification
                    try:
                        margin_used = abs(last_pos.get("amount", 0)) * entry_price / last_pos.get("leverage", 5)
                        telegram.send_take_profit_hit(
                            symbol=self.symbol,
                            side=side,
                            entry_price=entry_price,
                            exit_price=current_price,
                            quantity=abs(last_pos.get("amount", 0)),
                            pnl=last_pnl,
                            tp_level=tp_level,
                            leverage=last_pos.get("leverage", 5),
                            margin_used=margin_used,
                        )
                    except Exception as e:
                        logger.debug(f"Could not send TP notification: {e}")
                else:
                    reason = f"SL Hit (Exchange) ({last_pnl:.2f} USDT)"
                    logger.info(f"[{self.symbol}] 🛑 Exchange SL executed! Loss: ${last_pnl:.2f}")
                    
                    # Send notification + trigger retraining
                    try:
                        margin_used = abs(last_pos.get("amount", 0)) * entry_price / last_pos.get("leverage", 5)
                        telegram.send_stop_loss_hit(
                            symbol=self.symbol,
                            side=side,
                            entry_price=entry_price,
                            exit_price=current_price,
                            quantity=abs(last_pos.get("amount", 0)),
                            pnl=last_pnl,
                            leverage=last_pos.get("leverage", 5),
                            margin_used=margin_used,
                        )
                    except Exception as e:
                        logger.debug(f"Could not send SL notification: {e}")
                    
                    # 🎓 TRIGGER RETRAINING ON LOSS
                    try:
                        from loss_learning_engine import process_loss_for_learning
                        process_loss_for_learning(
                            symbol=self.symbol,
                            entry_price=entry_price,
                            exit_price=current_price,
                            side=side,
                            pnl=last_pnl,
                            klines=klines,
                        )
                        logger.info(f"[{self.symbol}] 🎓 Loss learning triggered after exchange SL")
                    except Exception as e:
                        logger.debug(f"Could not trigger loss learning: {e}")
                
                self._last_position = None  # Clear tracked position
                return {
                    "action": "closed",
                    "symbol": self.symbol,
                    "reason": reason,
                    "pnl": last_pnl,
                    "exit_price": current_price,
                }
            
            # Update last known position for next iteration
            if symbol_positions:
                self._last_position = symbol_positions[0].copy()

            # ── Max-position guard ────────────────────────────────────────
            total_positions = len(all_positions)
            if total_positions >= self.risk_config["max_positions"] * 3:
                logger.debug(
                    f"[{self.symbol}] Max positions reached ({total_positions})"
                )
                return None

            if symbol_positions:
                return None

            # ── Generate signal using 5-strategy confluence engine ─────────
            signal_data = self.strategy.generate_signal(klines_15m, klines_1h, klines_4h)
            
            signal = signal_data.get("signal")
            confidence = signal_data.get("confidence", 0)
            details = {
                "entry_price": signal_data.get("entry_price", current_price),
                "stop_loss": signal_data.get("stop_loss", current_price * 0.98),
                "take_profit_1": signal_data.get("take_profit", current_price * 1.02),
                "take_profit_2": signal_data.get("take_profit", current_price * 1.04),
                "risk_reward": signal_data.get("risk_reward", 1.5),
                "confluence_count": signal_data.get("confluence_count", 0),
                "strategies": signal_data.get("strategies", []),
                "metadata": signal_data.get("metadata", {}),
            }
            
            # Override TP levels for 1.5R to 2R targets
            if signal and details["risk_reward"] >= 1.5:
                # Set TP1 at 1.5R, TP2 at 2R
                risk = abs(details["entry_price"] - details["stop_loss"])
                details["take_profit_1"] = details["entry_price"] + (risk * 1.5) if signal == "LONG" else details["entry_price"] - (risk * 1.5)
                details["take_profit_2"] = details["entry_price"] + (risk * 2.0) if signal == "LONG" else details["entry_price"] - (risk * 2.0)

            result = {
                "symbol": self.symbol,
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }

            # Check confluence requirement (2-3 strategies)
            confluence_met = details["confluence_count"] >= 2
            confidence_met = confidence >= Config.MIN_SIGNAL_CONFIDENCE_MULTI
            
            if signal and confluence_met and confidence_met:
                logger.info(
                    f"[{self.symbol}] ✓ HIGH-CONFIDENCE {signal} signal: "
                    f"{confidence:.1f}% (min: {Config.MIN_SIGNAL_CONFIDENCE_MULTI}%) | "
                    f"Confluence: {details['confluence_count']}/5 strategies"
                )
                logger.info(f"[{self.symbol}] Active strategies: {', '.join(details['strategies'])}")

                # Calculate position size with 1-2% risk, max 10x leverage
                symbol_allocation = available_balance / len(Config.TOP_20_SYMBOLS)
                quantity = self.strategy.calculate_position_size(
                    symbol_allocation, details["entry_price"], details["stop_loss"]
                )
                position_usdt = quantity * current_price

                logger.info(f"[{self.symbol}] Trade details: qty={quantity:.4f}, position=${position_usdt:.2f}, "
                           f"available=${available_balance:.2f}, max_10%=${available_balance * 0.1:.2f}")

                if position_usdt > available_balance * 0.1:
                    logger.warning(f"[{self.symbol}] Position too large (${position_usdt:.2f} > ${available_balance * 0.1:.2f}), skipping")
                    return result

                side = "BUY" if signal == "LONG" else "SELL"
                logger.info(f"[{self.symbol}] Placing {side} order for {quantity}...")
                order = self.client.place_order(
                    side=side, order_type="MARKET", quantity=quantity, 
                    stop_loss=details["stop_loss"], take_profit_1=details["take_profit_1"],
                    take_profit_2=details["take_profit_2"]
                )

                if order:
                    logger.info(
                        f"[{self.symbol}] Order placed: {quantity} "
                        f"@ ${details['entry_price']:.2f}"
                    )

                    # ── Cache entry info for the tracker ──────────────────
                    self._pending_entry[self.symbol] = {
                        "signal": signal,
                        "confidence": confidence,
                        "entry_price": details["entry_price"],
                    }

                    logger.info(f"[{self.symbol}] 📤 Sending Telegram trade entry notification...")
                    telegram.send_trade_entry(
                        symbol=self.symbol,
                        side=signal,
                        entry_price=details["entry_price"],
                        quantity=quantity,
                        stop_loss=details["stop_loss"],
                        take_profit_1=details["take_profit_1"],
                        take_profit_2=details["take_profit_2"],
                        leverage=self.risk_config["leverage"],
                        confidence=confidence,
                    )
                    logger.info(f"[{self.symbol}] ✅ Telegram trade entry notification sent")

                    # Reset trailing stop flag for new position
                    self._trailing_stop_activated = False

                    result["action"] = "opened"
                    result["details"] = details
                    return result
                else:
                    logger.error(f"[{self.symbol}] ❌ Order FAILED to place! Check API/balance/limits.")

            elif signal:
                logger.debug(
                    f"[{self.symbol}] ✗ Signal rejected: {confidence:.1f}% "
                    f"< {Config.MIN_SIGNAL_CONFIDENCE_MULTI}%"
                )
                telegram.send_signal_detected(
                    symbol=self.symbol,
                    signal=signal,
                    confidence=confidence,
                    entry_price=details["entry_price"],
                    stop_loss=details["stop_loss"],
                    take_profit_1=details["take_profit_1"],
                    take_profit_2=details["take_profit_2"],
                    rsi=details.get("rsi", 0),
                    volume_ratio=details.get("volume_ratio", 0),
                )

            # Return position status for dynamic polling speed
            has_position = len(symbol_positions) > 0
            
            if signal:
                result["has_position"] = has_position
                return result
            elif has_position:
                # Return minimal result when in position but no signal (for fast TP/SL checking)
                return {"has_position": True, "action": "monitoring"}
            return None

        except Exception as e:
            if "Invalid symbol" in str(e) or "-1121" in str(e):
                logger.warning(f"[{self.symbol}] Symbol not available on testnet")
                return None
            logger.error(f"[{self.symbol}] Error in check_and_trade: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# MultiSymbolBot
# ─────────────────────────────────────────────────────────────────────────────

class MultiSymbolBot:
    """Manages trading across multiple symbols"""

    def __init__(self, symbols=None):
        self.client = BinanceFuturesClient()
        self.risk_config = Config.get_risk_config()
        self.symbols = symbols or Config.get_default_multi_symbols()
        self.traders = {}
        self.is_running = False
        self.threads = []
        self.status = {
            "running": False,
            "symbols_monitored": len(self.symbols),
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "open_positions": 0,
            "last_signals": {},
        }

        for symbol in self.symbols:
            self.traders[symbol] = SymbolTrader(symbol, self.client, self.risk_config)

        logger.info(f"MultiSymbolBot initialised for {len(self.symbols)} symbols")

    def monitor_symbol(self, symbol):
        trader = self.traders[symbol]
        has_open_position = False
        
        while self.is_running:
            try:
                result = trader.check_and_trade()
                if result:
                    self.status["last_signals"][symbol] = result
                    # Track position status for dynamic polling
                    has_open_position = result.get("has_position", False) or result.get("action") == "opened"
                    
                    # Only count COMPLETED trades (on close), not opens
                    if result.get("action") == "closed":
                        has_open_position = False  # Position just closed
                        self.status["total_trades"] += 1
                        # Track win/loss based on PnL
                        pnl = result.get("pnl", 0)
                        if pnl > 0:
                            self.status["winning_trades"] += 1
                        else:
                            self.status["losing_trades"] += 1
                        # Update win rate
                        total = self.status["winning_trades"] + self.status["losing_trades"]
                        if total > 0:
                            self.status["win_rate"] = round((self.status["winning_trades"] / total) * 100, 1)
                
                # DYNAMIC POLLING: Ultra-fast when in position (TP/SL), slow when waiting
                if has_open_position:
                    time.sleep(1)  # 1 second - ULTRA AGGRESSIVE when position open
                else:
                    time.sleep(30)  # 30 seconds - relaxed when waiting for signals
                    
            except Exception as e:
                logger.error(f"[{symbol}] Monitor error: {e}")
                time.sleep(5)

    def start(self):
        if self.is_running:
            return False

        self.is_running = True
        self.status["running"] = True

        logger.info("=" * 60)
        logger.info("FINAL PAIR STRATEGY - Optimized Rules + ML")
        logger.info("Strategy: Pair-Specific Optimized Rules + ML Predictions")
        logger.info("Risk: 1% TP | 2% SL | 5x Leverage | 75%+ Win Rate Target")
        logger.info("=" * 60)
        
        logger.info("✓ Using pre-trained pair-specific models from pair_models/")
        
        for symbol in self.symbols:
            try:
                self.client.symbol = symbol
                self.client.set_leverage()
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"Could not set leverage for {symbol}: {e}")

        for symbol in self.symbols:
            thread = threading.Thread(
                target=self.monitor_symbol, args=(symbol,)
            )
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            time.sleep(2)

        balance = self.client.get_account_balance()
        telegram.send_bot_started(
            symbol=f"{len(self.symbols)} pairs (Final Pair Strategy + ML)",
            risk_level=Config.RISK_LEVEL,
            leverage=self.risk_config["leverage"],
            balance=balance["total"] if balance else 0,
        )

        logger.info(
            f"✓ Multi-symbol bot started with {len(self.symbols)} pairs"
        )
        logger.info(
            "Strategy: Final Pair Strategy (Optimized Rules + ML) | "
            "1% TP | 2% SL | 5x Leverage | 75%+ Win Rate Target"
        )
        logger.info(f"Monitoring: {', '.join(self.symbols)}")
        return True

    def stop(self):
        if not self.is_running:
            return False

        self.is_running = False
        self.status["running"] = False

        for thread in self.threads:
            thread.join(timeout=5)

        telegram.send_bot_stopped(
            total_pnl=0,
            total_trades=self.status["total_trades"],
            winning_trades=0,
            losing_trades=0,
        )

        logger.info("Multi-symbol bot stopped")
        return True

    def get_status(self):
        try:
            positions = self.client.get_open_positions()
            self.status["open_positions"] = len(positions)
        except Exception:
            pass

        return self.status


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Strategy Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedStrategyWrapper:
    """Wrapper for the 5-strategy confluence engine"""
    
    def __init__(self, risk_config, symbol, model_path=None):
        self.risk_config = risk_config
        self.symbol = symbol
        self.model_path = model_path
        self.leverage = min(risk_config.get('leverage', 10), 10)  # Max 10x
        self.max_risk_percent = 1.5  # 1-2% risk per trade
    
    def generate_signal(self, klines_15m, klines_1h, klines_4h):
        """Generate signal using advanced 5-strategy confluence engine"""
        from advanced_strategies import generate_signal as adv_generate_signal
        
        result = adv_generate_signal(klines_15m, klines_1h, klines_4h)
        
        # Convert SignalType to string
        signal_str = None
        if result.signal == SignalType.LONG:
            signal_str = "LONG"
        elif result.signal == SignalType.SHORT:
            signal_str = "SHORT"
        
        # Format response to match expected structure
        return {
            "signal": signal_str,
            "confidence": result.confidence,
            "entry_price": result.entry_price,
            "stop_loss": result.stop_loss,
            "take_profit": result.take_profit,
            "risk_reward": result.risk_reward,
            "confluence_count": result.confluence_count,
            "strategies": list(result.strategy_signals.keys()),
            "metadata": result.metadata
        }
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss):
        """Calculate position size with max 10x leverage and 1-2% risk"""
        risk_amount = account_balance * (self.max_risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            price_risk = entry_price * 0.01  # Default 1%
        
        # Position size with leverage
        position_size = (risk_amount / price_risk) * entry_price * self.leverage
        
        return position_size


# ─────────────────────────────────────────────────────────────────────────────
# Global singleton helpers
# ─────────────────────────────────────────────────────────────────────────────

multi_bot = None


def set_multi_bot(bot_instance):
    global multi_bot
    multi_bot = bot_instance
    return multi_bot


def get_multi_bot(symbols=None):
    global multi_bot
    if multi_bot is None:
        multi_bot = MultiSymbolBot(symbols=symbols)
    return multi_bot