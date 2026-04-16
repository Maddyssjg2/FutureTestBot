import requests
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.enabled = Config.TELEGRAM_ENABLED
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if self.enabled:
            logger.info("Telegram notifications enabled")
        else:
            logger.info("Telegram notifications disabled (missing config)")
    
    def send_message(self, message, parse_mode='HTML'):
        """Send a message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Telegram message sent: {message[:50]}...")
                return True
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_entry(self, symbol, side, entry_price, quantity, stop_loss, take_profit_1, take_profit_2, leverage, confidence):
        """Notify on trade entry with margin and expected profit breakdown"""
        emoji = "🟢" if side == "LONG" else "🔴"
        position_value = quantity * entry_price
        margin_used = position_value / leverage

        # Calculate expected returns with leverage
        tp1_pct = abs((take_profit_1 - entry_price) / entry_price * 100)
        tp2_pct = abs((take_profit_2 - entry_price) / entry_price * 100)
        sl_pct = abs((stop_loss - entry_price) / entry_price * 100)

        tp1_profit = margin_used * (tp1_pct * leverage / 100)
        tp2_profit = margin_used * (tp2_pct * leverage / 100)
        sl_loss = margin_used * (sl_pct * leverage / 100)

        message = f"""
{emoji} <b>NEW TRADE ENTERED</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> ${entry_price:,.2f}
<b>Position Size:</b> {quantity:.4f} ({position_value:,.2f} USDT)
<b>Margin Used:</b> {margin_used:.2f} USDT
<b>Leverage:</b> {leverage}x
<b>Confidence:</b> {confidence:.1f}%

<b>Stop Loss:</b> ${stop_loss:,.2f} (Max Loss: {sl_loss:.2f} USDT)
<b>Take Profit 1:</b> ${take_profit_1:,.2f} (Profit: {tp1_profit:.2f} USDT)
<b>Take Profit 2:</b> ${take_profit_2:,.2f} (Profit: {tp2_profit:.2f} USDT)

📊 Risk/Reward: 1:{tp1_pct/sl_pct:.1f} to 1:{tp2_pct/sl_pct:.1f}
        """
        return self.send_message(message)
    
    def send_trade_exit(self, symbol, side, entry_price, exit_price, quantity, pnl, pnl_percent, exit_reason, leverage=10, margin_used=None):
        """Notify on trade exit with full margin and leverage breakdown"""
        emoji = "✅" if pnl >= 0 else "❌"
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"

        position_value = quantity * entry_price
        margin = margin_used if margin_used else position_value / leverage
        raw_pct = abs((exit_price - entry_price) / entry_price * 100)
        leveraged_pct = raw_pct * leverage
        actual_pnl = margin * (leveraged_pct / 100) * (1 if pnl >= 0 else -1)

        message = f"""
{emoji} <b>TRADE CLOSED</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>Position Size:</b> {quantity:.4f} ({position_value:,.2f} USDT)

<b>Trade Breakdown:</b>
• Margin Used: {margin:.2f} USDT
• Leverage: {leverage}x
• Price Move: {raw_pct:.2f}%
• Leveraged Return: {leveraged_pct:.2f}%

<b>Exit Reason:</b> {exit_reason}

{pnl_emoji} <b>Actual P&L: {actual_pnl:+.2f} USDT</b>
        """
        return self.send_message(message)
    
    def send_stop_loss_hit(self, symbol, side, entry_price, exit_price, quantity, pnl, leverage=10, margin_used=None):
        """Notify when stop loss is hit with full loss breakdown"""
        # Calculate values
        position_value = quantity * entry_price
        margin = margin_used if margin_used else position_value / leverage
        
        # Use actual PnL from Binance (more reliable than calculating from exit price)
        actual_loss = abs(pnl) if pnl < 0 else 0
        
        # Calculate price move from actual loss
        leveraged_loss_pct = (actual_loss / margin * 100) if margin > 0 else 0
        raw_loss_pct = leveraged_loss_pct / leverage if leverage > 0 else 0

        message = f"""
🛑 <b>STOP LOSS HIT</b> 🛑

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>Size:</b> {quantity:.4f} ({position_value:.2f} USDT)

📊 <b>Trade Breakdown:</b>
• Margin Invested: {margin:.2f} USDT
• Leverage: {leverage}x
• Est. Price Move: -{raw_loss_pct:.2f}%
• Leveraged Loss: -{leveraged_loss_pct:.2f}%

🔴 <b>Actual Loss: -{actual_loss:.2f} USDT</b>

Risk managed. Onto the next one! 💪
        """
        return self.send_message(message)
    
    def send_take_profit_hit(self, symbol, side, entry_price, exit_price, quantity, pnl, tp_level, leverage=10, margin_used=None):
        """Notify when take profit is hit with full profit breakdown"""
        # Calculate values
        position_value = quantity * entry_price
        margin = margin_used if margin_used else position_value / leverage
        
        # Use actual PnL from Binance (more reliable than calculating from exit price)
        actual_profit = pnl if pnl > 0 else 0
        
        # Calculate price move from actual profit
        leveraged_pnl_pct = (actual_profit / margin * 100) if margin > 0 else 0
        raw_pnl_pct = leveraged_pnl_pct / leverage if leverage > 0 else 0

        message = f"""
🎯 <b>TAKE PROFIT {tp_level} HIT!</b> 🎯

<b>Symbol:</b> {symbol}
<b>Side:</b> {side}
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>Size:</b> {quantity:.4f} ({position_value:.2f} USDT)

📊 <b>Trade Breakdown:</b>
• Margin Invested: {margin:.2f} USDT
• Leverage: {leverage}x
• Est. Price Move: {raw_pnl_pct:.2f}%
• Leveraged Return: {leveraged_pnl_pct:.2f}%

🟢 <b>Actual Profit: +{actual_profit:.2f} USDT</b>

Letting the rest run! 🚀
        """
        return self.send_message(message)
    
    def send_signal_detected(self, symbol, signal, confidence, entry_price, stop_loss, take_profit_1, take_profit_2, rsi, volume_ratio):
        """Notify when a signal is detected"""
        emoji = "🟢" if signal == "LONG" else "🔴"
        message = f"""
{emoji} <b>SIGNAL DETECTED</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal}
<b>Confidence:</b> {confidence:.1f}%

<b>Entry:</b> ${entry_price:,.2f}
<b>Stop Loss:</b> ${stop_loss:,.2f}
<b>TP1:</b> ${take_profit_1:,.2f}
<b>TP2:</b> ${take_profit_2:,.2f}

📊 RSI: {rsi}
📈 Volume: {volume_ratio:.2f}x avg

<i>Executing trade if conditions met...</i>
        """
        return self.send_message(message)
    
    def send_bot_started(self, symbol, risk_level, leverage, balance):
        """Notify when bot starts"""
        message = f"""
🤖 <b>TALON SNIPER BOT STARTED</b> 🤖

<b>Symbol:</b> {symbol}
<b>Risk Level:</b> {risk_level.upper()}
<b>Leverage:</b> {leverage}x
<b>Balance:</b> ${balance:,.2f} USDT

📊 <b>Strategy:</b> Talon Sniper v1 (Heikin Ashi)
• Signal 1: TEMA/DEMA Crossover + Adaptive Bounds
• Signal 2: ATR-Based Trend Following
• Trend Filter: EMA 13 Direction

🎯 <b>Take Profits:</b> Split (TP1: 2% effective, TP2: 4% effective)
🛑 <b>Stop Loss:</b> -2% effective (SL set on exchange)

🎓 <b>Adaptive Learning:</b> Active - Bot learns from every loss!
<i>Paper Trading on Binance Testnet</i>
        """
        return self.send_message(message)
    
    def send_bot_stopped(self, total_pnl, total_trades, winning_trades, losing_trades):
        """Notify when bot stops"""
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        message = f"""
🛑 <b>TRADING BOT STOPPED</b> 🛑

📊 <b>Session Summary:</b>

<b>Total Trades:</b> {total_trades}
<b>Winning:</b> {winning_trades} | <b>Losing:</b> {losing_trades}
<b>Win Rate:</b> {win_rate:.1f}%

{pnl_emoji} <b>Total P&L:</b> {total_pnl:+.2f} USDT

See you next session! 👋
        """
        return self.send_message(message)
    
    def send_daily_summary(self, balance, open_positions, daily_pnl, total_trades_today):
        """Send daily summary"""
        positions_text = "\n".join([
            f"• {p['side']} {p['symbol']}: {p['unrealized_pnl']:+.2f} USDT"
            for p in open_positions
        ]) if open_positions else "No open positions"
        
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        
        message = f"""
📅 <b>DAILY SUMMARY</b> 📅

<b>Balance:</b> ${balance:,.2f} USDT
<b>Daily P&L:</b> {pnl_emoji} {daily_pnl:+.2f} USDT
<b>Trades Today:</b> {total_trades_today}

<b>Open Positions:</b>
{positions_text}

Keep grinding! 💪
        """
        return self.send_message(message)

# Global instance
telegram = TelegramNotifier()
