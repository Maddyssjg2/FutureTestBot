from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
import logging
import time
from decimal import Decimal, ROUND_DOWN
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceFuturesClient:
    def __init__(self):
        self.testnet = Config.USE_TESTNET
        self.client = Client(
            api_key=Config.BINANCE_API_KEY,
            api_secret=Config.BINANCE_SECRET_KEY,
            testnet=self.testnet
        )
        
        # Track active SL/TP orders for trailing stop management
        self.active_sl_order_id = None
        self.active_tp1_order_id = None
        self.entry_price = None  # Store entry price for trailing stop calculations
        
        # Increase recv_window to handle clock skew
        self.client.RECV_WINDOW = 60000  # 60 seconds
        
        if self.testnet:
            self.client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'
            logger.info("Using Binance Futures TESTNET (Paper Trading)")
        
        self.symbol = Config.TRADING_SYMBOL
        self.risk_config = Config.get_risk_config()
        self._symbol_meta_cache = {}
        
        # Sync time with Binance server
        self._sync_time()

    @staticmethod
    def _coalesce_float(data, keys, default=0.0):
        """Return first parseable float from candidate keys."""
        for key in keys:
            if key in data and data[key] not in (None, ''):
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    continue
        return float(default)
    
    def _sync_time(self):
        """Synchronize local time with Binance server time"""
        try:
            server_time = self.client.futures_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            
            # Set timestamp offset for this client
            self.client.timestamp_offset = time_offset
            logger.info(f"Time sync complete. Offset: {time_offset}ms")
        except Exception as e:
            logger.warning(f"Could not sync time: {e}")
        
    def get_account_balance(self):
        try:
            account = self.client.futures_account()
            assets = account.get('assets', []) if isinstance(account, dict) else []
            usdt_balance = next((b for b in assets if b.get('asset') == 'USDT'), None)

            # Fallback to account balance endpoint if futures_account schema differs.
            if not usdt_balance:
                balances = self.client.futures_account_balance()
                usdt_balance = next((b for b in balances if b.get('asset') == 'USDT'), None)

            if usdt_balance:
                total = self._coalesce_float(
                    usdt_balance,
                    ['walletBalance', 'balance', 'crossWalletBalance'],
                    default=0.0
                )
                available = self._coalesce_float(
                    usdt_balance,
                    ['availableBalance', 'maxWithdrawAmount', 'withdrawAvailable'],
                    default=total
                )
                unrealized_pnl = self._coalesce_float(
                    usdt_balance,
                    ['unrealizedProfit', 'unRealizedProfit', 'crossUnPnl', 'unrealizedPnl'],
                    default=0.0
                )
                return {
                    'total': total,
                    'available': available,
                    'unrealized_pnl': unrealized_pnl
                }
            return None
        except BinanceAPIException as e:
            logger.error(f"Error getting balance: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected balance parse error: {e}")
            return None
    
    def get_symbol_info(self):
        try:
            info = self.client.futures_exchange_info()
            for symbol in info['symbols']:
                if symbol['symbol'] == self.symbol:
                    return symbol
            return None
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            return None

    def _get_symbol_meta(self):
        """Fetch and cache precision/filter metadata for current symbol."""
        if self.symbol in self._symbol_meta_cache:
            return self._symbol_meta_cache[self.symbol]

        info = self.get_symbol_info()
        if not info:
            return None

        lot_filter = next((f for f in info.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        price_filter = next((f for f in info.get('filters', []) if f.get('filterType') == 'PRICE_FILTER'), None)

        meta = {
            'quantity_precision': int(info.get('quantityPrecision', 3)),
            'price_precision': int(info.get('pricePrecision', 2)),
            'step_size': Decimal(lot_filter['stepSize']) if lot_filter else None,
            'min_qty': Decimal(lot_filter['minQty']) if lot_filter else None,
            'max_qty': Decimal(lot_filter['maxQty']) if lot_filter else None,
            'tick_size': Decimal(price_filter['tickSize']) if price_filter else None,
            'min_price': Decimal(price_filter['minPrice']) if price_filter else None,
            'max_price': Decimal(price_filter['maxPrice']) if price_filter else None,
        }
        self._symbol_meta_cache[self.symbol] = meta
        return meta

    def _normalize_quantity(self, quantity):
        """Normalize quantity to Binance lot-size precision."""
        meta = self._get_symbol_meta()
        if not meta:
            return float(quantity)

        qty = Decimal(str(abs(float(quantity))))
        step_size = meta.get('step_size')
        min_qty = meta.get('min_qty')
        max_qty = meta.get('max_qty')
        qty_precision = meta.get('quantity_precision', 3)

        if step_size and step_size > 0:
            qty = (qty / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size

        if min_qty and qty < min_qty:
            qty = min_qty
        if max_qty and qty > max_qty:
            qty = max_qty

        quant = Decimal(f"1e-{qty_precision}")
        qty = qty.quantize(quant, rounding=ROUND_DOWN)
        return float(qty)

    def _normalize_price(self, price):
        """Normalize stop/limit price to Binance tick-size precision."""
        meta = self._get_symbol_meta()
        if not meta:
            return float(price)

        px = Decimal(str(float(price)))
        tick_size = meta.get('tick_size')
        min_price = meta.get('min_price')
        max_price = meta.get('max_price')
        price_precision = meta.get('price_precision', 2)

        if tick_size and tick_size > 0:
            px = (px / tick_size).to_integral_value(rounding=ROUND_DOWN) * tick_size

        if min_price and px < min_price:
            px = min_price
        if max_price and px > max_price:
            px = max_price

        quant = Decimal(f"1e-{price_precision}")
        px = px.quantize(quant, rounding=ROUND_DOWN)
        return float(px)
    
    def set_leverage(self, leverage=None):
        try:
            lev = leverage or self.risk_config['leverage']
            response = self.client.futures_change_leverage(
                symbol=self.symbol,
                leverage=lev
            )
            logger.info(f"Leverage set to {lev}x for {self.symbol}")
            return response
        except BinanceAPIException as e:
            logger.error(f"Error setting leverage: {e}")
            logger.error(f"Error setting leverage for symbol: {self.symbol}")
            return None
    
    def get_mark_price(self):
        try:
            mark_price = self.client.futures_mark_price(symbol=self.symbol)
            return float(mark_price['markPrice'])
        except BinanceAPIException as e:
            logger.error(f"Error getting mark price: {e}")
            return None
    
    def get_klines(self, interval='1h', limit=100):
        try:
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            return klines
        except BinanceAPIException as e:
            logger.error(f"Error getting klines: {e}")
            return None
    
    def get_open_positions(self):
        try:
            positions = self.client.futures_position_information()
            open_positions = []
            for pos in positions:
                amount = self._coalesce_float(pos, ['positionAmt', 'position_amt'], default=0.0)
                if amount != 0:
                    entry_price = self._coalesce_float(pos, ['entryPrice', 'entry_price'], default=0.0)
                    mark_price = self._coalesce_float(pos, ['markPrice', 'mark_price'], default=entry_price)
                    leverage = int(self._coalesce_float(pos, ['leverage'], default=self.risk_config['leverage']))
                    unrealized_pnl = self._coalesce_float(
                        pos,
                        ['unrealizedProfit', 'unRealizedProfit', 'unrealizedPnl'],
                        default=(mark_price - entry_price) * amount if entry_price and mark_price else 0.0
                    )
                    open_positions.append({
                        'symbol': pos.get('symbol', self.symbol),
                        'amount': amount,
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,
                        'leverage': leverage,
                        'side': 'LONG' if amount > 0 else 'SHORT'
                    })
            return open_positions
        except BinanceAPIException as e:
            logger.error(f"Error getting positions: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected positions parse error: {e}")
            return []
    
    def place_order(self, side: str, order_type: str, quantity: float,
                   stop_loss=None, take_profit=None, take_profit_1=None, take_profit_2=None):
        try:
            normalized_qty = self._normalize_quantity(quantity)
            order_params = {
                'symbol': self.symbol,
                'side': side,
                'type': order_type,
                'quantity': normalized_qty
            }
            
            if order_type == 'LIMIT':
                order_params['timeInForce'] = 'GTC'
            
            # Step 1: Place main order
            order = self.client.futures_create_order(**order_params)
            logger.info(f"📝 Main order placed: {side} {normalized_qty} {self.symbol}, OrderID: {order.get('orderId')}")
            
            # Step 2: Wait for order to fill (poll for status)
            order_id = order.get('orderId')
            if order_id:
                filled = False
                for _ in range(10):  # Check up to 10 times
                    try:
                        order_status = self.client.futures_get_order(symbol=self.symbol, orderId=order_id)
                        status = order_status.get('status', '')
                        if status == 'FILLED':
                            filled = True
                            logger.info(f"✅ Main order FILLED! Proceeding to add TP/SL...")
                            break
                        elif status in ['CANCELED', 'EXPIRED', 'REJECTED']:
                            logger.warning(f"⚠️ Main order {status}, not adding TP/SL")
                            return order
                        time.sleep(0.5)  # Wait 500ms between checks
                    except Exception as e:
                        logger.debug(f"Error checking order status: {e}")
                        time.sleep(0.5)
                
                if not filled:
                    logger.warning(f"⚠️ Main order not filled within 5 seconds, adding TP/SL anyway...")
            
            # Step 3: Add TP/SL after main order fills (reduce-only)
            entry_price = self.get_mark_price()
            self.entry_price = entry_price  # Store for trailing stop calculations
            tp_sl_placed = []
            
            if entry_price:
                sl_side = 'SELL' if side == 'BUY' else 'BUY'
                
                # 1. STOP LOSS - reduce-only, close position
                if stop_loss:
                    try:
                        normalized_stop = self._normalize_price(stop_loss)
                        sl_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=sl_side,
                            type='STOP_MARKET',
                            stopPrice=normalized_stop,
                            closePosition=True,
                            reduceOnly=True
                        )
                        self.active_sl_order_id = sl_order.get('orderId')
                        tp_sl_placed.append(f"SL@{stop_loss}")
                        logger.info(f"🛡️ Reduce-only SL added at {normalized_stop} (orderId: {self.active_sl_order_id})")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not add reduce-only SL: {e} (will use bot-managed SL)")
                
                # 2. TAKE PROFIT 1 - reduce-only, 50% position
                if take_profit_1:
                    try:
                        normalized_tp1 = self._normalize_price(take_profit_1)
                        tp1_qty = self._normalize_quantity(quantity * 0.5)
                        tp1_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=sl_side,
                            type='TAKE_PROFIT_MARKET',
                            stopPrice=normalized_tp1,
                            quantity=tp1_qty,
                            reduceOnly=True
                        )
                        self.active_tp1_order_id = tp1_order.get('orderId')
                        tp_sl_placed.append(f"TP1@{take_profit_1}(50%)")
                        logger.info(f"🎯 Reduce-only TP1 added at {normalized_tp1} (orderId: {self.active_tp1_order_id})")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not add reduce-only TP1: {e} (will use bot-managed TP)")
                
                # 3. TAKE PROFIT 2 - reduce-only, 50% position
                if take_profit_2:
                    try:
                        normalized_tp2 = self._normalize_price(take_profit_2)
                        tp2_qty = self._normalize_quantity(quantity * 0.5)
                        tp2_order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=sl_side,
                            type='TAKE_PROFIT_MARKET',
                            stopPrice=normalized_tp2,
                            quantity=tp2_qty,
                            reduceOnly=True
                        )
                        tp_sl_placed.append(f"TP2@{take_profit_2}(50%)")
                        logger.info(f"🎯 Reduce-only TP2 added at {normalized_tp2}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not add reduce-only TP2: {e} (will use bot-managed TP)")
                
                if tp_sl_placed:
                    logger.info(f"✅ Reduce-only TP/SL added after fill: {', '.join(tp_sl_placed)}")
                else:
                    logger.info("⚠️ Reduce-only TP/SL failed, using bot-managed TP/SL (1-sec polling)")
            
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def close_position(self, position):
        try:
            side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            position_amount = abs(float(position['amount']))
            quantity = self._normalize_quantity(position_amount)
            
            # Get symbol meta to check max quantity
            meta = self._get_symbol_meta()
            max_qty = meta.get('max_qty') if meta else None
            
            logger.info(f"🔒 Closing {self.symbol}: Side={side}, PositionSize={position_amount}, "
                       f"NormalizedQty={quantity}, MaxQty={max_qty}")
            
            # If normalized quantity exceeds max_qty, cap it
            if max_qty and quantity > float(max_qty):
                logger.warning(f"⚠️ Quantity {quantity} exceeds max_qty {max_qty}, capping...")
                quantity = float(max_qty)
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            # Check if order filled completely
            executed_qty = float(order.get('executedQty', 0))
            order_status = order.get('status', 'UNKNOWN')
            
            logger.info(f"🔓 Close order: ID={order.get('orderId')}, Status={order_status}, "
                       f"Requested={quantity}, Executed={executed_qty}")
            
            if executed_qty < quantity * 0.99:  # Less than 99% filled
                logger.warning(f"⚠️ {self.symbol}: Close order PARTIAL FILL! "
                              f"Requested {quantity}, got {executed_qty}. Position NOT fully closed!")
            else:
                logger.info(f"✅ {self.symbol}: Position fully closed. {side} {executed_qty}")
            
            return order
        except BinanceAPIException as e:
            logger.error(f"Error closing position: {e}")
            # If max quantity error, try closing with smaller chunks
            if 'Quantity greater than max quantity' in str(e):
                logger.warning(f"⚠️ Max quantity error, trying to close with max_qty...")
                if meta and max_qty:
                    try:
                        order = self.client.futures_create_order(
                            symbol=self.symbol,
                            side=side,
                            type='MARKET',
                            quantity=float(max_qty)
                        )
                        logger.info(f"✅ {self.symbol}: Position closed with max_qty {max_qty}")
                        return order
                    except Exception as e2:
                        logger.error(f"⚠️ Still failed with max_qty: {e2}")
            return None
    
    def get_order_history(self, limit=50):
        try:
            orders = self.client.futures_get_all_orders(
                symbol=self.symbol,
                limit=limit
            )
            return orders
        except BinanceAPIException as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def get_income_history(self, limit=50, symbol=None):
        try:
            params = {'limit': limit}
            if symbol:
                params['symbol'] = symbol
            income = self.client.futures_income_history(**params)
            return income
        except BinanceAPIException as e:
            logger.error(f"Error getting income history: {e}")
            return []
    
    def check_tp1_filled(self):
        """Check if TP1 order has been filled (executed)"""
        if not self.active_tp1_order_id:
            return False
        
        try:
            order = self.client.futures_get_order(
                symbol=self.symbol,
                orderId=self.active_tp1_order_id
            )
            status = order.get('status', '')
            return status == 'FILLED'
        except Exception as e:
            logger.debug(f"Could not check TP1 status: {e}")
            return False
    
    def cancel_stop_loss(self):
        """Cancel the active stop loss order"""
        if not self.active_sl_order_id:
            return False
        
        try:
            self.client.futures_cancel_order(
                symbol=self.symbol,
                orderId=self.active_sl_order_id
            )
            logger.info(f"🗑️ Cancelled old SL order (ID: {self.active_sl_order_id})")
            self.active_sl_order_id = None
            return True
        except Exception as e:
            logger.warning(f"Could not cancel SL order: {e}")
            return False
    
    def set_trailing_stop(self, position, callback_rate=1.0):
        """
        Set a trailing stop order after TP1 hits
        callback_rate: Percentage below highest price to trail (1% = 0.01)
        """
        try:
            side = position['side']
            current_price = position.get('mark_price', self.get_mark_price())
            
            # Calculate trailing stop price
            if side == 'LONG':
                # For long, stop trails below highest price
                activation_price = current_price * 1.01  # Activate 1% above current
                trailing_price = current_price * (1 - callback_rate / 100)
                stop_side = 'SELL'
            else:
                # For short, stop trails above lowest price
                activation_price = current_price * 0.99  # Activate 1% below current
                trailing_price = current_price * (1 + callback_rate / 100)
                stop_side = 'BUY'
            
            # Get remaining position quantity (50% after TP1)
            remaining_qty = self._normalize_quantity(abs(position['amount']) * 0.5)
            
            # Place trailing stop order
            # Note: TRAILING_STOP_MARKET may not be available on testnet
            # Fall back to regular STOP_MARKET if needed
            try:
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side=stop_side,
                    type='TRAILING_STOP_MARKET',
                    callbackRate=str(callback_rate),  # 1% callback
                    quantity=remaining_qty
                )
                logger.info(f"📈 Trailing stop set: {callback_rate}% callback, qty={remaining_qty}")
                self.active_sl_order_id = order.get('orderId')
                return True
            except BinanceAPIException as e:
                if 'TRAILING_STOP' in str(e):
                    # Fallback to regular stop at current price
                    logger.info("TRAILING_STOP not supported, using regular STOP instead")
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side=stop_side,
                        type='STOP_MARKET',
                        stopPrice=self._normalize_price(trailing_price),
                        quantity=remaining_qty
                    )
                    logger.info(f"📈 Regular stop set at {trailing_price:.4f} (1% below current)")
                    self.active_sl_order_id = order.get('orderId')
                    return True
                else:
                    raise e
                    
        except Exception as e:
            logger.error(f"Could not set trailing stop: {e}")
            return False
