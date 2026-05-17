from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from multi_symbol_bot import MultiSymbolBot, get_multi_bot, set_multi_bot
from binance_client import BinanceFuturesClient
from data_downloader import DataDownloader
from performance_analyzer import TradePerformanceAnalyzer
from config import Config
import threading
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Background thread for emitting updates
thread = None
data_fetcher = None


def resolve_multi_symbols(symbol_count=None, explicit_symbols=None):
    """Resolve symbols for multi-bot while always honoring forced symbols (e.g., BTCUSDT)."""
    if explicit_symbols:
        requested = [s.strip().upper() for s in explicit_symbols if s and str(s).strip()]
        requested = [s for s in requested if s in Config.TOP_20_SYMBOLS]
        if not requested:
            requested = Config.get_default_multi_symbols(symbol_count=symbol_count)
    else:
        requested = Config.get_default_multi_symbols(symbol_count=symbol_count)

    resolved = []
    for forced_symbol in Config.FORCE_INCLUDE_SYMBOLS:
        if forced_symbol in Config.TOP_20_SYMBOLS and forced_symbol not in resolved:
            resolved.append(forced_symbol)
    for symbol in requested:
        if symbol not in resolved:
            resolved.append(symbol)
    return resolved

class DataFetcher:
    def __init__(self):
        self.client = BinanceFuturesClient()
        self.is_running = True
        self.last_data = {}
        self.chart_symbol = (
            Config.FORCE_INCLUDE_SYMBOLS[0]
            if Config.FORCE_INCLUDE_SYMBOLS else Config.TRADING_SYMBOL
        )
    
    def fetch_loop(self):
        while self.is_running:
            try:
                bot_instance = get_multi_bot()
                multi_status = bot_instance.get_status()

                balance = self.client.get_account_balance()
                positions = self.client.get_open_positions()
                self.client.symbol = self.chart_symbol
                price = self.client.get_mark_price()
                klines = self.client.get_klines(interval='1h', limit=50)
                chart_data = []
                if klines:
                    for k in klines:
                        chart_data.append({
                            'time': k[0],
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4]),
                            'volume': float(k[5])
                        })
                orders = self.client.get_order_history(limit=20)
                data = {
                    'balance': balance,
                    'positions': positions,
                    'current_price': price,
                    'chart_data': chart_data,
                    'orders': orders,
                    'bot_status': multi_status,
                    'multi_status': multi_status
                }
                self.last_data = data
                socketio.emit('market_update', data)
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error in data fetcher: {e}")
                time.sleep(5)
    
    def stop(self):
        self.is_running = False

def start_data_fetcher():
    global data_fetcher
    data_fetcher = DataFetcher()
    thread = threading.Thread(target=data_fetcher.fetch_loop)
    thread.daemon = True
    thread.start()

@app.route('/api/status', methods=['GET'])
def get_status():
    bot_instance = get_multi_bot()
    status = bot_instance.get_status()
    return jsonify({
        'bot_running': status.get('running', False),
        'status': status,
        'config': {
            'symbol': Config.FORCE_INCLUDE_SYMBOLS[0] if Config.FORCE_INCLUDE_SYMBOLS else Config.TRADING_SYMBOL,
            'risk_level': Config.RISK_LEVEL,
            'strategy_mode': Config.STRATEGY_MODE,
            'multi_only': True,
            'default_multi_symbol_count': Config.DEFAULT_MULTI_SYMBOL_COUNT,
            'default_symbols': Config.get_default_multi_symbols(),
        }
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    data = request.json or {}
    symbol_count = data.get('symbol_count', Config.DEFAULT_MULTI_SYMBOL_COUNT)
    explicit_symbols = data.get('symbols')
    symbols = resolve_multi_symbols(symbol_count=symbol_count, explicit_symbols=explicit_symbols)

    existing = get_multi_bot()
    if existing.is_running:
        existing.stop()

    bot_instance = MultiSymbolBot(symbols=symbols)
    set_multi_bot(bot_instance)
    success = bot_instance.start()
    return jsonify({
        'success': success,
        'message': f'Multi-symbol bot started with {len(symbols)} pairs' if success else 'Failed to start multi-symbol bot',
        'symbols': symbols
    })

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    bot_instance = get_multi_bot()
    success = bot_instance.stop()
    return jsonify({'success': success, 'message': 'Multi-symbol bot stopped' if success else 'Failed to stop multi-symbol bot'})

@app.route('/api/balance', methods=['GET'])
def get_balance():
    client = BinanceFuturesClient()
    balance = client.get_account_balance()
    return jsonify(balance)

@app.route('/api/positions', methods=['GET'])
def get_positions():
    client = BinanceFuturesClient()
    positions = client.get_open_positions()
    return jsonify(positions)

@app.route('/api/orders', methods=['GET'])
def get_orders():
    client = BinanceFuturesClient()
    orders = client.get_order_history()
    return jsonify(orders)

@app.route('/api/trade', methods=['POST'])
def manual_trade():
    return jsonify({
        'success': False,
        'error': 'Manual single-symbol trade is disabled in multi-only mode'
    }), 400

@app.route('/api/close-all', methods=['POST'])
def close_all():
    bot_instance = get_multi_bot()
    count = bot_instance.close_all_positions() if bot_instance else 0
    return jsonify({'success': True, 'closed_positions': count})

@app.route('/api/config', methods=['GET'])
def get_config():
    bot_instance = get_multi_bot()
    return jsonify({
        'symbol': Config.FORCE_INCLUDE_SYMBOLS[0] if Config.FORCE_INCLUDE_SYMBOLS else Config.TRADING_SYMBOL,
        'risk_level': Config.RISK_LEVEL,
        'strategy_mode': Config.STRATEGY_MODE,
        'leverage': bot_instance.risk_config['leverage'],
        'trade_percentage': bot_instance.risk_config['trade_percentage'],
        'stop_loss': bot_instance.risk_config['stop_loss'],
        'take_profit': bot_instance.risk_config['take_profit'],
        'max_positions': bot_instance.risk_config['max_positions'],
        'testnet': Config.USE_TESTNET,
        'top_20_symbols': Config.TOP_20_SYMBOLS,
        'multi_only': True,
        'default_multi_symbol_count': Config.DEFAULT_MULTI_SYMBOL_COUNT,
        'force_include_symbols': Config.FORCE_INCLUDE_SYMBOLS,
        'default_symbols': Config.get_default_multi_symbols()
    })

@app.route('/api/multi/start', methods=['POST'])
def start_multi_bot():
    try:
        data = request.json or {}
        symbol_count = data.get('symbol_count', Config.DEFAULT_MULTI_SYMBOL_COUNT)
        explicit_symbols = data.get('symbols')
        symbols = resolve_multi_symbols(symbol_count=symbol_count, explicit_symbols=explicit_symbols)
        existing = get_multi_bot()
        if existing.is_running:
            existing.stop()
        bot_instance = MultiSymbolBot(symbols=symbols)
        set_multi_bot(bot_instance)
        success = bot_instance.start()
        return jsonify({'success': success, 'message': f'Multi-symbol bot started with {len(symbols)} pairs' if success else 'Failed to start', 'symbols': symbols})
    except Exception as e:
        import traceback
        logger.error(f"Error starting multi-bot: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Error: {str(e)}', 'error': str(e)}), 500

@app.route('/api/multi/stop', methods=['POST'])
def stop_multi_bot():
    try:
        bot_instance = get_multi_bot()
        success = bot_instance.stop()
        return jsonify({'success': success, 'message': 'Multi-symbol bot stopped' if success else 'Failed to stop'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/multi/status', methods=['GET'])
def get_multi_status():
    try:
        bot_instance = get_multi_bot(symbols=Config.get_default_multi_symbols())
        status = bot_instance.get_status()
        status['symbols'] = bot_instance.symbols
        return jsonify(status)
    except Exception as e:
        return jsonify({'running': False, 'error': f'Multi-bot unavailable: {str(e)}'})

@app.route('/api/data/download', methods=['POST'])
def download_historical_data():
    data = request.json or {}
    days = data.get('days', 365)
    symbols = data.get('symbols', Config.TOP_20_SYMBOLS)
    def download_task():
        downloader = DataDownloader()
        downloader.download_all_symbols(symbols=symbols, days_back=days)
    thread = threading.Thread(target=download_task)
    thread.daemon = True
    thread.start()
    return jsonify({'success': True, 'message': f'Downloading {days} days of data for {len(symbols)} symbols...', 'symbols': symbols})

@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    downloader = DataDownloader()
    summary = downloader.get_data_summary()
    return jsonify(summary)

@app.route('/api/symbols', methods=['GET'])
def get_available_symbols():
    return jsonify({'top_20': Config.TOP_20_SYMBOLS, 'count': len(Config.TOP_20_SYMBOLS)})

@app.route('/api/performance/summary', methods=['GET'])
def get_performance_summary():
    try:
        limit = int(request.args.get('limit', 500))
    except ValueError:
        limit = 500
    limit = max(50, min(limit, 2000))
    analyzer = TradePerformanceAnalyzer(symbols=Config.TOP_20_SYMBOLS, income_limit=limit)
    report = analyzer.build_report()
    analyzer.save_report(report)
    return jsonify(report)

@app.route('/api/chart', methods=['GET'])
def get_chart_data():
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        symbol = Config.FORCE_INCLUDE_SYMBOLS[0] if Config.FORCE_INCLUDE_SYMBOLS else Config.TRADING_SYMBOL
    if symbol not in Config.TOP_20_SYMBOLS:
        return jsonify({'success': False, 'error': 'Unsupported symbol'}), 400
    try:
        limit = int(request.args.get('limit', 120))
    except ValueError:
        limit = 120
    limit = max(20, min(limit, 500))
    client = BinanceFuturesClient()
    client.symbol = symbol
    klines = client.get_klines(interval='1h', limit=limit) or []
    chart_data = []
    for k in klines:
        chart_data.append({'time': k[0], 'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])})
    return jsonify({'success': True, 'symbol': symbol, 'chart_data': chart_data})

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    if data_fetcher:
        emit('market_update', data_fetcher.last_data)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    bot_instance = get_multi_bot(symbols=Config.get_default_multi_symbols())
    if not bot_instance.is_running:
        try:
            success = bot_instance.start()
            if success:
                logger.info("✓ Multi-symbol bot auto-started with server")
        except Exception as e:
            logger.error(f"Auto-start failed: {e}")
    start_data_fetcher()
    socketio.run(app, host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.DEBUG, use_reloader=False)
