"""
Historical Data Downloader for Binance Futures
Downloads 1 year of OHLCV data for top 20 trading pairs
"""

import os
import json
import logging
from datetime import datetime, timedelta
from binance_client import BinanceFuturesClient
from config import Config
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self):
        self.client = BinanceFuturesClient()
        self.data_dir = os.path.join(os.path.dirname(__file__), 'historical_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_symbol_data(self, symbol, interval='1h', days_back=365):
        """Download historical klines for a symbol"""
        logger.info(f"Downloading {days_back} days of {interval} data for {symbol}...")
        
        all_klines = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Binance limits to 1000 candles per request
        # For 1h candles: 1000 hours = ~41 days per request
        limit = 1000
        
        days_downloaded = 0
        while days_downloaded < days_back:
            try:
                klines = self.client.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    endTime=end_time
                )
                
                if not klines or len(klines) == 0:
                    break
                
                all_klines = klines + all_klines
                
                # Update end_time for next batch (go further back)
                end_time = klines[0][0] - 1
                
                days_downloaded += len(klines) / 24  # 24 hours per day
                
                logger.info(f"  Downloaded {len(klines)} candles for {symbol} (total: {len(all_klines)})")
                
                # Rate limit protection
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
                time.sleep(2)
                continue
        
        # Save to file
        if all_klines:
            filename = f"{symbol}_{interval}_{days_back}d.json"
            filepath = os.path.join(self.data_dir, filename)
            
            # Convert to more readable format
            formatted_data = []
            for k in all_klines:
                formatted_data.append({
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                    'taker_buy_base': float(k[9]),
                    'taker_buy_quote': float(k[10])
                })
            
            with open(filepath, 'w') as f:
                json.dump(formatted_data, f)
            
            logger.info(f"✓ Saved {len(formatted_data)} candles to {filepath}")
            return len(formatted_data)
        
        return 0
    
    def download_all_symbols(self, symbols=None, interval='1h', days_back=365):
        """Download data for all symbols"""
        if symbols is None:
            symbols = Config.TOP_20_SYMBOLS
        
        logger.info(f"Starting download for {len(symbols)} symbols...")
        
        results = {}
        for symbol in symbols:
            try:
                count = self.download_symbol_data(symbol, interval, days_back)
                results[symbol] = count
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                results[symbol] = 0
            
            time.sleep(1)  # Rate limit between symbols
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*50)
        total_candles = 0
        for symbol, count in results.items():
            status = "✓" if count > 0 else "✗"
            logger.info(f"{status} {symbol}: {count:,} candles")
            total_candles += count
        
        logger.info(f"\nTotal: {total_candles:,} candles downloaded")
        logger.info(f"Data saved to: {self.data_dir}")
        
        return results
    
    def load_historical_data(self, symbol, interval='1h', days_back=365):
        """Load previously downloaded data for a symbol/interval."""
        filepath = os.path.join(self.data_dir, f"{symbol}_{interval}_{days_back}d.json")

        if not os.path.exists(filepath):
            # Fallback: load the largest available file for this symbol+interval
            prefix = f"{symbol}_{interval}_"
            candidates = [f for f in os.listdir(self.data_dir)] if os.path.exists(self.data_dir) else []
            candidates = [f for f in candidates if f.startswith(prefix) and f.endswith('d.json')]
            if not candidates:
                logger.warning(f"No historical data found for {symbol}")
                return None

            def extract_days(name):
                try:
                    return int(name.replace('.json', '').split('_')[-1].replace('d', ''))
                except Exception:
                    return 0

            candidates.sort(key=extract_days, reverse=True)
            filepath = os.path.join(self.data_dir, candidates[0])

        with open(filepath, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} historical candles for {symbol} from {os.path.basename(filepath)}")
        return data
    
    def get_data_summary(self):
        """Get summary of downloaded data"""
        if not os.path.exists(self.data_dir):
            return {}
        
        summary = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                parts = filename.replace('.json', '').split('_')
                symbol = parts[0]
                interval = parts[1] if len(parts) > 1 else 'unknown'
                
                summary[symbol] = {
                    'candles': len(data),
                    'interval': interval,
                    'date_range': {
                        'start': datetime.fromtimestamp(data[0]['timestamp']/1000).isoformat() if data else None,
                        'end': datetime.fromtimestamp(data[-1]['timestamp']/1000).isoformat() if data else None
                    }
                }
        
        return summary

if __name__ == '__main__':
    downloader = DataDownloader()
    
    # Download all top 20 symbols
    print("Starting 1-year historical data download...")
    print("This may take 5-10 minutes due to API rate limits.\n")
    
    results = downloader.download_all_symbols()
    
    print("\n✓ Download complete!")
    print(f"Data location: {downloader.data_dir}")
