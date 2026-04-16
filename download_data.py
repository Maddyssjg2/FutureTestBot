#!/usr/bin/env python3
"""
Download historical OHLCV data for all top 20 futures pairs
Usage: python download_data.py [--days 365] [--symbols all|top10]
"""

import sys
import argparse
sys.path.append('backend')

from backend.data_downloader import DataDownloader
from backend.config import Config

def main():
    parser = argparse.ArgumentParser(description='Download historical futures data')
    parser.add_argument('--days', type=int, default=365, help='Number of days to download (default: 365)')
    parser.add_argument('--symbols', type=str, default='all', 
                       help='Symbols to download: all, top10, top5, or comma-separated list')
    
    args = parser.parse_args()
    
    # Determine symbols
    if args.symbols == 'all':
        symbols = Config.TOP_20_SYMBOLS
    elif args.symbols == 'top10':
        symbols = Config.TOP_20_SYMBOLS[:10]
    elif args.symbols == 'top5':
        symbols = Config.TOP_20_SYMBOLS[:5]
    else:
        symbols = args.symbols.split(',')
    
    print(f"\n{'='*60}")
    print(f"Historical Data Downloader")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)} pairs")
    print(f"Days: {args.days}")
    print(f"Interval: 1h")
    print(f"{'='*60}\n")
    
    downloader = DataDownloader()
    results = downloader.download_all_symbols(symbols=symbols, days_back=args.days)
    
    print(f"\n{'='*60}")
    print("Download Complete!")
    print(f"{'='*60}")
    print(f"Data location: {downloader.data_dir}")
    
    # Show summary
    total = sum(results.values())
    successful = sum(1 for v in results.values() if v > 0)
    print(f"Successful: {successful}/{len(symbols)}")
    print(f"Total candles: {total:,}")

if __name__ == '__main__':
    main()
