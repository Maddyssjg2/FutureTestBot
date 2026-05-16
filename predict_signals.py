#!/usr/bin/env python3
"""
Run premium model predictions for futures pairs and print BUY/SELL signals.

This script fetches the latest market data (klines) from Binance, processes them into 
features using the PremiumSignalModel's utility functions, and generates trading 
signals based on a pre-trained model.
"""

import argparse
import os
import sys

# Ensure the backend directory is in the python path for imports
sys.path.append("backend")

from backend.binance_client import BinanceFuturesClient
from backend.config import Config
from backend.premium_model import PremiumSignalModel, latest_feature_row_from_klines


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Predict LONG/SHORT signals with premium model")
    parser.add_argument("--symbols", type=str, default="all", help="all, top10, top5, or CSV list")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("backend", "models", "premium_signal_model.json"),
        help="Path to trained model JSON",
    )
    parser.add_argument("--limit", type=int, default=120, help="Kline lookback candles (default: 120)")
    parser.add_argument("--top", type=int, default=10, help="Print top N signals by expected return")
    args = parser.parse_args()

    # Determine which symbols to analyze based on the --symbols argument
    if args.symbols == "all":
        symbols = Config.TOP_20_SYMBOLS
    elif args.symbols == "top10":
        symbols = Config.TOP_20_SYMBOLS[:10]
    elif args.symbols == "top5":
        symbols = Config.TOP_20_SYMBOLS[:5]
    else:
        # Allow a comma-separated list of symbols (e.g., "BTCUSDT,ETHUSDT")
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Verify that the model file exists before attempting to load it
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Load the pre-trained premium signal model and initialize the Binance client
    model = PremiumSignalModel.load(args.model_path)
    client = BinanceFuturesClient()

    rows = []
    for symbol in symbols:
        client.symbol = symbol
        # Fetch historical kline data; ensure we have at least 90 candles for feature calculation
        klines = client.get_klines(interval="1h", limit=max(args.limit, 90))
        if not klines:
            continue
        
        # Convert raw kline data into a single row of features for the model
        feature_row = latest_feature_row_from_klines(klines)
        if not feature_row:
            continue
        
        # Generate prediction using the loaded model
        pred = model.predict(feature_map=feature_row, symbol=symbol)
        price = client.get_mark_price()
        
        # Only keep signals that meet the model's internal confidence/threshold criteria
        if pred["signal"] is None:
            continue
            
        rows.append(
            {
                "symbol": symbol,
                "action": "BUY" if pred["signal"] == "LONG" else "SELL",
                "signal": pred["signal"],
                "confidence": pred["confidence"],
                "expected_return": pred["expected_return"],
                "edge": pred["edge"],
                "price": price,
            }
        )

    # Sort the resulting signals by expected return in descending order to highlight best opportunities
    rows.sort(key=lambda x: x["expected_return"], reverse=True)

    # Print a formatted table of the top N signals
    print("=" * 78)
    print(f"Premium Model Signals ({len(rows)} signals)")
    print("=" * 78)
    print("Symbol    Action  Signal  Confidence  ExpectedRet  Edge      Price")
    for row in rows[: args.top]:
        print(
            f"{row['symbol']:<9} {row['action']:<6} {row['signal']:<6} "
            f"{row['confidence']:>9.1f}%  {row['expected_return']*100:>9.3f}%  "
            f"{row['edge']*100:>6.3f}%  {row['price'] if row['price'] else 0:>10.4f}"
        )

    if not rows:
        print("No qualified signals at current thresholds.")


if __name__ == "__main__":
    main()