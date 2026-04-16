#!/usr/bin/env python3
"""
Run premium model predictions for futures pairs and print BUY/SELL signals.
"""

import argparse
import os
import sys

sys.path.append("backend")

from backend.binance_client import BinanceFuturesClient
from backend.config import Config
from backend.premium_model import PremiumSignalModel, latest_feature_row_from_klines


def main():
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

    if args.symbols == "all":
        symbols = Config.TOP_20_SYMBOLS
    elif args.symbols == "top10":
        symbols = Config.TOP_20_SYMBOLS[:10]
    elif args.symbols == "top5":
        symbols = Config.TOP_20_SYMBOLS[:5]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    model = PremiumSignalModel.load(args.model_path)
    client = BinanceFuturesClient()

    rows = []
    for symbol in symbols:
        client.symbol = symbol
        klines = client.get_klines(interval="1h", limit=max(args.limit, 90))
        if not klines:
            continue
        feature_row = latest_feature_row_from_klines(klines)
        if not feature_row:
            continue
        pred = model.predict(feature_map=feature_row, symbol=symbol)
        price = client.get_mark_price()
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

    rows.sort(key=lambda x: x["expected_return"], reverse=True)

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

