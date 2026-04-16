#!/usr/bin/env python3
"""
Train premium multi-pair signal model from historical candle data.
"""

import argparse
import os
import sys
from typing import Dict, List

sys.path.append("backend")

from backend.config import Config
from backend.data_downloader import DataDownloader
from backend.premium_model import PremiumSignalModelTrainer, build_symbol_training_frame


def _load_symbol_data(
    downloader: DataDownloader, symbols: List[str], interval: str, auto_download: bool, days: int
) -> Dict[str, list]:
    data_map: Dict[str, list] = {}
    missing: List[str] = []
    for symbol in symbols:
        data = downloader.load_historical_data(symbol=symbol, interval=interval, days_back=days)
        if data:
            data_map[symbol] = data
        else:
            missing.append(symbol)

    if missing and auto_download:
        print(f"Missing local candles for {len(missing)} symbols. Downloading now...")
        downloader.download_all_symbols(symbols=missing, interval=interval, days_back=days)
        for symbol in missing:
            data = downloader.load_historical_data(symbol=symbol, interval=interval, days_back=days)
            if data:
                data_map[symbol] = data

    return data_map


def main():
    parser = argparse.ArgumentParser(description="Train premium futures signal model")
    parser.add_argument("--symbols", type=str, default="all", help="all, top10, top5, or CSV list")
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval (default: 1h)")
    parser.add_argument("--days", type=int, default=365, help="Days to download if missing")
    parser.add_argument("--horizon", type=int, default=6, help="Forward candles used for target")
    parser.add_argument("--fee-bps", type=float, default=4.0, help="Round-trip trading fee in bps")
    parser.add_argument("--alpha", type=float, default=8.0, help="Ridge regularization alpha")
    parser.add_argument("--download-missing", action="store_true", help="Fetch missing pair data")
    parser.add_argument(
        "--model-out",
        type=str,
        default=os.path.join("backend", "models", "premium_signal_model.json"),
        help="Model output path",
    )

    args = parser.parse_args()

    if args.symbols == "all":
        symbols = Config.TOP_20_SYMBOLS
    elif args.symbols == "top10":
        symbols = Config.TOP_20_SYMBOLS[:10]
    elif args.symbols == "top5":
        symbols = Config.TOP_20_SYMBOLS[:5]
    else:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print("=" * 72)
    print("Premium Signal Trainer")
    print("=" * 72)
    print(f"Symbols: {len(symbols)}")
    print(f"Interval: {args.interval}")
    print(f"Horizon: {args.horizon} candles")
    print(f"Fee: {args.fee_bps} bps")
    print(f"Model output: {args.model_out}")
    print("=" * 72)

    downloader = DataDownloader()
    symbol_data = _load_symbol_data(
        downloader=downloader,
        symbols=symbols,
        interval=args.interval,
        auto_download=args.download_missing,
        days=args.days,
    )

    if not symbol_data:
        print("No historical data available. Download first with:")
        print("  py -3 download_data.py --days 365 --symbols all")
        print("Or run this trainer with --download-missing.")
        sys.exit(1)

    training_frames = {}
    for symbol, candles in symbol_data.items():
        frame = build_symbol_training_frame(
            candles=candles, symbol=symbol, horizon=args.horizon, fee_bps=args.fee_bps
        )
        if not frame.empty:
            training_frames[symbol] = frame
            print(f"{symbol}: {len(frame):,} feature rows")
        else:
            print(f"{symbol}: skipped (insufficient rows after indicators)")

    if len(training_frames) < 2:
        print("Need data from at least 2 symbols to train the premium model.")
        sys.exit(1)

    trainer = PremiumSignalModelTrainer(alpha=args.alpha, horizon=args.horizon, fee_bps=args.fee_bps)
    result = trainer.fit(training_frames)
    trainer.save_artifact(result.artifact, args.model_out)

    train_sel = result.train_metrics["selection"]
    val_sel = result.validation_metrics["selection"]
    print("\nTraining complete.")
    print(f"Samples used: {result.samples_used:,}")
    print(f"Train avg return/trade: {train_sel['avg_return'] * 100:.3f}%")
    print(f"Validation avg return/trade: {val_sel['avg_return'] * 100:.3f}%")
    print(f"Validation win rate: {val_sel['win_rate'] * 100:.2f}%")
    print(f"Validation trades: {val_sel['trade_count']}")
    print(f"Saved model: {args.model_out}")
    print("\nSet STRATEGY_MODE=ml to use this model in live bot signals.")


if __name__ == "__main__":
    main()
