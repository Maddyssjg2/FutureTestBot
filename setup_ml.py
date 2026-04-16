#!/usr/bin/env python3
"""
Setup script for Adaptive ML Trading
Initializes models before starting the bot
"""

import sys
sys.path.append('backend')

import logging
from adaptive_trainer import initialize_ml_models, adaptive_trainer
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("="*70)
    print("ADAPTIVE ML TRADING - INITIAL SETUP")
    print("="*70)
    print(f"Target: {adaptive_trainer.config.target_daily_trades} daily wins")
    print(f"Minimum Confidence: {adaptive_trainer.config.min_confidence}%")
    print(f"Target Win Rate: {adaptive_trainer.config.target_win_rate:.0%}")
    print(f"Symbols: {len(adaptive_trainer.symbols)} pairs")
    print(f"Retraining Interval: Every {adaptive_trainer.config.training_interval_hours} hour(s)")
    print("="*70)
    print()
    
    # Run initial training
    logger.info("Starting initial model training...")
    success = initialize_ml_models()
    
    if success:
        print("\n" + "="*70)
        print("✓ ML Models Initialized Successfully!")
        print("="*70)
        print("\nYou can now start the bot with:")
        print("  .\\start.bat          (Single symbol)")
        print("  py start_multi.py    (Multi-symbol with ML)")
        print("\nThe bot will automatically retrain models every hour.")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print("✗ Failed to initialize ML models")
        print("="*70)
        print("\nThe bot will fall back to rule-based strategies.")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
