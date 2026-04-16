#!/usr/bin/env python3
"""
Reset all models and retrain with Talon Sniper v1 strategy
This wipes old EnhancedCryptoModel files and trains fresh TalonMLModel
"""

import os
import sys
import glob
import shutil

sys.path.append('backend')

print("=" * 60)
print("TALON SNIPER v1 - MODEL RESET & RETRAIN")
print("=" * 60)

# 1. Clear old model files
models_dir = os.path.join('backend', 'ml_models')
if os.path.exists(models_dir):
    print("\n[1] Clearing old model files...")
    
    # Remove enhanced_*.pkl files
    old_models = glob.glob(os.path.join(models_dir, 'enhanced_*.pkl'))
    old_models += glob.glob(os.path.join(models_dir, 'talon_*.pkl'))
    
    for f in old_models:
        try:
            os.remove(f)
            print(f"   ✗ Removed: {os.path.basename(f)}")
        except Exception as e:
            print(f"   ⚠ Could not remove {f}: {e}")
    
    # Remove metadata
    metadata_file = os.path.join(models_dir, 'training_metadata.json')
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
        print(f"   ✗ Removed: training_metadata.json")
    
    print(f"   Cleared {len(old_models)} old model files")
else:
    os.makedirs(models_dir, exist_ok=True)
    print(f"\n[1] Created models directory: {models_dir}")

# 2. Clear model cache
print("\n[2] Clearing model cache...")
from talon_ml_model import _model_cache
_model_cache.clear()
print("   ✓ Model cache cleared")

# 3. Start training
print("\n[3] Starting Talon model training...")
print("   Strategy: Talon Sniper v1 (Heikin Ashi candles)")
print("   Features: TEMA/DEMA Signal 1 + ATR Trend Signal 2")
print("   Target: 75%+ confidence, 80% win rate")
print()

from adaptive_trainer import AdaptiveMLTrainer, TrainingConfig
from config import Config

config = TrainingConfig()
config.target_win_rate = 0.80  # 80% win rate target
config.min_confidence = 75.0
config.retrain_win_target = 0.80

trainer = AdaptiveMLTrainer(config=config)

# Force full retrain for all symbols
success = trainer.run_training_cycle(force_retrain=True)

if success:
    print("\n" + "=" * 60)
    print("✓ TALON SNIPER MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nModels saved to: {models_dir}")
    print(f"Symbols trained: {len(trainer.models)}")
    
    # Start hourly retraining
    print("\n[4] Starting hourly retraining scheduler...")
    trainer.schedule_hourly_training()
    print("   ✓ Scheduler started")
    
    print("\n" + "-" * 60)
    print("Press Ctrl+C to stop the trainer")
    print("-" * 60)
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        
else:
    print("\n" + "=" * 60)
    print("✗ TRAINING FAILED")
    print("=" * 60)
    sys.exit(1)
