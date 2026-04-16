#!/usr/bin/env python3
"""
Reset and Train Advanced 5-Strategy ML Models
Use this script to train models with the new confluence-based strategies
"""

import os
import sys
import shutil

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def main():
    print("\n" + "="*70)
    print("  ADVANCED STRATEGY ML MODEL TRAINING")
    print("="*70)
    print("\nThis will train ML models for the 5-strategy confluence system:")
    print("  1. EMA Trend Confluence")
    print("  2. RSI Divergence + MACD")
    print("  3. Bollinger Band Squeeze Breakout")
    print("  4. Fibonacci Golden Pocket Pullback")
    print("  5. ADX Multi-Timeframe Trend Filter")
    print("\nConfiguration:")
    print("  - Min confluence: 2 strategies")
    print("  - Max leverage: 10x")
    print("  - Risk per trade: 1-2%")
    print("  - SL: Strategy-specific (EMA34, BB middle, Fib 65%, DI swing)")
    print("  - TP: 1.5R to 2R minimum")
    print("="*70)
    print()
    
    # Clear old advanced models if they exist
    advanced_model_dir = os.path.join('backend', 'ml_models', 'advanced')
    if os.path.exists(advanced_model_dir):
        print(f"[1] Clearing old advanced models from {advanced_model_dir}...")
        shutil.rmtree(advanced_model_dir)
        print("    ✓ Old models cleared")
    else:
        print("[1] No old advanced models to clear")
    
    print()
    
    # Import and run training
    print("[2] Starting advanced ML training...")
    print()
    
    try:
        from advanced_ml_trainer import train_advanced_models
        
        # Train all models with force retrain
        results = train_advanced_models(force_retrain=True)
        
        # Count successful
        successful = sum(1 for r in results.values() if r.get('success'))
        total = len(results)
        
        print()
        print("="*70)
        print(f"  TRAINING COMPLETE: {successful}/{total} models trained successfully")
        print("="*70)
        print()
        
        if successful == total:
            print("✓ All models trained successfully!")
            print(f"\nModels saved to: {advanced_model_dir}")
            print("\nYou can now start the bot with the new advanced strategies.")
            return 0
        else:
            print(f"⚠ {total - successful} models failed to train")
            print("Check logs above for details.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    
    print()
    input("Press Enter to exit...")
    sys.exit(exit_code)
