#!/usr/bin/env python3
"""
Reset all bot data - trades, learning history, performance stats
Start completely fresh
"""

import os
import json
import shutil
from datetime import datetime

def reset_all_data():
    print("=" * 60)
    print("RESETTING ALL BOT DATA")
    print("=" * 60)
    
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    
    # 1. Clear adaptive strategy config (learning parameters)
    config_file = os.path.join(backend_dir, 'adaptive_strategy_config.json')
    if os.path.exists(config_file):
        os.remove(config_file)
        print("✓ Cleared: adaptive_strategy_config.json")
    
    # 2. Clear performance reports
    reports_dir = os.path.join(backend_dir, 'reports')
    if os.path.exists(reports_dir):
        for f in os.listdir(reports_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(reports_dir, f))
                print(f"✓ Cleared report: {f}")
    
    # 3. Clear trade performance logs
    trade_logs = [
        'trade_performance.json',
        'trade_history.json',
        'closed_trades.json'
    ]
    for log in trade_logs:
        log_path = os.path.join(backend_dir, log)
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"✓ Cleared: {log}")
    
    # 4. Clear loss learning history file
    loss_history = os.path.join(backend_dir, 'loss_learning_history.json')
    if os.path.exists(loss_history):
        os.remove(loss_history)
        print("✓ Cleared: loss_learning_history.json")
    
    # 5. Reset model metadata to trigger fresh training perception
    metadata_file = os.path.join(backend_dir, 'ml_models', 'training_metadata.json')
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
        print("✓ Cleared: training_metadata.json")
    
    # 6. Create fresh empty tracking files
    fresh_config = {
        'tema_dema_weight': 1.0,
        'signal_2_weight': 1.0,
        'atr_multiplier_sl': 2.0,
        'atr_multiplier_tp1': 0.5,
        'atr_multiplier_tp2': 1.0,
        'ema_trend_filter': 'ema_13',
        'ema_confirm_candles': 1,
        'min_confidence': 75,
        'max_atr_pct': 0.06,
        'require_signal_2_confirm': False,
        'volatility_filter': False,
        'custom_strategy_name': 'Talon Sniper v1 Fresh Start',
        'version': 1,
        'reset_at': datetime.utcnow().isoformat()
    }
    
    with open(config_file, 'w') as f:
        json.dump(fresh_config, f, indent=2)
    print("✓ Created: fresh adaptive_strategy_config.json")
    
    print("\n" + "=" * 60)
    print("ALL DATA CLEARED!")
    print("=" * 60)
    print("\nWhat's been reset:")
    print("• Trade counters (wins/losses/win rate)")
    print("• Loss learning history")
    print("• Strategy adaptations")
    print("• Performance reports")
    print("• Trade history logs")
    print("\nModels preserved - no need to retrain")
    print("Bot will start with fresh counters and default strategy params")
    print("=" * 60)

if __name__ == '__main__':
    reset_all_data()
