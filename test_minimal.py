#!/usr/bin/env python3
print("Starting...")

import os
import sys

print("Imports done")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("Path configured")

try:
    import tensorflow as tf
    print(f"TensorFlow loaded: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow error: {e}")

try:
    from binance_client import BinanceFuturesClient
    print("Binance client imported")
except Exception as e:
    print(f"Binance client error: {e}")

try:
    from advanced_strategies import AdvancedStrategyEngine
    print("Advanced strategies imported")
except Exception as e:
    print(f"Advanced strategies error: {e}")

print("All imports successful!")
