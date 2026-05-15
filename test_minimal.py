#!/usr/bin/env python3
# Minimal test script to verify imports and dependencies

print("Starting...")

import os
import sys

print("Imports done")

# Add the backend directory to the Python path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

print("Path configured")

# Try to load TensorFlow and print its version
try:
    import tensorflow as tf
    print(f"TensorFlow loaded: {tf.__version__}")
except Exception as e:
    print(f"TensorFlow error: {e}")

# Try to import the Binance Futures client
try:
    from binance_client import BinanceFuturesClient
    print("Binance client imported")
except Exception as e:
    print(f"Binance client error: {e}")

# Try to import the Advanced Strategy Engine
try:
    from advanced_strategies import AdvancedStrategyEngine
    print("Advanced strategies imported")
except Exception as e:
    print(f"Advanced strategies error: {e}")

print("All imports successful!")
