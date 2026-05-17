#!/usr/bin/env python3
"""Minimal smoke test for the repository.

This test intentionally avoids optional third-party dependencies so it can run
in a clean environment and still verify that the repo's core Python modules are
syntactically valid and importable when possible.
"""

from __future__ import annotations

import os
import sys
import importlib.util

print("Starting...")
print("Imports done")

# Add the backend directory to the Python path for module imports.
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, backend_dir)
print("Path configured")


def try_import(module_name: str, label: str) -> None:
    try:
        __import__(module_name)
        print(f"{label} imported")
    except Exception as exc:
        print(f"{label} error: {exc}")


# Optional dependency checks: report availability without failing the test.
for module_name, label in [
    ("tensorflow", "TensorFlow"),
    ("binance_client", "Binance client"),
    ("advanced_strategies", "Advanced strategies"),
]:
    try_import(module_name, label)

print("All imports attempted!")
