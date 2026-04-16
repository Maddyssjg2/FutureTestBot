#!/usr/bin/env python3
"""Start multi-symbol trading bot via Flask server API

This ensures the frontend stays in sync - the bot is managed by
the Flask server which the dashboard connects to.
"""
import sys
import time
import requests

API_BASE = 'http://localhost:5000'

print("="*50)
print("Starting Multi-Symbol Trading Bot")
print("="*50)

# Start bot via Flask API so frontend stays in sync
try:
    # First check if server is running
    try:
        resp = requests.get(f'{API_BASE}/api/multi/status', timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('running'):
                print("Bot is already running!")
                print(f"Monitoring {data.get('symbols_monitored', 0)} pairs")
                sys.exit(0)
    except requests.ConnectionError:
        print("Flask server not running. Start it first with: cd backend && py app.py")
        print("Or use start.bat which starts both backend and frontend.")
        sys.exit(1)
    
    # Start the bot via API
    resp = requests.post(f'{API_BASE}/api/start', json={}, timeout=30)
    data = resp.json()
    
    if data.get('success'):
        symbols = data.get('symbols', [])
        print(f"\n✓ Bot started successfully!")
        print(f"Monitoring {len(symbols)} futures pairs...")
        print(f"Pairs: {', '.join(symbols[:5])}...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping bot...")
            resp = requests.post(f'{API_BASE}/api/stop', timeout=10)
            print("✓ Bot stopped")
    else:
        print(f"\n✗ Failed to start bot: {data.get('message', 'unknown')}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)
