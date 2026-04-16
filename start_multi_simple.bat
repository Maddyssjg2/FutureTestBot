@echo off
echo Starting Multi-Symbol Bot (Python)
echo ===================================
cd backend
venv\Scripts\activate
py -c "from multi_symbol_bot import MultiSymbolBot; from config import Config; b = MultiSymbolBot(Config.TOP_20_SYMBOLS[:10]); b.start(); import time; 
print('Bot running - Press Ctrl+C in backend window to stop'); time.sleep(99999)"
