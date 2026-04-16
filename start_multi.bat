@echo off
echo Starting Multi-Symbol Trading Bot (Multi-Only)
echo ==============================================
echo.

REM Default to configured multi symbols (includes BTCUSDT)
curl -X POST http://localhost:5000/api/start -H "Content-Type: application/json" -d "{\"symbol_count\": 10}"

echo.
echo.
echo Bot started in multi-only mode! Check Telegram for notifications.
echo Dashboard: http://localhost:3000
echo.
pause
