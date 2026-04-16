@echo off
echo Starting Binance Futures Multi-Symbol Bot
echo =========================================

REM Check if virtual environment exists
if not exist backend\venv\Scripts\activate.bat (
    echo Please set up the backend first:
    echo cd backend
    echo py -m venv venv
    echo venv\Scripts\activate
    echo pip install -r requirements.txt
    pause
    exit /b 1
)

REM Start backend in new window
echo Starting Backend...
start "Trading Bot Backend" cmd /k "cd backend && venv\Scripts\activate && py app.py"

REM Wait for backend to start
timeout /t 3 /nobreak > nul

REM Check if node_modules exists
if not exist frontend\node_modules (
    echo Please install frontend dependencies first:
    echo cd frontend
    echo npm install
    pause
    exit /b 1
)

REM Start frontend in new window (LAN accessible)
echo Starting Dashboard...
start "Trading Bot Dashboard" cmd /k "cd frontend && set HOST=0.0.0.0&& set PORT=3000&& npm start"

REM Wait for backend to boot, then start multi bot by default
timeout /t 5 /nobreak > nul
curl -X POST http://localhost:5000/api/start -H "Content-Type: application/json" -d "{\"symbol_count\": 10}" > nul 2>&1

echo.
echo =========================================
echo Backend:  http://localhost:5000
echo Dashboard (local): http://localhost:3000
echo Dashboard (LAN):   http://YOUR_LOCAL_IP:3000
echo Multi-bot default: started (includes BTCUSDT)
echo.
echo Press any key to stop all services...
pause > nul

REM Kill processes
taskkill /FI "WINDOWTITLE eq Trading Bot Backend" /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Trading Bot Dashboard" /F > nul 2>&1

echo Services stopped.
