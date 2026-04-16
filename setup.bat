@echo off
echo Setting up Binance Futures Trading Bot
echo =====================================

REM Check Python
py --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check Node
node --version > nul 2>&1
if errorlevel 1 (
    echo Node.js is not installed. Please install Node.js 16+
    pause
    exit /b 1
)

REM Setup Backend
echo.
echo Setting up Backend...
cd backend

if not exist venv (
    echo Creating virtual environment...
    py -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing Python dependencies...
pip install -r requirements.txt

if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo.
    echo IMPORTANT: Edit backend\.env with your Binance Testnet API keys!
    echo Get keys from: https://testnet.binancefuture.com
)

cd ..

REM Setup Frontend
echo.
echo Setting up Frontend...
cd frontend

echo Installing Node dependencies...
npm install

cd ..

echo.
echo =====================================
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit backend\.env with your API keys
echo 2. Run start.bat to launch the bot
echo.
pause
