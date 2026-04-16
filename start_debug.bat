@echo off
echo Starting Trading Bot (Debug Mode)
echo =================================
echo.
echo This window will show all errors.
echo.

cd backend
venv\Scripts\activate
py app.py

pause
