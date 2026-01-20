@echo off
title Stix - AI Sticker Maker
color 0A

echo.
echo  ========================================
echo    STIX - AI STICKER MAKER
echo  ========================================
echo.

:: Set the project path
set PROJECT_PATH=G:\Projects\Stix-Sticker-Maker

:: Check if project exists
if not exist "%PROJECT_PATH%" (
    echo ERROR: Project not found at %PROJECT_PATH%
    echo Please update the PROJECT_PATH in this file.
    pause
    exit /b 1
)

:: Kill any existing Stix processes to ensure clean start
echo  [0/3] Cleaning up old processes...
taskkill /f /fi "WINDOWTITLE eq Stix Backend*" > nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Stix Frontend*" > nul 2>&1
timeout /t 2 /nobreak > nul

echo  [1/3] Starting Backend Server...
cd /d "%PROJECT_PATH%\backend"
start "Stix Backend" cmd /k "uvicorn main:app --reload --port 8000"

:: Wait for backend to start
echo  [2/3] Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo  [3/3] Starting Frontend...
cd /d "%PROJECT_PATH%\frontend"
start "Stix Frontend" cmd /k "npm run dev"

:: Wait for frontend to start
timeout /t 3 /nobreak > nul

echo.
echo  ========================================
echo    STIX IS READY!
echo  ========================================
echo.
echo  Opening browser...
echo.
echo  Frontend: http://localhost:5173
echo  Backend:  http://localhost:8000
echo.
echo  To stop: Close the terminal windows
echo  ========================================
echo.

:: Open browser
start http://localhost:5173

echo  Press any key to close this launcher...
pause > nul
