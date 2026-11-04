@echo off
setlocal
title Kalp Hastaligi Projesi
set "APP_URL=http://localhost:8501"
set "HEALTH_URL=%APP_URL%/_stcore/health"
echo Uygulama adresi: %APP_URL%
echo Durdurmak icin Ctrl+C tuslarina basin.

curl.exe --silent --fail --max-time 1 "%HEALTH_URL%" >nul 2>&1
if not errorlevel 1 (
    echo Uygulama zaten calisiyor; tarayicida aciliyor.
    start "" "%APP_URL%"
    exit /b 0
)

start "" /b powershell.exe -NoProfile -WindowStyle Hidden -Command "$url='%APP_URL%'; for($i=0; $i -lt 120; $i++){ try { $r=Invoke-WebRequest -UseBasicParsing -Uri ($url+'/_stcore/health') -TimeoutSec 1; if($r.StatusCode -eq 200){ Start-Process $url; exit 0 } } catch {}; Start-Sleep -Milliseconds 500 }"
wsl.exe -d Ubuntu --cd "%~dp0heart-disease-uci" -- .venv-wsl/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 (
    echo Uygulama baslatilamadi.
    pause
)
