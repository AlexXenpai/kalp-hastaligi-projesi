@echo off
title Kalp Hastaligi Projesi
echo Uygulama adresi: http://localhost:8501
echo Durdurmak icin Ctrl+C tuslarina basin.
wsl.exe -d Ubuntu --cd "%~dp0heart-disease-uci" -- .venv-wsl/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
if errorlevel 1 pause
