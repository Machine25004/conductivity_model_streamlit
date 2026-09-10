@echo off
setlocal
cd /d "%~dp0"

netstat -ano | findstr /r ":8501 .*LISTENING" >nul 2>nul
if not errorlevel 1 (
    start "" "http://localhost:8501"
    exit /b 0
)

echo Starting Streamlit app at http://localhost:8501
start "Yili Streamlit App" cmd /k "python -m streamlit run app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false"
timeout /t 5 /nobreak >nul
start "" "http://localhost:8501"
endlocal
exit /b 0
