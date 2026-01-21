@echo off
taskkill /IM python.exe /F 2>nul
taskkill /IM pythonw.exe /F 2>nul
taskkill /IM python3.13.exe /F 2>nul
taskkill /IM pythonw3.13.exe /F 2>nul
echo Koe stopped.
timeout /t 2
