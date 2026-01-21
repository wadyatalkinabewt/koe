@echo off
taskkill /IM python.exe /F 2>nul
taskkill /IM pythonw.exe /F 2>nul
echo Koe stopped.
timeout /t 2
