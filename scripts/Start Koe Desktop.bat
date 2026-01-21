@echo off
cd /d "%~dp0.."
pythonw src\server_launcher.py start
pythonw run.py
