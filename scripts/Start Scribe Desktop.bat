@echo off
cd /d "%~dp0.."
python src\server_launcher.py start
pythonw -m src.meeting.app
