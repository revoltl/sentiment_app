@echo off
setlocal
cd /d "%~dp0"
title 基于大语言模型的消费者评论情感分析系统

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY set "PY=python"

echo.
echo  使用解释器：%PY%
echo  正在启动，浏览器会自动打开，关闭本窗口即停止服务。
echo.

%PY% start.py --open

echo.
echo  服务已停止。若上方出现报错，请先运行「环境自检.bat」查看缺什么。
pause
