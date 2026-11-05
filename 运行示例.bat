@echo off
setlocal
cd /d "%~dp0"
title 运行示例：影评三分类

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    py -3.11 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.11"
)
if not defined PY set "PY=python"

echo.
echo  使用解释器：%PY%
echo  正在处理前 5 条影评，首次运行需要加载模型，请稍候。
echo.

%PY% src\pipeline_main.py ^
  --comment_csv data\samples\class\comments.csv ^
  --bg_info_txt data\samples\class\bg_info.txt ^
  --comment_mission class ^
  --id_col id --comment_col comment ^
  --model_name qwen3:8b ^
  --prompts_dir prompts ^
  --output_csv output\结果分析\sample_class\result.csv ^
  --ollama_config configs\llm_ollama_config.json ^
  --gen_config configs\llm_generation_config.json ^
  --lda_config configs\lda_config.json ^
  --top_k 3 ^
  --label_list "消极,中立,积极" ^
  --sample 5 ^
  --disable_think

echo.
echo  完成。结果在 output\结果分析\sample_class 目录。
pause
