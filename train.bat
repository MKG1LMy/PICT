@echo off
setlocal EnableExtensions DisableDelayedExpansion
title PICT Model Training
set "PICT_TRAIN_CODE=1"
set "PATH=%SystemRoot%\System32;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"
set "PYTHONIOENCODING=utf-8"
if not defined PICT_ENV set "PICT_ENV=pict_model"
call "%~dp0deploy\find_conda.bat"
if errorlevel 1 goto finish
echo Training environment: %PICT_ENV%
call "%PICT_CONDA%" run --no-capture-output -n "%PICT_ENV%" python "%~dp0deploy\run_training.py" %*
set "PICT_TRAIN_CODE=%ERRORLEVEL%"
if not "%PICT_TRAIN_CODE%"=="0" echo Training did not complete. Review the messages above.
:finish
echo.
if not "%PICT_TRAIN_NO_PAUSE%"=="1" pause
exit /b %PICT_TRAIN_CODE%
