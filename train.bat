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
set "CONDA_EXE=%PICT_CONDA%"
for %%I in ("%PICT_CONDA%") do set "PICT_BASE_PYTHON=%%~dpI..\python.exe"
if not exist "%PICT_BASE_PYTHON%" (
    echo Conda base Python was not found next to its Scripts directory.
    goto finish
)
"%PICT_BASE_PYTHON%" "%~dp0deploy\start_training.py" %*
set "PICT_TRAIN_CODE=%ERRORLEVEL%"
if not "%PICT_TRAIN_CODE%"=="0" echo Training did not complete. Review the messages above.
:finish
echo.
if not "%PICT_TRAIN_NO_PAUSE%"=="1" pause
exit /b %PICT_TRAIN_CODE%
