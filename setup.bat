@echo off
setlocal EnableExtensions DisableDelayedExpansion
title PICT Model Environment Setup
set "PICT_SETUP_CODE=1"
rem Desktop launchers may omit standard Windows tools from PATH.
set "PATH=%SystemRoot%\System32;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"

call "%~dp0deploy\find_conda.bat"
if errorlevel 1 goto finish

:conda_found
set "CONDA_EXE=%PICT_CONDA%"
set "PYTHONIOENCODING=utf-8"
echo PICT Model environment setup
echo Conda: "%PICT_CONDA%"
for %%I in ("%PICT_CONDA%") do set "PICT_CONDA_SCRIPTS=%%~dpI"
if exist "%PICT_CONDA_SCRIPTS%..\python.exe" goto base_python
call "%PICT_CONDA%" run --no-capture-output -n base python "%~dp0deploy\setup_environment.py" %*
goto result
:base_python
"%PICT_CONDA_SCRIPTS%..\python.exe" "%~dp0deploy\setup_environment.py" %*
:result
set "PICT_SETUP_CODE=%ERRORLEVEL%"
if not "%PICT_SETUP_CODE%"=="0" echo Installation did not complete. Review the messages above.

:finish
echo.
rem Set PICT_SETUP_NO_PAUSE=1 for unattended invocation.
if not "%PICT_SETUP_NO_PAUSE%"=="1" pause
exit /b %PICT_SETUP_CODE%
