@echo off
setlocal EnableExtensions DisableDelayedExpansion
title PICT Model Environment Setup
set "PICT_SETUP_CODE=1"
set "PICT_CONDA="
rem Desktop launchers may omit standard Windows tools from PATH.
set "PATH=%SystemRoot%\System32;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"

rem Prefer the active installation, then PATH and common installer locations.
if defined CONDA_EXE if exist "%CONDA_EXE%" set "PICT_CONDA=%CONDA_EXE%"
if defined PICT_CONDA goto conda_found
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined PICT_CONDA set "PICT_CONDA=%%I"
if defined PICT_CONDA goto conda_found
for %%I in ("%USERPROFILE%\miniconda3\Scripts\conda.exe" "%USERPROFILE%\anaconda3\Scripts\conda.exe" "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" "%ProgramData%\miniconda3\Scripts\conda.exe" "%ProgramData%\anaconda3\Scripts\conda.exe") do if not defined PICT_CONDA if exist "%%~I" set "PICT_CONDA=%%~I"
if defined PICT_CONDA goto conda_found
echo Conda was not found. Install Miniconda or Anaconda first.
echo For a custom installation, run this file from its Conda Prompt.
goto finish

:conda_found
set "CONDA_EXE=%PICT_CONDA%"
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
