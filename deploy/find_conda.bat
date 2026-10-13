@echo off
rem Called by a launcher; return PICT_CONDA in its local environment.
set "PICT_CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "PICT_CONDA=%CONDA_EXE%"
if defined PICT_CONDA exit /b 0
for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined PICT_CONDA set "PICT_CONDA=%%I"
if defined PICT_CONDA exit /b 0
for %%I in ("%USERPROFILE%\miniconda3\Scripts\conda.exe" "%USERPROFILE%\anaconda3\Scripts\conda.exe" "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" "%ProgramData%\miniconda3\Scripts\conda.exe" "%ProgramData%\anaconda3\Scripts\conda.exe") do if not defined PICT_CONDA if exist "%%~I" set "PICT_CONDA=%%~I"
if defined PICT_CONDA exit /b 0
echo Conda was not found. Install Miniconda or Anaconda first.
echo For a custom installation, run this file from its Conda Prompt.
exit /b 1
