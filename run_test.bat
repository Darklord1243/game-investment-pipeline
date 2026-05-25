@echo off
echo Running test script with Anaconda Python...
echo.

:: Try to find Anaconda Python
set ANACONDA_PATH=C:\Users\%USERNAME%\anaconda3
set PYTHON_EXE=%ANACONDA_PATH%\python.exe

if exist "%PYTHON_EXE%" (
    echo Found Anaconda Python at %PYTHON_EXE%
    echo.
    "%PYTHON_EXE%" test_fix.py
) else (
    echo Anaconda Python not found at %ANACONDA_PATH%
    echo Trying to activate conda environment...
    
    :: Try to use conda activate
    call conda activate base
    python test_fix.py
    
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo Failed to run with conda. Please run this script from an Anaconda prompt.
    )
)

echo.
echo Press any key to exit...
pause > nul 