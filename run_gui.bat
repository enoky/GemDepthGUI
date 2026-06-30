@echo off
REM Launch the GemDepth GUI using the project's virtual environment.
REM Always runs from the repo root so ./checkpoint and the model package resolve.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at ".venv\Scripts\activate.bat"
    echo Create it first, e.g.:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

python "gemdepth_gui.py"
set "EXITCODE=%ERRORLEVEL%"

REM Keep the window open if the GUI exited with an error so messages stay visible.
if not "%EXITCODE%"=="0" (
    echo.
    echo [GUI exited with code %EXITCODE%]
    pause
)

endlocal
