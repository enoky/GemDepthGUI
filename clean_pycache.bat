@echo off
REM Delete all __pycache__ folders and stray .pyc/.pyo files under the repo root.
REM Skips the .venv folder so the virtual environment's caches are left alone.

setlocal
cd /d "%~dp0"

echo Removing __pycache__ folders...
for /d /r %%d in (__pycache__) do (
    echo "%%d" | findstr /i /c:"\.venv" >nul
    if errorlevel 1 (
        if exist "%%d" (
            echo   %%d
            rd /s /q "%%d"
        )
    )
)

echo Removing stray .pyc / .pyo files...
for /r %%f in (*.pyc *.pyo) do (
    echo "%%f" | findstr /i /c:"\.venv" >nul
    if errorlevel 1 (
        if exist "%%f" del /f /q "%%f"
    )
)

echo Done.
endlocal
