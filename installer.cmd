@echo off

:: Get the current directory where the CMD file is executing
set "currentDir=%~dp0"

:: Define the path to the virtual environment
set "venvPath=%currentDir%venv"

:: Check if the virtual environment exists
if not exist "%venvPath%" (
    echo Virtual environment not found. Creating it...
    python -m venv "%venvPath%"
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Exiting.
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists. Skipping creation.
)

@echo Activating Environment
call "%venvPath%\scripts\activate"
python.exe -m pip install --upgrade pip

:: uncomment the following to install UV, or just remove the uv from the uv pip install command
:: pip install uv 
uv pip install -r requirements.txt

@echo Installation Complete, if you notice a command not found error, its because you need to install uv (pip install uv)
pause