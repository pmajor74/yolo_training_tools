@echo off

python -c "import sys; print(f'python version=={sys.version_info.major}.{sys.version_info.minor}')"

:: Check Python version
for /f "tokens=2 delims==" %%V in ('python -c "import sys; print(f'version=={sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%V

:: Extract just the version number
for /f "tokens=2 delims==" %%V in ("version==%PY_VER%") do set VER_NUM=%%V

:: Check minimum and maximum bounds
for /f "tokens=1,2 delims=." %%A in ("%VER_NUM%") do (
    set MAJOR=%%A
    set MINOR=%%B
)

if "%MAJOR%" NEQ "3" (
    echo Python 3.8 to 3.12 required. Found: %VER_NUM%
    goto exit_failure
)

if %MINOR% LSS 8 (
    echo Python 3.8 to 3.12 required. Found: %VER_NUM%
    goto exit_failure
)

if %MINOR% GTR 12 (
    echo Python 3.8 to 3.12 required. Found: %VER_NUM%
    goto exit_failure
)

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

    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists. Skipping creation.
)

@echo Activating Environment
call "%venvPath%\scripts\activate"
python.exe -m pip install --upgrade pip

pip install -r requirements.txt

@echo Installation Complete
@echo The CPU torch library is installed, if you have a GPU, then uninstall the local torch library and get the proper pytorch that @echo matches your CUDA installation (otherwise training will be very slow).
@echo for example if you have CUDA 12.8 the command would be: 
@echo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pause
exit /b 0

:exit_failure
pause
exit /b 1