@echo off
echo =========================
echo People Counter Setup
echo =========================
echo.

set "GPU_NAME="
for /f "skip=1 delims=" %%G in ('wmic path win32_VideoController get name 2^>nul') do (
    if not "%%G"=="" (
        echo %%G | find /I "NVIDIA" >nul
        if not errorlevel 1 (
            set "GPU_NAME=%%G"
        )
    )
)

if defined GPU_NAME (
    echo Detected NVIDIA GPU: %GPU_NAME%
    echo Recommended mode: GPU
) else (
    echo No NVIDIA GPU detected.
    echo Recommended mode: CPU
)

echo.
echo Select installation mode:
echo 1. GPU (CUDA - NVIDIA only)
echo 2. CPU (compatible with all machines)
echo.
set /p choice=Enter 1 or 2:

if not "%choice%"=="1" if not "%choice%"=="2" (
    echo Invalid choice. Defaulting to CPU mode.
    set "choice=2"
)

python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
if "%choice%"=="1" (
    echo Installing CUDA version of PyTorch...
    python -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing CPU version of PyTorch...
    python -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
    python -m pip install torch torchvision torchaudio
)

echo Installing other packages...
python -m pip install -r requirements.txt

if not exist outputs mkdir outputs
if not exist models mkdir models

echo.
echo =========================
echo Verifying model file...
echo =========================
if not exist models\yolov8s.pt (
    echo ERROR: models\yolov8s.pt was not found.
    pause
    exit /b 1
)

echo.
echo =========================
echo Verifying PyTorch / CUDA...
echo =========================
python -c "import torch; print('torch version:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

echo.
echo =========================
echo Setup complete!
echo =========================
pause
