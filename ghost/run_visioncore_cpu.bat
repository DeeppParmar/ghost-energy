@echo off
setlocal

REM One-click launcher for VisionCore (Windows CPU-only)
REM Usage: double-click or run "run_visioncore_cpu.bat"

cd /d "%~dp0"

REM Create venv if missing
if not exist "venv\Scripts\activate.bat" (
  echo [VisionCore] Creating virtual environment...
  python -m venv venv
)

call "venv\Scripts\activate.bat"

echo [VisionCore] Upgrading pip...
python -m pip install --upgrade pip

echo [VisionCore] Installing CPU PyTorch (if not already installed)...
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo [VisionCore] Installing app dependencies...
pip install flask opencv-python numpy ultralytics fpdf2 psutil waitress scikit-learn psycopg2-binary sqlalchemy requests

echo [VisionCore] Starting server on http://localhost:5000 ...
echo [VisionCore] Opening browser...
start "" "http://localhost:5000/monitor"
waitress-serve --listen=0.0.0.0:5000 --threads=4 app:app

endlocal

