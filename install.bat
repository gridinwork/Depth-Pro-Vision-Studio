@echo off
chcp 65001 >nul
setlocal DisableDelayedExpansion
cd /d "%~dp0"

echo ==========================================
echo Depth Pro Vision Studio - install
echo ==========================================

where py >nul 2>&1
if errorlevel 1 goto no_py

py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) else 1)"
if errorlevel 1 goto no_py311

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 goto failed
)

call ".venv\Scripts\activate.bat"
echo Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed

echo Installing PyTorch with CUDA 12.4...
".venv\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto failed

echo Installing application dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Installing official Depth Pro package from Apple GitHub...
".venv\Scripts\python.exe" -m pip install "git+https://github.com/apple/ml-depth-pro.git" --no-deps
if errorlevel 1 goto failed

if not exist checkpoints mkdir checkpoints
if not exist screenshots mkdir screenshots
if not exist depth_maps mkdir depth_maps
if not exist raw_depth mkdir raw_depth
if not exist demo_videos mkdir demo_videos
if not exist logs mkdir logs
if not exist config mkdir config

if not exist "checkpoints\depth_pro.pt" (
  echo Downloading official Depth Pro checkpoint...
  ".venv\Scripts\python.exe" -c "from utils.downloader import download_checkpoint; download_checkpoint()"
  if errorlevel 1 goto failed
)

echo Fetching ffmpeg...
".venv\Scripts\python.exe" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if errorlevel 1 goto failed

echo Running install checks...
".venv\Scripts\python.exe" scripts\verify_install.py
if errorlevel 1 goto failed

echo.
echo Install finished.
echo Start the application with start.bat
goto end

:no_py
echo Python launcher py.exe was not found.
goto failed_pause

:no_py311
echo Python 3.11 is required.
echo Install Python 3.11 from https://www.python.org/downloads/ and run install.bat again.
goto failed_pause

:failed
echo.
echo Install failed. Read the messages above.
echo Details may also be in logs\app.log
goto failed_pause

:failed_pause
if /I "%~1"=="nopause" exit /b 1
pause
exit /b 1

:end
if /I "%~1"=="nopause" exit /b 0
pause
exit /b 0
