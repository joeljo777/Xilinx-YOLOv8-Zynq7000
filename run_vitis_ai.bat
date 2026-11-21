@echo off
REM Launch Vitis-AI Docker Container with Project Mounted

echo ============================================================
echo   Launching Vitis-AI Docker Container
echo ============================================================
echo.
echo Your project will be mounted at /workspace inside the container
echo.

REM Add Docker to PATH
set "PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin"

"C:\Program Files\Docker\Docker\resources\bin\docker.exe" run -it --rm ^
  -v "%cd%":/workspace ^
  xilinx/vitis-ai-pytorch-cpu:latest ^
  bash

echo.
echo ============================================================
echo   Container Exited
echo ============================================================
pause
