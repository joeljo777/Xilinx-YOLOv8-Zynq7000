# Launch Vitis-AI Docker Container with Project Mounted
# PowerShell Script

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Launching Vitis-AI Docker Container" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your project will be mounted at /workspace inside the container"
Write-Host ""

# Refresh PATH environment variable
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Get current directory for volume mount
$currentDir = (Get-Location).Path

# Run Docker container
docker run -it --rm `
  -v "${currentDir}:/workspace" `
  xilinx/vitis-ai-pytorch-cpu:latest `
  bash

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Container Exited" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
