# Build script for logic-lang package (PowerShell version)

# Enable strict error handling
$ErrorActionPreference = "Stop"

Write-Host "Building logic-lang package..." -ForegroundColor Green

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
Get-ChildItem -Filter "*.egg-info" -Directory | Remove-Item -Recurse -Force

# Build the package
Write-Host "Building package..." -ForegroundColor Yellow
python -m build

Write-Host "Build complete! Distribution files are in dist/" -ForegroundColor Green
Write-Host "Contents of dist/:" -ForegroundColor Cyan
Get-ChildItem -Path "dist" -Force | Format-Table Name, Length, LastWriteTime

Write-Host ""
Write-Host "To upload to PyPI:" -ForegroundColor Magenta
Write-Host "1. Test upload: python -m twine upload --repository testpypi dist/*" -ForegroundColor White
Write-Host "2. Production upload: python -m twine upload dist/*" -ForegroundColor White
