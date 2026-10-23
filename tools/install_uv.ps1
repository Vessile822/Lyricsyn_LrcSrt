# LyricSync Windows 現代化極速安裝腳本 (整合 RTX 5060 Ti + uv + Python 3.14)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $projectRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  LyricSync 環境配置 (RTX 5060 Ti / CUDA 13.0)  " -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# 1. 檢查 uv
if (-not (Get-Command uv.exe -ErrorAction SilentlyContinue)) {
    Write-Host "[錯誤] 未找到 uv 套件管理器。請先安裝 uv。" -ForegroundColor Red
    exit 1
}
Write-Host "[1/4] 檢查 Python 與建立虛擬環境..." -ForegroundColor Green
& uv.exe venv --python "3.14" (Join-Path $projectRoot '.venv') --allow-existing

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "虛擬環境 Python 建立失敗: $venvPython"
}

# 2. 安裝 PyTorch (CUDA 13.0 - 支援 Blackwell RTX 50 系列)
Write-Host "[2/4] 安裝 PyTorch + CUDA 13.0 (利用本地快取極速連結)..." -ForegroundColor Green
& uv.exe pip install torch==2.14.0+cu130 torchaudio --index-url https://download.pytorch.org/whl/cu130 --python $venvPython

# 3. 安裝其餘相依套件
Write-Host "[3/4] 安裝 LyricSync 相依套件..." -ForegroundColor Green
& uv.exe pip install -r (Join-Path $PSScriptRoot 'requirements.txt') --python $venvPython

# 4. 驗證 GPU 運算與產生 runtime_config.json
Write-Host "[4/4] 驗證硬體加速運算與環境..." -ForegroundColor Green
& $venvPython (Join-Path $PSScriptRoot 'check_environment.py')

Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  LyricSync 安裝與配置成功！" -ForegroundColor Green
Write-Host "  拖曳歌曲與同名字幕檔至 DROP_HERE.bat 即可使用。" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green
