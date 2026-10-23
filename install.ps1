$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PIP_DEFAULT_TIMEOUT = '120'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$script:transcribing = $false
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}
function Test-Python {
    param([string]$Exe, [string[]]$Prefix = @())
    try {
        $probe = @(& $Exe @Prefix (Join-Path $PSScriptRoot 'python_probe.py') 2>$null)
        if ($LASTEXITCODE -eq 0 -and $probe.Count -gt 0) {
            $candidate = [string]$probe[-1]
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    } catch { }
    return $null
}
try {
    Start-Transcript -Path (Join-Path $PSScriptRoot 'install.log') -Force | Out-Null
    $script:transcribing = $true
    Write-Host 'LyricSync setup - keep this window open.' -ForegroundColor Cyan
    Write-Host 'Downloads may take 10-30 minutes. No PATH checkbox is needed.'
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    foreach ($required in @('requirements.txt','python_probe.py','check_environment.py','lyricsync.py','mms_refine.py','launcher.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $required))) { throw 'Extract the entire ZIP first, then open INSTALL.bat in the extracted folder.' }
    }
    Write-Host '[1/5] Finding Python 3.11 / 3.12 (64-bit)...'
    $pythonExe = $null
    $venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) {
        $pythonExe = Test-Python $venvPython
        if (-not $pythonExe) { throw 'The existing .venv is incompatible. Rename .venv to .venv_old and run INSTALL.bat again.' }
    }
    if (-not $pythonExe -and (Get-Command py.exe -ErrorAction SilentlyContinue)) {
        foreach ($version in @('-3.12','-3.11')) {
            $pythonExe = Test-Python 'py.exe' @($version)
            if ($pythonExe) { break }
        }
    }
    if (-not $pythonExe) {
        foreach ($candidate in @(
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe'),
            (Join-Path $env:LOCALAPPDATA 'LyricSyncRuntime\Python312\python.exe')
        )) {
            if (Test-Path -LiteralPath $candidate) { $pythonExe = Test-Python $candidate }
            if ($pythonExe) { break }
        }
    }
    if (-not $pythonExe) {
        foreach ($cmd in @('python.exe','python3.exe')) {
            $found = Get-Command $cmd -ErrorAction SilentlyContinue
            if ($found -and $found.Source -notlike '*\Microsoft\WindowsApps\*') { $pythonExe = Test-Python $found.Source }
            if ($pythonExe) { break }
        }
    }
    if (-not $pythonExe -and (Get-Command pymanager.exe -ErrorAction SilentlyContinue)) {
        Write-Host 'Python Install Manager found. Installing Python 3.12...'
        & pymanager.exe install 3.12
        if ($LASTEXITCODE -eq 0) { $pythonExe = Test-Python 'py.exe' @('-3.12') }
    }
    if (-not $pythonExe) {
        Write-Host 'Downloading official Python 3.12.10 for this user...'
        $target = Join-Path $env:LOCALAPPDATA 'LyricSyncRuntime\Python312'
        $installer = Join-Path $env:TEMP ('lyricsync-python-' + [guid]::NewGuid().ToString() + '.exe')
        Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer
        $signature = Get-AuthenticodeSignature -FilePath $installer
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') { throw 'Python installer signature verification failed.' }
        $installerArguments = '/quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Include_doc=0 Include_pip=1 TargetDir="' + $target + '"'
        $proc = Start-Process -FilePath $installer -ArgumentList $installerArguments -Wait -PassThru
        if ($proc.ExitCode -notin @(0,3010)) { throw "Python installation failed: $($proc.ExitCode)" }
        Remove-Item -LiteralPath $installer -Force
        $pythonExe = Test-Python (Join-Path $target 'python.exe')
        if (-not $pythonExe) { throw 'Python installation could not be verified.' }
    }
    Write-Host "Using: $pythonExe"
    Write-Host '[2/5] Preparing a separate environment...'
    if (-not (Test-Path -LiteralPath $venvPython)) { Invoke-Checked $pythonExe @('-m','venv',(Join-Path $PSScriptRoot '.venv')) }
    $env:PATH = (Join-Path $PSScriptRoot '.venv\Scripts') + ';' + $env:PATH
    Invoke-Checked $venvPython @('-m','ensurepip','--upgrade')
    Invoke-Checked $venvPython @('-m','pip','install','--upgrade','pip','setuptools','wheel')
    Write-Host '[3/5] Installing matching PyTorch packages...'
    $gpu = $false
    if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) {
        & nvidia-smi.exe -L
        $gpu = ($LASTEXITCODE -eq 0)
    }
    if ($gpu) {
        Write-Host 'NVIDIA detected. Trying CUDA 12.6 build...'
        & $venvPython -m pip install 'torch==2.6.0' 'torchaudio==2.6.0' --index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'GPU package installation failed; retrying with CPU build.' -ForegroundColor Yellow
            Invoke-Checked $venvPython @('-m','pip','install','--force-reinstall','torch==2.6.0','torchaudio==2.6.0','--index-url','https://download.pytorch.org/whl/cpu')
        }
    } else {
        Invoke-Checked $venvPython @('-m','pip','install','torch==2.6.0','torchaudio==2.6.0','--index-url','https://download.pytorch.org/whl/cpu')
    }
    Write-Host '[4/5] Installing LyricSync dependencies and FFmpeg...'
    Invoke-Checked $venvPython @('-m','pip','install','-r',(Join-Path $PSScriptRoot 'requirements.txt'))
    Write-Host '[5/5] Checking packages, FFmpeg, and GPU computation...'
    Invoke-Checked $venvPython @('-m','pip','check')
    Invoke-Checked $venvPython @((Join-Path $PSScriptRoot 'check_environment.py'))
    Write-Host ''
    Write-Host 'INSTALLATION SUCCESSFUL' -ForegroundColor Green
    Write-Host 'Drag a song and its same-name lyrics TXT onto DROP_HERE.bat.'
    Write-Host 'The first song will download AI models. It can take much longer.'
} catch {
    Write-Host ''
    Write-Host ('INSTALLATION FAILED: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Please send install.log back for help.'
    if ($script:transcribing) { Stop-Transcript | Out-Null }
    exit 1
}
if ($script:transcribing) { Stop-Transcript | Out-Null }
exit 0
