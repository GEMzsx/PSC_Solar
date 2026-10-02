param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if (-not $PythonPath) {
        $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
        if (Test-Path -LiteralPath $bundledPython) {
            $PythonPath = $bundledPython
        } else {
            $PythonPath = (Get-Command python -ErrorAction Stop).Source
        }
    }
    $venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $PythonPath -m venv --without-pip .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
    }
    & $PythonPath -m pip --python .venv install --index-url https://pypi.org/simple -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Write-Output 'Environment ready. Run scripts/run.ps1 to open the local tool.'
} finally {
    Pop-Location
}
