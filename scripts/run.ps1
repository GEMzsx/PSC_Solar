param([ValidateSet('app', 'train', 'test', 'small-sample')][string]$Action = 'app')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Project environment is missing. Run scripts/setup.ps1 first.'
}
Push-Location $projectRoot
try {
    switch ($Action) {
        'app' { & $venvPython -X utf8 -m streamlit run app.py }
        'train' { & $venvPython -X utf8 -m psc_solar train }
        'test' { & $venvPython -X utf8 -m unittest discover -s tests -v }
        'small-sample' { & $venvPython -X utf8 -m psc_solar small-sample }
    }
    if ($LASTEXITCODE -ne 0) { throw "Action failed: $Action" }
} finally {
    Pop-Location
}
