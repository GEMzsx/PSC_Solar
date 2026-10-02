$ErrorActionPreference = 'Stop'
$revision = '9b6ed4ca72910cbdf27b9832cf0c081caa936af3'
$expectedHash = 'e722ded60e6162a47b00746c6215498dcf0ce9dd004703fd01ab560dfc213ba2'
$expectedSize = 84659951
$projectRoot = Split-Path -Parent $PSScriptRoot
$rawDirectory = Join-Path $projectRoot 'data/raw'
$targetFile = Join-Path $rawDirectory 'Perovskite_database_content_all_data.csv'
$downloadFile = "$targetFile.download"
$url = "https://media.githubusercontent.com/media/Jesperkemist/perovskitedatabase_data/$revision/data/Perovskite_database_content_all_data.csv"

if (Test-Path -LiteralPath $targetFile) {
    if ((Get-FileHash -LiteralPath $targetFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
        throw 'Existing CSV does not match the pinned version. Preserve it and resolve the version before downloading.'
    }
    Write-Output "Verified existing data: $targetFile"
    return
}
if (Test-Path -LiteralPath $downloadFile) {
    throw "Partial download exists: $downloadFile. Inspect it before retrying."
}
New-Item -ItemType Directory -Path $rawDirectory -Force | Out-Null
Invoke-WebRequest -Uri $url -OutFile $downloadFile
if ((Get-Item -LiteralPath $downloadFile).Length -ne $expectedSize) {
    throw 'Downloaded file size does not match the pinned CSV.'
}
if ((Get-FileHash -LiteralPath $downloadFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'Downloaded file hash does not match the pinned CSV.'
}
Move-Item -LiteralPath $downloadFile -Destination $targetFile
Write-Output "Downloaded and verified: $targetFile"
