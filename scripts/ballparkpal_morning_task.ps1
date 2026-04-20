param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $repoRoot "scripts\run_ballparkpal_morning_refresh.ps1"

if (-not (Test-Path $runScript)) {
    throw "Run script not found at $runScript"
}

& $runScript
