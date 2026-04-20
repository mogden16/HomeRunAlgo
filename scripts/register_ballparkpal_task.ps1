param(
    [string]$TaskName = "HomeRunAlgoBallparkPalMorning",
    [string]$RunTime = "06:10",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$installer = Join-Path $PSScriptRoot "install_ballparkpal_startup_watcher.ps1"
if (-not (Test-Path $installer)) {
    throw "Installer script not found at $installer"
}

& $installer
