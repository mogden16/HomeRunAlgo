param(
    [string]$TaskPrefix = "HomeRunAlgoDashboardRefresh",
    [string]$PythonPath = "",
    [string[]]$RunTimes = @("07:00", "15:00")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($RunTimes.Count -ne 2) {
    throw "Provide exactly two run times, for example @('07:00', '15:00')."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$refreshScript = Join-Path $repoRoot "scripts\refresh_dashboard.ps1"
$resolvedPython = if ($PythonPath) {
    if ([System.IO.Path]::IsPathRooted($PythonPath)) { $PythonPath } else { Join-Path $repoRoot $PythonPath }
} else {
    Join-Path $repoRoot ".venv1\Scripts\python.exe"
}

if (-not (Test-Path $refreshScript)) {
    throw "Refresh script not found at $refreshScript"
}
if (-not (Test-Path $resolvedPython)) {
    throw "Python executable not found at $resolvedPython"
}

$settleCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$refreshScript`" -PythonPath `"$resolvedPython`" -Mode settle"
$publishCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$refreshScript`" -PythonPath `"$resolvedPython`" -Mode publish"
$taskNames = @("${TaskPrefix}-Settle", "${TaskPrefix}-Publish")
$taskCommands = @($settleCommand, $publishCommand)

for ($i = 0; $i -lt $taskNames.Count; $i++) {
    & schtasks.exe /Create /SC DAILY /TN $taskNames[$i] /TR $taskCommands[$i] /ST $RunTimes[$i] /F | Out-Null
    Write-Host "Registered $($taskNames[$i]) at $($RunTimes[$i])"
}
