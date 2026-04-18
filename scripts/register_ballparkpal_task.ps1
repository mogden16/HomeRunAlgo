param(
    [string]$TaskName = "HomeRunAlgoBallparkPalMorning",
    [string]$PythonPath = "",
    [string]$RunTime = "06:10",
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$runScript = Join-Path $repoRoot "scripts\run_ballparkpal_morning_refresh.ps1"
$resolvedPython = if ($PythonPath) {
    if ([System.IO.Path]::IsPathRooted($PythonPath)) { $PythonPath } else { Join-Path $repoRoot $PythonPath }
} else {
    Join-Path $repoRoot ".venv1\Scripts\python.exe"
}

if (-not (Test-Path $runScript)) {
    throw "Run script not found at $runScript"
}
if (-not (Test-Path $resolvedPython)) {
    throw "Python executable not found at $resolvedPython"
}

$parts = $RunTime.Split(":")
if ($parts.Count -ne 2) {
    throw "Invalid RunTime '$RunTime'. Use HH:mm in 24-hour local time."
}
$hours = [int]$parts[0]
$minutes = [int]$parts[1]
if ($hours -lt 0 -or $hours -gt 23 -or $minutes -lt 0 -or $minutes -gt 59) {
    throw "Invalid RunTime '$RunTime'. Use HH:mm in 24-hour local time."
}

$start = (Get-Date).Date.AddHours($hours).AddMinutes($minutes)
$trigger = New-ScheduledTaskTrigger -Daily -At $start
$actionArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$runScript`"",
    "-PythonPath", "`"$resolvedPython`"",
    "-EnvFile", "`"$EnvFile`""
)
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($actionArgs -join " ")
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4)

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Write-Host "Registered $TaskName at $RunTime using $runScript"
