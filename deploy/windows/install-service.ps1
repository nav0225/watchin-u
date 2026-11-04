# Registers watchin-u as a Windows service with NSSM (https://nssm.cc), run as Administrator:
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1 -Root C:\watchin-u
param(
    [string]$Root = "C:\watchin-u",
    [string]$Nssm = "nssm.exe",
    [string]$Name = "watchinu"
)
$ErrorActionPreference = "Stop"
$exe = Join-Path $Root ".venv\Scripts\watchinu.exe"
if (-not (Test-Path $exe)) { throw "not found: $exe (create the venv and pip install first)" }
New-Item -ItemType Directory -Force -Path (Join-Path $Root "data\logs") | Out-Null

& $Nssm install $Name $exe "run --config configs\camera01.yaml --log data\events.jsonl"
& $Nssm set $Name AppDirectory $Root
& $Nssm set $Name DisplayName "watchin-u lacing point monitor"
& $Nssm set $Name Start SERVICE_AUTO_START
& $Nssm set $Name AppStdout (Join-Path $Root "data\logs\service.log")
& $Nssm set $Name AppStderr (Join-Path $Root "data\logs\service.log")
& $Nssm set $Name AppRotateFiles 1
& $Nssm set $Name AppRotateBytes 10485760
& $Nssm set $Name AppExit Default Restart
& $Nssm set $Name AppRestartDelay 10000
& $Nssm start $Name
& $Nssm status $Name
