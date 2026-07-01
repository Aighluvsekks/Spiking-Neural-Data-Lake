# record_and_run.ps1 — record 3 gestures off the ESP32, replay them, then run the live loop.
# Windows PowerShell. Run from the repo root with the ESP32 connected.
# Usage:  .\record_and_run.ps1 -Port COM8            (defaults: COM8, 115200, 40s/gesture)
#         .\record_and_run.ps1 -Port COM5 -Duration 60
# Find your port:  [System.IO.Ports.SerialPort]::getportnames()
param(
    [string]$Port = "COM8",
    [int]$Baud = 115200,
    [int]$Duration = 40          # seconds recorded per gesture
)
$ErrorActionPreference = "Stop"
$logger   = "robot arm/sensor_reading.py"
$gestures = "IDLE", "HAND_APPROACH", "HAND_RETREAT"

# serial dep (no-op if already present)
python -m pip install --quiet pyserial
New-Item -ItemType Directory -Force captures | Out-Null

# 1. record each gesture for $Duration seconds
foreach ($g in $gestures) {
    Write-Host "`n=== $g  ($Duration s on $Port) ===" -ForegroundColor Cyan
    Read-Host "Press Enter, then do the '$g' gesture repeatedly for $Duration s"
    $p = Start-Process python -ArgumentList "`"$logger`"", $g, $Port -PassThru -NoNewWindow
    Start-Sleep -Seconds $Duration
    Stop-Process -Id $p.Id -Force
    Start-Sleep -Milliseconds 300
    if (Test-Path "$g.csv") {
        Move-Item -Force "$g.csv" "captures\$g.csv"
        $n = (Get-Content "captures\$g.csv").Count - 1
        Write-Host "saved captures\$g.csv  ($n rows)" -ForegroundColor Green
    } else {
        Write-Host "no $g.csv written -- check the port/connection" -ForegroundColor Red
    }
}

# 2. replay each capture through the full loop (hardware-free) — see the recognized command
foreach ($g in $gestures) {
    if (Test-Path "captures\$g.csv") {
        Write-Host "`n--- replay captures\$g.csv ---" -ForegroundColor Cyan
        python live_arm.py --csv "captures\$g.csv"
    }
}

# 3. live loop off the sensor (Ctrl-C to stop)
Write-Host "`n=== LIVE loop on $Port (Ctrl-C to stop) ===" -ForegroundColor Cyan
Read-Host "Press Enter to start the live loop"
python live_arm.py --serial $Port --baud $Baud
