# Create Koe Remote Package
$ErrorActionPreference = "Stop"
Set-Location "c:\dev\koe"

# Create a temporary directory for the remote package
$tempDir = "koe-remote-package"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Core files
Copy-Item "run.py" "$tempDir/"
Copy-Item "requirements-remote.txt" "$tempDir/"
if (Test-Path "config.yaml") { Copy-Item "config.yaml" "$tempDir/" }
if (Test-Path ".env") { Copy-Item ".env" "$tempDir/" }

# Assets
New-Item -ItemType Directory -Path "$tempDir/assets" | Out-Null
Copy-Item "assets/koe-icon.ico" "$tempDir/assets/"
Copy-Item "assets/koe-icon.png" "$tempDir/assets/"
Copy-Item "assets/beep.wav" "$tempDir/assets/"

# Scripts (remote only)
New-Item -ItemType Directory -Path "$tempDir/scripts" | Out-Null
if (Test-Path "scripts/Start Koe Remote.bat") { Copy-Item "scripts/Start Koe Remote.bat" "$tempDir/scripts/" }
if (Test-Path "scripts/Start Koe Remote.vbs") { Copy-Item "scripts/Start Koe Remote.vbs" "$tempDir/scripts/" }
if (Test-Path "scripts/Start Scribe Remote.bat") { Copy-Item "scripts/Start Scribe Remote.bat" "$tempDir/scripts/" }
if (Test-Path "scripts/Start Scribe Remote.vbs") { Copy-Item "scripts/Start Scribe Remote.vbs" "$tempDir/scripts/" }
if (Test-Path "scripts/Stop Koe.bat") { Copy-Item "scripts/Stop Koe.bat" "$tempDir/scripts/" }
if (Test-Path "scripts/Stop Koe.vbs") { Copy-Item "scripts/Stop Koe.vbs" "$tempDir/scripts/" }
if (Test-Path "scripts/create_shortcuts.ps1") { Copy-Item "scripts/create_shortcuts.ps1" "$tempDir/scripts/" }

# Shortcuts (if they exist)
if (Test-Path "Start Koe Remote.lnk") { Copy-Item "Start Koe Remote.lnk" "$tempDir/" }
if (Test-Path "Start Scribe Remote.lnk") { Copy-Item "Start Scribe Remote.lnk" "$tempDir/" }
if (Test-Path "Stop Koe.lnk") { Copy-Item "Stop Koe.lnk" "$tempDir/" }

# Source files
New-Item -ItemType Directory -Path "$tempDir/src" | Out-Null
Copy-Item "src/main.py" "$tempDir/src/"
Copy-Item "src/transcription.py" "$tempDir/src/"
Copy-Item "src/result_thread.py" "$tempDir/src/"
Copy-Item "src/key_listener.py" "$tempDir/src/"
Copy-Item "src/utils.py" "$tempDir/src/"
Copy-Item "src/logger.py" "$tempDir/src/"
Copy-Item "src/config_schema.yaml" "$tempDir/src/"
Copy-Item "src/transcription_client.py" "$tempDir/src/"

# UI files
New-Item -ItemType Directory -Path "$tempDir/src/ui" | Out-Null
Copy-Item "src/ui/*.py" "$tempDir/src/ui/"

# Meeting files (for Scribe remote)
New-Item -ItemType Directory -Path "$tempDir/src/meeting" | Out-Null
Copy-Item "src/meeting/__init__.py" "$tempDir/src/meeting/"
Copy-Item "src/meeting/app.py" "$tempDir/src/meeting/"
Copy-Item "src/meeting/capture.py" "$tempDir/src/meeting/"
Copy-Item "src/meeting/processor.py" "$tempDir/src/meeting/"
Copy-Item "src/meeting/transcript.py" "$tempDir/src/meeting/"
if (Test-Path "src/meeting/summary_status.py") { Copy-Item "src/meeting/summary_status.py" "$tempDir/src/meeting/" }
if (Test-Path "src/meeting/summarizer.py") { Copy-Item "src/meeting/summarizer.py" "$tempDir/src/meeting/" }
if (Test-Path "src/meeting/summarize_detached.py") { Copy-Item "src/meeting/summarize_detached.py" "$tempDir/src/meeting/" }

# Create empty directories that might be needed
New-Item -ItemType Directory -Path "$tempDir/Snippets" | Out-Null
New-Item -ItemType Directory -Path "$tempDir/Meetings" | Out-Null
New-Item -ItemType Directory -Path "$tempDir/logs" | Out-Null

# Create the zip
$zipPath = "koe-remote.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath }
Compress-Archive -Path "$tempDir/*" -DestinationPath $zipPath

# Cleanup temp directory
Remove-Item -Recurse -Force $tempDir

# Show result
Write-Host "`nCreated: $zipPath"
Get-Item $zipPath | Select-Object Name, @{N='Size(MB)';E={[math]::Round($_.Length/1MB, 2)}}, LastWriteTime | Format-Table -AutoSize
