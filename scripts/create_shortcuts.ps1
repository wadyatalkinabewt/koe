# Create shortcuts with custom Koe icon for all VBS launchers
# Script is in scripts/ folder, so go up one level to get Koe root
$scriptsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$koeRoot = Split-Path -Parent $scriptsDir
$iconPath = Join-Path $koeRoot "assets\koe-icon.ico"

$vbsFiles = @(
    "Start Koe Desktop.vbs",
    "Start Koe Remote.vbs",
    "Start Scribe Desktop.vbs",
    "Start Scribe Remote.vbs",
    "Stop Koe.vbs"
)

$shell = New-Object -ComObject WScript.Shell

foreach ($vbsFile in $vbsFiles) {
    $vbsPath = Join-Path $scriptsDir $vbsFile
    if (Test-Path $vbsPath) {
        $shortcutName = [System.IO.Path]::GetFileNameWithoutExtension($vbsFile) + ".lnk"
        $shortcutPath = Join-Path $koeRoot $shortcutName

        $shortcut = $shell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $vbsPath
        $shortcut.WorkingDirectory = $koeRoot
        $shortcut.IconLocation = "$iconPath,0"
        $shortcut.Save()

        Write-Host "Created: $shortcutName"
    }
}

Write-Host "`nDone! Shortcuts created with Koe icon."
