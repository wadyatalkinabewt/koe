$file = "c:\dev\koe\opus_chat.md"
$hash = (Get-FileHash $file).Hash
Write-Host "Monitoring opus_chat.md for changes..."
Write-Host "Initial hash: $hash"

for ($i = 0; $i -lt 100; $i++) {
    Start-Sleep -Seconds 3
    $newHash = (Get-FileHash $file).Hash
    if ($newHash -ne $hash) {
        Write-Host "`nFILE CHANGED!"
        Get-Content $file -Tail 30
        exit 0
    }
    Write-Host "." -NoNewline
}
Write-Host "`nTimeout after 5 minutes"
