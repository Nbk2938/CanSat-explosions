# Download raw video files from the GitHub release and place them in
# Data/Explosion Videos/ as expected by ground_truth_extraction.py.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File download_videos.ps1

$release = "https://github.com/Nbk2938/CanSat-explosions/releases/download/v1.0-data"
$dest    = "Data\Explosion Videos"

New-Item -ItemType Directory -Force -Path $dest | Out-Null

Write-Host "Downloading video files into '$dest'..."

$files = @(
    @{ remote = "20260205_140018000_iOS.MOV";      local = "20260205_140018000_iOS.MOV" },
    @{ remote = "20260205_150632000_iOS.MP4";      local = "20260205_150632000_iOS.MP4" },
    @{ remote = "TimeVideo_20260205_150217.3.mp4"; local = "TimeVideo_20260205_150217~3.mp4" }
)

foreach ($f in $files) {
    $url  = "$release/$($f.remote)"
    $out  = Join-Path $dest $f.local
    Write-Host "  $($f.remote) ..."
    Invoke-WebRequest -Uri $url -OutFile $out
}

Write-Host ""
Write-Host "Done. Files in '$dest':"
Get-ChildItem $dest | Format-Table Name, Length
