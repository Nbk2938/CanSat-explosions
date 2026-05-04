#!/usr/bin/env bash
# Download raw video files from the GitHub release and place them in
# Data/Explosion Videos/ as expected by ground_truth_extraction.py.

set -e

RELEASE="https://github.com/Nbk2938/CanSat-explosions/releases/download/v1.0-data"
DEST="Data/Explosion Videos"

mkdir -p "$DEST"

echo "Downloading video files into '$DEST'..."

curl -L --progress-bar -o "$DEST/20260205_140018000_iOS.MOV" \
    "$RELEASE/20260205_140018000_iOS.MOV"

curl -L --progress-bar -o "$DEST/20260205_150632000_iOS.MP4" \
    "$RELEASE/20260205_150632000_iOS.MP4"

# GitHub converts ~ to . in asset filenames — download and rename
curl -L --progress-bar -o "$DEST/TimeVideo_20260205_150217~3.mp4" \
    "$RELEASE/TimeVideo_20260205_150217.3.mp4"

echo ""
echo "Done. Files in '$DEST':"
ls -lh "$DEST"
