#!/bin/bash
echo "Installing dependencies..."
pip install numba numpy torch tqdm more-itertools tiktoken==0.3.3
pip install --no-deps whisper-at
pip install -r requirements.txt
echo "Installing ffmpeg..."
wget https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z -o ffmpeg.7z
7z x ffmpeg.7z
rm ffmpeg.7z

set PATH=%PATH%;ffmpeg-git-full/bin
echo "Running Whisper-AT..."
python whisper_at.py