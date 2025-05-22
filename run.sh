#!/bin/bash
echo "Installing dependencies..."
pip install --no-deps whisper-at
pip install -r requirements.txt


if [ ! -d ffmpeg-7.1.1-essentials_build ]; then
    echo "downloading ffmpeg..."
    curl -L -o ffmpeg.zip https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
    unzip ffmpeg.zip
    rm ffmpeg.zip
else
    echo "ffmpeg already exists, skipping download."
fi



export PATH="$PATH:$(pwd)/ffmpeg-7.1.1-essentials_build/bin"
echo "Running Whisper-AT..."
python at_whisper.py