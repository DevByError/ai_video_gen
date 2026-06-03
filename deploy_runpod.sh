#!/bin/bash

echo "========================================="
echo "AI Video Generator - RunPod Deployment"
echo "========================================="

# Set working directory
cd /workspace

# Update system
echo "Installing system dependencies..."
apt-get update
apt-get install -y ffmpeg libsndfile1 git wget curl

# Create necessary directories
mkdir -p /workspace/ai_video_gen/assets
mkdir -p /workspace/ai_video_gen/output

# Clone or copy the repository
if [ -d "/workspace/ai_video_gen/.git" ]; then
    echo "Repository already exists, pulling latest..."
    cd /workspace/ai_video_gen
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/DevByError/ai_video_gen.git /workspace/ai_video_gen
fi

cd /workspace/ai_video_gen

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download external repositories and models
echo "Downloading Wav2Lip, GFPGAN, and SadTalker..."
bash download_models.sh

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True)"

# Preload TTS model
echo "Preloading TTS model..."
python -c "from TTS.api import TTS; TTS('tts_models/en/ljspeech/tacotron2-DDC')"

# Check if base video exists
if [ ! -f "/workspace/ai_video_gen/assets/base_presenter.mp4" ]; then
    echo "WARNING: base_presenter.mp4 not found in assets folder!"
    echo "Please upload your base video to: /workspace/ai_video_gen/assets/"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "Starting API server..."
echo "========================================="

# Start the uvicorn server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1