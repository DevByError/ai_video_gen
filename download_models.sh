#!/bin/bash

echo "Downloading external repositories and models..."

cd /workspace/ai_video_gen

# Clone repositories if they don't exist
if [ ! -d "Wav2Lip" ]; then
    echo "Cloning Wav2Lip..."
    git clone https://github.com/Rudrabha/Wav2Lip.git
fi

if [ ! -d "GFPGAN" ]; then
    echo "Cloning GFPGAN..."
    git clone https://github.com/TencentARC/GFPGAN.git
fi

if [ ! -d "SadTalker" ]; then
    echo "Cloning SadTalker..."
    git clone https://github.com/OpenTalker/SadTalker.git
fi

# Install Wav2Lip dependencies
echo "Installing Wav2Lip dependencies..."
cd /workspace/ai_video_gen/Wav2Lip
pip install -r requirements.txt

# Download Wav2Lip model checkpoint
echo "Downloading Wav2Lip model..."
mkdir -p checkpoints
if [ ! -f "checkpoints/wav2lip_gan.pth" ]; then
    gdown --fuzzy https://drive.google.com/uc?id=1f4B5Yj6F3B5VKQn9Q5uY8G6KpQ7WlXV -O checkpoints/wav2lip_gan.pth
fi

# Install GFPGAN dependencies
echo "Installing GFPGAN dependencies..."
cd /workspace/ai_video_gen/GFPGAN
pip install -r requirements.txt

# Install SadTalker dependencies
echo "Installing SadTalker dependencies..."
cd /workspace/ai_video_gen/SadTalker
pip install -r requirements.txt

# Create inference_fixed.py for Wav2Lip (encoding fix)
echo "Creating inference_fixed.py for Wav2Lip..."
cd /workspace/ai_video_gen/Wav2Lip

cat > inference_fixed.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# Import original inference.py
import importlib.util
spec = importlib.util.spec_from_file_location("inference_original", "inference.py")
inference_original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_original)
EOF

echo "Download complete!"