FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install ffmpeg + nodejs (JS runtime for yt-dlp)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg nodejs && rm -rf /var/lib/apt/lists/*

# Install Python deps: runpod, yt-dlp, demucs, audio-separator (Mel-Band RoFormer)
# audio-separator needs onnxruntime-gpu for CUDA inference
RUN pip install --no-cache-dir runpod yt-dlp demucs audio-separator onnxruntime-gpu requests && pip cache purge

# Pre-download models so first request is fast
# 1. Demucs htdemucs model
RUN python3 -c "import torch; from demucs.pretrained import get_model; get_model('htdemucs')"
# 2. Mel-Band RoFormer default model (audio-separator downloads on first load)
RUN python3 -c "from audio_separator.separator import Separator; s = Separator(); s.load_model()"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]