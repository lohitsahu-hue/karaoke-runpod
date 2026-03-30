FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install ffmpeg + unzip + nodejs (JS runtime for yt-dlp)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg unzip nodejs && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir runpod yt-dlp demucs requests && pip cache purge

# Pre-download the htdemucs model so first request is fast
RUN python3 -c "import torch; from demucs.pretrained import get_model; get_model('htdemucs')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
