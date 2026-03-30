FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install ffmpeg + unzip (needed for deno installer)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg unzip && rm -rf /var/lib/apt/lists/*

# Install deno (JS runtime needed by yt-dlp for YouTube extraction)
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_DIR="/root/.deno"
ENV PATH="/root/.deno/bin:${PATH}"

# Install Python deps
RUN pip install --no-cache-dir runpod yt-dlp demucs && pip cache purge

# Pre-download the htdemucs model so first request is fast
RUN python3 -c "import torch; from demucs.pretrained import get_model; get_model('htdemucs')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
