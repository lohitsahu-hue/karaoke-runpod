FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install ffmpeg + unzip + nodejs (JS runtime for yt-dlp)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg unzip nodejs && rm -rf /var/lib/apt/lists/*

# Install Python deps (audio-separator for MDX Karaoke vocal split)
RUN pip install --no-cache-dir runpod yt-dlp demucs requests audio-separator==0.18.0 onnxruntime && pip cache purge

# Pre-download the htdemucs model so first request is fast
RUN python3 -c "import torch; from demucs.pretrained import get_model; get_model('htdemucs')"

# Pre-download the MDX Karaoke 2 model (53MB) for lead/backing vocal split
RUN python3 -c "\
from audio_separator.separator import Separator; \
sep = Separator(model_file_dir='/app/models'); \
sep.load_model('UVR_MDXNET_KARA_2.onnx'); \
print('MDX Karaoke 2 model cached')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
