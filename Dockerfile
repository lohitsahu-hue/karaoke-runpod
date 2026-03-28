FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN pip install --no-cache-dir runpod yt-dlp demucs && pip cache purge

RUN python3 -c "import torch; from demucs.pretrained import get_model; get_model('htdemucs')"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
