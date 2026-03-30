"""
RunPod Serverless Handler: YouTube -> Demucs stem separation pipeline.
Returns presigned URLs to stem files via RunPod's built-in upload.
"""
import runpod
import subprocess
import os
import tempfile
import traceback
import requests
import time

PRESIGNED_URL_ENDPOINT = "https://api.runpod.ai/v2/{}/upload/url"

def download_youtube(youtube_id, work_dir):
    out_path = os.path.join(work_dir, "audio.wav")
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0", "--no-playlist", "-o", out_path, url]
    print(f"[yt-dlp] Downloading {youtube_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-500:]}")
    if os.path.exists(out_path):
        return out_path
    for f in os.listdir(work_dir):
        if f.startswith("audio"):
            return os.path.join(work_dir, f)
    raise RuntimeError("yt-dlp: output file not found")

def separate_stems(audio_path, work_dir, model="htdemucs"):
    out_dir = os.path.join(work_dir, "stems")
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["python3", "-m", "demucs", "-n", model, "--out", out_dir, "--two-stems", "vocals", audio_path]
    print(f"[demucs] Separating stems...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[-500:]}")
    model_dir = os.path.join(out_dir, model)
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Demucs output not found at {model_dir}")
    song_dirs = os.listdir(model_dir)
    if not song_dirs:
        raise RuntimeError("Demucs produced no output")
    stem_dir = os.path.join(model_dir, song_dirs[0])
    stems = {}
    for f in os.listdir(stem_dir):
        name = os.path.splitext(f)[0]
        stems[name] = os.path.join(stem_dir, f)
    print(f"[demucs] Done -> {list(stems.keys())}")
    return stems

def upload_file(filepath, filename):
    """Upload a file using transfer.sh and return the download URL."""
    print(f"[upload] Uploading {filename} ({os.path.getsize(filepath)/1024/1024:.1f} MB)...")
    with open(filepath, "rb") as f:
        resp = requests.put(
            f"https://transfer.sh/{filename}",
            data=f,
            headers={"Max-Days": "1"}
        )
    if resp.status_code == 200:
        url = resp.text.strip()
        print(f"[upload] {filename} -> {url}")
        return url
    # Fallback: use file.io
    with open(filepath, "rb") as f:
        resp = requests.post("https://file.io", files={"file": (filename, f)})
    if resp.status_code == 200:
        url = resp.json().get("link", "")
        print(f"[upload] {filename} -> {url}")
        return url
    raise RuntimeError(f"Upload failed for {filename}: {resp.status_code} {resp.text[:200]}")

def handler(job):
    job_input = job["input"]
    youtube_id = job_input.get("youtube_id")
    job_id = job_input.get("job_id", "unknown")
    model = job_input.get("model", "htdemucs")
    if not youtube_id:
        return {"error": "youtube_id is required"}
    print(f"[handler] Job {job_id}: processing {youtube_id} with model {model}")
    work_dir = tempfile.mkdtemp(prefix=f"karaoke_{job_id}_")
    try:
        audio_path = download_youtube(youtube_id, work_dir)
        print(f"[handler] Download complete: {audio_path}")
        stems = separate_stems(audio_path, work_dir, model)
        # Upload each stem and collect URLs
        stem_urls = {}
        for name, filepath in stems.items():
            size = os.path.getsize(filepath)
            filename = f"{job_id}_{name}.wav"
            url = upload_file(filepath, filename)
            stem_urls[name] = {"url": url, "size_bytes": size, "filename": f"{name}.wav"}
        return {
            "job_id": job_id,
            "youtube_id": youtube_id,
            "stems": stem_urls,
            "stem_names": list(stem_urls.keys()),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "job_id": job_id}
    finally:
        subprocess.run(["rm", "-rf", work_dir], capture_output=True)

runpod.serverless.start({"handler": handler})
