"""
RunPod Serverless Handler: YouTube -> Demucs stem separation pipeline.
"""
import runpod
import subprocess
import os
import base64
import tempfile
import traceback

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

def encode_stems(stems):
    encoded = {}
    for name, filepath in stems.items():
        with open(filepath, "rb") as f:
            data = f.read()
        encoded[name] = {
            "base64": base64.b64encode(data).decode("utf-8"),
            "size_bytes": len(data),
            "filename": f"{name}.wav",
        }
        print(f"[encode] {name}: {len(data) / 1024 / 1024:.1f} MB")
    return encoded

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
        encoded = encode_stems(stems)
        return {
            "job_id": job_id,
            "youtube_id": youtube_id,
            "stems": encoded,
            "stem_names": list(encoded.keys()),
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "job_id": job_id}
    finally:
        subprocess.run(["rm", "-rf", work_dir], capture_output=True)

runpod.serverless.start({"handler": handler})
