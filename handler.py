"""
RunPod Serverless Handler: YouTube -> Demucs stem separation.
Returns stems as base64-encoded OGG files (small enough for response body).
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
    cmd = ["yt-dlp", "--js-runtimes", "nodejs", "-x", "--audio-format", "wav",
           "--audio-quality", "0", "--no-playlist", "-o", out_path, url]
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
    cmd = ["python3", "-m", "demucs", "-n", model, "--out", out_dir,
           "--two-stems", "vocals", audio_path]
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

def wav_to_ogg(wav_path, ogg_path):
    """Convert WAV to OGG Vorbis using ffmpeg (much smaller)."""
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libvorbis", "-q:a", "6", ogg_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")
    return ogg_path

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
        stems = separate_stems(audio_path, work_dir, model)
        encoded = {}
        for name, wav_path in stems.items():
            ogg_path = wav_path.replace(".wav", ".ogg")
            wav_to_ogg(wav_path, ogg_path)
            ogg_size = os.path.getsize(ogg_path)
            with open(ogg_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            encoded[name] = {
                "base64": b64,
                "size_bytes": ogg_size,
                "format": "ogg",
                "filename": f"{name}.ogg",
            }
            print(f"[encode] {name}: WAV {os.path.getsize(wav_path)/1024/1024:.1f}MB -> OGG {ogg_size/1024/1024:.1f}MB")
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
