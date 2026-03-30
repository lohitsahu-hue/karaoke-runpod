"""
RunPod Serverless Handler v6: Ensemble Pipeline for Karaoke Stem Separation

Pipeline architecture (based on audio research best practices):
  Phase 1: Mel-Band RoFormer (via audio-separator) -> ultra-clean vocals + instrumental
           SDR ~11.6 dB for vocals -- massive upgrade over Demucs alone (~8.9 dB)
  Phase 2: Demucs htdemucs on the instrumental -> drums, bass, other
           Demucs excels at transient-heavy percussion when vocals don't confuse it
  Phase 3: Mid-side decomposition on vocals -> lead_vocals + backing_vocals
           Center-panned lead (L+R)/2 vs side-panned harmonies (L-R)/2

Returns 5 OGG stems: lead_vocals, backing_vocals, drums, bass, other
All stems encoded as mono OGG quality 3 (~112kbps) to stay under 20 MB limit.
"""

import runpod
import subprocess
import os
import base64
import tempfile
import traceback


def download_youtube(youtube_id, work_dir):
    """Download audio from YouTube using yt-dlp."""
    out_path = os.path.join(work_dir, "audio.wav")
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp", "--js-runtimes", "node",
        "-x", "--audio-format", "wav", "--audio-quality", "0",
        "--no-playlist", "-o", out_path, url,
    ]
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

def extract_vocals_roformer(audio_path, work_dir, timeout=300):
    """Phase 1: Use audio-separator with Mel-Band RoFormer for vocal extraction.
    Runs in a subprocess with timeout to prevent hangs from ONNX runtime."""
    out_dir = os.path.join(work_dir, "roformer")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Phase 1] Mel-Band RoFormer: extracting vocals (timeout={timeout}s)...")

    # Run separation in a subprocess to enforce timeout
    script = f"""
import sys, json
from audio_separator.separator import Separator
separator = Separator(output_dir="{out_dir}")
separator.load_model()
output_files = separator.separate("{audio_path}")
print(json.dumps(output_files))
"""
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"RoFormer subprocess failed: {result.stderr[-500:]}")

    # Parse output files from the last line of stdout
    import json as _json
    stdout_lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
    output_files = _json.loads(stdout_lines[-1])
    print(f"[Phase 1] RoFormer output: {output_files}")

    vocals_path = None
    instrumental_path = None
    for f in output_files:
        fl = f.lower()
        if "vocal" in fl and "no_vocal" not in fl and "instrumental" not in fl:
            vocals_path = f
        elif "instrumental" in fl or "no_vocal" in fl or "accompaniment" in fl:
            instrumental_path = f
    if not vocals_path or not instrumental_path:
        if len(output_files) >= 2:
            vocals_path = output_files[0]
            instrumental_path = output_files[1]
        else:
            raise RuntimeError(f"RoFormer unexpected output: {output_files}")
    print(f"[Phase 1] Vocals: {vocals_path}")
    print(f"[Phase 1] Instrumental: {instrumental_path}")
    return vocals_path, instrumental_path

def separate_instrumental_demucs(instrumental_path, work_dir, model="htdemucs"):
    """Phase 2: Run Demucs on vocal-free instrumental for drums, bass, other."""
    out_dir = os.path.join(work_dir, "demucs_stems")
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["python3", "-m", "demucs", "-n", model, "--out", out_dir, instrumental_path]
    print("[Phase 2] Demucs: separating instrumental into drums, bass, other...")
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
        if name in ("drums", "bass", "other"):
            stems[name] = os.path.join(stem_dir, f)
    print(f"[Phase 2] Demucs instrumental stems: {list(stems.keys())}")
    return stems

def mid_side_split(vocals_path, work_dir):
    """Phase 3: Split vocals into lead (center/mid) and backing (sides)."""
    split_dir = os.path.join(work_dir, "vocal_split")
    os.makedirs(split_dir, exist_ok=True)
    lead_path = os.path.join(split_dir, "lead_vocals.wav")
    backing_path = os.path.join(split_dir, "backing_vocals.wav")
    cmd_mid = ["ffmpeg", "-y", "-i", vocals_path, "-af", "pan=mono|c0=0.5*c0+0.5*c1", lead_path]
    cmd_side = ["ffmpeg", "-y", "-i", vocals_path, "-af", "pan=mono|c0=0.5*c0-0.5*c1", backing_path]
    print("[Phase 3] Extracting lead vocals (center)...")
    r1 = subprocess.run(cmd_mid, capture_output=True, text=True, timeout=120)
    if r1.returncode != 0:
        raise RuntimeError(f"ffmpeg mid failed: {r1.stderr[-300:]}")
    print("[Phase 3] Extracting backing vocals (sides)...")
    r2 = subprocess.run(cmd_side, capture_output=True, text=True, timeout=120)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg side failed: {r2.stderr[-300:]}")
    return {"lead_vocals": lead_path, "backing_vocals": backing_path}

def wav_to_ogg(wav_path, ogg_path):
    """Convert WAV to mono OGG Vorbis at quality 3 (~112kbps)."""
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-ac", "1", "-c:a", "libvorbis", "-q:a", "3", ogg_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg ogg failed: {result.stderr[-300:]}")
    return ogg_path

def encode_stems_ogg(stems, work_dir):
    """Convert all WAV stems to OGG and base64-encode for transport."""
    ogg_dir = os.path.join(work_dir, "ogg")
    os.makedirs(ogg_dir, exist_ok=True)
    encoded = {}
    total_bytes = 0
    for name, wav_path in stems.items():
        ogg_path = os.path.join(ogg_dir, f"{name}.ogg")
        wav_to_ogg(wav_path, ogg_path)
        with open(ogg_path, "rb") as f:
            data = f.read()
        encoded[name] = {
            "base64": base64.b64encode(data).decode("utf-8"),
            "size_bytes": len(data),
            "format": "ogg",
        }
        total_bytes += len(data)
        print(f"[encode] {name}: {len(data) / 1024 / 1024:.1f} MB (OGG mono q3)")
    estimated_payload = total_bytes * 4 / 3
    print(f"[encode] Total raw: {total_bytes / 1024 / 1024:.1f} MB, "
          f"estimated base64 payload: {estimated_payload / 1024 / 1024:.1f} MB")
    return encoded

def fallback_demucs_pipeline(audio_path, work_dir, model="htdemucs"):
    """Fallback: original Demucs-only pipeline if RoFormer OOMs."""
    print("[FALLBACK] Running Demucs-only pipeline...")
    out_dir = os.path.join(work_dir, "fallback_stems")
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["python3", "-m", "demucs", "-n", model, "--out", out_dir, audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs fallback failed: {result.stderr[-500:]}")
    model_dir = os.path.join(out_dir, model)
    song_dirs = os.listdir(model_dir)
    stem_dir = os.path.join(model_dir, song_dirs[0])
    stems = {}
    for f in os.listdir(stem_dir):
        name = os.path.splitext(f)[0]
        stems[name] = os.path.join(stem_dir, f)
    if "vocals" in stems:
        vocal_splits = mid_side_split(stems["vocals"], work_dir)
        del stems["vocals"]
        stems["lead_vocals"] = vocal_splits["lead_vocals"]
        stems["backing_vocals"] = vocal_splits["backing_vocals"]
    return stems

def handler(job):
    """RunPod serverless handler -- Ensemble Pipeline v6."""
    job_input = job["input"]
    youtube_id = job_input.get("youtube_id")
    job_id = job_input.get("job_id", "unknown")
    if not youtube_id:
        return {"error": "youtube_id is required"}
    print(f"[handler] Job {job_id}: processing {youtube_id} (ensemble pipeline v6)")
    work_dir = tempfile.mkdtemp(prefix=f"karaoke_{job_id}_")
    try:
        audio_path = download_youtube(youtube_id, work_dir)
        print(f"[handler] Download complete: {audio_path}")
        try:
            vocals_path, instrumental_path = extract_vocals_roformer(audio_path, work_dir)
            stems = separate_instrumental_demucs(instrumental_path, work_dir)
            vocal_splits = mid_side_split(vocals_path, work_dir)
            stems["lead_vocals"] = vocal_splits["lead_vocals"]
            stems["backing_vocals"] = vocal_splits["backing_vocals"]
            print(f"[handler] Ensemble pipeline complete: {list(stems.keys())}")
        except Exception as e:
            print(f"[handler] Ensemble failed ({e}), falling back to Demucs-only...")
            traceback.print_exc()
            stems = fallback_demucs_pipeline(audio_path, work_dir)
        encoded = encode_stems_ogg(stems, work_dir)
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
