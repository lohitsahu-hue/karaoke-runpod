"""
RunPod Serverless Handler: YouTube → Demucs 4-stem + MDX Karaoke lead/backing split.

Pipeline:
  1. Download audio from YouTube via yt-dlp
  2. Demucs htdemucs → vocals, drums, bass, other
  3. MDX Karaoke 2 on vocals → lead_vocals, backing_vocals
  4. Encode all 5 stems as mono OGG

Returns 5 OGG stems: lead_vocals, backing_vocals, drums, bass, other
"""

import runpod
import subprocess
import os
import base64
import tempfile
import traceback


MODEL_DIR = "/app/models"


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


def separate_stems(audio_path, work_dir, model="htdemucs"):
    """Run Demucs full 4-stem separation (vocals, drums, bass, other)."""
    out_dir = os.path.join(work_dir, "stems")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python3", "-m", "demucs",
        "-n", model,
        "--out", out_dir,
        audio_path,
    ]

    print("[demucs] Separating 4 stems (vocals, drums, bass, other)...")
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

    print(f"[demucs] Done → {list(stems.keys())}")
    return stems


def mdx_karaoke_split(vocals_path, work_dir):
    """Split vocals into lead and backing using MDX Karaoke 2 model."""
    from audio_separator.separator import Separator

    split_dir = os.path.join(work_dir, "vocal_split")
    os.makedirs(split_dir, exist_ok=True)

    print("[mdx-karaoke] Splitting vocals into lead/backing...")
    sep = Separator(output_dir=split_dir, model_file_dir=MODEL_DIR)
    sep.load_model("UVR_MDXNET_KARA_2.onnx")
    output_files = sep.separate(vocals_path)

    lead_path = None
    backing_path = None
    for f in output_files:
        full = os.path.join(split_dir, f)
        if "Vocal" in f and "Inst" not in f:
            lead_path = full
        elif "Inst" in f:
            backing_path = full

    if not lead_path or not backing_path:
        raise RuntimeError(f"MDX Karaoke split failed, got: {output_files}")

    print(f"[mdx-karaoke] Done → lead: {os.path.basename(lead_path)}, "
          f"backing: {os.path.basename(backing_path)}")
    return {"lead_vocals": lead_path, "backing_vocals": backing_path}


def mid_side_split(vocals_path, work_dir):
    """Fallback: split vocals via mid-side if MDX Karaoke fails."""
    split_dir = os.path.join(work_dir, "vocal_split")
    os.makedirs(split_dir, exist_ok=True)

    lead_path = os.path.join(split_dir, "lead_vocals.wav")
    backing_path = os.path.join(split_dir, "backing_vocals.wav")

    cmd_mid = [
        "ffmpeg", "-y", "-i", vocals_path,
        "-af", "pan=mono|c0=0.5*c0+0.5*c1", lead_path,
    ]
    cmd_side = [
        "ffmpeg", "-y", "-i", vocals_path,
        "-af", "pan=mono|c0=0.5*c0-0.5*c1", backing_path,
    ]

    print("[mid-side] Fallback: extracting lead vocals (center)...")
    r1 = subprocess.run(cmd_mid, capture_output=True, text=True, timeout=120)
    if r1.returncode != 0:
        raise RuntimeError(f"ffmpeg mid failed: {r1.stderr[-300:]}")

    print("[mid-side] Fallback: extracting backing vocals (sides)...")
    r2 = subprocess.run(cmd_side, capture_output=True, text=True, timeout=120)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg side failed: {r2.stderr[-300:]}")

    return {"lead_vocals": lead_path, "backing_vocals": backing_path}


def wav_to_ogg(wav_path, ogg_path):
    """Convert WAV to mono OGG Vorbis at quality 3 (~112kbps) for small transfer."""
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ac", "1",
        "-c:a", "libvorbis",
        "-q:a", "3",
        ogg_path,
    ]
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


def handler(job):
    """RunPod serverless handler."""
    job_input = job["input"]
    youtube_id = job_input.get("youtube_id")
    job_id = job_input.get("job_id", "unknown")
    model = job_input.get("model", "htdemucs")

    if not youtube_id:
        return {"error": "youtube_id is required"}

    print(f"[handler] Job {job_id}: processing {youtube_id} with model {model}")
    work_dir = tempfile.mkdtemp(prefix=f"karaoke_{job_id}_")

    try:
        # 1. Download from YouTube
        audio_path = download_youtube(youtube_id, work_dir)
        print(f"[handler] Download complete: {audio_path}")

        # 2. Full 4-stem separation
        stems = separate_stems(audio_path, work_dir, model)

        # 3. Split vocals into lead + backing
        if "vocals" in stems:
            try:
                vocal_splits = mdx_karaoke_split(stems["vocals"], work_dir)
            except Exception as e:
                print(f"[handler] MDX Karaoke failed ({e}), falling back to mid-side")
                vocal_splits = mid_side_split(stems["vocals"], work_dir)
            del stems["vocals"]
            stems["lead_vocals"] = vocal_splits["lead_vocals"]
            stems["backing_vocals"] = vocal_splits["backing_vocals"]

        # 4. Compress to mono OGG + base64 encode
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
