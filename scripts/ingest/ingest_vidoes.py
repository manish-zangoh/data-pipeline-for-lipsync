import os
import json
import subprocess
import uuid
from pathlib import Path
from datetime import datetime
import shutil

INPUT_DIR = "dataset/input_videos"        # your source videos
OUTPUT_DIR = "dataset/raw"
MANIFEST_PATH = "dataset/manifests/videos.jsonl"
LOG_PATH = "dataset/logs/ingest.log"

def run_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None

    return json.loads(result.stdout)


def extract_metadata(ffprobe_data):
    video_stream = None

    for stream in ffprobe_data["streams"]:
        if stream["codec_type"] == "video":
            video_stream = stream
            break

    if video_stream is None:
        return None

    fps = eval(video_stream.get("r_frame_rate", "0/1"))
    width = video_stream.get("width")
    height = video_stream.get("height")

    duration = float(ffprobe_data["format"].get("duration", 0))

    return {
        "fps": fps,
        "resolution": [width, height],
        "duration": duration
    }

def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.now()} - {msg}\n")

def has_audio(ffprobe_data):
    for stream in ffprobe_data["streams"]:
        if stream["codec_type"] == "audio":
            return True
    return False

import hashlib

def compute_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:12]

def load_existing_ids(manifest_path):
    existing_ids = set()
    if not os.path.exists(manifest_path):
        return existing_ids

    with open(manifest_path, "r") as f:
        for line in f:
            data = json.loads(line)
            existing_ids.add(data["video_id"])

    return existing_ids

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(MANIFEST_PATH)

    with open(MANIFEST_PATH, "a") as manifest:

        for file in os.listdir(INPUT_DIR):
            if not file.endswith((".mp4", ".mov", ".mkv")):
                continue

            input_path = os.path.join(INPUT_DIR, file)
            video_id = f"vid_{compute_hash(input_path)}"

            if video_id in existing_ids:
                print(f"Skipping existing: {file}")
                continue

            input_path = os.path.join(INPUT_DIR, file)

            print(f"Processing: {file}")

            ffprobe_data = run_ffprobe(input_path)
            bool_has_audio = has_audio(ffprobe_data)
            if ffprobe_data is None:
                log(f"FFPROBE FAILED: {file}")
                continue

            metadata = extract_metadata(ffprobe_data)

            if metadata is None:
                log(f"NO VIDEO STREAM: {file}")
                continue

            video_id = f"vid_{compute_hash(input_path)}"
            output_path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")

            shutil.copy(input_path, output_path)

            record = {
                "video_id": video_id,
                "original_name": file,
                "path": output_path,
                "fps": metadata["fps"],
                "resolution": metadata["resolution"],
                "duration": metadata["duration"],
                "has_audio": bool_has_audio,
                "ingested_at": datetime.now().isoformat()
            }
            manifest.write(json.dumps(record) + "\n")
            print(f"INGESTED: {video_id} -> {output_path}")

if __name__ == "__main__":
    main()