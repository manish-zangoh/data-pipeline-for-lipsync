import os
import json
from pathlib import Path
from xml.parsers.expat import model
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import subprocess
from faster_whisper import WhisperModel
RAW_DIR = "dataset/raw"
CLIP_DIR = "dataset/processed/clips"
MANIFEST_PATH = "dataset/manifests/clips.jsonl"

MIN_DURATION = 1.0   # seconds
MAX_DURATION = 20.0  # seconds


def detect_scenes(video_path):
    video = open_video(video_path,backend='pyav')
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    return scene_list


def save_clip(video_path, start, end, clip_id):
    output_dir = os.path.join(CLIP_DIR, clip_id)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "video.mp4")
    audio_output_path = os.path.join(output_dir,  "audio.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ss", str(start),
        "-to", str(end),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p", # 🔥 Vital for QuickTime and OpenCV
        "-profile:v", "baseline", # 🔥 Maximum compatibility
        "-level", "3.0",
        "-c:a", "aac",
        output_path
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run(["ffmpeg", "-y", "-i", output_path, "-vn", "-acodec", "pcm_s16le", audio_output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_path

def extract_metadata(video_path):
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


    data =  json.loads(result.stdout)
    video_stream = None
    has_audio = False
    for stream in data["streams"]:
        if stream["codec_type"] == "video":
            video_stream = stream
        elif stream["codec_type"] == "audio":
            has_audio = True
    
    if video_stream is None:
        return None
    
    fps = eval(video_stream.get("r_frame_rate", "0/1"))
    width = video_stream.get("width")
    height = video_stream.get("height")
    duration = float(data["format"].get("duration", 0))

    return {
        "fps": fps,
        "resolution": [width, height],
        "has_audio": has_audio,
        "duration": duration
    }
whisper_model = WhisperModel("base",compute_type="int8")
def detect_language(audio_path):
    try:
        segments, info = whisper_model.transcribe(audio_path, beam_size=1)
        return info.language
    except Exception as e:
        print(f"Language detection failed: {e}")
        return None

def main():
    Path(CLIP_DIR).mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            for line in f:
                existing_ids.add(json.loads(line)["clip_id"])

    with open(MANIFEST_PATH, "a") as manifest:

        for file in os.listdir(RAW_DIR):
            video_path = os.path.join(RAW_DIR, file)
            video_id = file.split(".")[0]

            print(f"Processing {video_id}")

            scenes = detect_scenes(video_path)
            is_fallback = False

            if not scenes:
                is_fallback = True
                # If no scenes detected, treat the whole video as one scene
                metadata_full = extract_metadata(video_path)
                if metadata_full:
                    scenes = [(0.0, metadata_full["duration"])]
                else:
                    print(f"⚠️ Could not extract metadata for {video_id}, skipping.")
                    continue

            for scene_tuple in scenes:
                start_time, end_time = scene_tuple
                if hasattr(start_time, "get_seconds"):
                    start = start_time.get_seconds()
                    end = end_time.get_seconds()
                else:
                    start = float(start_time)
                    end = float(end_time)

                duration = end - start

                # Bypass MAX_DURATION filter if this is a fallback for a video with no detected scenes
                if not is_fallback:
                    if duration < MIN_DURATION or duration > MAX_DURATION:
                        continue
                elif duration < MIN_DURATION:
                    continue

                clip_id = f"{video_id}_{int(start*1000)}_{int(end*1000)}"

                if clip_id in existing_ids:
                    continue

                video_out = save_clip(video_path, start, end, clip_id)
                audio_path = video_out.replace("video.mp4", "audio.wav")
                language = detect_language(audio_path)
                metadata = extract_metadata(video_out)

                if metadata is None:
                    continue

                record = {
                    "clip_id": clip_id,
                    "video_id": video_id,
                    "source_video": video_path,

                    "start": start,
                    "end": end,
                    "duration": duration,

                    "fps": metadata["fps"],
                    "resolution": metadata["resolution"],
                    "has_audio": metadata["has_audio"],

                    "language": language,
                    "speaker_count": None,
                    "shot_type": None,

                    "path": video_out
                }

                manifest.write(json.dumps(record) + "\n")

                print(f"✅ Clip created: {clip_id}")


if __name__ == "__main__":
    main()