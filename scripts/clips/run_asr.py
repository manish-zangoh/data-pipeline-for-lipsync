import os
import json
from faster_whisper import WhisperModel

CLIPS_MANIFEST = "dataset/manifests/filtered_clips.jsonl"
OUTPUT_DIR = "dataset/sidecars/transcripts"
REJECTED_OUT = "dataset/manifests/rejected_asr.jsonl"

RAW_DIR = "dataset/processed/clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = WhisperModel("medium", compute_type="int8")

def process_clip(clip, rejected_file):
    clip_id = clip["clip_id"]
    clip_folder = os.path.join(RAW_DIR, clip_id)
    audio_path = os.path.join(clip_folder, "audio.wav")

    print(f"Processing ASR for {clip_id}")

    if not os.path.exists(audio_path):
        rejected_file.write(json.dumps({
            "clip_id": clip_id,
            "reason": "missing_audio"
        }) + "\n")
        return

    try:
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,  # 🔥 IMPORTANT
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True
        )
    except Exception as e:
        rejected_file.write(json.dumps({
            "clip_id": clip_id,
            "reason": "asr_failure"
        }) + "\n")
        return

    full_text = []
    seg_list = []
    word_list = []

    for seg in segments:
        text = seg.text.strip()
        if text:
            full_text.append(text)

        seg_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": text
        })

        if seg.words:
            for w in seg.words:
                word = w.word.strip()
                if word:
                    word_list.append({
                        "word": word,
                        "start": w.start,
                        "end": w.end
                    })

    final_text = " ".join(full_text).strip()

    clip_duration = clip.get("duration", 0)

    speech_duration = sum([w["end"] - w["start"] for w in word_list])

    if clip_duration > 0:
        speech_ratio = speech_duration / clip_duration
    else:
        speech_ratio = 0

    result = {
        "clip_id": clip_id,
        "language": info.language,
        "text": final_text,
        "segments": seg_list,
        "words": word_list,
        "speech_ratio": speech_ratio
    }

    with open(os.path.join(OUTPUT_DIR, f"{clip_id}.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ ASR done: {clip_id}")

def main():
    with open(CLIPS_MANIFEST, "r") as f, open(REJECTED_OUT, "w") as rj:

        for line in f:
            clip = json.loads(line)
            process_clip(clip, rj)


if __name__ == "__main__":
    main()