import os
import json
import nltk
from g2p_en import G2p
from pathlib import Path

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

TRANSCRIPT_DIR = "dataset/sidecars/transcripts"
PHONEME_DIR = "dataset/sidecars/phonemes"
MANIFEST_PATH = "dataset/manifests/filtered_clips.jsonl"

os.makedirs(PHONEME_DIR, exist_ok=True)

g2p = G2p()

def get_phonemes_for_word(word):
    # g2p returns a list of phonemes for a string
    phonemes = g2p(word)
    # Remove spaces and filter out non-phoneme characters if necessary
    return [p for p in phonemes if p.strip()]

def process_clip(clip_id):
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{clip_id}.json")
    if not os.path.exists(transcript_path):
        return

    with open(transcript_path, "r") as f:
        data = json.load(f)

    words = data.get("words", [])
    phoneme_spans = []

    for w in words:
        word_text = w["word"].strip(".,!?\"")
        if not word_text:
            continue
            
        word_phonemes = get_phonemes_for_word(word_text)
        if not word_phonemes:
            continue

        start = w["start"]
        end = w["end"]
        duration = end - start
        
        ph_duration = duration / len(word_phonemes)
        
        for i, ph in enumerate(word_phonemes):
            phoneme_spans.append({
                "phoneme": ph,
                "start": start + i * ph_duration,
                "end": start + (i + 1) * ph_duration
            })

    result = {
        "clip_id": clip_id,
        "phonemes": phoneme_spans
    }

    output_path = os.path.join(PHONEME_DIR, f"{clip_id}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r") as f:
        for line in f:
            clip = json.loads(line)
            clip_id = clip["clip_id"]
            print(f"Processing phonemes for {clip_id}")
            process_clip(clip_id)

if __name__ == "__main__":
    main()