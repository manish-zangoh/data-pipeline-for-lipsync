import os
import json

TRANSCRIPT_DIR = "dataset/sidecars/transcripts"
PHONEME_DIR = "dataset/sidecars/phonemes"
MANIFEST_PATH = "dataset/manifests/filtered_clips.jsonl"
OUTPUT_PATH = "dataset/manifests/final_manifest.jsonl"
REJECTED_PATH = "dataset/manifests/rejected_alignment.jsonl"

def score_clip(clip_id, transcript, phonemes):
    speech_ratio = transcript.get("speech_ratio", 0)
    if speech_ratio < 0.2 or speech_ratio > 1.0:
        return 0.3, "suspicious_speech_ratio"

    num_phonemes = len(phonemes.get("phonemes", []))
    duration = transcript.get("words", [-1])[-1].get("end", 0) - transcript.get("words", [{}])[0].get("start", 0) if transcript.get("words") else 0
    
    if num_phonemes == 0:
        return 0.0, "no_phonemes"
        
    if duration > 0:
        density = num_phonemes / duration
        if density > 15: # Too many phonemes per second
            return 0.4, "high_phoneme_density"
        if density < 2: # Too few phonemes
            return 0.4, "low_phoneme_density"

    return 1.0, "ok"

def main():
    if not os.path.exists(MANIFEST_PATH):
        print("Filtered manifest not found.")
        return

    with open(MANIFEST_PATH, "r") as f, open(OUTPUT_PATH, "w") as fout, open(REJECTED_PATH, "w") as rj:
        for line in f:
            clip = json.loads(line)
            clip_id = clip["clip_id"]
            
            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{clip_id}.json")
            phoneme_path = os.path.join(PHONEME_DIR, f"{clip_id}.json")
            
            if not os.path.exists(transcript_path) or not os.path.exists(phoneme_path):
                rj.write(json.dumps({"clip_id": clip_id, "reason": "missing_sidecars"}) + "\n")
                continue
                
            with open(transcript_path, "r") as tf, open(phoneme_path, "r") as pf:
                transcript = json.load(tf)
                phonemes = json.load(pf)
                
            score, reason = score_clip(clip_id, transcript, phonemes)
            
            clip["alignment_score"] = score
            clip["alignment_note"] = reason
            
            if score < 0.5:
                rj.write(json.dumps(clip) + "\n")
                print(f"❌ Rejected {clip_id}: {reason}")
            else:
                fout.write(json.dumps(clip) + "\n")
                print(f"✅ Finalized {clip_id}")

if __name__ == "__main__":
    main()