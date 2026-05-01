import subprocess
import os

def run_script(script_path):
    print(f"🚀 Running {script_path}...")
    result = subprocess.run(["python3", script_path])
    if result.returncode != 0:
        print(f"❌ Error running {script_path}")
        return False
    return True

def main():
    scripts = [
        "scripts/ingest/ingest_vidoes.py",
        "scripts/clips/split_shots.py",
        "scripts/tracking/face_tracking.py",
        "scripts/clips/filter_clips.py",
        "scripts/clips/run_asr.py",
        "scripts/clips/get_phoneme.py",
        "scripts/clips/score_alignment.py"
    ]

    for script in scripts:
        if not run_script(script):
            break
    else:
        print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()