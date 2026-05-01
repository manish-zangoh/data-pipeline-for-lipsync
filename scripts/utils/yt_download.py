import os
import subprocess
import shutil
import sys

OUTPUT_DIR = "dataset/input_videos"

URLS = [
    "https://youtu.be/6O0fySNw-Lw?si=mQqoftfXtotXU33x",
    # add more URLs
]


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("❌ ffmpeg not found. Install it first.")
        print("Mac: brew install ffmpeg")
        sys.exit(1)


def download_video(url):
    cmd = [
        "yt-dlp",

        # 🔥 Ensure BOTH video + audio are downloaded
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",

        # 🔥 Force merge into mp4
        "--merge-output-format", "mp4",

        # 🔥 Avoid re-downloading
        "--no-overwrites",

        # 🔥 Ensure ffmpeg is used
        "--postprocessor-args", "ffmpeg:-loglevel error",

        # 🔥 Fail if no audio
        "--abort-on-error",

        # 🔥 Optional: limit duration (increased to 10 mins)
        "--match-filter", "duration < 600",

        # 🔥 Clean naming
        "-o", f"{OUTPUT_DIR}/%(id)s.%(ext)s",

        # Debug (you can remove later)
        "-v",

            url
        ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if "skipping .." in result.stdout:
            print(f"⚠️  Skipped (did not match filters): {url}")
        else:
            print(f"✅ Downloaded successfully: {url}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {url}")
        print(e.stderr)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    check_ffmpeg()

    for url in URLS:
        download_video(url)


if __name__ == "__main__":
    main()