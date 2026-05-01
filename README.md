# Lipsync Data Pipeline

A professional-grade pipeline for ingesting, processing, and validating talking-head video datasets for lipsync model training (e.g., Wav2Lip, LivePortrait).

## 🚀 Overview

This pipeline automates the transformation of raw videos into high-quality, training-ready clips. It handles scene detection, face tracking, pose estimation, ASR (Automatic Speech Recognition), phoneme extraction, and alignment quality scoring.

## 🛠️ Prerequisites

- **Python 3.10+**
- **FFmpeg**: Required for video/audio processing.
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
- **yt-dlp**: Required for downloading videos.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd lipsync
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Directory Structure

```text
/
├── dataset/
│   ├── input_videos/     # Put your raw .mp4 files here
│   ├── raw/              # Ingested videos with unique IDs
│   ├── processed/
│   │   ├── clips/        # Individual shot-level clips
│   │   └── debug/        # Visualizations (face mesh overlays)
│   ├── sidecars/         # Metadata (face tracks, transcripts, phonemes)
│   └── manifests/        # JSONL files for dataset tracking
└── scripts/
    ├── ingest/           # Video ingestion & hashing
    ├── clips/            # Splitting, ASR, Phonemes, Filtering
    ├── tracking/         # MediaPipe Face Tracking
    └── utils/            # Download utilities
```

## ⚙️ How to Run

You can run the entire pipeline with a single command:

```bash
python3 run_pipeline.py
```

This master script executes the following steps in order:

1. **Ingest:** Hashing and moving videos to `dataset/raw`.
2. **Split:** Detecting shots/scenes and creating 1-5s clips.
3. **Track:** Extracting face landmarks and mouth bounding boxes.
4. **Filter:** Rejecting clips based on face size, head pose, and stability.
5. **ASR:** Generating text transcripts and word-level timestamps.
6. **Phoneme:** Converting transcripts to phoneme spans for training.
7. **Score:** Final quality check for audio-visual alignment.

## 🔍 Quality Gates (Filter Logic)

The pipeline is designed to be strict to ensure high-quality training data:
- **Face Coverage:** Face must be detected in at least 80% of frames (adjustable).
- **Face Size:** Face must occupy at least 2% of the frame area.
- **Pose:** Rejects extreme profile views (Yaw > 0.35) or head tilts.
- **Alignment:** Rejects clips where the speech ratio or phoneme density is unrealistic.

## 📺 Visualization

Check `dataset/processed/debug/` after running the pipeline to see `.mp4` files with face mesh and lip overlays. These are compatible with macOS QuickTime and standard media players.

## 📄 License

This project is intended for internal research and data processing.
