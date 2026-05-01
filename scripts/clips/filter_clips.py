import os
import json
import math
import cv2
from statistics import mean

SIDECAR_DIR = "dataset/sidecars/face_tracks"
CLIP_MANIFEST = "dataset/manifests/clips.jsonl"
RAW_DIR = "dataset/processed/clips"

FILTERED_OUT = "dataset/manifests/filtered_clips.jsonl"
REJECTED_OUT = "dataset/manifests/rejected_clips.jsonl"



def estimate_pose(frame):
    le = frame["landmarks"]["left_eye"]
    re = frame["landmarks"]["right_eye"]
    nose = frame["landmarks"]["nose_tip"]

    eye_dx = re[0] - le[0]
    eye_dy = re[1] - le[1]

    tilt = abs(eye_dy / (eye_dx + 1e-6))

    mid_eye_x = (le[0] + re[0]) / 2
    yaw = abs(nose[0] - mid_eye_x) / (abs(eye_dx) + 1e-6)

    return yaw, tilt



def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        total = int(fps * duration)

    cap.release()
    return total


def evaluate_clip(frames_data, frame_w, frame_h, total_frames):
    valid_frames = len(frames_data)

    # -------- Basic checks --------
    if valid_frames < 5:
        return False, "too_few_frames", None

    if total_frames <= 0:
        return False, "invalid_total_frames", None

    coverage = valid_frames / total_frames

    if coverage < 0.8:
        return False, "low_face_coverage", None

    # -------- Metrics --------
    face_ratios = []
    shifts = []
    yaw_vals = []
    tilt_vals = []

    prev_center = None

    frame_indices = []

    for f in frames_data:
        x1, y1, x2, y2 = f["face_bbox"]

        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_w * frame_h

        face_ratios.append(face_area / frame_area)

        # center shift
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if prev_center:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            shifts.append(math.sqrt(dx * dx + dy * dy))

        prev_center = (cx, cy)

        # pose
        yaw, tilt = estimate_pose(f)
        yaw_vals.append(yaw)
        tilt_vals.append(tilt)

        frame_indices.append(f["frame_idx"])

    avg_face = mean(face_ratios)
    avg_shift = mean(shifts) if shifts else 0
    avg_yaw = mean(yaw_vals)
    avg_tilt = mean(tilt_vals)

    # -------- Fragmentation check --------
    gaps = [
        frame_indices[i] - frame_indices[i - 1]
        for i in range(1, len(frame_indices))
    ]

    if gaps and max(gaps) > 10:
        return False, "fragmented_tracking", None

    # -------- Rejection rules --------
    if avg_face < 0.02: # Lowered from 0.05
        return False, "face_too_small", None

    if avg_shift > 20:
        return False, "unstable_tracking", None

    if avg_yaw > 0.35:
        return False, "side_face", None

    if avg_tilt > 0.3:
        return False, "head_tilt", None

    if avg_face > 0.15: # Lowered from 0.25
        scale = "close_up"
    elif avg_face > 0.05: # Lowered from 0.10
        scale = "medium"
    else:
        scale = "wide"

    if avg_yaw < 0.2:
        pose = "frontal"
    elif avg_yaw < 0.4:
        pose = "three_quarter"
    else:
        pose = "profile"

    shot_type = f"{scale}_{pose}"

    return True, "ok", shot_type



def load_clips():
    clips = {}
    with open(CLIP_MANIFEST, "r") as f:
        for line in f:
            data = json.loads(line)
            clips[data["clip_id"]] = data
    return clips


# -----------------------------
# Main
# -----------------------------
def main():
    clips = load_clips()

    os.makedirs(os.path.dirname(FILTERED_OUT), exist_ok=True)

    kept, rejected = 0, 0

    with open(FILTERED_OUT, "w") as fout, open(REJECTED_OUT, "w") as rj:

        for fname in os.listdir(SIDECAR_DIR):
            if not fname.endswith(".json"):
                continue

            path = os.path.join(SIDECAR_DIR, fname)

            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except:
                continue

            clip_id = data.get("clip_id")
            frames_data = data.get("frames", [])

            if not clip_id or clip_id not in clips:
                continue

            clip_meta = clips[clip_id]

            video_folder = os.path.join(RAW_DIR, f"{clip_id}")
            clip_path = os.path.join(video_folder, "video.mp4")
            print(f"video_path: {clip_path}")
            total_frames = get_total_frames(clip_path)
            print(f"Processing {clip_id}: {len(frames_data)} valid frames out of {total_frames} total frames")
            w, h = clip_meta["resolution"]

            ok, reason, shot_type = evaluate_clip(
                frames_data,
                w,
                h,
                total_frames
            )

            if not ok:
                rejected += 1
                rj.write(json.dumps({
                    "clip_id": clip_id,
                    "reason": reason
                }) + "\n")

                print(f"❌ Reject {clip_id}: {reason}")
                continue

            kept += 1

            clip_meta["shot_type"] = shot_type

            fout.write(json.dumps(clip_meta) + "\n")

            print(f"✅ Keep {clip_id} ({shot_type})")

    print("\n📊 Summary")
    print(f"Kept: {kept}")
    print(f"Rejected: {rejected}")


if __name__ == "__main__":
    main()