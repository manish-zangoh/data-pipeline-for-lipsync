import os
import json
import cv2
from tqdm import tqdm
import mediapipe as mp

CLIP_DIR = "dataset/processed/clips"
SIDECAR_DIR = "dataset/sidecars/face_tracks"

os.makedirs(SIDECAR_DIR, exist_ok=True)

mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh



LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95
]

NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263

def to_pixel(lm, w, h):
    return [int(lm.x * w), int(lm.y * h)]


def get_bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]

def get_mouth_bbox(landmarks, w, h):
    mouth_indices = [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 308, 324, 318
    ]

    xs = [landmarks[i].x * w for i in mouth_indices]
    ys = [landmarks[i].y * h for i in mouth_indices]

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    return [x1, y1, x2, y2]



def process_clip(clip_path, clip_id):
    cap = cv2.VideoCapture(clip_path)

    frames_data = []
    frame_idx = 0

    with mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None:
                print("Bad frame at index", frame_idx)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark


                all_points = [to_pixel(lm, w, h) for lm in landmarks]
                face_bbox = get_bbox_from_points(all_points)

                lip_points = [to_pixel(landmarks[i], w, h) for i in LIP_INDICES]

                mouth_bbox = get_bbox_from_points(lip_points)

                pad = 5
                x1, y1, x2, y2 = mouth_bbox
                mouth_bbox = [
                    max(0, x1 - pad),
                    max(0, y1 - pad),
                    min(w, x2 + pad),
                    min(h, y2 + pad)
                ]


                nose_tip = to_pixel(landmarks[NOSE_TIP], w, h)
                left_eye = to_pixel(landmarks[LEFT_EYE], w, h)
                right_eye = to_pixel(landmarks[RIGHT_EYE], w, h)

                frames_data.append({
                    "frame_idx": frame_idx,
                    "face_bbox": face_bbox,
                    "mouth_bbox": mouth_bbox,
                    "landmarks": {
                        "lips": lip_points,
                        "nose_tip": nose_tip,
                        "left_eye": left_eye,
                        "right_eye": right_eye
                    },
                    "confidence": 1.0
                })

            frame_idx += 1

    cap.release()
    return frames_data

def visualize_advanced(clip_path, clip_id):
    cap = cv2.VideoCapture(clip_path)

    output_path = f"dataset/processed/debug/{clip_id}_mesh.mp4"
    os.makedirs("dataset/processed/debug", exist_ok=True)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'avc1'), # 🔥 More compatible with Mac/QuickTime
        fps,
        (w, h)
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:

                    # 🔵 draw full face mesh
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0),
                            thickness=1,
                            circle_radius=1
                        )
                    )

                    # 🔴 highlight lips
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 0, 255),
                            thickness=2
                        )
                    )

            out.write(frame)

    cap.release()
    out.release()

    print(f"🎥 Advanced debug saved: {output_path}")


def main():
    clips = sorted(os.listdir(CLIP_DIR))
    for clip_id in tqdm(clips):
        clip_path = os.path.join(CLIP_DIR, clip_id, "video.mp4")

        if not os.path.exists(clip_path):
            continue

        output_json = os.path.join(SIDECAR_DIR, f"{clip_id}.json")
        output_debug = f"dataset/processed/debug/{clip_id}_mesh.mp4"

        # skip only if BOTH exist to prevent out-of-sync debug videos
        if os.path.exists(output_json) and os.path.exists(output_debug):
            continue

        print(f"\n🎬 Processing: {clip_id}")
        
        # Sanity check: can we actually read this video?
        temp_cap = cv2.VideoCapture(clip_path)
        if not temp_cap.isOpened():
            print(f"⚠️ Could not open video: {clip_path}")
            temp_cap.release()
            continue
        temp_cap.release()

        frames_data = process_clip(clip_path, clip_id)
        
        # Always re-visualize if we are here
        visualize_advanced(clip_path, clip_id)

        # reject clips with no face
        if len(frames_data) == 0:
            print(f"❌ No face detected in: {clip_id}")
            continue

        result = {
            "clip_id": clip_id,
            "track_id": 0,
            "frames": frames_data
        }

        with open(output_json, "w") as f:
            json.dump(result, f)

        print(f"✅ Success: {clip_id}")


if __name__ == "__main__":
    main()