import cv2
import os
import numpy as np
import time
from insightface.app import FaceAnalysis

# =========================
# QUALITY FUNCTIONS
# =========================

def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def face_quality(face_crop):
    h, w = face_crop.shape[:2]
    size_score = min(w, h)
    blur = blur_score(face_crop)
    return size_score + blur


def is_frontal_face(face):

    kps = face.kps

    left_eye = kps[0]
    right_eye = kps[1]

    eye_distance = abs(right_eye[0] - left_eye[0])
    eye_height_diff = abs(right_eye[1] - left_eye[1])

    if eye_height_diff > eye_distance * 0.25:
        return False

    if eye_distance < 20:
        return False

    return True


# =========================
# CONFIG
# =========================

# VIDEO_SOURCE = "./source/VIDEO DEMO 6.mp4"
VIDEO_SOURCE = "https://sumsel.smart-gateway.net/storage/vms/videos/4ff1f636-3d2d-4497-b680-be6cbf9cabf2/8e5ee3a8f03427de6945ee08aed138fb.mp4"
OUTPUT_DIR = "faces_clustered_2"

FRAME_SKIP = 5
CLUSTER_THRESHOLD = 0.75
BLUR_THRESHOLD = 120

TARGET_SIZE = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0)

# =========================
# VARIABLES
# =========================

clusters = []
cluster_count = 0
frame_count = 0

best_faces = {}

cap = cv2.VideoCapture(VIDEO_SOURCE)

# =========================
# PROCESS VIDEO
# =========================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        continue

    faces = face_app.get(frame)

    for face in faces:

        if not is_frontal_face(face):
            continue

        x1, y1, x2, y2 = face.bbox.astype(int)

        margin = 20

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        

        # blur filter
        if blur_score(face_crop) < BLUR_THRESHOLD:
            continue

        embedding = face.embedding

        matched_cluster = None

        for i, cluster in enumerate(clusters):

            dist = np.linalg.norm(cluster["center"] - embedding)

            if dist < CLUSTER_THRESHOLD:
                matched_cluster = i
                break

        # new person
        if matched_cluster is None:

            clusters.append({
                "center": embedding
            })

            matched_cluster = cluster_count
            cluster_count += 1

        score = face_quality(face_crop)

        # best frame selection
        if matched_cluster not in best_faces:

            best_faces[matched_cluster] = {
                "score": score,
                "image": face_crop
            }

        else:

            if score > best_faces[matched_cluster]["score"]:

                best_faces[matched_cluster] = {
                    "score": score,
                    "image": face_crop
                }

cap.release()

# =========================
# SAVE BEST FACES
# =========================

for person_id, data in best_faces.items():

    face_crop = data["image"]

    h, w = face_crop.shape[:2]

    scale = TARGET_SIZE / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    face_resized = cv2.resize(
        face_crop,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    timestamp = int(time.time()*1000)

    filename = os.path.join(
        OUTPUT_DIR,
        f"person{person_id}_{timestamp}.jpg"
    )

    cv2.imwrite(filename, face_resized)

print("Total person detected:", cluster_count)
print("Best faces saved:", len(best_faces))