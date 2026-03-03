import cv2
import time
import os
import math
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from threading import Thread
from queue import Queue
import json

load_dotenv()

CHANNEL_ID = os.getenv("CHANNEL_ID")
CLIENT_ID = os.getenv("CLIENT_ID")

MODEL_PATH = os.getenv("MODEL_PATH")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))

INPUT_TYPE = os.getenv("INPUT_TYPE", "webcam").lower()
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
RTSP_URL = os.getenv("RTSP_URL")

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))

ENABLE_RESIZE = os.getenv("ENABLE_RESIZE", "true").lower() == "true"
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", 640))
RESIZE_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 360))

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 800))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 450))

WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", 5))

FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("=== APP STARTED ===", flush=True)
print("CHANNEL_ID:", CHANNEL_ID, flush=True)
print("CLIENT_ID:", CLIENT_ID, flush=True)
print("RTSP_URL:", RTSP_URL, flush=True)

# ==============================
# WEBHOOK ASYNC
# ==============================
queue = Queue(maxsize=50)

def webhook_worker():
    while True:
        item = queue.get()
        if item is None:
            break

        face_path, frame_path, ts_iso, bbox = item

        try:
            with open(face_path, "rb") as f1, open(frame_path, "rb") as f2:

                data_payload = {
                    "timestamp": ts_iso,
                    "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "channel_id": CHANNEL_ID,
                    "client_id": CLIENT_ID
                }

                log_payload = {
                    "url": WEBHOOK_URL,
                    "face_file": os.path.basename(face_path),
                    "frame_file": os.path.basename(frame_path),
                    "data": data_payload
                }

                print("\n=== WEBHOOK SEND ===")
                print(json.dumps(data_payload, indent=4))
                print("====================\n")

                try:
                    response = requests.post(
                        WEBHOOK_URL,
                        files=[
                            ("files", (os.path.basename(face_path), f1, "image/jpeg")),
                            ("files", (os.path.basename(frame_path), f2, "image/jpeg")),
                        ],
                        data=data_payload,
                        timeout=WEBHOOK_TIMEOUT
                    )

                    print("Status:", response.status_code)
                except Exception as e:
                    print("Request error:", e)

        except Exception as e:
            print("Webhook error:", e)

        queue.task_done()

Thread(target=webhook_worker, daemon=True).start()
# ==============================
# HELPERS
# ==============================
def enforce_limit(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".jpg")]
    if len(files) <= MAX_IMAGES:
        return
    files.sort(key=os.path.getmtime)
    for f in files[:len(files)-MAX_IMAGES]:
        os.remove(f)

def iso_name(prefix):
    ts = datetime.now().replace(microsecond=0).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{prefix}_{ts}.jpg"

def center(box):
    x,y,w,h = box
    return (x+w//2, y+h//2)

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

face_last_time = {}

def can_save(box):
    c = center(box)
    now = time.time()
    for saved_center in list(face_last_time.keys()):
        if distance(c, saved_center) < 80:
            if now - face_last_time[saved_center] >= SAVE_INTERVAL:
                face_last_time[saved_center] = now
                return True
            return False
    face_last_time[c] = now
    return True

# ==============================
# VIDEO SOURCE
# ==============================
if INPUT_TYPE == "rtsp":
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Video tidak bisa dibuka")
    exit()

detector = cv2.FaceDetectorYN_create(
    MODEL_PATH, "", (320, 320),
    score_threshold=SCORE_THRESHOLD,
    nms_threshold=0.3,
    top_k=5000
)

if ENABLE_VIEW:
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("System ready")

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(1)
        continue

    original = frame
    oh, ow = original.shape[:2]

    if ENABLE_RESIZE:
        resized = cv2.resize(original, (RESIZE_WIDTH, RESIZE_HEIGHT))
        sx = ow / RESIZE_WIDTH
        sy = oh / RESIZE_HEIGHT
    else:
        resized = original
        sx = sy = 1

    h, w = resized.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(resized)

    boxes = []
    if faces is not None and len(faces) > 0:

        # =========================
        # AMBIL WAJAH TERBESAR
        # =========================
        largest_face = max(faces, key=lambda f: f[2] * f[3])

        x, y, fw, fh = largest_face[:4].astype(int)

        # scale ke ukuran original
        x = int(x * sx)
        y = int(y * sy)
        fw = int(fw * sx)
        fh = int(fh * sy)

        # =========================
        # TAMBAH MARGIN
        # =========================
        margin = 0.30          # 30% kanan kiri bawah
        extra_top = 0.40       # 40% tambahan ke atas (rambut)

        pad_w = int(fw * margin)
        pad_h = int(fh * margin)
        extra_h = int(fh * extra_top)

        x_new = x - pad_w
        y_new = y - pad_h - extra_h
        fw_new = fw + (2 * pad_w)
        fh_new = fh + (2 * pad_h) + extra_h

        # =========================
        # CLAMP AGAR TIDAK KELUAR FRAME
        # =========================
        h_img, w_img = original.shape[:2]

        x_new = max(0, x_new)
        y_new = max(0, y_new)

        if x_new + fw_new > w_img:
            fw_new = w_img - x_new

        if y_new + fh_new > h_img:
            fh_new = h_img - y_new

        boxes.append((x_new, y_new, fw_new, fh_new))

    # === FRAME UNTUK DISIMPAN (SELALU DIGAMBAR) ===
    frame_with_box = original.copy()
    for (x,y,fw,fh) in boxes:
        cv2.rectangle(frame_with_box, (x,y), (x+fw,y+fh), (0,255,0), 2)

    # === PREVIEW ===
    if ENABLE_VIEW:
        preview = frame_with_box.copy()

    # === SAVE PER FACE ===
    for box in boxes:
        x,y,fw,fh = box
        if can_save(box):
            face_img = original[y:y+fh, x:x+fw]

            face_name = iso_name("face")
            frame_name = iso_name("frame")

            face_path = os.path.join(FACE_FOLDER, face_name)
            frame_path = os.path.join(FRAME_FOLDER, frame_name)

            if cv2.imwrite(face_path, face_img) and cv2.imwrite(frame_path, frame_with_box):
                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                ts = face_name.replace(".jpg","").replace("face_","")

                if WEBHOOK_URL and not queue.full():
                    queue.put((face_path, frame_path, ts, box))

                print("Saved:", face_name)

    if ENABLE_VIEW:
        display = cv2.resize(preview, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow("Face Detection", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
if ENABLE_VIEW:
    cv2.destroyAllWindows()