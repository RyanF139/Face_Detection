# =========================================================
# MULTI CAMERA FACE CAPTURE (FULL STABLE FINAL VERSION)
# CPU ONLY (YuNet)
import cv2
import time
import os
import math
import glob
import requests
import numpy as np
import json

from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue

load_dotenv()

# =========================================================
# ====================== CONFIG (.env) =====================
# =========================================================

SERVICE_ID = os.getenv("SERVICE_ID")
ENDPOINT_URL = os.getenv("CCTV_ENDPOINT")

MODEL_PATH = os.getenv("MODEL_PATH")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 0)) 

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))

TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH", 640))
TARGET_MAX_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 360))
MIN_SIZE_CAPTURE = int(os.getenv("MIN_SIZE_CAPTURE", 0))

FRAME_FPS = int(os.getenv("FRAME_FPS", 12))
CROP_PADDING = float(os.getenv("CROP_PADDING", 0.40)) # 0.0 - 1.0, seberapa besar area crop dibanding bbox deteksi

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 800))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_FOLDER = os.getenv("DEBUG_VIDEO_FOLDER", "./sample_video")
DEBUG_LOOP = os.getenv("DEBUG_LOOP", "true").lower() == "true"

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

print("ENABLE_VIEW:", ENABLE_VIEW)

# RTSP lebih stabil
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# =========================================================
# ================= SAFE MODEL PATH ========================
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model ONNX tidak ditemukan: {MODEL_PATH}")

print("MODEL_PATH:", MODEL_PATH)

# =========================================================
# ================= FOLDER SETUP ===========================
# =========================================================

FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("=== APP STARTED ===")
print("SERVICE_ID:", SERVICE_ID)

# =========================================================
# ================= GLOBAL STORAGE =========================
# =========================================================

preview_frames = {}
preview_lock = Lock()

active_cameras = {}
camera_lock = Lock()

queue = Queue(maxsize=300)

# =========================================================
# ================= WEBHOOK WORKER =========================
# =========================================================
def webhook_worker():
    while True:
        item = queue.get()
        if item is None:
            break

        face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, cctv_id, client_id = item

        try:
            data_payload = {
                "timestamp": ts_iso,
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "channel_id": cctv_id,
                "client_id": client_id
            }

            log_payload = {
                "url": WEBHOOK_URL,
                "face_file": face_name,
                "frame_file": frame_name,
                "data": data_payload
            }

            print(f"\n[WEBHOOK] Sending:\n{json.dumps(log_payload, indent=4)}")

            response = requests.post(
                WEBHOOK_URL,
                files=[
                    ("files", (face_name, face_bytes, "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=data_payload,
                timeout=10
            )

            print(f"[WEBHOOK] Status: {response.status_code}")

            if response.status_code >= 400:
                print("[WEBHOOK] ERROR RESPONSE:", response.text)

        except requests.exceptions.Timeout:
            print("[WEBHOOK] TIMEOUT")
        except requests.exceptions.ConnectionError:
            print("[WEBHOOK] CONNECTION ERROR")
        except Exception as e:
            print("[WEBHOOK] ERROR:", e)

        finally:
            queue.task_done()


for _ in range(3):
    Thread(target=webhook_worker, daemon=True).start()


# =========================================================
# ================= HELPERS ================================
# =========================================================

def iso_name(prefix):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prefix}_{ts}.jpg"


def enforce_limit(folder):
    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )
    while len(files) > MAX_IMAGES:
        os.remove(files.pop(0))

def expand_crop_bbox(x, y, w, h, img_w, img_h):
    """
    Perbesar area crop saja.
    Rectangle preview tetap pakai bbox asli.
    """
    pad_w = int(w * CROP_PADDING)
    pad_h = int(h * CROP_PADDING)

    x -= pad_w
    y -= pad_h
    w += pad_w * 2
    h += pad_h * 2

    # safety clamp
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    return x, y, w, h

def calc_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# =========================================================
# ================= DEBUG VIDEO LOADER =====================
# =========================================================

def load_debug_sources(folder):
    files = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        files += glob.glob(os.path.join(folder, ext))

    sources = []
    for i, path in enumerate(files):
        sources.append({
            "cctv_id": f"debug_{i}",
            "client_id": "debug",
            "stream_url": path
        })

    print("[DEBUG MODE] Found", len(sources), "files")
    return sources


# =========================================================
# ================= CAMERA WORKER ==========================
# =========================================================

class CameraWorker:

    def __init__(self, cid, client_id, url):

        self.cid = cid
        self.client_id = client_id

        self.rtsp = url if DEBUG_MODE else "rtsp://" + url

        self.running = True

        self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_interval = 1 / FRAME_FPS
        self.last_time = 0

        self.bad = 0
        self.max_bad = 20

        self.last_save = 0

        self.face_last_time = {}

        # YuNet detector (CPU only)
        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (320, 320),
            score_threshold=SCORE_THRESHOLD
        )

        # detector warmup
        self.detector.setInputSize((320, 320))
        self.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

        print("[CAMERA START]", cid)


    # ================= adaptive resize =================
    def resize_adaptive(self, frame):
        h, w = frame.shape[:2]

        if w > TARGET_MAX_WIDTH:
            scale = TARGET_MAX_WIDTH / w
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            return small, 1/scale, 1/scale

        return frame, 1, 1


    # ================= main loop =================
    def run(self):

        while self.running:

            if time.time() - self.last_time < self.frame_interval:
                continue

            self.last_time = time.time()

            # flush buffer
            for _ in range(2):
                self.cap.grab()

            ret, frame = self.cap.read()

            if not ret:
                self.bad += 1

                if self.bad >= self.max_bad:
                    print("[RECONNECT]", self.cid)
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
                    self.bad = 0
                continue

            self.bad = 0

            resized, sx, sy = self.resize_adaptive(frame)

            h, w = resized.shape[:2]
            self.detector.setInputSize((w, h))

            _, faces = self.detector.detect(resized)

            boxes = []

            if faces is not None:
                for f in faces:
                    score = float(f[4])  # confidence YuNet
                    label = f"C:{score:.2f}"
                    print (label)
                    if score < SCORE_THRESHOLD:
                        continue
                    x, y, fw, fh = f[:4].astype(int)

                    x = int(x*sx)
                    y = int(y*sy)
                    fw = int(fw*sx)
                    fh = int(fh*sy)

                    if fw < MIN_SIZE_CAPTURE:
                        continue

                    # clamp safety
                    x = max(0, x)
                    y = max(0, y)
                    fw = min(fw, frame.shape[1]-x)
                    fh = min(fh, frame.shape[0]-y)

                    boxes.append((x,y,fw,fh))

            view = frame.copy()

            for box in boxes:

                x,y,fw,fh = box

                cv2.rectangle(view,(x,y),(x+fw,y+fh),(0,255,0),2)


                cx = x + fw//2
                cy = y + fh//2

                bucket = (cx//150, cy//150)
                now = time.time()

                # cooldown per wajah
                if bucket in self.face_last_time and now - self.face_last_time[bucket] < SAVE_INTERVAL:
                    continue
                self.face_last_time[bucket] = now
                # =========================
                # crop diperbesar (padding)
                # =========================
                pad_w = int(fw * CROP_PADDING)
                pad_h = int(fh * CROP_PADDING)

                cx = max(0, x - pad_w)
                cy = max(0, y - pad_h)
                cw = min(frame.shape[1] - cx, fw + pad_w*2)
                ch = min(frame.shape[0] - cy, fh + pad_h*2)

                face_img = frame[cy:cy+ch, cx:cx+cw]

                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < BLUR_THRESHOLD:
                    continue

                face_name = iso_name("face")
                frame_name = iso_name("frame")

                fb1 = cv2.imencode(".jpg", face_img)[1].tobytes()
                fb2 = cv2.imencode(".jpg", view)[1].tobytes()

                open(os.path.join(FACE_FOLDER, face_name),"wb").write(fb1)
                open(os.path.join(FRAME_FOLDER, frame_name),"wb").write(fb2)

                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                if WEBHOOK_URL:
                    queue.put((fb1,fb2,face_name,frame_name,face_name,box,self.cid,self.client_id))

                print(f"[{self.cid}] Saved:", face_name)

                self.last_save = time.time()

            with preview_lock:
                preview_frames[self.cid] = view


# =========================================================
# ================= FETCH CAMERA LIST ======================
# =========================================================

def fetch_cameras():
    try:
        r = requests.get(f"{ENDPOINT_URL}?service_id={SERVICE_ID}", timeout=10)
        return r.json()["data"]
    except:
        return []


# =========================================================
# ================= CAMERA MANAGER =========================
# =========================================================

def camera_manager():

    while True:

        cams = load_debug_sources(DEBUG_VIDEO_FOLDER) if DEBUG_MODE else fetch_cameras()

        with camera_lock:
            existing = set(active_cameras.keys())

            for c in cams:
                if c["cctv_id"] not in existing:
                    w = CameraWorker(c["cctv_id"], c["client_id"], c["stream_url"])
                    Thread(target=w.run, daemon=True).start()
                    active_cameras[c["cctv_id"]] = w

        time.sleep(CAMERA_REFRESH_INTERVAL)


Thread(target=camera_manager, daemon=True).start()

print("System ready")


# =========================================================
# ================= MAIN LOOP ==============================
# =========================================================

if ENABLE_VIEW:

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    while True:

        with preview_lock:
            frames = list(preview_frames.values())

        if not frames:
            time.sleep(0.1)
            continue

        rows = math.ceil(math.sqrt(len(frames)))
        cols = math.ceil(len(frames)/rows)

        tw = DISPLAY_WIDTH//cols
        th = DISPLAY_HEIGHT//rows

        imgs = [cv2.resize(f,(tw,th)) for f in frames]

        while len(imgs) < rows*cols:
            imgs.append(np.zeros((th,tw,3),dtype=np.uint8))

        grid=[]
        idx=0
        for r in range(rows):
            grid.append(cv2.hconcat(imgs[idx:idx+cols]))
            idx+=cols

        cv2.imshow("Face Detection", cv2.vconcat(grid))

        if cv2.waitKey(1)==ord("q"):
            break

    cv2.destroyAllWindows()

else:
    print("Running headless mode")
    while True:
        time.sleep(1)
