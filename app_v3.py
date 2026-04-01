import cv2
import time
import os
import math
import requests
import numpy as np
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue

# ================= PERFORMANCE =================
cv2.setNumThreads(1)
cv2.setUseOptimized(True)

load_dotenv()

# ================= ENV =================

SERVICE_ID = os.getenv("SERVICE_ID")

MODEL_PATH = os.getenv("MODEL_PATH")

ENDPOINT_URL = os.getenv("CCTV_ENDPOINT")
CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

WEBHOOK_URL = os.getenv("WEBHOOK_URL")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.35))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 0))

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))
CROP_PADDING = float(os.getenv("CROP_PADDING", 0.35))

TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH", 960))
MIN_SIZE_CAPTURE = int(os.getenv("MIN_SIZE_CAPTURE", 0))
MAX_FACE_SIZE = int(os.getenv("MAX_FACE_SIZE", 800))

FRAME_FPS = int(os.getenv("FRAME_FPS", 12))
IDLE_FPS = int(os.getenv("IDLE_FPS", 3))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 10))

# ================= ANTI SPAM =================

FACE_COOLDOWN = float(os.getenv("FACE_COOLDOWN", 8))
FACE_MOVE_THRESHOLD = int(os.getenv("FACE_MOVE_THRESHOLD", 80))
FACE_BUCKET_SIZE = int(os.getenv("FACE_BUCKET_SIZE", 220))

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 800))

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_DIR = os.getenv("DEBUG_VIDEO_DIR", "./sample_videos")


# ================= FOLDER =================

FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ================= GLOBAL =================
JAKARTA_TZ = timezone(timedelta(hours=7))
active_cameras = {}
camera_lock = Lock()

preview_frames = {}
preview_lock = Lock()

queue = Queue(maxsize=1000)

# ================= UTIL =================

# def iso_name(prefix):
#     return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

def iso_name(prefix):
    ts_iso = datetime.now(JAKARTA_TZ).isoformat(timespec="seconds")
    ts_iso = ts_iso.replace(":", "-")  # supaya aman untuk nama file
    return f"{prefix}_{ts_iso}.jpg"

def enforce_limit(folder):

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )

    while len(files) > MAX_IMAGES:
        os.remove(files.pop(0))


def expand_crop_bbox(x, y, w, h, img_w, img_h):

    pad_w = int(w * CROP_PADDING)
    pad_h = int(h * CROP_PADDING)

    x -= pad_w
    y -= pad_h

    w += pad_w * 2
    h += pad_h * 2

    x = max(0, x)
    y = max(0, y)

    w = min(w, img_w - x)
    h = min(h, img_h - y)

    return x, y, w, h


def calc_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ================= FACE QUALITY FILTER =================

def is_valid_face(f, x, y, w, h, scale):

    lx, ly = f[4] * scale, f[5] * scale
    rx, ry = f[6] * scale, f[7] * scale
    nx, ny = f[8] * scale, f[9] * scale

    eye_angle = abs(math.degrees(math.atan2(ry - ly, rx - lx)))

    if eye_angle > 40:
        return False

    ratio = w / float(h)

    if ratio < 0.50 or ratio > 1.50:
        return False

    nose_ratio = (nx - x) / float(w)

    if nose_ratio < 0.15 or nose_ratio > 0.85:
        return False

    return True


# ================= WEBHOOK WORKER =================

def webhook_worker():

    while True:

        item = queue.get()

        if item is None:
            break

        try:

            face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, score, cid, client_id = item

            payload = {
                "timestamp": ts_iso,
                "type": "face_detection_service",
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(score, 4),
                "channel_id": cid,
                "client_id": client_id
            }

            log_payload = {
                "url": WEBHOOK_URL,
                #"face_file": face_name,
                "frame_file": frame_name,
                "data": payload
            }

            print(f"\n[WEBHOOK] Sending:\n{json.dumps(log_payload, indent=4)}")

            requests.post(
                WEBHOOK_URL,
                files=[
                    #("files", (face_name, face_bytes, "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=payload,
                timeout=10
            )

            print(f"[WEBHOOK OK] {face_name} | conf={score:.3f}")

        except Exception as e:
            print(f"[WEBHOOK FAIL] log_payload={log_payload}")
            print("[WEBHOOK ERROR]", e)

        finally:
            queue.task_done()


if WEBHOOK_URL:
    Thread(target=webhook_worker, daemon=True).start()


# ================= CAMERA WORKER =================

class CameraWorker:

    def __init__(self, cid, client_id, url):

        self.cid = cid
        self.client_id = client_id

        self.stream_source = url if DEBUG_MODE else "rtsp://" + url

        self.running = True

        self.cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.bad = 0
        self.max_bad = 10

        self.frame_interval = 1 / FRAME_FPS
        self.last_time = 0

        self.face_memory = {}

        self.last_face_time = time.time()

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH,
            "",
            (640, 640),
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=0.4,
            top_k=5000
        )

        # detector warmup
        dummy = np.zeros((640,640,3), dtype=np.uint8)
        self.detector.setInputSize((640,640))
        self.detector.detect(dummy)

        print(f"[CAMERA START] {cid}")

    def stop(self):

        self.running = False
        self.cap.release()

        print(f"[CAMERA STOP] {self.cid}")

    def resize_adaptive(self, frame):

        h, w = frame.shape[:2]

        max_width = TARGET_MAX_WIDTH

        scale = min(1.0, max_width / w)

        resized = cv2.resize(frame, None, fx=scale, fy=scale)

        if resized.shape[1] < 640:
            scale_up = 640 / resized.shape[1]
            resized = cv2.resize(resized, None, fx=scale_up, fy=scale_up)
            scale = scale / scale_up

        return resized, 1 / scale

    def cleanup_memory(self):

        now = time.time()

        remove = []

        for k, v in self.face_memory.items():
            if now - v[0] > 60:
                remove.append(k)

        for k in remove:
            del self.face_memory[k]

    def adaptive_fps(self):

        if time.time() - self.last_face_time > IDLE_TIMEOUT:
            return 1 / IDLE_FPS

        return 1 / FRAME_FPS

    def run(self):

        while self.running:

            self.frame_interval = self.adaptive_fps()

            if time.time() - self.last_time < self.frame_interval:
                continue

            self.last_time = time.time()

            for _ in range(2):
                self.cap.grab()

            ret, frame = self.cap.read()

            if not ret:

                self.bad += 1

                if self.bad >= self.max_bad:

                    print(f"[RECONNECT] {self.cid}")

                    self.cap.release()
                    time.sleep(1)

                    self.cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG)

                    self.bad = 0

                continue

            self.bad = 0

            view = frame.copy()

            resized, scale = self.resize_adaptive(frame)

            h, w = resized.shape[:2]

            self.detector.setInputSize((w, h))

            _, faces = self.detector.detect(resized)

            boxes = []

            if faces is not None:

                for f in faces:

                    score = float(f[14])

                    if score < SCORE_THRESHOLD:
                        continue

                    x, y, fw, fh = f[:4].astype(int)

                    x = int(x * scale)
                    y = int(y * scale)
                    fw = int(fw * scale)
                    fh = int(fh * scale)

                    if fw < MIN_SIZE_CAPTURE:
                        continue

                    if fw > MAX_FACE_SIZE:
                        continue

                    if not is_valid_face(f, x, y, fw, fh, scale):
                        continue

                    x = max(0, x)
                    y = max(0, y)

                    fw = min(fw, frame.shape[1] - x)
                    fh = min(fh, frame.shape[0] - y)

                    boxes.append((x, y, fw, fh, score))

            for box in boxes:

                x, y, fw, fh, score = box

                # cv2.rectangle(view, (x, y), (x+fw, y+fh), (0,255,0), 2)

                cx = x + fw // 2
                cy = y + fh // 2

                bucket = (cx // FACE_BUCKET_SIZE, cy // FACE_BUCKET_SIZE)

                now = time.time()

                if bucket in self.face_memory:

                    last_time, last_pos = self.face_memory[bucket]

                    dist = math.hypot(cx-last_pos[0], cy-last_pos[1])

                    if now - last_time < FACE_COOLDOWN:
                        continue

                    if dist < FACE_MOVE_THRESHOLD:
                        continue

                self.face_memory[bucket] = (now, (cx,cy))

                self.last_face_time = time.time()

                cx2, cy2, cw, ch = expand_crop_bbox(
                    x, y, fw, fh,
                    frame.shape[1],
                    frame.shape[0]
                )

                face_img = frame[cy2:cy2+ch, cx2:cx2+cw]

                if calc_sharpness(face_img) < BLUR_THRESHOLD:
                    continue

                face_name = iso_name("face")
                frame_name = iso_name("frame")

                fb1 = cv2.imencode(".jpg", face_img)[1].tobytes()
                fb2 = cv2.imencode(".jpg", view)[1].tobytes()

                open(os.path.join(FACE_FOLDER, face_name), "wb").write(fb1)
                open(os.path.join(FRAME_FOLDER, frame_name), "wb").write(fb2)

                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                if WEBHOOK_URL:
                    try:
                        queue.put_nowait(
                            (fb1, fb2, face_name, frame_name,
                            ts_iso := datetime.now(JAKARTA_TZ).isoformat(timespec="seconds"),
                            (x,y,fw,fh),
                            score,
                            self.cid,
                            self.client_id)
                        )
                    except:
                        print("[QUEUE FULL] webhook queue penuh")

                print(f"[{self.cid}] SAVED {face_name}")

            self.cleanup_memory()

            with preview_lock:
                preview_frames[self.cid] = view


# ================= LOAD CAMERA =================

def load_cameras():

    if DEBUG_MODE:

        print("[DEBUG MODE] Load video dari folder")

        videos = []

        for file in os.listdir(DEBUG_VIDEO_DIR):

            if file.lower().endswith(".mp4"):

                videos.append({
                    "cctv_id": file,
                    "client_id": "debug",
                    "stream_url": os.path.join(DEBUG_VIDEO_DIR, file)
                })

        print(f"[DEBUG] Total video ditemukan: {len(videos)}")

        return videos

    else:

        print("[PRODUCTION MODE] Load kamera dari endpoint")

        try:

            r = requests.get(f"{ENDPOINT_URL}?service_id={SERVICE_ID}", timeout=10)

            data = r.json()["data"]

            print(f"[API] Total kamera dari server: {len(data)}")

            return data

        except Exception as e:

            print("[API ERROR]", e)

            return []


# ================= CAMERA MANAGER =================

def camera_manager():

    print("\n=== CAMERA MANAGER STARTED ===")

    while True:

        cams = load_cameras()

        api_ids = set([c["cctv_id"] for c in cams])

        with camera_lock:

            active_ids = set(active_cameras.keys())

            new_ids = api_ids - active_ids

            for c in cams:

                if c["cctv_id"] in new_ids:

                    try:

                        w = CameraWorker(
                            c["cctv_id"],
                            c["client_id"],
                            c["stream_url"]
                        )

                        Thread(target=w.run, daemon=True).start()

                        active_cameras[c["cctv_id"]] = w

                        print(f"[NEW CAMERA STARTED] {c['cctv_id']}")

                    except Exception as e:

                        print(f"[FAILED START] {c['cctv_id']} -> {e}")

            removed_ids = active_ids - api_ids

            for rid in removed_ids:

                try:

                    active_cameras[rid].stop()
                    del active_cameras[rid]
                    print(f"[CAMERA REMOVED] {rid}")
                except:
                    pass

            print("------ STATUS ------")
            print("Total Source :", len(api_ids))
            print("Active Worker:", len(active_cameras))
            print("--------------------\n")

        time.sleep(CAMERA_REFRESH_INTERVAL)


Thread(target=camera_manager, daemon=True).start()


# ================= VIEW =================

if ENABLE_VIEW:

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    while True:

        with preview_lock:
            frames = list(preview_frames.values())

        if not frames:
            time.sleep(0.1)
            continue

        total = len(frames)

        cols = math.ceil(math.sqrt(total))
        rows = math.ceil(total / cols)

        tw = DISPLAY_WIDTH // cols
        th = DISPLAY_HEIGHT // rows

        imgs = []

        for f in frames:

            try:
                imgs.append(cv2.resize(f, (tw, th)))
            except:
                imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        grid = []
        idx = 0

        for r in range(rows):

            row_imgs = imgs[idx:idx+cols]

            try:
                grid.append(cv2.hconcat(row_imgs))
            except:
                grid.append(np.zeros((th, tw * cols, 3), dtype=np.uint8))

            idx += cols

        try:
            final = cv2.vconcat(grid)
        except:
            final = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        cv2.imshow("Face Detection", final)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:

    while True:
        time.sleep(1)