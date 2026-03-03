import cv2
import time
import os
import math
import glob
import requests
import numpy as np 
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue
import json

load_dotenv()

# ==============================
# CONFIG
# ==============================
SERVICE_ID = os.getenv("SERVICE_ID")
ENDPOINT_URL = os.getenv("CCTV_ENDPOINT")

MODEL_PATH = os.getenv("MODEL_PATH")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.6))

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))

ENABLE_RESIZE = os.getenv("ENABLE_RESIZE", "true").lower() == "true"
RESIZE_WIDTH = int(os.getenv("RESIZE_WIDTH", 640))
RESIZE_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 360))
MIN_SIZE_CAPTURE = int(os.getenv("MIN_SIZE_CAPTURE", 25))

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 400))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 300))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_FOLDER = os.getenv("DEBUG_VIDEO_FOLDER", "./sample_video")
DEBUG_LOOP = os.getenv("DEBUG_LOOP", "true").lower() == "true"

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", 5))





print("ENABLE_VIEW:", ENABLE_VIEW)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ==============================
# SAFE MODEL PATH
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)

MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model ONNX tidak ditemukan: {MODEL_PATH}")

print("MODEL_PATH:", MODEL_PATH, flush=True)

# ==============================
# FOLDER SETUP
# ==============================
FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")
os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("=== APP STARTED ===")
print("SERVICE_ID:", SERVICE_ID)

# ==============================
# GLOBAL STORAGE
# ==============================
preview_frames = {}
preview_lock = Lock()
active_cameras = {}
camera_lock = Lock()

#====debug mode load video files as camera sources======
def load_debug_sources(folder):
    video_files = []
    extensions = ("*.mp4", "*.avi", "*.mkv")

    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(folder, ext)))

    video_files.sort()

    sources = []
    for idx, path in enumerate(video_files):
        # sources.append({
        #     "cctv_id": f"debug_{idx+1}",
        #     "client_id": "debug",
        #     "stream_url": path   # ⚠ HARUS stream_url
        # })

         sources.append({
            "cctv_id": "126a73ff-6d23-44a8-a496-194256bcf5db",
            "client_id": "1102a7be-c4a5-42d6-ba0e-3905693e0722",
            "stream_url": path   # ⚠ HARUS stream_url
        })

    print(f"[DEBUG MODE] Found {len(sources)} video files")
    return sources
# ==============================
# WEBHOOK ASYNC
# ==============================
queue = Queue(maxsize=200)

def webhook_worker():
    while True:
        item = queue.get()
        if item is None:
            break

        #face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, cctv_id, client_id = item
        face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, cctv_id, client_id, is_debug = item

        try:
            data_payload = {
                "timestamp": ts_iso,
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "channel_id": cctv_id,
                "client_id": client_id,
                "is_debug": str(is_debug).lower()
            }

            log_payload = {
                "url": WEBHOOK_URL,
                "face_file": face_name,
                "frame_file": frame_name,
                "data": data_payload
            }
            print(f"\n[WEBHOOK] Sending {json.dumps(log_payload, indent=4)}")

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

        except requests.exceptions.Timeout:
            print("[WEBHOOK] TIMEOUT")
        except requests.exceptions.ConnectionError:
            print("[WEBHOOK] CONNECTION ERROR")
        except Exception as e:
            print("[WEBHOOK] ERROR:", e)

        queue.task_done()

for _ in range(3):
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
    ts = datetime.now().replace(microsecond=0).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{prefix}_{ts}.jpg"

# ==============================
# CAMERA WORKER
# ==============================
class CameraWorker:

    def __init__(self, cctv_id, client_id, rtsp_url):
        self.cctv_id = cctv_id
        self.client_id = client_id
        if DEBUG_MODE:
            self.rtsp_url = rtsp_url   # langsung path video
        else:
            self.rtsp_url = "rtsp://" + rtsp_url
        self.face_last_time = {}
        self.face_memory_timeout = SAVE_INTERVAL
        self.last_global_save = 0
        self.running = True
        
        self.frame_interval = 1 / 15
        self.last_frame_time = 0
        self.bad_frame_count = 0
        self.max_bad_frames = 10

        print(f"[CAMERA START] {cctv_id}")

        print(f"[CAMERA START] {cctv_id}")

        self.cap = None   # WAJIB ADA

        if DEBUG_MODE:
            self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open stream: {self.rtsp_url}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (320, 320),
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=0.3,
            top_k=5000
        )

    def stop(self):
        print(f"[CAMERA STOP] {self.cctv_id}")
        self.running = False
        if self.cap is not None:
            self.cap.release()

    def can_save(self, box):
        now = time.time()

        if now - self.last_global_save < SAVE_INTERVAL:
            return False
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2

        bucket_x = cx // 150
        bucket_y = cy // 150

        face_id = (bucket_x, bucket_y)

        if face_id in self.face_last_time:
            if now - self.face_last_time[face_id] >= SAVE_INTERVAL:
                self.face_last_time[face_id] = now
                self.last_global_save = now
                return True
            return False
        else:
            self.face_last_time[face_id] = now
            self.last_global_save = now
            return True

    def run(self):
        while self.running:

            try:
                # Flush buffer supaya ambil frame terbaru
                if not DEBUG_MODE:
                    for _ in range(2):
                        self.cap.grab()

                ret, frame = self.cap.read()

                if not ret or frame is None:

                    if DEBUG_MODE and DEBUG_LOOP:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    self.bad_frame_count += 1
                    print(f"[{self.cctv_id}] Bad frame ({self.bad_frame_count})")

                    if DEBUG_MODE and DEBUG_LOOP:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    self.bad_frame_count += 1

                if not ret or frame is None:
                    self.bad_frame_count += 1
                    print(f"[{self.cctv_id}] Bad frame ({self.bad_frame_count})")

                    if self.bad_frame_count >= self.max_bad_frames:
                        print(f"[{self.cctv_id}] Too many bad frames. FULL RECONNECT...")
                        self.cap.release()
                        time.sleep(1)
                    
                        if DEBUG_MODE:
                            self.cap = cv2.VideoCapture(self.rtsp_url)
                        else:
                            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FPS, 15)

                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FPS, 15)

                        self.bad_frame_count = 0

                    continue

                # Reset counter kalau frame normal
                self.bad_frame_count = 0

                # FPS limiter hanya untuk RTSP
                if not DEBUG_MODE:
                    now = time.time()
                    if now - self.last_frame_time < self.frame_interval:
                        continue
                    self.last_frame_time = now

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
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(resized)

                boxes = []
                if faces is not None and len(faces) > 0:

                    for face in faces:

                        x, y, fw, fh = face[:4].astype(int)

                        # convert scale kalau resize aktif
                        x = int(x * sx)
                        y = int(y * sy)
                        fw = int(fw * sx)
                        fh = int(fh * sy)

                        # skip face terlalu kecil (optional tapi bagus)
                        if fw < MIN_SIZE_CAPTURE or fh < MIN_SIZE_CAPTURE:
                            continue

                        margin = 0.30
                        extra_top = 0.40

                        pad_w = int(fw * margin)
                        pad_h = int(fh * margin)
                        extra_h = int(fh * extra_top)

                        x_new = x - pad_w
                        y_new = y - pad_h - extra_h
                        fw_new = fw + (2 * pad_w)
                        fh_new = fh + (2 * pad_h) + extra_h

                        h_img, w_img = original.shape[:2]

                        x_new = max(0, x_new)
                        y_new = max(0, y_new)

                        if x_new + fw_new > w_img:
                            fw_new = w_img - x_new

                        if y_new + fh_new > h_img:
                            fh_new = h_img - y_new

                        boxes.append((x_new, y_new, fw_new, fh_new))

                frame_with_box = original.copy()
                for (x,y,fw,fh) in boxes:
                    cv2.rectangle(frame_with_box, (x,y), (x+fw,y+fh), (0,255,0), 2)

                if ENABLE_VIEW:
                    with preview_lock:
                        preview_frames[self.cctv_id] = frame_with_box.copy()

                for box in boxes:
                    if self.can_save(box):
                        x,y,fw,fh = box
                        face_img = original[y:y+fh, x:x+fw]

                        face_name = iso_name("face")
                        frame_name = iso_name("frame")

                        face_path = os.path.join(FACE_FOLDER, face_name)
                        frame_path = os.path.join(FRAME_FOLDER, frame_name)

                        ret1, face_buffer = cv2.imencode(".jpg", face_img)
                        ret2, frame_buffer = cv2.imencode(".jpg", frame_with_box)

                        if ret1 and ret2:

                            with open(face_path, "wb") as f:
                                f.write(face_buffer.tobytes())

                            with open(frame_path, "wb") as f:
                                f.write(frame_buffer.tobytes())

                            enforce_limit(FACE_FOLDER)
                            enforce_limit(FRAME_FOLDER)

                            ts = face_name.replace(".jpg","").replace("face_","")

                            if WEBHOOK_URL and not queue.full():
                                print("QUEUE SIZE:", queue.qsize())
                                queue.put((
                                    face_buffer.tobytes(),
                                    frame_buffer.tobytes(),
                                    face_name,
                                    frame_name,
                                    ts,
                                    box,
                                    self.cctv_id,
                                    self.client_id,
                                    DEBUG_MODE   # ⬅ tambahkan ini
                                ))

                            print(f"[{self.cctv_id}] Saved:", face_name)

            except Exception as e:
                print(f"[{self.cctv_id}] CRITICAL STREAM ERROR:", e)
                time.sleep(1)

# ==============================
# FETCH CAMERA LIST
# ==============================
def fetch_cameras():
    url = f"{ENDPOINT_URL}?service_id={SERVICE_ID}"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            print(f"[CAMERA API] Status error: {r.status_code}")
            return None

        data = r.json()

        if "data" not in data:
            print("[CAMERA API] Invalid response format")
            return None

        return data["data"]

    except requests.exceptions.Timeout:
        print("[CAMERA API] TIMEOUT")
    except requests.exceptions.ConnectionError:
        print("[CAMERA API] CONNECTION ERROR")
    except Exception as e:
        print("[CAMERA API] ERROR:", e)

    return None

# ==============================
# CAMERA MANAGER
# ==============================
def camera_manager():
    while True:
        start_time = time.time()

        try:
            if DEBUG_MODE:
                cameras = load_debug_sources(DEBUG_VIDEO_FOLDER)
            else:
                cameras = fetch_cameras()

            if cameras is None:
                print("[CAMERA MANAGER] API down, retry in 10s")
                time.sleep(10)
                continue

            print(f"[CAMERA MANAGER] Reload camera list ({len(cameras)} cameras)")

            with camera_lock:
                existing = set(active_cameras.keys())
                incoming = set(cam["cctv_id"] for cam in cameras)

                # Tambah kamera baru
                for cam in cameras:
                    cid = cam["cctv_id"]
                    if cid not in existing:
                        print("[CAMERA ADDED]", cid)
                        worker = CameraWorker(cid, cam["client_id"], cam["stream_url"])
                        active_cameras[cid] = worker
                        #Thread(target=worker.run, daemon=True).start()
                        t = Thread(target=worker.run, daemon=True)
                        worker.thread = t
                        t.start()

                # Hapus kamera yang tidak ada lagi
                for cid in list(existing - incoming):
                    print("[CAMERA REMOVED]", cid)

                    worker = active_cameras.pop(cid, None)
                    if worker:
                        worker.stop()
                        if hasattr(worker, "thread"):
                            worker.thread.join(timeout=2)

                    with preview_lock:
                        preview_frames.pop(cid, None)

        except Exception as e:
            print("[CAMERA MANAGER ERROR]", e)

        # Hitung sisa waktu supaya interval presisi
        elapsed = time.time() - start_time
        sleep_time = max(0, CAMERA_REFRESH_INTERVAL - elapsed)
        time.sleep(sleep_time)

Thread(target=camera_manager, daemon=True).start()

print("System ready")


#=========config auto grid based on camera count===========
def calculate_grid(n):
    if n <= 0:
        return 1, 1
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    return rows, cols

# ==============================
# GRID VIEW LOOP (WINDOWS SAFE)
# ==============================

if ENABLE_VIEW:
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

while True:

    if ENABLE_VIEW:

        with preview_lock:
            frames_raw = list(preview_frames.values())

        cam_count = len(frames_raw)

        if cam_count == 0:
            blank = 255 * np.ones(
                (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype="uint8"
            )
            cv2.resizeWindow("Face Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)
            cv2.imshow("Face Detection", blank)
            cv2.waitKey(1)
            time.sleep(0.1)
            continue

        # =====================================
        # DYNAMIC LANDSCAPE GRID
        # =====================================
        best_rows, best_cols = 1, cam_count
        best_ratio_diff = 999

        for rows in range(1, cam_count + 1):
            cols = math.ceil(cam_count / rows)

            window_ratio = cols / rows
            target_ratio = DISPLAY_WIDTH / DISPLAY_HEIGHT

            ratio_diff = abs(window_ratio - target_ratio)

            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_rows = rows
                best_cols = cols

        rows = best_rows
        cols = best_cols

        # =====================================
        # TILE SIZE (LOCK 16:9)
        # =====================================
        tile_width = DISPLAY_WIDTH // cols
        tile_height = int(tile_width * 9 / 16)

        # Jika tinggi melebihi batas, hitung ulang dari tinggi
        if tile_height * rows > DISPLAY_HEIGHT:
            tile_height = DISPLAY_HEIGHT // rows
            tile_width = int(tile_height * 16 / 9)

        # =====================================
        # REAL WINDOW SIZE
        # =====================================
        window_width = tile_width * cols
        window_height = tile_height * rows

        # Clamp agar tidak melebihi ENV
        window_width = min(window_width, DISPLAY_WIDTH)
        window_height = min(window_height, DISPLAY_HEIGHT)

        cv2.resizeWindow("Face Detection", window_width, window_height)

        # =====================================
        # RESIZE FRAMES
        # =====================================
        frames = []

        for f in frames_raw:
            resized = cv2.resize(f, (tile_width, tile_height))
            frames.append(resized)

        total_slots = rows * cols

        while len(frames) < total_slots:
            frames.append(
                255 * np.ones((tile_height, tile_width, 3), dtype="uint8")
            )

        frames = frames[:total_slots]

        grid_rows = []
        idx = 0
        for r in range(rows):
            row = cv2.hconcat(frames[idx:idx+cols])
            grid_rows.append(row)
            idx += cols

        grid = cv2.vconcat(grid_rows)

        cv2.imshow("Face Detection", grid)

        if cv2.waitKey(1) == ord("q"):
            break

    time.sleep(0.03)

if ENABLE_VIEW:
    cv2.destroyAllWindows()