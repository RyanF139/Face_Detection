import cv2
import time
import os
import math
import requests
import numpy as np
import json
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock, Semaphore
from queue import Queue
from urllib.parse import urlparse

# ================= PERFORMANCE =================
cv2.setNumThreads(1)
cv2.setUseOptimized(True)

load_dotenv()

# ================= ENV =================

SERVICE_ID   = os.getenv("SERVICE_ID")
ENDPOINT_URL = os.getenv("VIDEO_ENDPOINT")
MODEL_PATH   = os.getenv("MODEL_PATH")

SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD", 0.35))
BLUR_THRESHOLD_VIDEO   = float(os.getenv("BLUR_THRESHOLD_VIDEO", 0))

SAVE_FOLDER   = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES    = int(os.getenv("MAX_IMAGES", 150))

TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH", 960))
FRAME_FPS        = int(os.getenv("FRAME_FPS", 12))
MIN_SIZE_CAPTURE_VIDEO = int(os.getenv("MIN_SIZE_CAPTURE_VIDEO", 0))
MAX_FACE_SIZE    = int(os.getenv("MAX_FACE_SIZE", 800))

CROP_PADDING = float(os.getenv("CROP_PADDING", 0.35))

VIDEO_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))
WEBHOOK_URL    = os.getenv("WEBHOOK_URL_VIDEO")
WEBHOOK_STATUS = os.getenv("WEBHOOK_STATUS")

ENABLE_VIEW    = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH  = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 800))

VIDEO_DOWNLOAD_FOLDER = os.getenv("VIDEO_DOWNLOAD_FOLDER", "downloaded_videos")
MAX_VIDEO_WORKERS     = int(os.getenv("MAX_VIDEO_WORKERS", 3))
VIDEO_LOOP            = os.getenv("VIDEO_LOOP", "false").lower() == "true"

# ================= ANTI SPAM =================

FACE_COOLDOWN       = float(os.getenv("FACE_COOLDOWN", 8))
FACE_MOVE_THRESHOLD = int(os.getenv("FACE_MOVE_THRESHOLD", 80))
FACE_BUCKET_SIZE    = int(os.getenv("FACE_BUCKET_SIZE", 220))

# ================= ADAPTIVE FPS =================

IDLE_FPS     = int(os.getenv("IDLE_FPS", 3))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 10))

# ================= FOLDER =================

FACE_FOLDER  = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(VIDEO_DOWNLOAD_FOLDER, exist_ok=True)

# ================= GLOBAL =================

active_videos    = {}
done_videos      = set()   # video_id yang sudah selesai diproses, skip saat reload
video_lock       = Lock()

preview_frames   = {}
preview_lock     = Lock()

queue            = Queue(maxsize=300)
worker_semaphore = Semaphore(MAX_VIDEO_WORKERS)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

# ================= UTIL =================

def iso_name(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"


def enforce_limit(folder):

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )

    while len(files) > MAX_IMAGES:
        os.remove(files.pop(0))

def resize_image(frame):
        h, w = frame.shape[:2]

        scale = 300 / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        face_resized = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )
        return face_resized


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


def delete_video(local_path, vid):
    """Hapus file video dari folder download setelah selesai diproses."""
    try:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            print(f"[VIDEO DELETED] {vid} -> {local_path}")
    except Exception as e:
        print(f"[VIDEO DELETE ERROR] {vid} -> {e}")


def scan_local_videos():
    """
    Scan folder download, kembalikan list file video yang sudah ada
    dan belum sedang diproses (tidak ada di active_videos).
    """
    found = []

    try:
        for filename in os.listdir(VIDEO_DOWNLOAD_FOLDER):
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                local_path = os.path.join(VIDEO_DOWNLOAD_FOLDER, filename)
                found.append((filename, local_path))
    except Exception as e:
        print(f"[SCAN ERROR] {e}")

    return found


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

            face_bytes, frame_bytes, face_name, frame_name, bbox, score, vid, client_id, progress_pct = item

            payload = {
                "timestamp":  face_name,
                "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(score, 4),
                "video_id":   vid,
                "client_id":  client_id,
                "progress":   f"{progress_pct:.1f}%"
            }

            log_payload = {
                "url":        WEBHOOK_URL,
                "face_file":  face_name,
                "frame_file": frame_name,
                "data":       payload
            }

            print(f"\n[WEBHOOK] Sending:\n{json.dumps(log_payload, indent=4)}")

            requests.post(
                WEBHOOK_URL,
                files=[
                    ("files", (face_name,  face_bytes,  "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=payload,
                timeout=10
            )

            print(f"[WEBHOOK OK] {face_name} | conf={score:.3f} | progress={progress_pct:.1f}%")

        except Exception as e:
            print("[WEBHOOK ERROR]", e)

        finally:
            queue.task_done()


# def send_done_webhook(vid, client_id, video_filename):
#     """Kirim status done setelah video selesai diproses."""
#     if not WEBHOOK_URL:
#         return

#     try:

#         payload = {
#             "video_id":   vid,
#             "client_id":  client_id,
#             "status":     "done",
#             "progress":   "100.0%",
#             "video_file": video_filename,
#             "timestamp":  datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#         }

#         print(f"\n[WEBHOOK DONE] {json.dumps(payload, indent=4)}")

#         requests.post(
#             WEBHOOK_URL,
#             data=payload,
#             timeout=10
#         )

#         print(f"[WEBHOOK DONE OK] {vid} | video={video_filename}")

#     except Exception as e:
#         print("[WEBHOOK DONE ERROR]", e)


def send_status_webhook(video_id, client_id, status):
    """Kirim progress status per frame ke WEBHOOK_STATUS (non-blocking)."""
    if not WEBHOOK_STATUS:
        return

    def _send():
        try:
            payload = {
                "video_id":  video_id,
                "client_id": client_id,
                "status": "done" if status == 100 else "pending",
                "progress": status,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            requests.post(WEBHOOK_STATUS, json=payload, timeout=5)
            print(f"[STATUS] {video_id} | {status}")
        except Exception as e:
            print(f"[STATUS ERROR] {video_id} -> {e}")

    Thread(target=_send, daemon=True).start()


if WEBHOOK_URL:
    Thread(target=webhook_worker, daemon=True).start()


# ================= VIDEO DOWNLOADER =================

def download_video(url, vid):
    """Download video dari URL, simpan ke folder lokal. Return (local_path, filename) atau (None, None)."""

    try:

        parsed   = urlparse(url)
        filename = os.path.basename(parsed.path)

        if not filename or not filename.lower().endswith(VIDEO_EXTENSIONS):
            filename = f"{vid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        local_path = os.path.join(VIDEO_DOWNLOAD_FOLDER, filename)

        if os.path.exists(local_path):
            print(f"[DOWNLOAD SKIP] Sudah ada: {local_path}")
            return local_path, filename

        print(f"[DOWNLOAD START] {vid} -> {url}")

        with requests.get(url, stream=True, timeout=60) as r:

            r.raise_for_status()

            total      = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(local_path, "wb") as f:

                for chunk in r.iter_content(chunk_size=1024 * 1024):

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            pct = downloaded / total * 100
                            print(f"[DOWNLOAD] {vid} {pct:.1f}% ({downloaded}/{total} bytes)")

        print(f"[DOWNLOAD DONE] {vid} -> {local_path}")

        return local_path, filename

    except Exception as e:

        print(f"[DOWNLOAD ERROR] {vid} -> {e}")

        return None, None

# ================= VIDEO WORKER =================

class VideoWorker:

    def __init__(self, vid, client_id, url, local_path=None):

        self.vid            = vid
        self.video_id       = vid        # alias eksplisit untuk payload webhook
        self.client_id      = client_id
        self.video_url      = url

        # Jika sudah ada file lokal (dari scan), skip download
        self.preset_path    = local_path

        self.running        = True

        self.cap            = None
        self.local_path     = None
        self.video_filename = None
        self.total_frames   = 0

        self.face_memory    = {}
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
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector.setInputSize((640, 640))
        self.detector.detect(dummy)

        print(f"[VIDEO START] {vid}")

    def stop(self):

        self.running = False

        if self.cap:
            self.cap.release()

        print(f"[VIDEO STOP] {self.vid}")

    def resize_adaptive(self, frame):

        h, w  = frame.shape[:2]
        scale = min(1.0, TARGET_MAX_WIDTH / w)

        resized = cv2.resize(frame, None, fx=scale, fy=scale)

        if resized.shape[1] < 640:
            scale_up = 640 / resized.shape[1]
            resized  = cv2.resize(resized, None, fx=scale_up, fy=scale_up)
            scale    = scale / scale_up

        return resized, 1 / scale

    def cleanup_memory(self):

        now    = time.time()
        remove = [k for k, v in self.face_memory.items() if now - v[0] > 60]

        for k in remove:
            del self.face_memory[k]

    def get_progress(self, current_frame):
        """Hitung persentase progress berdasarkan frame saat ini vs total frame."""

        if self.total_frames <= 0:
            return 0.0

        # return min(100.0, (current_frame / self.total_frames) * 100.0)
        return min(100, round((current_frame / self.total_frames) * 100))

    

    def process(self):
        """Buka video, deteksi wajah per frame, kirim webhook."""

        self.cap = cv2.VideoCapture(self.local_path)

        if not self.cap.isOpened():
            print(f"[VIDEO ABORT] {self.vid} - tidak bisa buka video {self.local_path}")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps         = self.cap.get(cv2.CAP_PROP_FPS) or FRAME_FPS

        print(f"[VIDEO INFO] {self.vid} | frames={self.total_frames} | fps={video_fps:.1f}")

        frame_skip    = max(1, int(video_fps / FRAME_FPS))
        current_frame = 0

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                print(f"[VIDEO END] {self.vid}")
                break

            current_frame += 1

            # Skip frame untuk sesuaikan target FPS
            if (current_frame % frame_skip) != 0:
                continue

            view = frame.copy()

            resized = view #self.resize_adaptive(frame)
            scale = 1.0
            # keep original frame for final crop
            # original_frame = frame.copy()

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

                    x  = int(x  * scale)
                    y  = int(y  * scale)
                    fw = int(fw * scale)
                    fh = int(fh * scale)

                    if fw < MIN_SIZE_CAPTURE_VIDEO:
                        continue

                    if fw > MAX_FACE_SIZE:
                        continue

                    if not is_valid_face(f, x, y, fw, fh, scale):
                        continue

                    x  = max(0, x)
                    y  = max(0, y)
                    fw = min(fw, frame.shape[1] - x)
                    fh = min(fh, frame.shape[0] - y)

                    boxes.append((x, y, fw, fh, score))

            for box in boxes:

                x, y, fw, fh, score = box

                cv2.rectangle(view, (x, y), (x+fw, y+fh), (0, 255, 0), 2)

                cx = x + fw // 2
                cy = y + fh // 2

                bucket = (cx // FACE_BUCKET_SIZE, cy // FACE_BUCKET_SIZE)

                now = time.time()

                if bucket in self.face_memory:

                    last_time, last_pos = self.face_memory[bucket]

                    dist = math.hypot(cx - last_pos[0], cy - last_pos[1])

                    if now - last_time < FACE_COOLDOWN:
                        continue

                    if dist < FACE_MOVE_THRESHOLD:
                        continue

                self.face_memory[bucket] = (now, (cx, cy))
                self.last_face_time      = time.time()

                cx2, cy2, cw, ch = expand_crop_bbox(
                    x, y, fw, fh,
                    frame.shape[1],
                    frame.shape[0]
                )

                face_img = frame[cy2:cy2+ch, cx2:cx2+cw]

                if face_img.size == 0:
                    continue

                

                if calc_sharpness(face_img) < BLUR_THRESHOLD_VIDEO:
                    continue

                face_name  = iso_name("face")
                frame_name = iso_name("frame")

                resize_face = resize_image(face_img)

                fb1 = cv2.imencode(".jpg", resize_face)[1].tobytes()
                fb2 = cv2.imencode(".jpg", view)[1].tobytes()

                open(os.path.join(FACE_FOLDER,  face_name),  "wb").write(fb1)
                open(os.path.join(FRAME_FOLDER, frame_name), "wb").write(fb2)

                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                progress_pct = self.get_progress(current_frame)

                if WEBHOOK_URL:
                    try:
                        queue.put_nowait(
                            (fb1, fb2, face_name, frame_name,
                             (x, y, fw, fh),
                             score,
                             self.vid,
                             self.client_id,
                             progress_pct)
                        )
                    except Exception:
                        print("[QUEUE FULL] webhook queue penuh")

                print(f"[{self.vid}] SAVED {face_name} | progress={progress_pct:.1f}%")

            self.cleanup_memory()

            # Kirim progress status tiap frame yang diproses
            progress_pct = self.get_progress(current_frame)
            if progress_pct % 10 == 0:
                send_status_webhook(
                    self.video_id,
                    self.client_id,
                    f"{progress_pct}"
                )

            with preview_lock:
                preview_frames[self.vid] = view

        if self.cap:
            self.cap.release()
            self.cap = None

    def run(self):

        worker_semaphore.acquire()

        try:

            if self.preset_path:

                # ---- File sudah ada di lokal (dari scan folder) ----
                self.local_path     = self.preset_path
                self.video_filename = os.path.basename(self.preset_path)

                print(f"[VIDEO LOCAL] {self.vid} - pakai file lokal: {self.local_path}")

            else:

                # ---- Download video dari URL ----
                self.local_path, self.video_filename = download_video(self.video_url, self.vid)

                if not self.local_path:
                    print(f"[VIDEO ABORT] {self.vid} - download gagal")
                    return

            # ---- Proses video (dengan opsional loop) ----
            while self.running:

                self.face_memory = {}

                self.process()

                if not VIDEO_LOOP or not self.running:
                    break

                print(f"[VIDEO LOOP] {self.vid} - mengulang dari awal")

            # # ---- Selesai ----
            # send_done_webhook(self.video_id, self.client_id, self.video_filename)

            delete_video(self.local_path, self.video_id)

            print(f"[{self.video_id}] PROCESSING DONE")

        finally:

            worker_semaphore.release()

            # Catat done SEBELUM hapus dari active_videos,
            # supaya manager tidak sempat menganggap video ini sebagai baru
            done_videos.add(self.video_id)

            with video_lock:
                active_videos.pop(self.video_id, None)

            with preview_lock:
                preview_frames.pop(self.vid, None)


# ================= LOAD VIDEOS =================

def load_videos():

    print("[PRODUCTION MODE] Load video dari endpoint")

    try:

        r = requests.get(f"{ENDPOINT_URL}?service_id={SERVICE_ID}", timeout=10)

        data = r.json()["data"]

        print(f"[API] Total video dari server: {len(data)}")

        return data

    except Exception as e:

        print("[API ERROR]", e)

        return []


# ================= VIDEO MANAGER =================

def video_manager():

    print("\n=== VIDEO MANAGER STARTED ===")

    while True:

        # ---- STEP 1: Ambil daftar dari endpoint dulu ----
        videos = load_videos()

        # Buat mapping: filename -> video entry dari API
        # strip query string agar basename cocok dengan nama file yang didownload
        api_filename_map = {}
        for v in videos:
            parsed    = urlparse(v["video_url"])
            api_fname = os.path.basename(parsed.path)   # ambil dari path saja, bukan query
            if api_fname:
                api_filename_map[api_fname] = v
                # fallback tanpa ekstensi untuk jaga-jaga
                api_filename_map[os.path.splitext(api_fname)[0]] = v

        api_ids = set([v["video_id"] for v in videos])

        # ---- STEP 2: Cek file lokal yang belum diproses ----
        local_files = scan_local_videos()

        with video_lock:
            active_ids = set(active_videos.keys())

        pending_local = []

        for fname, fpath in local_files:

            # Cek apakah file ini punya pasangan video_id dari API
            # coba match: nama file lengkap dulu, lalu nama tanpa ekstensi
            api_entry = api_filename_map.get(fname) or api_filename_map.get(os.path.splitext(fname)[0])

            if api_entry:
                vid       = api_entry["video_id"]
                client_id = api_entry["client_id"]
            else:
                # Tidak ada di API — log warning, skip saja (jangan proses tanpa video_id valid)
                print(f"[SCAN SKIP] {fname} - tidak ditemukan di endpoint, skip")
                continue

            # Skip jika sudah aktif atau sudah selesai (pakai video_id yg benar)
            if vid in active_ids or vid in done_videos:
                continue

            pending_local.append((vid, client_id, fpath))

        if pending_local:

            print(f"[SCAN] Ditemukan {len(pending_local)} file lokal belum diproses, proses dulu...")

            with video_lock:

                for vid, client_id, local_path in pending_local:

                    if vid in active_videos:
                        continue

                    try:

                        w = VideoWorker(
                            vid=vid,
                            client_id=client_id,
                            url="",
                            local_path=local_path
                        )

                        Thread(target=w.run, daemon=True).start()

                        active_videos[vid] = w

                        print(f"[LOCAL VIDEO STARTED] {vid} -> {local_path}")

                    except Exception as e:

                        print(f"[FAILED START LOCAL] {vid} -> {e}")

            # Tunggu semua file lokal selesai sebelum lanjut ke endpoint
            print("[SCAN] Menunggu file lokal selesai diproses...")

            while True:

                with video_lock:
                    still_running = any(
                        vid in active_videos for vid, _, _ in pending_local
                    )

                if not still_running:
                    print("[SCAN] Semua file lokal selesai, lanjut ke endpoint...")
                    break

                time.sleep(2)

        # ---- STEP 3: Proses video baru dari endpoint ----
        with video_lock:

            active_ids = set(active_videos.keys())

            new_ids = api_ids - active_ids - done_videos

            for v in videos:

                if v["video_id"] in new_ids:

                    # Juga skip jika filenya sudah ada di folder download
                    parsed    = urlparse(v["video_url"])
                    api_fname = os.path.basename(parsed.path)
                    local_path = os.path.join(VIDEO_DOWNLOAD_FOLDER, api_fname)

                    if os.path.exists(local_path):
                        print(f"[SKIP] {v['video_id']} - file sudah ada, tunggu giliran proses lokal")
                        continue

                    try:

                        w = VideoWorker(
                            v["video_id"],
                            v["client_id"],
                            v["video_url"]
                        )

                        Thread(target=w.run, daemon=True).start()

                        active_videos[v["video_id"]] = w

                        print(f"[NEW VIDEO STARTED] {v['video_id']}")

                    except Exception as e:

                        print(f"[FAILED START] {v['video_id']} -> {e}")

            # Hanya hentikan video yang hilang dari API dan belum/sedang diproses
            removed_ids = active_ids - api_ids - done_videos

            for rid in removed_ids:

                try:

                    active_videos[rid].stop()
                    del active_videos[rid]
                    print(f"[VIDEO REMOVED] {rid}")
                except Exception:
                    pass

            print("------ STATUS ------")
            print("Total Source :", len(api_ids))
            print("Active Worker:", len(active_videos))
            print("--------------------\n")

        time.sleep(VIDEO_REFRESH_INTERVAL)


Thread(target=video_manager, daemon=True).start()


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

        tw = DISPLAY_WIDTH  // cols
        th = DISPLAY_HEIGHT // rows

        imgs = []

        for f in frames:

            try:
                imgs.append(cv2.resize(f, (tw, th)))
            except Exception:
                imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        grid = []
        idx  = 0

        for r in range(rows):

            row_imgs = imgs[idx:idx+cols]

            try:
                grid.append(cv2.hconcat(row_imgs))
            except Exception:
                grid.append(np.zeros((th, tw * cols, 3), dtype=np.uint8))

            idx += cols

        try:
            final = cv2.vconcat(grid)
        except Exception:
            final = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        cv2.imshow("Face Detection", final)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:

    while True:
        time.sleep(1)