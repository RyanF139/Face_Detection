# =========================================================
# MULTI VIDEO FACE CAPTURE
# CPU ONLY (YuNet) — Source: MP4 URL / YouTube
# Result: annotated MP4 → upload S3 → webhook done + s3_url
# =========================================================
import cv2
import time
import os
import math
import glob
import requests
import numpy as np
import json
import subprocess
import shutil
import tempfile
import boto3

from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue

load_dotenv()

# =========================================================
# ====================== CONFIG (.env) =====================
# =========================================================

SERVICE_ID        = os.getenv("SERVICE_ID")
ENDPOINT_URL      = os.getenv("VIDEO_ENDPOINT")
MODEL_PATH        = os.getenv("MODEL_PATH")

SCORE_THRESHOLD   = float(os.getenv("SCORE_THRESHOLD", 0.6))
BLUR_THRESHOLD    = float(os.getenv("BLUR_THRESHOLD", 0))

SAVE_FOLDER       = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL     = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES        = int(os.getenv("MAX_IMAGES", 150))

TARGET_MAX_WIDTH  = int(os.getenv("RESIZE_WIDTH", 640))
TARGET_MAX_HEIGHT = int(os.getenv("RESIZE_HEIGHT", 360))
MIN_SIZE_CAPTURE  = int(os.getenv("MIN_SIZE_CAPTURE", 0))

FRAME_FPS         = int(os.getenv("FRAME_FPS", 12))
CROP_PADDING      = float(os.getenv("CROP_PADDING", 0.40))

ENABLE_VIEW       = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH     = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT    = int(os.getenv("DISPLAY_HEIGHT", 800))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

DEBUG_MODE         = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_FOLDER = os.getenv("DEBUG_VIDEO_FOLDER", "./sample_video")

# BUG FIX #1: WEBHOOK_API diambil dari env, bukan import dari module lain
# Sebelumnya: from video_worker_v2 import WEBHOOK_API  ← ini menyebabkan
# WEBHOOK_API mungkin None / stale / berbeda environment
WEBHOOK_API = os.getenv("WEBHOOK_API", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

VIDEO_LOOP  = os.getenv("VIDEO_LOOP", "false").lower() == "true"

# ── S3 ────────────────────────────────────────────────────
S3_ENABLED        = os.getenv("S3_ENABLED", "false").lower() == "true"
S3_BUCKET         = os.getenv("S3_BUCKET", "")
S3_REGION         = os.getenv("S3_REGION", "ap-southeast-1")
S3_PREFIX         = os.getenv("S3_PREFIX", "results")
S3_ACCESS_KEY     = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY     = os.getenv("S3_SECRET_KEY", "")
S3_ENDPOINT_URL   = os.getenv("S3_ENDPOINT_URL", "")
S3_PUBLIC_BASE    = os.getenv("S3_PUBLIC_BASE", "")

# ── Result MP4 ────────────────────────────────────────────
RESULT_VIDEO_FPS     = int(os.getenv("RESULT_VIDEO_FPS", str(FRAME_FPS)))
RESULT_VIDEO_BITRATE = os.getenv("RESULT_VIDEO_BITRATE", "1000k")

# ── Startup diagnostics ───────────────────────────────────
print("ENABLE_VIEW  :", ENABLE_VIEW)
print("S3_ENABLED   :", S3_ENABLED)
print("VIDEO_LOOP   :", VIDEO_LOOP)
print("WEBHOOK_API  :", WEBHOOK_API)
print("S3_BUCKET    :", S3_BUCKET)

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

FACE_FOLDER   = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER  = os.path.join(SAVE_FOLDER, "frame")
RESULT_FOLDER = os.path.join(SAVE_FOLDER, "result")

os.makedirs(FACE_FOLDER,   exist_ok=True)
os.makedirs(FRAME_FOLDER,  exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

print("=== APP STARTED ===")
print("SERVICE_ID:", SERVICE_ID)

# =========================================================
# ================= GLOBAL STORAGE =========================
# =========================================================

preview_frames = {}
preview_lock   = Lock()

active_cameras = {}
camera_lock    = Lock()

webhook_queue  = Queue(maxsize=300)

# =========================================================
# ================= S3 UPLOADER ============================
# =========================================================

class S3Uploader:
    def __init__(self):
        kwargs = dict(
            region_name           = S3_REGION,
            aws_access_key_id     = S3_ACCESS_KEY,
            aws_secret_access_key = S3_SECRET_KEY,
        )
        if S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = S3_ENDPOINT_URL

        self._client = boto3.client("s3", **kwargs)
        print(f"[S3] Client ready — bucket: {S3_BUCKET}, region: {S3_REGION}")

    def upload_file(self, local_path: str, s3_key: str,
                    content_type: str = "video/mp4") -> str:
        print(f"[S3] Uploading {os.path.basename(local_path)} → s3://{S3_BUCKET}/{s3_key}")
        self._client.upload_file(
            local_path, S3_BUCKET, s3_key,
            ExtraArgs={"ContentType": content_type}
        )
        url = self._build_url(s3_key)
        print(f"[S3] Upload OK → {url}")
        return url

    def _build_url(self, s3_key: str) -> str:
        if S3_PUBLIC_BASE:
            return f"{S3_PUBLIC_BASE.rstrip('/')}/{s3_key}"
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=604800
        )


# BUG FIX #2: Validasi S3 config sebelum inisialisasi
# Sebelumnya: langsung S3Uploader() tanpa cek → boto3 error diam-diam
# atau upload ke bucket kosong
if S3_ENABLED:
    if not S3_BUCKET or not S3_ACCESS_KEY or not S3_SECRET_KEY:
        print("[S3] WARNING: S3_ENABLED=true tapi config tidak lengkap!")
        print(f"     S3_BUCKET={S3_BUCKET!r}")
        print(f"     S3_ACCESS_KEY={'set' if S3_ACCESS_KEY else 'KOSONG'}")
        print(f"     S3_SECRET_KEY={'set' if S3_SECRET_KEY else 'KOSONG'}")
        s3_uploader = None
    else:
        try:
            s3_uploader = S3Uploader()
        except Exception as e:
            print(f"[S3] INIT ERROR: {e}")
            s3_uploader = None
else:
    s3_uploader = None

print(f"[S3] s3_uploader = {s3_uploader}")

# =========================================================
# ================= FRAME VIDEO WRITER =====================
# =========================================================

class FrameVideoWriter:
    def __init__(self, video_id: str):
        self.video_id  = video_id
        self.frame_idx = 0
        self.tmp_dir   = tempfile.mkdtemp(prefix=f"frames_{video_id}_")
        print(f"[WRITER] {video_id} → tmp: {self.tmp_dir}")

    def write(self, frame: np.ndarray):
        path = os.path.join(self.tmp_dir, f"{self.frame_idx:07d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.frame_idx += 1

    def render(self) -> str | None:
        if self.frame_idx == 0:
            print(f"[WRITER] {self.video_id} — tidak ada frame, skip render")
            self._cleanup()
            return None

        ts       = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(RESULT_FOLDER, f"result_{self.video_id}_{ts}.mp4")

        print(f"[WRITER] Rendering {self.frame_idx} frames → {out_path}")

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            "-framerate", str(RESULT_VIDEO_FPS),
            "-i", os.path.join(self.tmp_dir, "%07d.jpg"),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-b:v", RESULT_VIDEO_BITRATE,
            "-movflags", "+faststart",
            out_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[WRITER] FFmpeg error:\n{result.stderr}")
                return None
            print(f"[WRITER] Render OK: {out_path}")
            return out_path
        except subprocess.TimeoutExpired:
            print(f"[WRITER] FFmpeg timeout: {self.video_id}")
            return None
        except FileNotFoundError:
            print("[WRITER] ffmpeg tidak ditemukan. Install: sudo apt install ffmpeg")
            return None
        finally:
            self._cleanup()

    def _cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        print(f"[WRITER] Temp cleaned: {self.tmp_dir}")


# =========================================================
# ================= WEBHOOK WORKER (face events) ===========
# =========================================================

def webhook_worker():
    """Thread pool untuk kirim face detection event."""
    while True:
        item = webhook_queue.get()
        if item is None:
            break

        face_bytes, frame_bytes, face_name, frame_name, ts_iso, bbox, video_id, client_id = item

        try:
            # BUG FIX #3: webhook face event tidak boleh di-comment
            # Sebelumnya seluruh requests.post di-comment → tidak ada yang dikirim
            if not WEBHOOK_API:
                print("[WEBHOOK] WEBHOOK_API tidak di-set, skip face event")
                continue

            data_payload = {
                "timestamp":  ts_iso,
                "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "video_id": video_id,
                "client_id":  client_id
            }

            print(f"[WEBHOOK] Sending face event: {face_name}")

            response = requests.post(
                WEBHOOK_API + "/webhook/detection-video",
                files=[
                    ("files", (face_name,  face_bytes,  "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=data_payload,
                timeout=10
            )

            print(f"[WEBHOOK FACE] Status: {response.status_code}")
            if response.status_code >= 400:
                print("[WEBHOOK FACE] ERROR:", response.text)

        except requests.exceptions.Timeout:
            print("[WEBHOOK FACE] TIMEOUT")
        except requests.exceptions.ConnectionError:
            print("[WEBHOOK FACE] CONNECTION ERROR")
        except Exception as e:
            print("[WEBHOOK FACE] ERROR:", e)
        finally:
            webhook_queue.task_done()


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


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def resolve_stream_url(video_url: str) -> str:
    if is_youtube_url(video_url):
        if shutil.which("yt-dlp") is None:
            print("[WARNING] yt-dlp tidak ditemukan. Install: pip install yt-dlp")
            return video_url
        try:
            result = subprocess.run(
                ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                 "--get-url", video_url],
                capture_output=True, text=True, timeout=30
            )
            urls = result.stdout.strip().splitlines()
            if urls:
                print(f"[yt-dlp] Resolved → {urls[0][:80]}...")
                return urls[0]
        except Exception as e:
            print(f"[yt-dlp] Error: {e}")
    return video_url


# =========================================================
# ================= DEBUG VIDEO LOADER =====================
# =========================================================

def load_debug_sources(folder):
    files = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        files += glob.glob(os.path.join(folder, ext))
    sources = []
    for i, path in enumerate(files):
        sources.append({"video_id": f"debug_{i}", "client_id": "debug", "video_url": path})
    print("[DEBUG MODE] Found", len(sources), "files")
    return sources


# =========================================================
# ================= VIDEO WORKER ===========================
# =========================================================

class VideoWorker:

    def __init__(self, video_id: str, client_id: str, video_url: str):
        self.video_id   = video_id
        self.client_id  = client_id
        self.video_url  = video_url
        self.running    = True

        self.stream_url = resolve_stream_url(video_url)
        self.cap        = self._open_cap(self.stream_url)

        self.frame_interval = 1 / FRAME_FPS
        self.last_time      = 0
        self.bad            = 0
        self.max_bad        = 20
        self.last_save      = 0
        self.face_last_time = {}

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (320, 320),
            score_threshold=SCORE_THRESHOLD
        )
        self.detector.setInputSize((320, 320))
        self.detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

        self.writer = FrameVideoWriter(video_id)

        print(f"[VIDEO START] {video_id} | {video_url[:70]}...")

    def _open_cap(self, url: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(url) if os.path.isfile(url) \
              else cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def resize_adaptive(self, frame):
        h, w = frame.shape[:2]
        if w > TARGET_MAX_WIDTH:
            scale = TARGET_MAX_WIDTH / w
            return cv2.resize(frame, None, fx=scale, fy=scale), 1/scale, 1/scale
        return frame, 1.0, 1.0

    # ── destroy ─────────────────────────────────────────────

    def destroy(self, reason: str = "done"):
        """
        Stop worker lalu spawn thread _finalize.
        Webhook HANYA dikirim dari _finalize setelah render + S3 selesai.

        BUG FIX #4: daemon=False agar thread _finalize tidak ikut mati
        ketika main thread / program exit sebelum upload selesai.
        Sebelumnya daemon=True → jika program exit saat upload sedang
        berjalan, thread langsung dibunuh → webhook tidak pernah terkirim.
        """
        self.running = False

        try:
            self.cap.release()
        except Exception:
            pass

        with preview_lock:
            preview_frames.pop(self.video_id, None)

        print(f"[{self.video_id}] Destroyed. Reason: {reason}")

        Thread(
            target=self._finalize,
            args=(reason,),
            daemon=False,                    # ← PENTING: bukan daemon
            name=f"finalize-{self.video_id}"
        ).start()

    # ── finalize pipeline ───────────────────────────────────

    def _finalize(self, reason: str):
        """
        Urutan eksekusi (sequential, di thread non-daemon):
        1. render() → MP4 lokal
        2. upload ke S3 → dapat s3_url
        3. kirim webhook done dengan s3_url
        """
        print(f"[FINALIZE] Start: {self.video_id} reason={reason}")

        s3_url     = None
        local_path = None

        # ── Step 1: Render MP4 ─────────────────────────────
        try:
            local_path = self.writer.render()
            print(f"[FINALIZE] Render result: {local_path}")
        except Exception as e:
            print(f"[FINALIZE] Render error: {e}")

        # ── Step 2: Upload S3 ──────────────────────────────
        if local_path and s3_uploader:
            try:
                filename = os.path.basename(local_path)
                s3_key   = f"{S3_PREFIX}/{self.client_id}/{self.video_id}/{filename}"
                s3_url   = s3_uploader.upload_file(local_path, s3_key)
            except (BotoCoreError, ClientError) as e:
                print(f"[FINALIZE] S3 boto error: {e}")
            except Exception as e:
                print(f"[FINALIZE] S3 unexpected error: {e}")
        else:
            # Log kenapa S3 di-skip agar mudah debug
            print(f"[FINALIZE] S3 skip — "
                  f"local_path={local_path!r}, "
                  f"s3_uploader={'ready' if s3_uploader else 'None (S3_ENABLED=false atau config salah)'}")

        # ── Step 3: Webhook done ───────────────────────────
        # Selalu kirim, bahkan jika S3 gagal (s3_url=None)
        self._send_done_webhook(reason, s3_url, local_path)

        print(f"[FINALIZE] Done: {self.video_id}")

    def _send_done_webhook(self, reason: str, s3_url, local_path):
        if not WEBHOOK_API:
            print("[WEBHOOK DONE] WEBHOOK_API tidak di-set, skip")
            return

        payload = {
            "video_id":   self.video_id,
            "client_id":  self.client_id,
            "status":     reason,
            "timestamp":  datetime.utcnow().isoformat() + "Z",
            "result": {
                "s3_url":     s3_url,
                "local_path": local_path,
                "uploaded":   s3_url is not None,
            }
        }

        print(f"[WEBHOOK DONE] Sending:\n{json.dumps(payload, indent=2)}")

        try:
            resp = requests.post(
                WEBHOOK_API + "/webhook/receive-status-video",
                json=payload,
                timeout=15
            )
            print(f"[WEBHOOK DONE] Status: {resp.status_code}")
            if resp.status_code >= 400:
                print("[WEBHOOK DONE] ERROR:", resp.text)
        except requests.exceptions.Timeout:
            print("[WEBHOOK DONE] TIMEOUT")
        except requests.exceptions.ConnectionError:
            print("[WEBHOOK DONE] CONNECTION ERROR")
        except Exception as e:
            print("[WEBHOOK DONE] ERROR:", e)

    # ── main loop ───────────────────────────────────────────

    def run(self):
        while self.running:

            now = time.time()
            if now - self.last_time < self.frame_interval:
                time.sleep(0.005)
                continue

            self.last_time = time.time()

            ret, frame = self.cap.read()

            if not ret:
                self.bad += 1

                if self.bad >= self.max_bad:
                    print(f"[{self.video_id}] Stream ended. Loop={VIDEO_LOOP}")

                    if VIDEO_LOOP:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if not os.path.isfile(self.stream_url):
                            self.cap.release()
                            time.sleep(2)
                            self.stream_url = resolve_stream_url(self.video_url)
                            self.cap = self._open_cap(self.stream_url)
                        self.bad = 0
                    else:
                        self.destroy(reason="done")
                        with camera_lock:
                            active_cameras.pop(self.video_id, None)
                continue

            self.bad = 0

            resized, sx, sy = self.resize_adaptive(frame)
            h, w = resized.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(resized)

            boxes = []
            if faces is not None:
                for f in faces:
                    score = float(f[4])
                    if score < SCORE_THRESHOLD:
                        continue
                    x, y, fw, fh = f[:4].astype(int)
                    x  = int(x * sx);  y  = int(y * sy)
                    fw = int(fw * sx); fh = int(fh * sy)
                    if fw < MIN_SIZE_CAPTURE:
                        continue
                    x  = max(0, x);  y  = max(0, y)
                    fw = min(fw, frame.shape[1] - x)
                    fh = min(fh, frame.shape[0] - y)
                    boxes.append((x, y, fw, fh))

            view = frame.copy()
            for box in boxes:
                x, y, fw, fh = box

                cv2.rectangle(view, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(view, f"{score:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                bucket = (x // 150, y // 150)
                if bucket in self.face_last_time and \
                   time.time() - self.face_last_time[bucket] < SAVE_INTERVAL:
                    continue
                self.face_last_time[bucket] = time.time()

                pad_w = int(fw * CROP_PADDING)
                pad_h = int(fh * CROP_PADDING)
                cx    = max(0, x - pad_w)
                cy    = max(0, y - pad_h)
                cw    = min(frame.shape[1] - cx, fw + pad_w * 2)
                ch    = min(frame.shape[0] - cy, fh + pad_h * 2)

                face_img  = frame[cy:cy+ch, cx:cx+cw]
                gray      = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < BLUR_THRESHOLD:
                    continue

                face_name  = iso_name("face")
                frame_name = iso_name("frame")

                fb1 = cv2.imencode(".jpg", face_img)[1].tobytes()
                fb2 = cv2.imencode(".jpg", view)[1].tobytes()

                open(os.path.join(FACE_FOLDER,  face_name),  "wb").write(fb1)
                open(os.path.join(FRAME_FOLDER, frame_name), "wb").write(fb2)

                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                ts_iso = datetime.utcnow().isoformat() + "Z"
                webhook_queue.put((
                    fb1, fb2, face_name, frame_name,
                    ts_iso, box, self.video_id, self.client_id
                ))

                print(f"[{self.video_id}] Face: {face_name}")
                self.last_save = time.time()

            self.writer.write(view)

            with preview_lock:
                preview_frames[self.video_id] = view


# =========================================================
# ================= FETCH VIDEO LIST =======================
# =========================================================

def fetch_videos():
    try:
        r = requests.get(WEBHOOK_API + "/webhook/videos", timeout=10)
        payload = r.json()
        if payload.get("ok"):
            return payload["data"]
        print("[fetch_videos] ok=false:", payload)
        return []
    except Exception as e:
        print("[fetch_videos] ERROR:", e)
        return []


# =========================================================
# ================= VIDEO MANAGER ==========================
# =========================================================

def video_manager():
    while True:
        sources = load_debug_sources(DEBUG_VIDEO_FOLDER) if DEBUG_MODE \
                  else fetch_videos()

        with camera_lock:
            existing    = set(active_cameras.keys())
            current_ids = {c["video_id"] for c in sources}

            for c in sources:
                vid = c["video_id"]
                if vid not in existing:
                    w = VideoWorker(vid, c["client_id"], c["video_url"])
                    Thread(target=w.run, daemon=True).start()
                    active_cameras[vid] = w
                    print(f"[MANAGER] Started: {vid}")

            for vid in list(existing):
                if vid not in current_ids:
                    print(f"[MANAGER] Removing: {vid}")
                    active_cameras[vid].destroy(reason="removed")
                    del active_cameras[vid]

        time.sleep(CAMERA_REFRESH_INTERVAL)


Thread(target=video_manager, daemon=True).start()
print("System ready — waiting for frames...")

# =========================================================
# ================= MAIN DISPLAY LOOP ======================
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
        cols = math.ceil(len(frames) / rows)

        tw = DISPLAY_WIDTH  // cols
        th = DISPLAY_HEIGHT // rows

        imgs = [cv2.resize(f, (tw, th)) for f in frames]

        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        grid = []
        idx  = 0
        for r in range(rows):
            grid.append(cv2.hconcat(imgs[idx:idx+cols]))
            idx += cols

        cv2.imshow("Face Detection", cv2.vconcat(grid))

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:
    print("Running headless mode")
    while True:
        time.sleep(1)