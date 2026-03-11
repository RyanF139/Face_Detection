# =========================================================
# MULTI VIDEO FACE CAPTURE — PRODUCTION READY
# CPU Only (YuNet) | YouTube + MP4 URL + File Lokal
# Arsitektur: Download → ReaderThread → Queue → DetectorThread
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
import logging
import signal
import sys

from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock, Event
from queue import Queue, Empty

load_dotenv()

# =========================================================
# =================== LOGGING SETUP =======================
# =========================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("FaceCapture")

# =========================================================
# ====================== CONFIG (.env) ====================
# =========================================================

SERVICE_ID        = os.getenv("SERVICE_ID", "default-service")
WEBHOOK_API       = os.getenv("WEBHOOK_API", "")
WEBHOOK_URL       = os.getenv("WEBHOOK_URL", "")

MODEL_PATH        = os.getenv("MODEL_PATH", "face_detection_yunet_2023mar.onnx")
SCORE_THRESHOLD   = float(os.getenv("SCORE_THRESHOLD", "0.6"))
BLUR_THRESHOLD    = float(os.getenv("BLUR_THRESHOLD", "0"))
MIN_SIZE_CAPTURE  = int(os.getenv("MIN_SIZE_CAPTURE", "0"))

SAVE_FOLDER       = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL     = float(os.getenv("SAVE_INTERVAL", "2.0"))
MAX_IMAGES        = int(os.getenv("MAX_IMAGES", "150"))

TARGET_MAX_WIDTH  = int(os.getenv("RESIZE_WIDTH", "640"))
TARGET_MAX_HEIGHT = int(os.getenv("RESIZE_HEIGHT", "360"))

FRAME_FPS         = int(os.getenv("FRAME_FPS", "12"))
CROP_PADDING      = float(os.getenv("CROP_PADDING", "0.40"))

ENABLE_VIEW       = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH     = int(os.getenv("DISPLAY_WIDTH", "1200"))
DISPLAY_HEIGHT    = int(os.getenv("DISPLAY_HEIGHT", "800"))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", "60"))

DEBUG_MODE         = os.getenv("DEBUG_MODE", "false").lower() == "true"
DEBUG_VIDEO_FOLDER = os.getenv("DEBUG_VIDEO_FOLDER", "./sample_video")

VIDEO_LOOP        = os.getenv("VIDEO_LOOP", "false").lower() == "true"

# ── Download config ───────────────────────────────────────
DOWNLOAD_FOLDER      = os.getenv("DOWNLOAD_FOLDER", "/tmp/video_cache")
DOWNLOAD_TIMEOUT_YT  = int(os.getenv("DOWNLOAD_TIMEOUT_YT", "300"))   # 5 menit
DOWNLOAD_TIMEOUT_HTTP= int(os.getenv("DOWNLOAD_TIMEOUT_HTTP", "120"))  # 2 menit
DOWNLOAD_MAX_SIZE_MB = int(os.getenv("DOWNLOAD_MAX_SIZE_MB", "500"))   # 500 MB limit
YT_MAX_HEIGHT        = int(os.getenv("YT_MAX_HEIGHT", "720"))

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

# ── Detection input size (fixed) ─────────────────────────
DETECT_W = int(os.getenv("DETECT_W", "640"))
DETECT_H = int(os.getenv("DETECT_H", "360"))

# ── Startup diagnostics ───────────────────────────────────
log.info(f"SERVICE_ID    : {SERVICE_ID}")
log.info(f"ENABLE_VIEW   : {ENABLE_VIEW}")
log.info(f"S3_ENABLED    : {S3_ENABLED}")
log.info(f"VIDEO_LOOP    : {VIDEO_LOOP}")
log.info(f"WEBHOOK_API   : {WEBHOOK_API or '(not set)'}")
log.info(f"S3_BUCKET     : {S3_BUCKET or '(not set)'}")
log.info(f"DOWNLOAD_FOLDER: {DOWNLOAD_FOLDER}")

# =========================================================
# ================= SAFE MODEL PATH =======================
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)
MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model ONNX tidak ditemukan: {MODEL_PATH}")

log.info(f"MODEL_PATH    : {MODEL_PATH}")

# =========================================================
# ================= FOLDER SETUP ==========================
# =========================================================

FACE_FOLDER   = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER  = os.path.join(SAVE_FOLDER, "frame")
RESULT_FOLDER = os.path.join(SAVE_FOLDER, "result")

for folder in (FACE_FOLDER, FRAME_FOLDER, RESULT_FOLDER, DOWNLOAD_FOLDER):
    os.makedirs(folder, exist_ok=True)

log.info("=== APP STARTED ===")

# =========================================================
# ================= GLOBAL STORAGE ========================
# =========================================================

preview_frames: dict[str, np.ndarray] = {}
preview_lock   = Lock()

active_cameras: dict[str, "VideoWorker"] = {}
camera_lock    = Lock()

shutdown_event = Event()

# =========================================================
# ================= ASYNC QUEUES ==========================
# =========================================================

# Queue untuk kirim face detection event ke webhook
webhook_queue = Queue(maxsize=300)

# Queue untuk async file write (tidak blocking di detector thread)
file_write_queue = Queue(maxsize=500)

# =========================================================
# ================= S3 UPLOADER ===========================
# =========================================================

class S3Uploader:
    def __init__(self):
        kwargs: dict = dict(
            region_name           = S3_REGION,
            aws_access_key_id     = S3_ACCESS_KEY,
            aws_secret_access_key = S3_SECRET_KEY,
        )
        if S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = S3_ENDPOINT_URL
        self._client = boto3.client("s3", **kwargs)
        log.info(f"[S3] Client ready — bucket={S3_BUCKET}, region={S3_REGION}")

    def upload_file(self, local_path: str, s3_key: str,
                    content_type: str = "video/mp4") -> str:
        log.info(f"[S3] Uploading {os.path.basename(local_path)} → s3://{S3_BUCKET}/{s3_key}")
        self._client.upload_file(
            local_path, S3_BUCKET, s3_key,
            ExtraArgs={"ContentType": content_type}
        )
        url = self._build_url(s3_key)
        log.info(f"[S3] Upload OK → {url}")
        return url

    def _build_url(self, s3_key: str) -> str:
        if S3_PUBLIC_BASE:
            return f"{S3_PUBLIC_BASE.rstrip('/')}/{s3_key}"
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=604800
        )


def _init_s3() -> "S3Uploader | None":
    if not S3_ENABLED:
        return None
    if not S3_BUCKET or not S3_ACCESS_KEY or not S3_SECRET_KEY:
        log.warning("[S3] S3_ENABLED=true tapi config tidak lengkap — S3 dinonaktifkan")
        log.warning(f"     S3_BUCKET     = {S3_BUCKET!r}")
        log.warning(f"     S3_ACCESS_KEY = {'set' if S3_ACCESS_KEY else 'KOSONG'}")
        log.warning(f"     S3_SECRET_KEY = {'set' if S3_SECRET_KEY else 'KOSONG'}")
        return None
    try:
        return S3Uploader()
    except Exception as e:
        log.error(f"[S3] Init error: {e}")
        return None


s3_uploader = _init_s3()
log.info(f"[S3] uploader = {'ready' if s3_uploader else 'disabled'}")

# =========================================================
# ================= DOWNLOAD MANAGER =====================
# =========================================================

class DownloadManager:
    """
    Menangani download video dari berbagai sumber ke lokal.
    - File lokal  → return path langsung
    - YouTube     → yt-dlp download ke /tmp/video_cache/<video_id>.mp4
    - MP4 URL     → requests chunked download ke /tmp/video_cache/<video_id>.mp4

    Download dilakukan sekali, hasil di-cache berdasarkan video_id.
    """

    _lock = Lock()

    def resolve(self, video_id: str, video_url: str) -> "str | None":
        """
        Return local path siap diproses, atau None jika gagal.
        Thread-safe: video_id yang sama tidak akan didownload dua kali.
        """
        # ── File lokal — langsung pakai ──────────────────
        if os.path.isfile(video_url):
            log.info(f"[DL:{video_id}] Local file: {video_url}")
            return video_url

        cached = self._cache_path(video_id)

        with self._lock:
            # Double-check setelah acquire lock
            if os.path.exists(cached) and os.path.getsize(cached) > 10_000:
                log.info(f"[DL:{video_id}] Cache hit: {cached} "
                         f"({os.path.getsize(cached)//1_000_000}MB)")
                return cached

            # Hapus partial download jika ada
            if os.path.exists(cached):
                os.remove(cached)

            if self._is_youtube(video_url):
                return self._download_youtube(video_id, video_url, cached)
            else:
                return self._download_http(video_id, video_url, cached)

    def evict(self, video_id: str):
        """Hapus cache untuk video_id tertentu."""
        cached = self._cache_path(video_id)
        if os.path.exists(cached):
            os.remove(cached)
            log.info(f"[DL:{video_id}] Cache evicted: {cached}")

    def _cache_path(self, video_id: str) -> str:
        # Sanitasi video_id agar aman sebagai filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_id)
        return os.path.join(DOWNLOAD_FOLDER, f"{safe_id}.mp4")

    @staticmethod
    def _is_youtube(url: str) -> bool:
        return "youtube.com" in url or "youtu.be" in url

    def _download_youtube(self, video_id: str, url: str, out_path: str) -> "str | None":
        if shutil.which("yt-dlp") is None:
            log.error("[DL] yt-dlp tidak ditemukan. Install: pip install yt-dlp")
            return None

        log.info(f"[DL:{video_id}] YouTube download dimulai...")

        # Gunakan direktori tmp terpisah agar mudah scan hasil file
        tmp_dir  = out_path + "_tmpdir"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_tmpl = os.path.join(tmp_dir, "video.%(ext)s")

        cmd = [
            "yt-dlp",
            "-f", (
                f"bestvideo[height<={YT_MAX_HEIGHT}][ext=mp4]"
                f"+bestaudio[ext=m4a]"
                f"/bestvideo[height<={YT_MAX_HEIGHT}]+bestaudio"
                f"/best[height<={YT_MAX_HEIGHT}]"
            ),
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--no-part",           # jangan buat file .part
            "--newline",           # progress per baris (mudah di-parse)
            "-o", tmp_tmpl,
            url
        ]

        log.debug(f"[DL:{video_id}] cmd: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=DOWNLOAD_TIMEOUT_YT
            )

            # Selalu log stderr untuk debug
            if result.stderr.strip():
                log.debug(f"[DL:{video_id}] yt-dlp stderr:\n{result.stderr[-800:]}")
            if result.stdout.strip():
                log.debug(f"[DL:{video_id}] yt-dlp stdout:\n{result.stdout[-400:]}")

            if result.returncode != 0:
                log.error(f"[DL:{video_id}] yt-dlp exit={result.returncode}")
                log.error(f"[DL:{video_id}] stderr:\n{result.stderr[-800:]}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None

            # Scan semua file hasil di tmp_dir
            found_files = [
                os.path.join(tmp_dir, f)
                for f in os.listdir(tmp_dir)
                if os.path.isfile(os.path.join(tmp_dir, f))
            ]

            log.debug(f"[DL:{video_id}] Files di tmp_dir: {found_files}")

            if not found_files:
                log.error(f"[DL:{video_id}] yt-dlp selesai tapi tidak ada file di {tmp_dir}")
                log.error(f"[DL:{video_id}] stdout: {result.stdout[-400:]}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None

            # Pilih file terbesar (hasil merge biasanya paling besar)
            actual_file = max(found_files, key=os.path.getsize)
            size_mb     = os.path.getsize(actual_file) / 1_000_000

            log.info(f"[DL:{video_id}] File ditemukan: {os.path.basename(actual_file)} "
                     f"({size_mb:.1f}MB)")

            if size_mb > DOWNLOAD_MAX_SIZE_MB:
                log.warning(f"[DL:{video_id}] File terlalu besar: {size_mb:.1f}MB "
                             f"(limit {DOWNLOAD_MAX_SIZE_MB}MB) — skip")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None

            # Pindahkan ke out_path final
            shutil.move(actual_file, out_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)

            log.info(f"[DL:{video_id}] YouTube OK: {size_mb:.1f}MB → {out_path}")
            return out_path

        except subprocess.TimeoutExpired:
            log.error(f"[DL:{video_id}] YouTube timeout ({DOWNLOAD_TIMEOUT_YT}s)")
        except Exception as e:
            log.error(f"[DL:{video_id}] YouTube error: {e}", exc_info=True)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(out_path):
            os.remove(out_path)
        return None

    def _download_http(self, video_id: str, url: str, out_path: str) -> "str | None":
        log.info(f"[DL:{video_id}] HTTP download: {url[:80]}...")
        tmp_path = out_path + ".tmp"

        try:
            with requests.get(url, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()

                total_bytes = int(r.headers.get("content-length", 0))
                max_bytes   = DOWNLOAD_MAX_SIZE_MB * 1_000_000

                if total_bytes and total_bytes > max_bytes:
                    log.warning(f"[DL:{video_id}] Content-Length {total_bytes//1_000_000}MB "
                                 f"melebihi limit {DOWNLOAD_MAX_SIZE_MB}MB — skip")
                    return None

                downloaded = 0
                last_log   = 0

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):  # 512KB
                        if shutdown_event.is_set():
                            log.info(f"[DL:{video_id}] Download dibatalkan (shutdown)")
                            return None

                        f.write(chunk)
                        downloaded += len(chunk)

                        if downloaded > max_bytes:
                            log.warning(f"[DL:{video_id}] Download melebihi size limit, stop")
                            return None

                        if total_bytes and time.time() - last_log > 5:
                            pct = downloaded / total_bytes * 100
                            log.info(f"[DL:{video_id}] {pct:.0f}% "
                                     f"({downloaded//1_000_000}MB/{total_bytes//1_000_000}MB)")
                            last_log = time.time()

            os.rename(tmp_path, out_path)
            size_mb = os.path.getsize(out_path) / 1_000_000
            log.info(f"[DL:{video_id}] HTTP OK: {size_mb:.1f}MB → {out_path}")
            return out_path

        except requests.exceptions.Timeout:
            log.error(f"[DL:{video_id}] HTTP timeout")
        except requests.exceptions.ConnectionError as e:
            log.error(f"[DL:{video_id}] HTTP connection error: {e}")
        except requests.exceptions.HTTPError as e:
            log.error(f"[DL:{video_id}] HTTP error {e.response.status_code}: {url}")
        except Exception as e:
            log.error(f"[DL:{video_id}] HTTP unexpected error: {e}")

        for p in (tmp_path, out_path):
            if os.path.exists(p):
                os.remove(p)
        return None


download_manager = DownloadManager()

# =========================================================
# ================= FRAME VIDEO WRITER ====================
# =========================================================

class FrameVideoWriter:
    """
    Kumpulkan frame JPEG di temp dir, render ke MP4 via ffmpeg.
    Digunakan di _finalize() setelah worker selesai.
    """

    def __init__(self, video_id: str):
        self.video_id  = video_id
        self.frame_idx = 0
        self.tmp_dir   = tempfile.mkdtemp(prefix=f"frames_{video_id}_")
        self._lock     = Lock()
        log.debug(f"[WRITER:{video_id}] tmp_dir={self.tmp_dir}")

    def write(self, frame: np.ndarray):
        with self._lock:
            path = os.path.join(self.tmp_dir, f"{self.frame_idx:07d}.jpg")
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.frame_idx += 1

    def render(self) -> "str | None":
        with self._lock:
            count = self.frame_idx

        if count == 0:
            log.info(f"[WRITER:{self.video_id}] Tidak ada frame, skip render")
            self._cleanup()
            return None

        ts       = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(RESULT_FOLDER, f"result_{self.video_id}_{ts}.mp4")

        log.info(f"[WRITER:{self.video_id}] Rendering {count} frame → {out_path}")

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
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                log.error(f"[WRITER:{self.video_id}] FFmpeg error:\n{result.stderr}")
                return None
            log.info(f"[WRITER:{self.video_id}] Render OK: {out_path}")
            return out_path
        except subprocess.TimeoutExpired:
            log.error(f"[WRITER:{self.video_id}] FFmpeg timeout")
            return None
        except FileNotFoundError:
            log.error("[WRITER] ffmpeg tidak ditemukan. Install: sudo apt install ffmpeg")
            return None
        finally:
            self._cleanup()

    def _cleanup(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        log.debug(f"[WRITER:{self.video_id}] Temp cleaned")


# =========================================================
# ================= WEBHOOK WORKER (face events) ==========
# =========================================================

def webhook_worker():
    """Thread pool kirim face detection event ke WEBHOOK_API."""
    while not shutdown_event.is_set():
        try:
            item = webhook_queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            webhook_queue.task_done()
            break

        face_bytes, frame_bytes, face_name, frame_name, \
            ts_iso, bbox, video_id, client_id, score = item

        try:
            if not WEBHOOK_API:
                log.debug("[WEBHOOK] WEBHOOK_API tidak di-set, skip face event")
                continue

            resp = requests.post(
                WEBHOOK_API + "/webhook/detection-video",
                files=[
                    ("files", (face_name,  face_bytes,  "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data={
                    "timestamp":  ts_iso,
                    "bbox":       f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "confidence": round(score, 4),
                    "video_id":   video_id,
                    "client_id":  client_id,
                },
                timeout=10
            )

            if resp.status_code >= 400:
                log.warning(f"[WEBHOOK FACE:{video_id}] HTTP {resp.status_code}: {resp.text[:200]}")
            else:
                log.debug(f"[WEBHOOK FACE:{video_id}] OK {resp.status_code}")

        except requests.exceptions.Timeout:
            log.warning(f"[WEBHOOK FACE:{video_id}] Timeout")
        except requests.exceptions.ConnectionError:
            log.warning(f"[WEBHOOK FACE:{video_id}] Connection error")
        except Exception as e:
            log.error(f"[WEBHOOK FACE:{video_id}] Error: {e}")
        finally:
            webhook_queue.task_done()


for _ in range(3):
    Thread(target=webhook_worker, daemon=True, name="webhook-face").start()


# =========================================================
# ================= FILE WRITER WORKER ====================
# =========================================================

def file_writer_worker():
    """Thread async untuk simpan JPEG ke disk — tidak blocking detector."""
    while not shutdown_event.is_set():
        try:
            item = file_write_queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            file_write_queue.task_done()
            break

        face_path, face_bytes, frame_path, frame_bytes = item
        try:
            with open(face_path,  "wb") as f:
                f.write(face_bytes)
            with open(frame_path, "wb") as f:
                f.write(frame_bytes)
        except OSError as e:
            log.error(f"[FILE WRITER] OS error: {e}")
        except Exception as e:
            log.error(f"[FILE WRITER] Error: {e}")
        finally:
            file_write_queue.task_done()


for _ in range(2):
    Thread(target=file_writer_worker, daemon=True, name="file-writer").start()


# =========================================================
# ================= HELPERS ===============================
# =========================================================

def iso_name(prefix: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S-%fZ")[:-4]  # ms precision
    return f"{prefix}_{ts}.jpg"


def enforce_limit(folder: str):
    """Hapus file terlama jika melebihi MAX_IMAGES."""
    try:
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))],
            key=os.path.getmtime
        )
        while len(files) > MAX_IMAGES:
            os.remove(files.pop(0))
    except Exception as e:
        log.warning(f"[ENFORCE LIMIT] {folder}: {e}")


# =========================================================
# ================= VIDEO WORKER ==========================
# =========================================================

class VideoWorker:
    """
    Alur kerja per video:
    1. DownloadManager.resolve() → local path
    2. _reader_thread()  → baca frame → frame_queue
    3. run()             → ambil dari queue → detect → save → webhook
    4. destroy()         → stop → _finalize() di thread non-daemon
    5. _finalize()       → render MP4 → upload S3 → webhook done
    """

    def __init__(self, video_id: str, client_id: str, video_url: str):
        self.video_id   = video_id
        self.client_id  = client_id
        self.video_url  = video_url
        self.running    = False
        self.cap        = None
        self.writer     = None
        self.local_path = None

        self.frame_queue    = Queue(maxsize=60)  # ~5 detik buffer di 12fps
        self.frame_interval = 1.0 / FRAME_FPS
        self.last_time      = 0.0
        self.last_save      = 0.0
        self.face_last_time: dict[tuple, float] = {}
        self.bad_frame_count = 0
        self.MAX_BAD_FRAMES  = 30

        self._reader_done = Event()

        # ── Download video ────────────────────────────────
        log.info(f"[{video_id}] Resolving: {video_url[:80]}...")
        self.local_path = download_manager.resolve(video_id, video_url)

        if self.local_path is None:
            log.error(f"[{video_id}] Download gagal, worker tidak dijalankan")
            return

        # ── Open capture dari file lokal ─────────────────
        self.cap = cv2.VideoCapture(self.local_path)
        if not self.cap.isOpened():
            log.error(f"[{video_id}] Tidak bisa buka: {self.local_path}")
            self.local_path = None
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

        # Log info video
        fps    = self.cap.get(cv2.CAP_PROP_FPS)
        total  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"[{video_id}] Video info: {width}x{height} @ {fps:.1f}fps, "
                 f"{total} frames ({total/max(fps,1)/60:.1f} menit)")

        # ── Result writer (annotated MP4) ─────────────────
        # self.writer = FrameVideoWriter(video_id)

        # ── YuNet face detector ───────────────────────────
        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH, "", (DETECT_W, DETECT_H),
            score_threshold=SCORE_THRESHOLD
        )
        self.detector.setInputSize((DETECT_W, DETECT_H))
        # Warmup inference
        self.detector.detect(np.zeros((DETECT_H, DETECT_W, 3), dtype=np.uint8))

        self.running = True
        log.info(f"[{video_id}] Worker siap ✓")

    # ──────────────────────────────────────────────────────
    # READER THREAD — hanya baca frame, tidak ada logic lain
    # ──────────────────────────────────────────────────────

    def _reader_thread(self):
        log.info(f"[READER:{self.video_id}] Thread started")
        consecutive_bad = 0

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()

            if not ret:
                consecutive_bad += 1
                if consecutive_bad >= self.MAX_BAD_FRAMES:
                    log.info(f"[READER:{self.video_id}] EOF / stream ended "
                             f"(bad={consecutive_bad})")
                    break
                time.sleep(0.01)
                continue

            consecutive_bad = 0

            # Jika queue penuh → drop frame lama (prioritaskan frame terbaru)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    log.debug(f"[READER:{self.video_id}] Frame dropped (queue full)")
                except Empty:
                    pass

            self.frame_queue.put(frame)

        # Kirim sinyal EOF ke detector thread
        self.frame_queue.put(None)
        self._reader_done.set()
        log.info(f"[READER:{self.video_id}] Thread selesai")

    # ──────────────────────────────────────────────────────
    # MAIN RUN — detector thread
    # ──────────────────────────────────────────────────────

    def run(self):
        if not self.running:
            log.warning(f"[{self.video_id}] Worker tidak siap, skip run()")
            return

        # Start reader thread
        Thread(
            target=self._reader_thread,
            daemon=True,
            name=f"reader-{self.video_id}"
        ).start()

        log.info(f"[{self.video_id}] Detector loop started")
        frames_processed = 0
        frames_skipped   = 0

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=15)
            except Empty:
                log.warning(f"[{self.video_id}] Frame queue timeout (15s)")
                break

            if frame is None:
                log.info(f"[{self.video_id}] EOF diterima dari reader")
                if VIDEO_LOOP and self.cap is not None:
                    log.info(f"[{self.video_id}] Loop — rewind ke frame 0")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._reader_done.clear()
                    Thread(
                        target=self._reader_thread,
                        daemon=True,
                        name=f"reader-{self.video_id}-loop"
                    ).start()
                    continue
                break

            # ── Throttle FPS ──────────────────────────────
            now = time.time()
            if now - self.last_time < self.frame_interval:
                frames_skipped += 1
                continue
            self.last_time = now

            self._process_frame(frame)
            frames_processed += 1

        log.info(f"[{self.video_id}] Loop selesai. "
                 f"Processed={frames_processed}, Skipped={frames_skipped}")

        self.destroy(reason="done")
        with camera_lock:
            active_cameras.pop(self.video_id, None)

    # ──────────────────────────────────────────────────────
    # PROCESS FRAME — detect + annotate + save
    # ──────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]

        # ── Resize ke fixed detection size ────────────────
        small = cv2.resize(frame, (DETECT_W, DETECT_H),
                           interpolation=cv2.INTER_LINEAR)
        sx = orig_w / DETECT_W
        sy = orig_h / DETECT_H

        _, faces = self.detector.detect(small)

        view = frame.copy()

        if faces is not None:
            for f_data in faces:
                score = float(f_data[4])
                if score < SCORE_THRESHOLD:
                    continue

                # Scale koordinat ke original frame
                x  = int(f_data[0] * sx)
                y  = int(f_data[1] * sy)
                fw = int(f_data[2] * sx)
                fh = int(f_data[3] * sy)

                if fw < MIN_SIZE_CAPTURE:
                    continue

                # Clamp ke boundary frame
                x  = max(0, x)
                y  = max(0, y)
                fw = min(fw, orig_w - x)
                fh = min(fh, orig_h - y)

                if fw <= 0 or fh <= 0:
                    continue

                # Annotasi
                cv2.rectangle(view, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                # cv2.putText(
                #     view, f"{score:.2f}", (x, max(0, y-5)),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                #     cv2.LINE_AA
                # )

                # ── Throttle save per spatial bucket ─────
                bucket = (x // 150, y // 150)
                now    = time.time()
                if bucket in self.face_last_time and \
                   now - self.face_last_time[bucket] < SAVE_INTERVAL:
                    continue
                self.face_last_time[bucket] = now

                # ── Crop wajah dengan padding ─────────────
                pad_w = int(fw * CROP_PADDING)
                pad_h = int(fh * CROP_PADDING)
                cx = max(0, x - pad_w)
                cy = max(0, y - pad_h)
                cw = min(orig_w - cx, fw + pad_w * 2)
                ch = min(orig_h - cy, fh + pad_h * 2)

                if cw <= 0 or ch <= 0:
                    continue

                face_img = frame[cy:cy+ch, cx:cx+cw]

                # ── Blur check ────────────────────────────
                if BLUR_THRESHOLD > 0:
                    gray      = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if sharpness < BLUR_THRESHOLD:
                        log.debug(f"[{self.video_id}] Blur skip: {sharpness:.1f}")
                        continue

                # ── JPEG encode ───────────────────────────
                ok1, buf1 = cv2.imencode(".jpg", face_img,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                ok2, buf2 = cv2.imencode(".jpg", view,
                                         [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok1 or not ok2:
                    continue

                face_bytes  = buf1.tobytes()
                frame_bytes = buf2.tobytes()

                face_name  = iso_name("face")
                frame_name = iso_name("frame")

                # ── Async file write ──────────────────────
                if not file_write_queue.full():
                    file_write_queue.put((
                        os.path.join(FACE_FOLDER,  face_name),  face_bytes,
                        os.path.join(FRAME_FOLDER, frame_name), frame_bytes,
                    ))
                else:
                    log.warning(f"[{self.video_id}] file_write_queue penuh, skip save")

                enforce_limit(FACE_FOLDER)
                enforce_limit(FRAME_FOLDER)

                # ── Webhook face event ────────────────────
                ts_iso = datetime.utcnow().isoformat() + "Z"
                if not webhook_queue.full():
                    webhook_queue.put((
                        face_bytes, frame_bytes,
                        face_name, frame_name,
                        ts_iso, (x, y, fw, fh),
                        self.video_id, self.client_id, score
                    ))
                else:
                    log.warning(f"[{self.video_id}] webhook_queue penuh, skip event")

                log.info(f"[{self.video_id}] Face: {face_name} score={score:.3f} "
                         f"size={fw}x{fh}")
                self.last_save = now

        # ── Tulis ke annotated video ──────────────────────
        # if self.writer:
        #     self.writer.write(view)

        # ── Update preview ────────────────────────────────
        with preview_lock:
            preview_frames[self.video_id] = view

    # ──────────────────────────────────────────────────────
    # DESTROY — stop worker, spawn finalize thread
    # ──────────────────────────────────────────────────────

    def destroy(self, reason: str = "done"):
        if not self.running:
            return  # sudah di-destroy sebelumnya

        self.running = False

        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

        with preview_lock:
            preview_frames.pop(self.video_id, None)

        log.info(f"[{self.video_id}] Destroyed. Reason: {reason}")

        # daemon=False agar _finalize tidak terbunuh saat main thread exit
        Thread(
            target=self._finalize,
            args=(reason,),
            daemon=False,
            name=f"finalize-{self.video_id}"
        ).start()

    # ──────────────────────────────────────────────────────
    # FINALIZE — render → upload S3 → webhook done
    # ──────────────────────────────────────────────────────

    def _finalize(self, reason: str):
        log.info(f"[FINALIZE:{self.video_id}] Start (reason={reason})")

        s3_url     = None
        local_path = None

        # ── Step 1: Render annotated MP4 ─────────────────
        # if self.writer:
        #     try:
        #         local_path = self.writer.render()
        #         log.info(f"[FINALIZE:{self.video_id}] Render: {local_path}")
        #     except Exception as e:
        #         log.error(f"[FINALIZE:{self.video_id}] Render error: {e}")

        # ── Step 2: Upload ke S3 ──────────────────────────
        # if local_path and s3_uploader:
        #     try:
        #         filename = os.path.basename(local_path)
        #         s3_key   = f"{S3_PREFIX}/{self.client_id}/{self.video_id}/{filename}"
        #         s3_url   = s3_uploader.upload_file(local_path, s3_key)
        #     except (BotoCoreError, ClientError) as e:
        #         log.error(f"[FINALIZE:{self.video_id}] S3 boto error: {e}")
        #     except Exception as e:
        #         log.error(f"[FINALIZE:{self.video_id}] S3 error: {e}")
        # else:
        #     log.info(
        #         f"[FINALIZE:{self.video_id}] S3 skip — "
        #         f"local_path={local_path!r}, "
        #         f"s3={'ready' if s3_uploader else 'disabled'}"
        #     )

        # ── Step 3: Hapus cache download (opsional) ───────
        # Uncomment jika ingin hemat disk setelah selesai:
        # download_manager.evict(self.video_id)

        # ── Step 4: Webhook done ──────────────────────────
        self._send_done_webhook(reason, s3_url, local_path)

        log.info(f"[FINALIZE:{self.video_id}] Complete")

    def _send_done_webhook(self, reason: str, s3_url: "str | None",
                           local_path: "str | None"):
        if not WEBHOOK_API:
            log.info("[WEBHOOK DONE] WEBHOOK_API tidak di-set, skip")
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

        log.info(f"[WEBHOOK DONE:{self.video_id}] Sending status={reason} "
                 f"uploaded={s3_url is not None}")
        log.debug(f"[WEBHOOK DONE] Payload:\n{json.dumps(payload, indent=2)}")

        for attempt in range(3):
            try:
                resp = requests.post(
                    WEBHOOK_API + "/webhook/receive-status-video",
                    json=payload,
                    timeout=15
                )
                if resp.status_code < 400:
                    log.info(f"[WEBHOOK DONE:{self.video_id}] OK {resp.status_code}")
                    return
                log.warning(f"[WEBHOOK DONE:{self.video_id}] HTTP {resp.status_code}: "
                             f"{resp.text[:200]}")
            except requests.exceptions.Timeout:
                log.warning(f"[WEBHOOK DONE:{self.video_id}] Timeout (attempt {attempt+1})")
            except requests.exceptions.ConnectionError:
                log.warning(f"[WEBHOOK DONE:{self.video_id}] "
                             f"Connection error (attempt {attempt+1})")
            except Exception as e:
                log.error(f"[WEBHOOK DONE:{self.video_id}] Error: {e}")

            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s retry backoff

        log.error(f"[WEBHOOK DONE:{self.video_id}] Gagal setelah 3 percobaan")


# =========================================================
# ================= DEBUG VIDEO LOADER ====================
# =========================================================

def load_debug_sources(folder: str) -> list[dict]:
    files: list[str] = []
    for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
        files += glob.glob(os.path.join(folder, ext))

    sources = []
    for i, path in enumerate(sorted(files)):
        sources.append({
            "video_id":  f"debug-{i:03d}",
            "client_id": "debug-client",
            "video_url": path
        })

    log.info(f"[DEBUG] Found {len(sources)} file(s) di {folder}")
    return sources


# =========================================================
# ================= FETCH VIDEO LIST ======================
# =========================================================

def fetch_videos() -> list[dict]:
    if not WEBHOOK_API:
        log.warning("[fetch_videos] WEBHOOK_API tidak di-set")
        return []
    try:
        r = requests.get(
            WEBHOOK_API + "/webhook/videos",
            timeout=10
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("ok"):
            return payload.get("data", [])
        log.warning(f"[fetch_videos] ok=false: {payload}")
        return []
    except requests.exceptions.Timeout:
        log.warning("[fetch_videos] Timeout")
        return []
    except requests.exceptions.ConnectionError as e:
        log.warning(f"[fetch_videos] Connection error: {e}")
        return []
    except Exception as e:
        log.error(f"[fetch_videos] Error: {e}")
        return []


# =========================================================
# ================= VIDEO MANAGER =========================
# =========================================================

def video_manager():
    """
    Loop poll sumber video setiap CAMERA_REFRESH_INTERVAL detik.
    - Tambah worker untuk video baru
    - Hentikan worker untuk video yang dihapus dari list
    """
    log.info("[MANAGER] Thread started")

    while not shutdown_event.is_set():
        sources = load_debug_sources(DEBUG_VIDEO_FOLDER) if DEBUG_MODE \
                  else fetch_videos()

        log.info(f"[MANAGER] {len(sources)} video(s) ditemukan")

        with camera_lock:
            existing    = set(active_cameras.keys())
            current_ids = {c["video_id"] for c in sources}

            # ── Start worker baru ─────────────────────────
            for c in sources:
                vid = c["video_id"]
                if vid not in existing:
                    log.info(f"[MANAGER] Starting worker: {vid}")
                    try:
                        w = VideoWorker(vid, c["client_id"], c["video_url"])
                        if w.running:
                            Thread(
                                target=w.run,
                                daemon=True,
                                name=f"worker-{vid}"
                            ).start()
                            active_cameras[vid] = w
                            log.info(f"[MANAGER] Worker started: {vid}")
                        else:
                            log.error(f"[MANAGER] Worker gagal init: {vid}")
                    except Exception as e:
                        log.error(f"[MANAGER] Error starting {vid}: {e}")

            # ── Stop worker yang sudah tidak ada di list ──
            for vid in list(existing):
                if vid not in current_ids:
                    log.info(f"[MANAGER] Stopping removed video: {vid}")
                    try:
                        active_cameras[vid].destroy(reason="removed")
                    except Exception as e:
                        log.warning(f"[MANAGER] Error destroying {vid}: {e}")
                    del active_cameras[vid]

        shutdown_event.wait(timeout=CAMERA_REFRESH_INTERVAL)

    log.info("[MANAGER] Thread stopped")


Thread(target=video_manager, daemon=True, name="video-manager").start()
log.info("System ready — waiting for frames...")

# =========================================================
# ================= GRACEFUL SHUTDOWN =====================
# =========================================================

def _shutdown_handler(signum, frame):
    log.info(f"[SHUTDOWN] Signal {signum} diterima, stopping...")
    shutdown_event.set()

    with camera_lock:
        for vid, worker in list(active_cameras.items()):
            log.info(f"[SHUTDOWN] Stopping worker: {vid}")
            worker.destroy(reason="shutdown")

    # Drain queues
    webhook_queue.put(None)
    file_write_queue.put(None)


signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

# =========================================================
# ================= MAIN DISPLAY LOOP =====================
# =========================================================

if ENABLE_VIEW:
    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    log.info(f"[DISPLAY] Window {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

    while not shutdown_event.is_set():
        with preview_lock:
            frames = list(preview_frames.values())

        if not frames:
            time.sleep(0.05)
            # Tampilkan layar hitam jika belum ada frame
            blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for video...",
                        (DISPLAY_WIDTH//2 - 120, DISPLAY_HEIGHT//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            cv2.imshow("Face Detection", blank)
            if cv2.waitKey(100) == ord("q"):
                break
            continue

        n   = len(frames)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        tw   = DISPLAY_WIDTH  // cols
        th   = DISPLAY_HEIGHT // rows

        # Resize semua tile
        tiles = [cv2.resize(f, (tw, th), interpolation=cv2.INTER_LINEAR)
                 for f in frames]

        # Pad ke rows*cols jika perlu
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((th, tw, 3), dtype=np.uint8))

        # Susun grid
        grid_rows = []
        for r in range(rows):
            row_tiles = tiles[r * cols: (r + 1) * cols]
            grid_rows.append(cv2.hconcat(row_tiles))
        grid = cv2.vconcat(grid_rows)

        cv2.imshow("Face Detection", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            log.info("[DISPLAY] 'q' ditekan, shutdown...")
            _shutdown_handler(0, None)
            break

    cv2.destroyAllWindows()

else:
    log.info("[HEADLESS] Running tanpa display. Ctrl+C untuk stop.")
    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown_handler(signal.SIGINT, None)

log.info("=== APP STOPPED ===")