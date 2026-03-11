import cv2
import time
import os
import math
import requests
import numpy as np
import shutil
import json

from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue


# ================= PERFORMANCE =================

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

load_dotenv()


# ================= ENV =================

VIDEO_ENDPOINT = os.getenv("VIDEO_ENDPOINT")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

MODEL_PATH = os.getenv("MODEL_PATH")

DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "videos")

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.35))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 0))

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
SAVE_INTERVAL = float(os.getenv("SAVE_INTERVAL", 2.0))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))

TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH", 960))
FRAME_FPS = int(os.getenv("FRAME_FPS", 10))

MIN_SIZE_CAPTURE = int(os.getenv("MIN_SIZE_CAPTURE", 0))
CROP_PADDING = float(os.getenv("CROP_PADDING", 0.35))

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 800))


# ================= FOLDER =================

FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ================= GLOBAL =================

active_videos = {}
video_lock = Lock()

preview_frames = {}
preview_lock = Lock()

queue = Queue(maxsize=300)


# ================= WEBHOOK WORKER =================

def webhook_worker():

    while True:

        item = queue.get()

        if item is None:
            break

        try:

            (
                face_bytes,
                frame_bytes,
                face_name,
                frame_name,
                bbox,
                score,
                vid,
                client_id,
            ) = item

            payload = {
                "timestamp": face_name,
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "confidence": round(score, 4),
                "video_id": vid,
                "client_id": client_id,
            }

            print("\n[WEBHOOK SEND]")
            print(json.dumps(payload, indent=4))

            requests.post(
                WEBHOOK_URL,
                files=[
                    ("files", (face_name, face_bytes, "image/jpeg")),
                    ("files", (frame_name, frame_bytes, "image/jpeg")),
                ],
                data=payload,
                timeout=10,
            )

            print("[WEBHOOK OK]", face_name)

        except Exception as e:

            print("[WEBHOOK ERROR]", e)

        finally:

            queue.task_done()


if WEBHOOK_URL:
    Thread(target=webhook_worker, daemon=True).start()


# ================= HELPERS =================

def enforce_limit(folder):

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime,
    )

    while len(files) > MAX_IMAGES:
        os.remove(files.pop(0))


def download_video(url, video_id):

    path = os.path.join(DOWNLOAD_DIR, f"{video_id}.mp4")

    if os.path.exists(path):
        return path

    try:

        print("[DOWNLOAD]", url)

        r = requests.get(url, stream=True, timeout=120)

        with open(path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        print("[DOWNLOAD OK]", path)

        return path

    except Exception as e:

        print("[DOWNLOAD ERROR]", e)
        return None


# ================= VIDEO WORKER =================

class VideoWorker:

    def __init__(self, video_id, client_id, path):

        self.video_id = video_id
        self.client_id = client_id
        self.path = path

        self.running = True

        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ===== VIDEO FPS =====

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.video_fps == 0:
            self.video_fps = 25

        self.skip_frames = int(self.video_fps / FRAME_FPS)

        if self.skip_frames < 1:
            self.skip_frames = 1

        self.frame_id = 0

        self.face_last_time = {}
        self.face_index = 0

        # ===== YUNET =====

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH,
            "",
            (320, 320),
            score_threshold=SCORE_THRESHOLD,
        )

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detector.setInputSize((640, 640))
        self.detector.detect(dummy)

        print("[VIDEO START]", video_id)


    def resize_adaptive(self, frame):

        h, w = frame.shape[:2]

        if w > TARGET_MAX_WIDTH:

            scale = TARGET_MAX_WIDTH / w
            small = cv2.resize(frame, None, fx=scale, fy=scale)

            return small, 1 / scale, 1 / scale

        return frame, 1, 1


    def run(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:

                print("[VIDEO FINISHED]", self.video_id)

                try:

                    requests.post(
                        WEBHOOK_URL,
                        json={
                            "video_id": self.video_id,
                            "client_id": self.client_id,
                            "status": "done",
                        },
                        timeout=10,
                    )

                except:
                    pass

                try:
                    os.remove(self.path)
                    print("[VIDEO DELETED]", self.path)
                except:
                    pass

                return

            # ===== FRAME SKIP =====

            self.frame_id += 1

            if self.frame_id % self.skip_frames != 0:
                continue

            view = frame.copy()

            resized, sx, sy = self.resize_adaptive(frame)

            h, w = resized.shape[:2]

            self.detector.setInputSize((w, h))

            _, faces = self.detector.detect(resized)

            if faces is not None:

                for f in faces:

                    score = float(f[14])

                    if score < SCORE_THRESHOLD:
                        continue

                    x, y, fw, fh = f[:4].astype(int)

                    x = int(x * sx)
                    y = int(y * sy)
                    fw = int(fw * sx)
                    fh = int(fh * sy)

                    cv2.rectangle(view, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

                    face_img = frame[y:y + fh, x:x + fw]

                    sharpness = cv2.Laplacian(
                        cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY),
                        cv2.CV_64F,
                    ).var()

                    if sharpness < BLUR_THRESHOLD:
                        continue

                    self.face_index += 1

                    face_name = f"{self.video_id}_{self.face_index}_face.jpg"
                    frame_name = f"{self.video_id}_{self.face_index}_frame.jpg"

                    fb1 = cv2.imencode(".jpg", face_img)[1].tobytes()
                    fb2 = cv2.imencode(".jpg", view)[1].tobytes()

                    open(os.path.join(FACE_FOLDER, face_name), "wb").write(fb1)
                    open(os.path.join(FRAME_FOLDER, frame_name), "wb").write(fb2)

                    enforce_limit(FACE_FOLDER)
                    enforce_limit(FRAME_FOLDER)

                    queue.put(
                        (
                            fb1,
                            fb2,
                            face_name,
                            frame_name,
                            (x, y, fw, fh),
                            score,
                            self.video_id,
                            self.client_id,
                        )
                    )

                    print("[FACE SAVED]", face_name)

            small = cv2.resize(view, (640, 360))

            with preview_lock:
                preview_frames[self.video_id] = small


# ================= LOAD VIDEOS =================

def load_videos():

    try:

        r = requests.get(VIDEO_ENDPOINT, timeout=10)

        data = r.json()["data"]

        videos = []

        for v in data:

            local = download_video(
                v["video_url"],
                v["video_id"],
            )

            if local:

                videos.append(
                    {
                        "video_id": v["video_id"],
                        "client_id": v["client_id"],
                        "path": local,
                    }
                )

        print("[VIDEOS FOUND]", len(videos))

        return videos

    except Exception as e:

        print("[VIDEO API ERROR]", e)

        return []


# ================= VIDEO MANAGER =================

def video_manager():

    print("\n=== VIDEO MANAGER STARTED ===")

    while True:

        vids = load_videos()

        with video_lock:

            for v in vids:

                vid = v["video_id"]

                if vid not in active_videos:

                    worker = VideoWorker(
                        vid,
                        v["client_id"],
                        v["path"],
                    )

                    Thread(target=worker.run, daemon=True).start()

                    active_videos[vid] = worker

                    print("[NEW VIDEO]", vid)

        time.sleep(30)


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

        tw = DISPLAY_WIDTH // cols
        th = DISPLAY_HEIGHT // rows

        imgs = []

        for f in frames:
            imgs.append(cv2.resize(f, (tw, th)))

        while len(imgs) < rows * cols:
            imgs.append(np.zeros((th, tw, 3), dtype=np.uint8))

        grid = []

        idx = 0

        for r in range(rows):

            row_imgs = imgs[idx : idx + cols]

            grid.append(cv2.hconcat(row_imgs))

            idx += cols

        final = cv2.vconcat(grid)

        cv2.imshow("Face Detection", final)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

else:

    while True:
        time.sleep(1)