# =========================================================
# MULTI VIDEO FACE CAPTURE (MERGED VERSION)
# Detection Engine : CODE 1
# Video Pipeline   : CODE 2
# Source           : URL / S3 / Local file
# =========================================================

import cv2
import time
import os
import math
import requests
import numpy as np
import json
import subprocess
import shutil
import tempfile
import boto3

from datetime import datetime
from dotenv import load_dotenv
from threading import Thread, Lock
from queue import Queue
from botocore.exceptions import BotoCoreError, ClientError

# ================= PERFORMANCE =================

cv2.setNumThreads(1)
cv2.setUseOptimized(True)

load_dotenv()

# =========================================================
# ================= ENV CONFIG =============================
# =========================================================

SERVICE_ID = os.getenv("SERVICE_ID")
MODEL_PATH = os.getenv("MODEL_PATH")

FRAME_FPS = int(os.getenv("FRAME_FPS", 12))

SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.35))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", 0))

SAVE_FOLDER = os.getenv("SAVE_FOLDER", "image_face")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 150))

TARGET_MAX_WIDTH = int(os.getenv("RESIZE_WIDTH", 960))
MIN_SIZE_CAPTURE = int(os.getenv("MIN_SIZE_CAPTURE", 0))
MAX_FACE_SIZE = int(os.getenv("MAX_FACE_SIZE", 800))

CROP_PADDING = float(os.getenv("CROP_PADDING", 0.35))

WEBHOOK_API = os.getenv("WEBHOOK_API")

ENABLE_VIEW = os.getenv("ENABLE_VIEW", "true").lower() == "true"
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", 1200))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", 800))

CAMERA_REFRESH_INTERVAL = int(os.getenv("CAMERA_REFRESH_INTERVAL", 60))

# ================= ANTI SPAM =================

FACE_COOLDOWN = float(os.getenv("FACE_COOLDOWN", 8))
FACE_MOVE_THRESHOLD = int(os.getenv("FACE_MOVE_THRESHOLD", 80))
FACE_BUCKET_SIZE = int(os.getenv("FACE_BUCKET_SIZE", 220))

# ================= S3 =================

S3_ENABLED = os.getenv("S3_ENABLED", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
S3_PREFIX = os.getenv("S3_PREFIX")

S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_PUBLIC_BASE = os.getenv("S3_PUBLIC_BASE")

# =========================================================
# ================= FOLDER ================================
# =========================================================

FACE_FOLDER = os.path.join(SAVE_FOLDER, "face")
FRAME_FOLDER = os.path.join(SAVE_FOLDER, "frame")
RESULT_FOLDER = os.path.join(SAVE_FOLDER, "result")

os.makedirs(FACE_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# =========================================================
# ================= GLOBAL ================================
# =========================================================

preview_frames = {}
preview_lock = Lock()

active_workers = {}
camera_lock = Lock()

webhook_queue = Queue(maxsize=500)

# =========================================================
# ================= UTIL ==================================
# =========================================================

def iso_name(prefix):
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.jpg"


def enforce_limit(folder):

    files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)],
        key=os.path.getmtime
    )

    while len(files) > MAX_IMAGES:
        os.remove(files.pop(0))


def calc_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


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


# =========================================================
# ================= S3 UPLOADER ===========================
# =========================================================

class S3Uploader:

    def __init__(self):

        self.client = boto3.client(
            "s3",
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            endpoint_url=S3_ENDPOINT_URL
        )

    def upload(self, file_path, key):

        self.client.upload_file(file_path, S3_BUCKET, key)

        if S3_PUBLIC_BASE:
            return f"{S3_PUBLIC_BASE}/{key}"

        return key


s3 = S3Uploader() if S3_ENABLED else None


# =========================================================
# ================= VIDEO WRITER ==========================
# =========================================================

class FrameVideoWriter:

    def __init__(self, video_id):

        self.video_id = video_id
        self.tmp_dir = tempfile.mkdtemp(prefix=f"frames_{video_id}_")
        self.idx = 0

    def write(self, frame):

        path = os.path.join(self.tmp_dir, f"{self.idx:07d}.jpg")
        cv2.imwrite(path, frame)

        self.idx += 1

    def render(self):

        if self.idx == 0:
            return None

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        out = os.path.join(
            RESULT_FOLDER,
            f"result_{self.video_id}_{ts}.mp4"
        )

        cmd = [
            "ffmpeg","-y",
            "-framerate","12",
            "-i", os.path.join(self.tmp_dir,"%07d.jpg"),
            "-c:v","libx264",
            "-pix_fmt","yuv420p",
            out
        ]

        subprocess.run(cmd)

        shutil.rmtree(self.tmp_dir)

        return out


# =========================================================
# ================= WEBHOOK WORKER ========================
# =========================================================

def webhook_worker():

    while True:

        item = webhook_queue.get()

        if item is None:
            break

        face_bytes, frame_bytes, face_name, frame_name, bbox, score, vid, client = item

        payload = {
            "video_id": vid,
            "client_id": client,
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "confidence": score
        }

        try:

            requests.post(
                WEBHOOK_API + "/webhook/detection-video",
                files=[
                    ("files",(face_name,face_bytes,"image/jpeg")),
                    ("files",(frame_name,frame_bytes,"image/jpeg"))
                ],
                data=payload,
                timeout=10
            )

        except Exception as e:
            print("WEBHOOK ERROR",e)

        webhook_queue.task_done()


for _ in range(3):
    Thread(target=webhook_worker,daemon=True).start()


# =========================================================
# ================= VIDEO WORKER ==========================
# =========================================================

class VideoWorker:

    def __init__(self, video_id, client_id, url):

        self.video_id = video_id
        self.client_id = client_id
        self.url = url

        self.cap = cv2.VideoCapture(url)

        self.running = True

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH,"",(640,640),
            score_threshold=SCORE_THRESHOLD
        )

        self.face_memory = {}

        self.writer = FrameVideoWriter(video_id)

        print("VIDEO START",video_id)

    def run(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                break

            resized, scale = self.resize(frame)

            h,w = resized.shape[:2]

            self.detector.setInputSize((w,h))

            _,faces = self.detector.detect(resized)

            view = frame.copy()

            if faces is not None:

                for f in faces:

                    score = float(f[14])

                    if score < SCORE_THRESHOLD:
                        continue

                    x,y,fw,fh = f[:4].astype(int)

                    x = int(x*scale)
                    y = int(y*scale)
                    fw = int(fw*scale)
                    fh = int(fh*scale)

                    if fw < MIN_SIZE_CAPTURE:
                        continue

                    cv2.rectangle(view,(x,y),(x+fw,y+fh),(0,255,0),2)

                    cx = x + fw//2
                    cy = y + fh//2

                    bucket = (cx//FACE_BUCKET_SIZE, cy//FACE_BUCKET_SIZE)

                    now = time.time()

                    if bucket in self.face_memory:

                        last,last_pos = self.face_memory[bucket]

                        dist = math.hypot(cx-last_pos[0],cy-last_pos[1])

                        if now-last < FACE_COOLDOWN:
                            continue

                        if dist < FACE_MOVE_THRESHOLD:
                            continue

                    self.face_memory[bucket]=(now,(cx,cy))

                    cx2,cy2,cw,ch = expand_crop_bbox(
                        x,y,fw,fh,
                        frame.shape[1],
                        frame.shape[0]
                    )

                    face_img = frame[cy2:cy2+ch, cx2:cx2+cw]

                    if calc_sharpness(face_img) < BLUR_THRESHOLD:
                        continue

                    face_name = iso_name("face")
                    frame_name = iso_name("frame")

                    fb1=cv2.imencode(".jpg",face_img)[1].tobytes()
                    fb2=cv2.imencode(".jpg",view)[1].tobytes()

                    open(os.path.join(FACE_FOLDER,face_name),"wb").write(fb1)
                    open(os.path.join(FRAME_FOLDER,frame_name),"wb").write(fb2)

                    enforce_limit(FACE_FOLDER)
                    enforce_limit(FRAME_FOLDER)

                    webhook_queue.put(
                        (fb1,fb2,face_name,frame_name,(x,y,fw,fh),score,self.video_id,self.client_id)
                    )

            self.writer.write(view)

            with preview_lock:
                preview_frames[self.video_id]=view

        self.finish()

    def resize(self,frame):

        h,w=frame.shape[:2]

        scale=min(1.0,TARGET_MAX_WIDTH/w)

        resized=cv2.resize(frame,None,fx=scale,fy=scale)

        return resized,1/scale


    def finish(self):

        print("VIDEO FINISHED",self.video_id)

        path=self.writer.render()

        s3_url=None

        if path and s3:

            key=f"{S3_PREFIX}/{self.client_id}/{self.video_id}/{os.path.basename(path)}"

            s3_url=s3.upload(path,key)

        payload={
            "video_id":self.video_id,
            "client_id":self.client_id,
            "status":"done",
            "s3_url":s3_url
        }

        requests.post(
            WEBHOOK_API+"/webhook/receive-status-video",
            json=payload
        )


# =========================================================
# ================= VIDEO MANAGER =========================
# =========================================================

def fetch_videos():

    r=requests.get(WEBHOOK_API+"/webhook/videos")

    data=r.json()

    return data["data"] if data["ok"] else []


def video_manager():

    while True:

        sources=fetch_videos()

        with camera_lock:

            existing=set(active_workers.keys())

            for s in sources:

                vid=s["video_id"]

                if vid not in existing:

                    w=VideoWorker(
                        vid,
                        s["client_id"],
                        s["video_url"]
                    )

                    Thread(target=w.run,daemon=True).start()

                    active_workers[vid]=w

        time.sleep(30)


Thread(target=video_manager,daemon=True).start()

# =========================================================
# ================= DISPLAY ===============================
# =========================================================

if ENABLE_VIEW:

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    while True:

        with preview_lock:
            frames=list(preview_frames.values())

        if not frames:
            time.sleep(0.1)
            continue

        rows=math.ceil(math.sqrt(len(frames)))
        cols=math.ceil(len(frames)/rows)

        tw=DISPLAY_WIDTH//cols
        th=DISPLAY_HEIGHT//rows

        imgs=[cv2.resize(f,(tw,th)) for f in frames]

        while len(imgs)<rows*cols:
            imgs.append(np.zeros((th,tw,3),dtype=np.uint8))

        grid=[]
        idx=0

        for r in range(rows):
            grid.append(cv2.hconcat(imgs[idx:idx+cols]))
            idx+=cols

        cv2.imshow("Face Detection",cv2.vconcat(grid))

        if cv2.waitKey(1)==ord("q"):
            break

    cv2.destroyAllWindows()

else:

    while True:
        time.sleep(1)