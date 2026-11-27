"""Inference review video: camera views above measured and requested joint traces."""
from collections import deque
from datetime import datetime
import logging
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from ..policy_protocol import joint_state_vector


FPS = 30
WINDOW = 10
STATE = (190, 200, 76)
ACTION = (60, 165, 245)
JOINTS = ("Shoulder pitch", "Shoulder roll", "Shoulder yaw", "Elbow",
          "Wrist roll", "Wrist pitch", "Wrist yaw", "Gripper")


def label(canvas, message, x, y, color=(222, 225, 225), size=.47):
    cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 1, cv2.LINE_AA)


class ReviewVideo:
    def __init__(self, directory, cameras, controller, prompt):
        directory = Path(directory).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / datetime.now().strftime("inference_%Y%m%d_%H%M%S_%f.mp4")
        self.cameras = cameras
        self.controller = controller
        self.prompt = prompt
        self.target = np.full(16, np.nan)
        self.stop = threading.Event()
        self.history = deque(maxlen=FPS * WINDOW + 2)

    def __enter__(self):
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 768))
        if not self.writer.isOpened():
            raise OSError(f"Could not open MP4 writer: {self.path}")
        self.thread = threading.Thread(target=self._record, name="review-video", daemon=True)
        self.thread.start()
        logging.info("Recording inference to %s", self.path)
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.writer.release()

    def _record(self):
        start = time.monotonic()
        written = 0
        try:
            while not self.stop.is_set():
                elapsed = time.monotonic() - start
                frame = self._frame(elapsed)
                due = int(elapsed * FPS) + 1
                for _ in range(due - written):
                    self.writer.write(frame)
                written = due
                self.stop.wait(max(0, start + written / FPS - time.monotonic()))
        except Exception:
            logging.exception("Recording stopped: %s", self.path)

    def _frame(self, elapsed):
        canvas = np.full((768, 1280, 3), (25, 29, 28), np.uint8)
        label(canvas, self.prompt, 10, 20)
        label(canvas, f"{elapsed:.2f} s", 1170, 20)
        try:
            frames = self.cameras.snapshot()
        except RuntimeError:
            frames = {}
        views = {}
        mapping = self.controller.config["policy"]["image_keys"]
        for name, frame in frames.items():
            views[mapping[name].split("/")[-1]] = frame["image"]
            if "right" in frame.get("head_views", {}):
                views["right_eye"] = frame["head_views"]["right"]
        for i, (key, title) in enumerate(zip(("high", "right_eye", "left_wrist", "right_wrist"),
                                              ("HEAD LEFT", "HEAD RIGHT", "WRIST LEFT", "WRIST RIGHT"))):
            x = i * 320
            label(canvas, title, x+6, 47)
            if key in views:
                rgb = views[key]
                canvas[51:291, x+40:x+280] = cv2.resize(rgb[:,:,::-1], (240,240))
            else:
                label(canvas, "Unavailable", x+100, 170)
        try:
            state = joint_state_vector(self.controller.backend.read())
        except RuntimeError:
            state = np.full(16, np.nan)
        self.history.append((elapsed, np.stack((state, self.target))))
        times = np.array([x[0] for x in self.history])
        data = np.stack([x[1] for x in self.history])
        label(canvas, "ABSOLUTE POSITION   |   arm joints: rad   |   grippers: motor position", 10, 320)
        label(canvas, "Measured", 920, 320, STATE)
        label(canvas, "Policy target", 1060, 320, ACTION)
        for j in range(16):
            side, k = divmod(j, 8)
            x, y, w = (k*182+4, 336+side*133, 174) if k < 7 else (side*640+4, 604, 628)
            label(canvas, ("R " if side else "L ")+JOINTS[k], x+4, y+16)
            v = data[:,:,j]
            finite = v[np.isfinite(v)]
            if not len(finite):
                continue
            low, high = float(finite.min()), float(finite.max())
            pad = max(.015, (high-low)*.1)
            low, high = low-pad, high+pad
            xx = x+2+(times-max(0, elapsed-WINDOW))*(w-4)/WINDOW
            for n,color in enumerate((STATE,ACTION)):
                valid = np.isfinite(v[:,n])
                yy = y+104-(v[:,n]-low)*77/(high-low)
                pts = np.column_stack((xx[valid], yy[valid])).astype(np.int32)
                if len(pts)>1:
                    cv2.polylines(canvas,[pts],False,color,1,cv2.LINE_AA)
            label(canvas,f"{high:.2f}",x+2,y+34,size=.32)
            label(canvas,f"{low:.2f}",x+2,y+115,size=.32)
        label(canvas, "Time (s)   %.1f  -  %.1f" % (max(0,elapsed-WINDOW),max(WINDOW,elapsed)), 10, 753)
        return canvas
