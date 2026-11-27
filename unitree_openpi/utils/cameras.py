"""Persistent TeleImager single-JPEG ZMQ subscribers, extracted from robotskills.

All timestamps describe host receipt, never sensor exposure. RGB crops and
letterboxing are shared by real and mock observations.
"""

import threading
import time

import numpy as np


def prepare_rgb(bgr, name, settings):
    import cv2

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("camera must produce HWC BGR images")
    if name == "head" and settings["head_view"] != "full":
        if bgr.shape[1] % 2:
            raise ValueError("binocular head image width must be even")
        half = bgr.shape[1] // 2
        bgr = bgr[:, :half] if settings["head_view"] == "left" else bgr[:, half:]
    width, height = settings["image_size"]
    scale = min(width / bgr.shape[1], height / bgr.shape[0])
    resized = cv2.resize(bgr, (max(1, round(bgr.shape[1] * scale)),
                               max(1, round(bgr.shape[0] * scale))))
    result = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized[:, :, ::-1]
    return result


class CameraCapture:
    def __init__(self, settings, *, head_views=()):
        self.settings = settings
        self.head_views = {view: dict(settings, head_view=view) for view in head_views}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames, self._errors = {}, {}
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("camera capture already started")
        self._stop.clear()
        with self._lock:
            self._frames.clear()
            self._errors.clear()
        self._thread = threading.Thread(target=self._run, name="g1d-cameras", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + self.settings["startup_timeout_s"]
        while True:
            try:
                self.snapshot()
                return
            except RuntimeError:
                if time.monotonic() >= deadline:
                    self.close()
                    raise TimeoutError(f"three camera streams not ready: {self._errors}")
                self._stop.wait(0.02)

    def _run(self):
        import cv2
        import zmq

        context, sockets = zmq.Context(), {}
        poller = zmq.Poller()
        try:
            for name, endpoint in self.settings["streams"].items():
                socket = context.socket(zmq.SUB)
                sockets[socket] = name
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.CONFLATE, 1)
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.connect(endpoint)
                poller.register(socket, zmq.POLLIN)
            while not self._stop.is_set():
                for socket, event in poller.poll(50):
                    if not event & zmq.POLLIN:
                        continue
                    name, payload = sockets[socket], socket.recv()
                    received, wall = time.monotonic(), time.time()
                    try:
                        if not payload.startswith(b"\xff\xd8"):
                            raise ValueError("expected a single JPEG ZMQ message")
                        bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None:
                            raise ValueError("invalid JPEG")
                        rgb = prepare_rgb(bgr, name, self.settings)
                        extra = {}
                        if name == "head" and self.head_views:
                            extra["head_views"] = {
                                view: prepare_rgb(bgr, name, settings)
                                for view, settings in self.head_views.items()
                            }
                        with self._lock:
                            previous = self._frames.get(name)
                            seq = 1 if previous is None else previous["sequence"] + 1
                            self._frames[name] = dict(image=rgb, sequence=seq,
                                received_monotonic_s=received, received_at=wall, **extra)
                            self._errors.pop(name, None)
                    except (ValueError, cv2.error) as exc:
                        with self._lock:
                            self._errors[name] = str(exc)
        except Exception as exc:
            with self._lock:
                self._errors["transport"] = str(exc)
        finally:
            for socket in sockets:
                socket.close(0)
            context.term()

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            frames = {name: dict(frame) for name, frame in self._frames.items()}
        names = self.settings["streams"]
        missing = [name for name in names if name not in frames or
                   now - frames[name]["received_monotonic_s"] > self.settings["stale_after_s"]]
        if missing:
            raise RuntimeError(f"missing or stale cameras: {missing}")
        times = [frames[name]["received_monotonic_s"] for name in names]
        if max(times) - min(times) > self.settings["max_skew_s"]:
            raise RuntimeError("camera receipt skew exceeds max_skew_s")
        return frames

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if not self._thread.is_alive():
                self._thread = None


class MockCameras:
    def __init__(self, settings, *, head_views=()):
        self.settings, self.sequence = settings, 0
        self.head_views = tuple(head_views)

    def start(self):
        pass

    def snapshot(self):
        self.sequence += 1
        width, height = self.settings["image_size"]
        now, wall = time.monotonic(), time.time()
        frames = {}
        for index, name in enumerate(self.settings["streams"]):
            rgb = np.zeros((height, width, 3), np.uint8)
            rgb[:, :, index] = (self.sequence % 128) + 64
            frames[name] = dict(image=rgb, sequence=self.sequence,
                               received_monotonic_s=now, received_at=wall)
            if name == "head" and self.head_views:
                frames[name]["head_views"] = {view: rgb.copy() for view in self.head_views}
        return frames

    def close(self):
        pass
