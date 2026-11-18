"""Optional G1-D AGV velocity stream and watchdog, extracted from navigation.

Not invoked by the EEF/VLA runtime. Initialize DDS once in the calling process
before constructing this controller. No synthesized odometry is reported as
measured absolute robot state.
"""

import threading
import time


class ChassisController:
    def __init__(self, client=None, hz=20.0, watchdog_s=0.3):
        if hz <= 0 or watchdog_s <= 0:
            raise ValueError("hz and watchdog_s must be positive")
        if client is None:
            from .agv import AgvClient
            client = AgvClient()
            client.SetTimeout(0.3)
            client.Init()
        self.client, self.period, self.watchdog_s = client, 1.0 / hz, watchdog_s
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._command, self._received = None, None
        self.error = None
        self._thread = threading.Thread(target=self._loop, name="g1d-agv", daemon=True)
        self._thread.start()

    def send_command(self, vx=0.0, vyaw=0.0):
        import math
        vx, vyaw = float(vx), float(vyaw)
        if not all(math.isfinite(v) for v in (vx, vyaw)) or abs(vx) > 1.5 or abs(vyaw) > 0.6:
            raise ValueError("invalid AGV velocity")
        with self._lock:
            self._command, self._received = (vx, vyaw), time.monotonic()

    def height_adjust(self, vz):
        return self.client.HeightAdjust(vz)

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                command, received = self._command, self._received
            if command is not None:
                if time.monotonic() - received > self.watchdog_s:
                    command = (0.0, 0.0)
                try:
                    result = self.client.Move(command[0], 0.0, command[1])
                    code = result[0] if isinstance(result, tuple) else result
                    if code not in (0, None):
                        raise RuntimeError(f"AGV returned {code}")
                except Exception as exc:
                    self.error = str(exc)
                    with self._lock:
                        self._command = (0.0, 0.0)
                        self._received = time.monotonic()
            self._stop.wait(self.period)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        try:
            self.client.Move(0.0, 0.0, 0.0)
        finally:
            self.client.HeightAdjust(0.0)
