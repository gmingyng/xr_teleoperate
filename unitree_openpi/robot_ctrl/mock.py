"""Deterministic offline plant; never imports SDK/Pinocchio or uses DDS."""

import time
import numpy as np

from ..utils.transforms import matrix_pose, pose_matrix, vector


class MockKinematics:
    frame_id = "mock_base"

    def update_state(self, state):
        pass

    def fk(self, q):
        # Test-only mapping, not G1-D geometry.
        return {side: pose_matrix(np.r_[q[start:start+3] + [0.3, sign*0.2, 0.1],
                                        q[start+3:start+6]])
                for side, start, sign in (("left", 0, 1), ("right", 7, -1))}

    def solve(self, targets, seed):
        q = vector(seed, 14)
        for side, target in targets.items():
            start, sign = (0, 1) if side == "left" else (7, -1)
            pose = matrix_pose(target)
            q[start:start+3] = pose[:3] - [0.3, sign*0.2, 0.1]
            q[start+3:start+6] = pose[3:]
        return q

    def gravity(self, q):
        return np.zeros(14)


class MockBackend:
    is_simulation = True

    def __init__(self, config):
        self.config, self.q, self.grip = config, np.zeros(14), np.zeros(2)
        self.connected = self.acquired = self.stopped = False
        self.commands = []

    def connect(self):
        self.connected = True

    def read(self):
        if not self.connected:
            raise RuntimeError("mock backend is disconnected")
        return dict(q14=self.q.copy(), dq14=np.zeros(14), waist_q3=np.zeros(3),
                    waist_dq3=np.zeros(3), grip_q2=self.grip.copy(), grip_dq2=np.zeros(2),
                    received_monotonic_s=time.monotonic(), received_at=time.time())

    def acquire(self, grip_q2, waist_q3, tau14):
        self.acquired = True

    def command(self, q14, tau14, grip_q2=None):
        if not self.acquired or self.stopped:
            raise RuntimeError("mock backend is not acquired")
        self.q = vector(q14, 14)
        if grip_q2 is not None:
            self.grip = vector(grip_q2, 2)
        self.commands.append(self.q.copy())

    def freeze(self):
        pass

    def hold(self):
        self.stopped = True

    def health(self):
        return {"simulation": True, "acquired": self.acquired, "stopped": self.stopped}

    def close(self):
        self.connected = False
