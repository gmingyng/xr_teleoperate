"""Measured absolute EEF observations and bounded relative-action execution."""

import time

import numpy as np

from ..utils.transforms import matrix_pose, relative_target, vector


def parse_actions(response, settings):
    """Reject a complete malformed chunk before sending its first command."""
    if "actions" not in response:
        raise ValueError("policy response is missing actions")
    actions = np.asarray(response["actions"], dtype=float)
    width = 14 if settings["include_grippers"] else 12
    if actions.ndim == 1:
        actions = actions[None, :]
    if (actions.ndim != 2 or actions.shape[1] != width or not len(actions) or
            len(actions) > settings["max_chunk_steps"] or not np.isfinite(actions).all()):
        raise ValueError(f"actions must be a finite (T,{width}) array")
    if settings["include_grippers"] and np.any((actions[:, [6, 13]] < 0) |
                                                               (actions[:, [6, 13]] > 1)):
        raise ValueError("gripper actions must be absolute normalized values in [0,1]")
    return actions


class EEFController:
    def __init__(self, config, backend, kinematics, execute=False):
        self.config, self.backend, self.kin = config, backend, kinematics
        self.execute = execute
        self.last_action = {"status": "idle"}

    def connect(self):
        self.backend.connect()
        state, _ = self.snapshot()
        if self.execute:
            self.backend.acquire(state["grip_q2"], state["waist_q3"],
                                 self.kin.gravity(state["q14"]))

    def snapshot(self):
        state = self.backend.read()
        self.kin.update_state(state)
        poses = self.kin.fk(state["q14"])
        return state, poses

    def state_vector(self, state, poses):
        values = []
        settings = self.config["policy"]
        for index, side in enumerate(("left", "right")):
            values.extend(matrix_pose(poses[side]))
            if settings["include_grippers"]:
                closed, opened = settings["closed_q2"][index], settings["open_q2"][index]
                values.append(float(np.clip((state["grip_q2"][index]-closed)/(opened-closed), 0, 1)))
        return np.asarray(values, dtype=np.float32)

    def validate_deltas(self, actions):
        stride = 7 if self.config["policy"]["include_grippers"] else 6
        motion = self.config["motion"]
        for start in (0, stride):
            if np.any(np.linalg.norm(actions[:, start:start+3], axis=1) > motion["max_translation_m"]):
                raise ValueError("relative translation exceeds max_translation_m")
            if np.any(np.linalg.norm(actions[:, start+3:start+6], axis=1) > motion["max_rotation_rad"]):
                raise ValueError("relative rotation exceeds max_rotation_rad")

    def apply(self, action, expires_at):
        settings, motion = self.config["policy"], self.config["motion"]
        width, stride = (14, 7) if settings["include_grippers"] else (12, 6)
        action = vector(action, width, "relative EEF action")
        self.validate_deltas(action[None])
        if time.monotonic() >= expires_at:
            raise TimeoutError("action expired before IK")
        state, poses = self.snapshot()
        targets = {side: relative_target(poses[side], action[start:start+6], settings["action_frame"])
                   for side, start in (("left", 0), ("right", stride))}
        for side, target in targets.items():
            lower, upper = np.asarray(self.config["model"]["workspace"][side])
            if np.any(target[:3, 3] < lower) or np.any(target[:3, 3] > upper):
                raise ValueError(f"{side} target exceeds configured EEF workspace")
        q = self.kin.solve(targets, state["q14"])
        if np.max(np.abs(q - state["q14"])) > motion["max_joint_step_rad"]:
            raise ValueError("IK joint step exceeds max_joint_step_rad")
        tau = self.kin.gravity(q)
        if time.monotonic() >= expires_at:
            raise TimeoutError("action expired while computing IK")
        # Recheck feedback after a potentially slow numerical solve.
        latest = self.backend.read()
        if np.max(np.abs(latest["q14"] - state["q14"])) > motion["arrival_tolerance_rad"]:
            raise RuntimeError("robot moved during IK; target must be replanned")
        if np.max(np.abs(latest["waist_q3"] - state["waist_q3"])) > motion["waist_tolerance_rad"]:
            raise RuntimeError("waist moved during IK; target must be replanned")
        grip = None
        if settings["include_grippers"]:
            normalized = action[[6, 13]]
            if np.any((normalized < 0) | (normalized > 1)):
                raise ValueError("normalized gripper values must be in [0,1]")
            grip = np.array(settings["closed_q2"]) + normalized * (
                np.array(settings["open_q2"]) - settings["closed_q2"])
        if not self.execute:
            return {"status": "observing", "executed": False}
        self.backend.command(q, tau, grip)
        deadline = min(expires_at, time.monotonic() + motion["settle_timeout_s"])
        while True:
            measured = self.backend.read()
            arm_ok = np.max(np.abs(measured["q14"] - q)) <= motion["arrival_tolerance_rad"]
            grip_ok = grip is None or np.max(np.abs(measured["grip_q2"] - grip)) <= motion["grip_tolerance"]
            if arm_ok and grip_ok:
                return {"status": "reached", "executed": True,
                        "q14": measured["q14"].tolist()}
            if time.monotonic() >= deadline:
                self.backend.freeze()
                raise TimeoutError("action timed out before measured arrival")
            time.sleep(1.0 / motion["publish_hz"])

    def freeze(self):
        if self.execute:
            self.backend.freeze()

    def close(self):
        try:
            if self.execute:
                self.backend.hold()
        finally:
            self.backend.close()
